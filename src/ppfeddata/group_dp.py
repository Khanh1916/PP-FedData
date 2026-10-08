"""Follow-up round, item 1: group-level DP for FedDP-Marginal (a whole TCP stream, or a whole capture, is the protected unit).

Record-level epsilon protects one packet, but the packets of one stream and of one capture are correlated. With a unit that adds at most m
records to the released tables, the sensitivity of a release of T tables is D1 = m T, D2 = m sqrt(T) (`marginal.sigma_for`, `skellam_rdp`
with `rows` = m), so the same noise calibration gives epsilon for the unit.

Two requirements make the bound hold:
- every unit is held by ONE client (true in a deployment: a capture is recorded at one gateway). The row-wise Dirichlet partition of the
  study splits streams and captures over the clients, so here the partition draws whole units: per class, Dirichlet proportions over the
  clients, then each unit goes to a client with those probabilities (`unit_partition`);
- every client keeps at most m rows of each of its units (`cap`), chosen at random with a seed (the client's own data, no privacy cost).
The distributed variant releases only sums, so the released tables do not depend on which client holds which unit (O4.2); the partition
matters only for the local variant and for the time / bytes.

Units: `stream` = (capture group, TCP stream) of `train.npz`, 76,463 in the train split, 90 % with one row; `capture` = capture group, 147,
62-1,277 rows each.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UNITS = ("stream", "capture")


def unit_ids(groups: np.ndarray, streams: np.ndarray, unit: str) -> np.ndarray:
    if unit == "capture":
        return pd.factorize(np.asarray(groups))[0]
    if unit == "stream":
        return pd.factorize(pd.MultiIndex.from_arrays([np.asarray(groups), np.asarray(streams)]))[0]
    raise ValueError(f"unit must be one of {UNITS}")


def unit_partition(y: np.ndarray, units: np.ndarray, K: int, alpha: float, seed: int, min_size: int = 200, max_tries: int = 2000) -> list[np.ndarray]:
    """Row indices of K clients; every unit goes whole to one client. Per class (the majority class of the unit), Dirichlet(alpha) proportions
    over the clients, then each unit drawn to a client with them; redrawn until every client has `min_size` rows."""
    df = pd.DataFrame({"u": units, "y": y, "i": np.arange(len(y))})
    ucls = df.groupby("u")["y"].agg(lambda s: int(s.value_counts().idxmax()))
    rng = np.random.default_rng([int(seed), 6007])
    for _ in range(max_tries):
        owner = np.empty(int(units.max()) + 1, dtype=np.int64)
        for c in np.unique(ucls.values):
            us = ucls.index[ucls.values == c].to_numpy()
            owner[us] = rng.choice(K, size=len(us), p=rng.dirichlet([alpha] * K))
        cl = owner[units]
        parts = [np.flatnonzero(cl == k) for k in range(K)]
        if min(len(p) for p in parts) >= min_size:
            return parts
    raise RuntimeError(f"no unit partition with {K} clients of at least {min_size} rows after {max_tries} draws")


def cap(idx: np.ndarray, units: np.ndarray, m: int, seed: int) -> np.ndarray:
    """At most m rows of each unit among `idx` (rows of one client), chosen at random."""
    rng = np.random.default_rng([int(seed), 6011])
    idx = np.asarray(idx)
    perm = idx[rng.permutation(len(idx))]
    keep = pd.Series(units[perm]).groupby(units[perm]).cumcount().to_numpy() < int(m)
    return np.sort(perm[keep])


def capped_parts(cfg: dict[str, Any], unit: str, m: int, seed: int, K: int | None = None, alpha: float | None = None) -> tuple[list[np.ndarray], Path]:
    """The unit partition of the train split, capped at m rows per unit; written to the partitions folder (so the Flower clients read it)."""
    from ppfeddata.data.preprocess import processed_dir

    z = np.load(processed_dir(cfg) / "train.npz", allow_pickle=True)
    y, u = z["y"].astype(np.int64), unit_ids(z["group_id"], z["stream_id"], unit)
    K = int(cfg["fl"]["num_clients"] if K is None else K)
    alpha = float(cfg["fl"]["dirichlet_alpha"] if alpha is None else alpha)
    parts = [cap(p, u, m, seed + k) for k, p in enumerate(unit_partition(y, u, K, alpha, seed))]
    out = Path(cfg["paths"]["work_dir"]) / "partitions" / f"units_{unit}_m{m}_alpha{alpha:g}_seed{seed}_k{K}_{cfg['label_mode']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"unit": unit, "cap": int(m), "alpha": alpha, "seed": int(seed), "num_clients": K, "sizes": [len(p) for p in parts],
                               "rows_kept": int(sum(len(p) for p in parts)), "n_rows": int(len(y)), "parts": [p.tolist() for p in parts]}), encoding="utf-8")
    return parts, out
