"""Phase 9: privacy accounting for client-side DP-SGD (Opacus 1.6.0, RDP accountant).

Per client i with n_i rows and batch size B: Opacus' Poisson loader uses q_i = 1 / ceil(n_i / B) and ceil(n_i / B) steps per
epoch (checked in the tests against a real `DPDataLoader`). sigma_i is calibrated ONCE for the whole run from
(target epsilon, delta, q_i, planned steps T_i = rounds x local_epochs x ceil(n_i / B)). A fresh PrivacyEngine is built every
round, which resets Opacus' own accountant, so Opacus' per-round `get_epsilon` is never used: the final epsilon is
recomputed here from the step counters (sigma_i, q_i, total steps), which are stored in the checkpoint, so a resumed run
reports exactly the epsilon of an uninterrupted one.

Meaning of epsilon (state this in every report): record-level (packet) DP for the training of the model weights, for an
observer who sees the updates of that one client. Packets of one TCP stream are strongly correlated, so a whole attack
session is protected far less than the number suggests (group privacy). The class label (the conditioning input) is not
protected by DP-SGD. epsilon of a configuration = max over clients (worst case); the median is reported next to it.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


def steps_per_epoch(n: int, batch_size: int) -> int:
    return int(math.ceil(n / batch_size))


def sample_rate(n: int, batch_size: int) -> float:
    return 1.0 / steps_per_epoch(n, batch_size)


def planned_steps(n: int, batch_size: int, rounds: int, local_epochs: int) -> int:
    return int(rounds * local_epochs * steps_per_epoch(n, batch_size))


def check_delta(sizes: Sequence[int], delta: float) -> None:
    """delta must be smaller than 1/n_i for every client (otherwise 'delta' allows releasing a record outright)."""
    bad = [int(n) for n in sizes if delta >= 1.0 / n]
    if bad:
        raise ValueError(f"delta={delta} is not < 1/n_i for clients with n_i={bad}")


def epsilon(sigma: float, q: float, steps: int, delta: float) -> float:
    """RDP-accountant epsilon of `steps` Poisson-subsampled Gaussian steps. 0 steps -> 0; sigma = 0 -> infinity."""
    if steps <= 0:
        return 0.0
    if sigma <= 0:
        return float("inf")
    acc = RDPAccountant()
    acc.history = [(float(sigma), float(q), int(steps))]
    return float(acc.get_epsilon(delta=delta))


def calibrate_sigma(target_eps: float, delta: float, n: int, batch_size: int, rounds: int, local_epochs: int) -> float:
    return float(get_noise_multiplier(target_epsilon=float(target_eps), target_delta=float(delta), sample_rate=sample_rate(n, batch_size),
                                      steps=planned_steps(n, batch_size, rounds, local_epochs), accountant="rdp"))


def calibrate_clients(sizes: Sequence[int], target_eps: float, delta: float, batch_size: int, rounds: int,
                      local_epochs: int) -> list[float]:
    """One sigma per client so that each client's epsilon over the planned run is <= target_eps."""
    check_delta(sizes, delta)
    return [calibrate_sigma(target_eps, delta, int(n), batch_size, rounds, local_epochs) for n in sizes]


def epsilon_table(sizes: Sequence[int], sigmas: Sequence[float], dp_steps: dict[Any, int], batch_size: int, delta: float,
                  target_eps: float | None = None) -> dict[str, Any]:
    """Per-client epsilon from the counted steps; the configuration's epsilon is the maximum over clients."""
    rows = []
    for i, (n, s) in enumerate(zip(sizes, sigmas)):
        steps = int(dp_steps.get(i, dp_steps.get(str(i), 0)))
        e = epsilon(s, sample_rate(int(n), batch_size), steps, delta)
        rows.append({"client": i, "n": int(n), "sigma": float(s), "q": sample_rate(int(n), batch_size), "steps": steps, "epsilon": e})
    eps = [r["epsilon"] for r in rows]
    out = {"rows": rows, "eps_max": float(max(eps)), "eps_median": float(np.median(eps)), "delta": float(delta)}
    if target_eps is not None:
        out["target_eps"] = float(target_eps)
        out["max_ratio_to_target"] = float(max(eps) / target_eps)
        out["within_2pct"] = bool(max(eps) <= 1.02 * target_eps)
    return out
