"""Phase 8: split the train pool across FL clients with a per-class Dirichlet(alpha).

For every class, the shares of that class's rows are drawn from Dirichlet(alpha * 1_K) over the K clients; small alpha =
strong label skew (non-IID), large alpha = nearly IID. The whole draw is repeated until every client has at least
`fl.min_client_size` rows. Indices refer to rows of the train split (`data/processed/<mode>/train.npz`) and are stored in
`<work_dir>/partitions/alpha{alpha}_seed{seed}[_{mode}].json`. Rows are split regardless of TCP stream, so the same stream
can be seen by several clients (stated as a limitation: client data is not stream-disjoint).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("ppfeddata.partition")


def dirichlet_partition(y: np.ndarray, num_clients: int, alpha: float, seed: int, min_size: int,
                        n_classes: int | None = None, max_tries: int = 2000) -> tuple[list[np.ndarray], int]:
    """Returns (list of index arrays, number of Dirichlet draws needed). Every row is assigned to exactly one client."""
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    n_classes = int(n_classes if n_classes is not None else y.max() + 1)
    if len(y) < num_clients * min_size:
        raise ValueError(f"{len(y)} rows cannot give {num_clients} clients at least {min_size} rows each")
    by_class = [np.flatnonzero(y == c) for c in range(n_classes)]
    rng = np.random.default_rng(seed)
    for attempt in range(1, max_tries + 1):
        parts: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
        for idx in by_class:
            if len(idx) == 0:
                continue
            idx = rng.permutation(idx)
            p = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].append(chunk)
        out = [np.sort(np.concatenate(p)) if p else np.zeros(0, dtype=np.int64) for p in parts]
        if min(len(o) for o in out) >= min_size:
            return out, attempt
    raise RuntimeError(f"no Dirichlet({alpha}) draw gave every client >= {min_size} rows in {max_tries} tries")


def label_table(y: np.ndarray, parts: list[np.ndarray], class_names: list[str]) -> pd.DataFrame:
    """Rows = clients, columns = classes (counts), plus the client size."""
    rows = [np.bincount(y[p], minlength=len(class_names)) for p in parts]
    df = pd.DataFrame(rows, columns=class_names)
    df.insert(0, "client", range(len(parts)))
    df["n"] = [len(p) for p in parts]
    return df


def partition_path(cfg: dict[str, Any], alpha: float, seed: int) -> Path:
    suffix = "" if cfg["label_mode"] == "6class" else f"_{cfg['label_mode']}"
    return Path(cfg["paths"]["work_dir"]) / "partitions" / f"alpha{alpha:g}_seed{seed}{suffix}.json"


def save_partition(path: Path, parts: list[np.ndarray], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**meta, "parts": [p.tolist() for p in parts]}), encoding="utf-8")


def load_partition(path: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    parts = [np.asarray(p, dtype=np.int64) for p in d.pop("parts")]
    return parts, d


def make_partition(cfg: dict[str, Any], y: np.ndarray, class_names: list[str], alpha: float | None = None,
                   seed: int = 0, num_clients: int | None = None, force: bool = False) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Load the stored partition if it matches, otherwise draw and store it (plus the label table)."""
    fl = cfg["fl"]
    alpha = float(fl["dirichlet_alpha"] if alpha is None else alpha)
    k = int(fl["num_clients"] if num_clients is None else num_clients)
    path = partition_path(cfg, alpha, seed)
    if k != int(fl["num_clients"]):
        path = path.with_name(path.stem + f"_k{k}.json")
    if path.exists() and not force:
        parts, meta = load_partition(path)
        if meta["num_clients"] == k and meta["n_rows"] == len(y):
            return parts, meta
    parts, tries = dirichlet_partition(y, k, alpha, seed, int(fl["min_client_size"]), len(class_names))
    table = label_table(y, parts, class_names)
    meta = {"alpha": alpha, "seed": seed, "num_clients": k, "min_client_size": int(fl["min_client_size"]), "n_rows": int(len(y)),
            "draws": tries, "sizes": [len(p) for p in parts], "label_mode": cfg["label_mode"],
            "classes": class_names, "table": table.drop(columns="client").values.tolist()}
    save_partition(path, parts, meta)
    logger.info("partition alpha=%g seed=%d: sizes %s (%d draws) -> %s", alpha, seed, meta["sizes"], tries, path)
    return parts, meta


def plot_heatmap(table: pd.DataFrame, out: str | Path, title: str = "") -> Path:
    """Heatmap of the share of each class held by each client (rows sum to 1 per class column)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = [c for c in table.columns if c not in ("client", "n")]
    counts = table[classes].to_numpy(dtype=float)
    share = counts / np.maximum(counts.sum(axis=0, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(classes), 1.0 + 0.6 * len(table)))
    im = ax.imshow(share, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(table)), [f"client {i} (n={int(n)})" for i, n in zip(table["client"], table["n"])])
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            ax.text(j, i, f"{int(counts[i, j])}", ha="center", va="center", fontsize=7,
                    color="white" if share[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="share of the class held by the client")
    ax.set_title(title)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return Path(out)
