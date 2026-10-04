"""Phase 6: empirical privacy of synthetic data (not a formal guarantee; the DP epsilon comes from Phase 9).

- duplicate rate: share of synthetic rows that equal a train row after rounding
- DCR (distance to closest record) and the ratio median DCR(syn->train) / median DCR(syn->val): ~1 is good,
  << 1 means the generator copies training rows
- MIA: score = -DCR(record -> synthetic set); members = train rows, non-members = val rows; AUC per class, then mean.
  AUC ~ 0.5 is good. A positive control (a generator that memorises a small training set) must give an AUC clearly
  above 0.5, otherwise the attack is too weak to support any claim.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def _row_hash(X: np.ndarray, decimals: int) -> np.ndarray:
    return pd.util.hash_pandas_object(pd.DataFrame(np.round(X.astype("float64"), decimals)), index=False).to_numpy()


def duplicate_rate(Xs: np.ndarray, Xtrain: np.ndarray, decimals: int = 3) -> float:
    return float(np.isin(_row_hash(Xs, decimals), _row_hash(Xtrain, decimals)).mean()) if len(Xs) else 0.0


def dcr(Xq: np.ndarray, Xref: np.ndarray) -> np.ndarray:
    """Euclidean distance from each query row to its nearest reference row."""
    return NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Xref).kneighbors(Xq)[0][:, 0]


def dcr_ratio(Xs: np.ndarray, Xtrain: np.ndarray, Xval: np.ndarray) -> dict[str, float]:
    a, b = float(np.median(dcr(Xs, Xtrain))), float(np.median(dcr(Xs, Xval)))
    return {"median_dcr_train": a, "median_dcr_val": b, "ratio": a / b if b > 0 else float("inf")}


def mia_auc(Xs: np.ndarray, Xmember: np.ndarray, Xnon: np.ndarray) -> float:
    """AUC of the distance-to-synthetic attack with equal numbers of members and non-members."""
    n = min(len(Xmember), len(Xnon))
    scores = -np.r_[dcr(Xmember[:n], Xs), dcr(Xnon[:n], Xs)]
    return float(roc_auc_score(np.r_[np.ones(n), np.zeros(n)], scores))


def mia_per_class(Xs, ys, Xtr, ytr, Xva, yva, n_classes: int, cap: int = 2000, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    per = {}
    for c in range(n_classes):
        s, m, v = Xs[ys == c], np.flatnonzero(ytr == c), np.flatnonzero(yva == c)
        n = min(len(m), len(v), cap)
        if len(s) == 0 or n < 5:
            continue
        per[c] = mia_auc(s, Xtr[rng.choice(m, n, replace=False)], Xva[rng.choice(v, n, replace=False)])
    return {"per_class": per, "mean": float(np.mean(list(per.values()))) if per else float("nan")}


def privacy_report(Xs, ys, Xtr, ytr, Xva, yva, class_names: list[str], seed: int = 0, decimals: int = 3) -> dict[str, Any]:
    k = len(class_names)
    rep: dict[str, Any] = {"duplicate_rate": duplicate_rate(Xs, Xtr, decimals), "per_class": {}}
    for c, name in enumerate(class_names):
        s, t, v = Xs[ys == c], Xtr[ytr == c], Xva[yva == c]
        if len(s) and len(t) and len(v):
            rep["per_class"][name] = {**dcr_ratio(s, t, v), "duplicate_rate": duplicate_rate(s, t, decimals)}
    ratios = [d["ratio"] for d in rep["per_class"].values()]
    rep["dcr_ratio_mean"] = float(np.mean(ratios)) if ratios else float("nan")
    mia = mia_per_class(Xs, ys, Xtr, ytr, Xva, yva, k, seed=seed)
    rep["mia_auc_mean"] = mia["mean"]
    rep["mia_auc_per_class"] = {class_names[c]: v for c, v in mia["per_class"].items()}
    return rep


def positive_control(generator: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]],
                     Xtr, ytr, Xva, yva, n_classes: int, n_members: int = 500, seed: int = 0,
                     stratified: bool = False) -> dict[str, Any]:
    """MIA against a generator that is trained on only `n_members` rows (so it can memorise them).

    `generator(X_members, y_members, seed) -> (X_syn, y_syn)`. Members are the rows the generator saw; non-members
    come from val. The AUC must be clearly above 0.5; if not, the MIA is too weak and must be reported as such.
    Phase 7 supplies the deliberately over-fitted CVAE; tests here use a copy-with-noise generator.
    """
    rng = np.random.default_rng(seed)
    if stratified:      # equal number of members per class, so rare classes are testable too
        per = max(1, n_members // n_classes)
        m = np.concatenate([rng.choice(np.flatnonzero(ytr == c), min(per, int((ytr == c).sum())), replace=False)
                            for c in range(n_classes) if (ytr == c).any()])
    else:
        m = rng.choice(len(Xtr), min(n_members, len(Xtr)), replace=False)
    Xm, ym = Xtr[m], ytr[m]
    Xs, ys = generator(Xm, ym, seed)
    res = mia_per_class(Xs, ys, Xm, ym, Xva, yva, n_classes, cap=len(Xm), seed=seed)
    return {"mia_auc_mean": res["mean"], "per_class": res["per_class"], "n_members": int(len(Xm))}
