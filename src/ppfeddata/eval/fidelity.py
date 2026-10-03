"""Phase 6: fidelity of synthetic data (computed in the encoded feature space, per class).

Numeric columns are already divided by the train std (standardised), so the Wasserstein distance is in units of the
train std. Binary / is_na columns and categorical groups use Jensen-Shannon divergence (base 2, 0 = identical,
1 = disjoint). C2ST: a Random Forest tries to tell real from synthetic rows (5-fold AUC, 0.5 is the best case).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _cols(schema: dict[str, Any], types: tuple[str, ...]) -> list[int]:
    return [b["start"] + i for b in schema["blocks"] if b["type"] in types for i in range(b["width"])]


def wasserstein_numeric(Xr: np.ndarray, Xs: np.ndarray, schema: dict[str, Any]) -> float:
    cols = _cols(schema, ("numeric",))
    return float(np.mean([wasserstein_distance(Xr[:, c], Xs[:, c]) for c in cols])) if cols else 0.0


def js_discrete(Xr: np.ndarray, Xs: np.ndarray, schema: dict[str, Any]) -> float:
    """Mean JS divergence over binary / is_na columns (Bernoulli) and categorical groups (category frequencies)."""
    vals = []
    for b in schema["blocks"]:
        if b["type"] == "numeric":
            continue
        sl = slice(b["start"], b["start"] + b["width"])
        if b["type"] == "categorical":
            p, q = Xr[:, sl].mean(0), Xs[:, sl].mean(0)
        else:
            pr, qs = float(Xr[:, sl].mean()), float(Xs[:, sl].mean())
            p, q = np.array([pr, 1 - pr]), np.array([qs, 1 - qs])
        vals.append(float(jensenshannon(p, q, base=2) ** 2) if p.sum() > 0 and q.sum() > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def correlation_distance(Xr: np.ndarray, Xs: np.ndarray) -> float:
    """Mean |difference| of Pearson correlations over column pairs; constant columns are ignored."""
    keep = np.flatnonzero((Xr.std(0) > 1e-8) & (Xs.std(0) > 1e-8))
    if len(keep) < 2:
        return 0.0
    cr, cs = np.corrcoef(Xr[:, keep], rowvar=False), np.corrcoef(Xs[:, keep], rowvar=False)
    iu = np.triu_indices(len(keep), k=1)
    return float(np.mean(np.abs(cr[iu] - cs[iu])))


def c2st_auc(Xr: np.ndarray, Xs: np.ndarray, seed: int = 0, folds: int = 5, n_trees: int = 100,
             max_rows: int = 10_000) -> float:
    """AUC of a RF separating real from synthetic rows (equal sizes, 5-fold out-of-fold probabilities)."""
    rng = np.random.default_rng(seed)
    n = min(len(Xr), len(Xs), max_rows)
    a, b = Xr[rng.choice(len(Xr), n, replace=False)], Xs[rng.choice(len(Xs), n, replace=False)]
    X, y = np.vstack([a, b]), np.r_[np.zeros(n), np.ones(n)]
    p = cross_val_predict(RandomForestClassifier(n_trees, n_jobs=-1, random_state=seed), X, y,
                          cv=StratifiedKFold(folds, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def fidelity_report(Xr, yr, Xs, ys, schema, class_names: list[str], seed: int = 0, with_c2st: bool = True,
                    n_trees: int = 100) -> dict[str, Any]:
    """Per-class fidelity (real = train rows of that class, synthetic = synthetic rows of that class) + the mean."""
    out: dict[str, Any] = {}
    for c, name in enumerate(class_names):
        r, s = Xr[yr == c], Xs[ys == c]
        if len(r) < 10 or len(s) < 10:
            continue
        out[name] = {"wasserstein": wasserstein_numeric(r, s, schema), "js": js_discrete(r, s, schema),
                     "corr_dist": correlation_distance(r, s)}
        if with_c2st:
            out[name]["c2st_auc"] = c2st_auc(r, s, seed=seed, n_trees=n_trees)
    keys = next(iter(out.values())).keys() if out else []
    out["mean"] = {k: float(np.mean([v[k] for n, v in out.items() if n != "mean"])) for k in keys}
    return out
