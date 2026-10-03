"""Phase 6: stratified bootstrap on the test set (CI of macro-F1 and CI of paired differences)."""
from __future__ import annotations

import numpy as np

from ppfeddata.eval.utility import confusion, macro_f1_from_confusion


def _class_indices(y: np.ndarray, n_classes: int) -> list[np.ndarray]:
    return [np.flatnonzero(y == c) for c in range(n_classes)]


def _resample(rng: np.random.Generator, by_class: list[np.ndarray]) -> np.ndarray:
    """One stratified resample: every class keeps its own size."""
    return np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_class if len(ix)])


def bootstrap_macro_f1(y_true, y_pred, n_classes: int, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    by_class = _class_indices(y_true, n_classes)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        ix = _resample(rng, by_class)
        vals[b] = macro_f1_from_confusion(confusion(y_true[ix], y_pred[ix], n_classes))
    point = macro_f1_from_confusion(confusion(y_true, y_pred, n_classes))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"macro_f1": point, "lo": float(lo), "hi": float(hi), "n_boot": n_boot}


def paired_bootstrap_diff(y_true, pred_a, pred_b, n_classes: int, n_boot: int = 1000, seed: int = 0,
                          alpha: float = 0.05) -> dict:
    """CI of macro-F1(a) - macro-F1(b) on the SAME resampled test rows (paired)."""
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    by_class = _class_indices(y_true, n_classes)
    d = np.empty(n_boot)
    for b in range(n_boot):
        ix = _resample(rng, by_class)
        d[b] = (macro_f1_from_confusion(confusion(y_true[ix], pred_a[ix], n_classes))
                - macro_f1_from_confusion(confusion(y_true[ix], pred_b[ix], n_classes)))
    point = (macro_f1_from_confusion(confusion(y_true, pred_a, n_classes))
             - macro_f1_from_confusion(confusion(y_true, pred_b, n_classes)))
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0),
            "p_positive": float((d > 0).mean()), "n_boot": n_boot}
