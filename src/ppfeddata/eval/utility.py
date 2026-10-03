"""Phase 6: classifiers, evaluation protocols (TRTR / TSTR / TAug) and utility metrics.

All protocols are evaluated on the REAL test split. Features come from `data/processed/<mode>` (Preprocessor fit on
train only). Predictions of every run are saved so that paired bootstrap can be done afterwards.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger("ppfeddata.eval.utility")


# --------------------------------------------------------------------------------------------------
# Classifiers
# --------------------------------------------------------------------------------------------------
def make_classifier(name: str, cfg: dict[str, Any], seed: int, balanced: bool = False):
    """RF (200 trees) or MLP (128-64, early stopping). `balanced` = class_weight="balanced" (RF only)."""
    e = cfg["eval"]
    if name == "rf":
        return RandomForestClassifier(n_estimators=int(e["rf"]["n_estimators"]), n_jobs=-1, random_state=seed,
                                      class_weight="balanced" if balanced else None)
    if name == "mlp":
        if balanced:
            raise ValueError("sklearn MLPClassifier has no class_weight; B1a is defined for RF only")
        m = e["mlp"]
        return MLPClassifier(hidden_layer_sizes=tuple(m["hidden"]), max_iter=int(m["max_iter"]),
                             early_stopping=bool(m["early_stopping"]), random_state=seed)
    raise ValueError(f"unknown classifier {name!r}")


def full_proba(clf, X: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba with one column per label (zeros for labels the model never saw)."""
    p = clf.predict_proba(X)
    out = np.zeros((len(X), n_classes), dtype=np.float32)
    out[:, clf.classes_.astype(int)] = p
    return out


# --------------------------------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------------------------------
def _take(rng: np.random.Generator, idx: np.ndarray, k: int) -> np.ndarray:
    return rng.choice(idx, size=min(k, len(idx)), replace=False)


def build_train_set(protocol: str, X_real, y_real, X_syn, y_syn, n_classes: int, syn_per_class: int,
                    target_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Training set for a protocol.

    TRTR: real only.  TSTR: synthetic only, `syn_per_class` rows per class.
    TAug: all real rows + synthetic rows so that every class reaches at least `target_per_class` (large classes are
    never reduced). Returns (X, y, info) where info lists how many synthetic rows were used and any shortage.
    """
    rng = np.random.default_rng(seed)
    if protocol == "TRTR":
        return X_real, y_real, {"n_synthetic": {}, "shortage": {}}
    if protocol not in ("TSTR", "TAug"):
        raise ValueError(f"unknown protocol {protocol!r}")
    if X_syn is None or len(X_syn) == 0:
        raise ValueError(f"{protocol} needs synthetic data")
    used, short, parts_X, parts_y = {}, {}, [], []
    real_counts = np.bincount(y_real, minlength=n_classes)
    for c in range(n_classes):
        want = syn_per_class if protocol == "TSTR" else max(0, target_per_class - int(real_counts[c]))
        if want == 0:
            continue
        pool = np.flatnonzero(y_syn == c)
        pick = _take(rng, pool, want)
        parts_X.append(X_syn[pick])
        parts_y.append(y_syn[pick])
        used[c] = int(len(pick))
        if len(pick) < want:
            short[c] = int(want - len(pick))
    Xs = np.vstack(parts_X) if parts_X else np.zeros((0, X_real.shape[1]), X_real.dtype)
    ys = np.concatenate(parts_y) if parts_y else np.zeros(0, y_real.dtype)
    if protocol == "TSTR":
        return Xs, ys, {"n_synthetic": used, "shortage": short}
    return np.vstack([X_real, Xs]), np.concatenate([y_real, ys]), {"n_synthetic": used, "shortage": short}


def snap_encoded(X: np.ndarray, schema: dict[str, Any]) -> np.ndarray:
    """Make interpolated rows valid again: argmax one-hot per categorical group, 0/1 for binary and is_na flags."""
    X = X.copy()
    for b in schema["blocks"]:
        sl = slice(b["start"], b["start"] + b["width"])
        if b["type"] == "categorical":
            idx = np.argmax(X[:, sl], axis=1)
            X[:, sl] = 0.0
            X[np.arange(len(X)), b["start"] + idx] = 1.0
        elif b["type"] in ("binary", "na_flag"):
            X[:, sl] = (X[:, sl] >= 0.5).astype(X.dtype)
    return X


def smote_oversample(X, y, schema, target_per_class: int, seed: int, k_neighbors: int = 5):
    """B1b: SMOTE in the encoded space up to `target_per_class` for every class below it (no class is reduced),
    then snap the new rows back to valid one-hot / binary values."""
    from imblearn.over_sampling import SMOTE

    counts = np.bincount(y)
    strategy = {int(c): int(target_per_class) for c in range(len(counts)) if 0 < counts[c] < target_per_class}
    if not strategy:
        return X, y
    k = int(min(k_neighbors, max(1, min(counts[c] for c in strategy) - 1)))
    Xr, yr = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed).fit_resample(X, y)
    n0 = len(X)
    Xr = Xr.astype(X.dtype, copy=False)
    Xr[n0:] = snap_encoded(Xr[n0:], schema)
    return Xr, yr


# --------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------
def macro_f1_from_confusion(conf: np.ndarray) -> float:
    tp = np.diag(conf).astype(float)
    denom = 2 * tp + (conf.sum(0) - tp) + (conf.sum(1) - tp)
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)))


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(y_true) * n_classes + np.asarray(y_pred), minlength=n_classes ** 2).reshape(n_classes, n_classes)


def compute_metrics(y_true, y_pred, proba, class_names: list[str], normal_label: int = 0) -> dict[str, Any]:
    """Utility metrics. Confusion matrix rows = true class. The binary table is derived by collapsing the
    multi-class prediction (Normal vs everything else)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k = len(class_names)
    labels = list(range(k))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    conf = confusion(y_true, y_pred, k)
    out: dict[str, Any] = {
        "macro_f1": float(np.mean(f)), "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": {class_names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i in labels},
        "confusion": conf.tolist(),
    }
    if proba is not None:
        aps = [average_precision_score((y_true == c).astype(int), proba[:, c]) for c in labels if (y_true == c).any()]
        out["pr_auc_macro"] = float(np.mean(aps))
    tb, pb = y_true != normal_label, y_pred != normal_label
    tp, fp, fn = int((tb & pb).sum()), int((~tb & pb).sum()), int((tb & ~pb).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    out["binary"] = {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
                     "accuracy": float((tb == pb).mean())}
    return out
