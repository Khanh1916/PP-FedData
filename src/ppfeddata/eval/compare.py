"""Phase 12: paired bootstrap comparisons of runs on the real test split.

Every run is evaluated on the same test rows, so one set of resamples serves all runs and the difference of two runs is paired by
construction. Two resampling schemes:

- `rows`: each class keeps its size and its rows are drawn with replacement (the stratified bootstrap of Phase 6, the paired bootstrap of the spec);
- `streams`: whole TCP streams are drawn with replacement inside each class. Packets of one stream are strongly correlated, so the row bootstrap
  understates the uncertainty; the number of rows of a class now varies from resample to resample.

Neither scheme reflects the choice of capture files (the test split comes from a handful of capture groups; sensitivity A4) or the randomness of
training beyond the seeds that were run: the seed-to-seed spread is handled by the second criterion of `effect_of`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from scipy import sparse

CHUNK = 100                   # resamples per block in the stream bootstrap (bounds the memory of the dense multiplicity matrix)


# --------------------------------------------------------------------------------------------------
# Metrics of a confusion matrix (rows = true class); work on [..., k, k] arrays so a whole bootstrap is one call
# --------------------------------------------------------------------------------------------------
def per_class_f1(conf) -> np.ndarray:
    conf = np.asarray(conf, dtype=float)
    tp = np.einsum("...ii->...i", conf)
    den = 2 * tp + (conf.sum(-2) - tp) + (conf.sum(-1) - tp)
    return np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)


def per_class_recall(conf) -> np.ndarray:
    conf = np.asarray(conf, dtype=float)
    tp = np.einsum("...ii->...i", conf)
    tot = conf.sum(-1)
    return np.divide(tp, tot, out=np.zeros_like(tp), where=tot > 0)


@dataclass(frozen=True)
class Metric:
    """`macro_f1`, or `recall` = mean recall over `classes` (one class = its recall)."""
    kind: str = "macro_f1"
    classes: tuple[int, ...] = ()

    def __call__(self, conf) -> np.ndarray:
        if self.kind == "macro_f1":
            return per_class_f1(conf).mean(-1)
        if self.kind == "recall":
            if not self.classes:
                raise ValueError("Metric('recall') needs at least one class")
            return per_class_recall(conf)[..., list(self.classes)].mean(-1)
        raise ValueError(f"unknown metric {self.kind!r}")


MACRO_F1 = Metric("macro_f1")


def recall_of(*classes: int) -> Metric:
    return Metric("recall", tuple(int(c) for c in classes))


# --------------------------------------------------------------------------------------------------
# Shared resamples
# --------------------------------------------------------------------------------------------------
class PairedBootstrap:
    """Bootstrap confusion matrices of many runs under the SAME resamples of one test set.

    `add({key: y_pred})` computes, for every new key, the confusion matrix of the full test set (`point[key]`, [k, k]) and of each of the `n_boot`
    resamples (`conf3(key)`, [n_boot, k, k]). The resamples depend only on the seed and the test set, never on the keys, so runs added in separate
    calls are still paired.
    """

    def __init__(self, y_true, n_classes: int, n_boot: int = 1000, seed: int = 0, mode: str = "rows", clusters=None):
        self.y = np.asarray(y_true, dtype=np.int64)
        self.k, self.B, self.seed, self.mode = int(n_classes), int(n_boot), int(seed), mode
        self.point: dict[Hashable, np.ndarray] = {}
        self.conf: dict[Hashable, np.ndarray] = {}
        if mode == "streams":
            if clusters is None:
                raise ValueError("mode 'streams' needs the stream id of every test row")
            uniq, self.codes = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
            self.G = len(uniq)
            self.cls_of = np.full(self.G, -1, dtype=np.int64)
            self.cls_of[self.codes] = self.y
            if not np.array_equal(self.cls_of[self.codes], self.y):
                raise ValueError("a stream holds rows of several classes")
        elif mode != "rows":
            raise ValueError(f"unknown mode {mode!r}")

    def add(self, preds: Mapping[Hashable, Any]) -> None:
        k = self.k
        keys = [key for key in preds if key not in self.conf]
        if not keys:
            return
        P = {key: np.asarray(preds[key], dtype=np.int64) for key in keys}
        for key, p in P.items():
            if p.shape != self.y.shape:
                raise ValueError(f"{key}: {p.shape} predictions for {self.y.shape} test rows")
            if p.min() < 0 or p.max() >= k:
                raise ValueError(f"{key}: predicted label outside 0..{k - 1}")
            self.point[key] = np.bincount(self.y * k + p, minlength=k * k).reshape(k, k)
        out = {key: np.empty((self.B, k * k), dtype=np.int32) for key in keys}
        rng = np.random.default_rng(self.seed)
        if self.mode == "rows":
            by_class = [ix for ix in (np.flatnonzero(self.y == c) for c in range(k)) if len(ix)]
            yk = self.y * k
            for b in range(self.B):
                ix = np.concatenate([rng.choice(c, size=len(c), replace=True) for c in by_class])
                base = yk[ix]
                for key in keys:
                    out[key][b] = np.bincount(base + P[key][ix], minlength=k * k)
        else:
            members = [m for m in (np.flatnonzero(self.cls_of == c) for c in range(k)) if len(m)]
            # [k*k, G] sparse: column g holds the confusion counts of stream g
            mats = {key: sparse.csr_matrix((np.ones(len(self.y)), (self.codes, self.y * k + P[key])), shape=(self.G, k * k)).T.tocsr() for key in keys}
            for s in range(0, self.B, CHUNK):
                nb = min(CHUNK, self.B - s)
                W = np.zeros((self.G, nb))
                for m in members:                                            # streams are drawn inside their own class
                    W[m, :] = rng.multinomial(len(m), np.full(len(m), 1.0 / len(m)), size=nb).T
                for key in keys:
                    out[key][s:s + nb] = np.rint(mats[key] @ W).T
        self.conf.update(out)

    def conf3(self, key: Hashable) -> np.ndarray:
        return self.conf[key].reshape(self.B, self.k, self.k)


# --------------------------------------------------------------------------------------------------
# Comparison of two configurations over their seeds
# --------------------------------------------------------------------------------------------------
def compare_runs(boot: PairedBootstrap, a_keys: Sequence[Hashable], b_keys: Sequence[Hashable], metric: Metric = MACRO_F1, alpha: float = 0.05) -> dict[str, Any]:
    """Seed-averaged difference metric(a) - metric(b); `a_keys[i]` and `b_keys[i]` are the runs of the same seed.

    The interval is the percentile interval of the seed-averaged difference over the shared resamples. `sigma` is the larger of the two
    configurations' std over seeds (ddof = 0, as in every table of the report).
    """
    if len(a_keys) != len(b_keys) or not len(a_keys):
        raise ValueError("a_keys and b_keys must be non-empty and of equal length")
    pa = np.array([float(metric(boot.point[k])) for k in a_keys])
    pb = np.array([float(metric(boot.point[k])) for k in b_keys])
    d = (np.stack([metric(boot.conf3(k)) for k in a_keys]) - np.stack([metric(boot.conf3(k)) for k in b_keys])).mean(axis=0)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    delta = float((pa - pb).mean())
    sigma = float(max(pa.std(), pb.std()))
    out = {"delta": delta, "lo": float(lo), "hi": float(hi), "sigma": sigma, "mean_a": float(pa.mean()), "mean_b": float(pb.mean()),
           "std_a": float(pa.std()), "std_b": float(pb.std()), "per_seed": (pa - pb).tolist(), "n_seeds": len(a_keys), "mode": boot.mode,
           "ci_excludes_zero": bool(lo > 0 or hi < 0), "exceeds_seed_std": bool(abs(delta) > sigma)}
    out["effect"] = effect_of(out)
    return out


def effect_of(c: Mapping[str, Any]) -> str:
    """The rule of the spec (R1, R2): the interval of the difference excludes 0 AND |difference| is larger than the std over seeds.
    Returns 'better' / 'worse' (sign of the difference) or 'none'."""
    if c["lo"] > 0 and c["delta"] > c["sigma"]:
        return "better"
    if c["hi"] < 0 and -c["delta"] > c["sigma"]:
        return "worse"
    return "none"
