"""Phase 7: sampling synthetic rows from a trained CVAE, back into the encoded feature space.

`generate(model, schema, class_counts, seed)` draws z ~ N(0, I) for the requested labels and decodes:
- numeric: decoder mean -> raw units (inverse of standardise / log1p / scale) -> clip to the train [min, max] ->
  round for integer columns -> encode again (so the rows follow the same rules as real rows);
- binary and `_is_na` flags: Bernoulli(sigmoid(logit));
- categorical: sample from the softmax, never from a category with 0 train rows (dead OTHER / NONE slots);
- where a column's `_is_na` is 1, the column takes its "not applicable" value (the fill used by the Preprocessor).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from scipy.special import expit

from ppfeddata.models.cvae import CVAE, one_hot


def _to_counts(class_counts: Sequence[int] | dict[int, int], n_classes: int) -> np.ndarray:
    if isinstance(class_counts, dict):
        out = np.zeros(n_classes, dtype=np.int64)
        for k, v in class_counts.items():
            out[int(k)] = int(v)
        return out
    out = np.asarray(class_counts, dtype=np.int64)
    if len(out) != n_classes:
        raise ValueError(f"class_counts has {len(out)} entries, the model has {n_classes} classes")
    return out


def encode_numeric(raw_scaled: np.ndarray, p: dict[str, Any], clip: float) -> np.ndarray:
    """Same arithmetic as Preprocessor._numeric_z for a vector that is already multiplied by `scale`."""
    v = np.asarray(raw_scaled, dtype="float64")
    if p["nonneg_clip"] or p["log1p"]:
        v = np.maximum(v, 0.0)
    t = np.log1p(v) if p["log1p"] else v
    return np.clip((t - p["mean"]) / p["std"], -clip, clip)


def decode_numeric(z: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Standardised value -> raw units (inverse of encode_numeric, before clipping)."""
    t = np.asarray(z, dtype="float64") * p["std"] + p["mean"]
    return (np.expm1(np.minimum(t, 700.0)) if p["log1p"] else t) / p["scale"]


def na_default_z(p: dict[str, Any], clip: float) -> float:
    """Encoded value of a 'not applicable' cell: 0 raw for normal columns, 0 for ultra-sparse ones."""
    return 0.0 if p.get("ultra_sparse") else float(encode_numeric(np.zeros(1), p, clip)[0])


def dead_category_mask(schema: dict[str, Any]) -> np.ndarray:
    """Boolean vector over D: True for one-hot slots whose category has no train row (never to be sampled)."""
    dead = np.zeros(schema["n_features"], dtype=bool)
    for b in schema["blocks"]:
        if b["type"] != "categorical":
            continue
        counts = schema["params"][b["column"]]["train_counts"]
        for i, cat in enumerate(b["categories"]):
            if int(counts.get(cat, 0)) == 0:
                dead[b["start"] + i] = True
    return dead


@dataclass
class GenStats:
    """Generation-side statistics estimated on the TRAIN split (no extra model parameters, nothing from val/test).

    support: (class, block start) -> sorted encoded values seen in train in that class, for numeric columns with
             <= `max_unique` values there (window sizes, lengths, keep-alive ...). The MSE decoder predicts a mean that
             falls between those values; snapping to the nearest seen value restores the discrete support.
    residual_std: [n_classes, n_num] std of (encoded value - decoder mean at the posterior mean) per class. The MSE
             decoder is a Gaussian with a fixed variance; sampling adds noise with this std so the spread is not lost.
    """
    support: dict[tuple[int, int], np.ndarray]
    residual_std: np.ndarray | None


def gen_stats_from_cfg(model: CVAE, X: np.ndarray, y: np.ndarray, schema: dict[str, Any], gcfg: dict[str, Any]) -> GenStats | None:
    """`generate.residual_noise` / `generate.snap_support` switches (see GenStats). None = plain decoder means."""
    if not (gcfg.get("residual_noise") or gcfg.get("snap_support")):
        return None
    st = fit_gen_stats(model, X, y, schema, int(gcfg.get("snap_max_unique", 256)), snap=bool(gcfg.get("snap_support")))
    return GenStats(st.support, st.residual_std if gcfg.get("residual_noise") else None)


def fit_gen_stats(model: CVAE, X: np.ndarray, y: np.ndarray, schema: dict[str, Any], max_unique: int = 256,
                  snap: bool = True) -> GenStats:
    support: dict[tuple[int, int], np.ndarray] = {}
    for c in range(model.n_classes if snap else 0):
        for b in schema["blocks"]:
            if b["type"] == "numeric" and (y == c).any():
                u = np.unique(X[y == c, b["start"]])
                if len(u) <= max_unique:
                    support[(c, b["start"])] = u.astype(np.float64)
    n = model.layout.n_num
    res = np.zeros((model.n_classes, n))
    model.eval()
    with torch.no_grad():
        Xt, yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
        mean = []
        for i in range(0, len(Xt), 8192):
            xb, yb = Xt[i:i + 8192], one_hot(yt[i:i + 8192], model.n_classes)
            mu, _ = model.encode(xb, yb)
            mean.append(model.decode(mu, yb)[:, :n].numpy())
    diff = X[:, :n] - np.vstack(mean)
    for c in range(model.n_classes):
        if (y == c).any():
            res[c] = diff[y == c].std(axis=0)
    return GenStats(support, res)


def postprocess(raw_out: np.ndarray, schema: dict[str, Any], rng: np.random.Generator,
                stats: GenStats | None = None, y: np.ndarray | None = None) -> np.ndarray:
    """Decoder output (means / logits) -> valid encoded rows. Pure numpy so it can be tested without torch."""
    clip = float(schema["settings"]["clip_sigma"])
    if stats is not None and stats.residual_std is not None and y is not None:
        raw_out = raw_out.copy()
        n_num = stats.residual_std.shape[1]
        raw_out[:, :n_num] += rng.standard_normal((len(raw_out), n_num)) * stats.residual_std[y]
    n = len(raw_out)
    X = np.zeros((n, schema["n_features"]), dtype=np.float32)
    flags = {b["column"]: b for b in schema["blocks"] if b["type"] == "na_flag"}
    dead = dead_category_mask(schema)

    for b in schema["blocks"]:                         # flags first: other blocks depend on them
        if b["type"] in ("binary", "na_flag"):
            prob = expit(raw_out[:, b["start"]])
            X[:, b["start"]] = (rng.random(n) < prob).astype(np.float32)

    for b in schema["blocks"]:
        s, w = b["start"], b["width"]
        if b["type"] == "numeric":
            p = schema["params"][b["column"]]
            raw = decode_numeric(raw_out[:, s], p)
            raw = np.clip(raw, p["raw_min"], p["raw_max"])
            if p["is_integer"]:
                raw = np.round(raw)
            z = encode_numeric(raw * p["scale"], p, clip)
            if stats is not None and y is not None:
                for c in np.unique(y):
                    sup = stats.support.get((int(c), s))
                    if sup is None:
                        continue
                    m = y == c
                    if len(sup) == 1:
                        z[m] = sup[0]
                        continue
                    j = np.clip(np.searchsorted(sup, z[m]), 1, len(sup) - 1)
                    zc = z[m]
                    z[m] = np.where(np.abs(zc - sup[j - 1]) <= np.abs(zc - sup[j]), sup[j - 1], sup[j])
            if b["column"] in flags:
                na = X[:, flags[b["column"]]["start"]] >= 0.5
                z = np.where(na, na_default_z(p, clip), z)
            X[:, s] = z.astype(np.float32)
        elif b["type"] == "binary" and b["column"] in flags:
            X[:, s] = np.where(X[:, flags[b["column"]]["start"]] >= 0.5, 0.0, X[:, s])
        elif b["type"] == "categorical":
            logits = raw_out[:, s:s + w].astype("float64")
            logits = np.where(dead[s:s + w], -np.inf, logits)
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)
            pick = (prob.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
            X[np.arange(n), s + pick] = 1.0
    return X


@torch.no_grad()
def generate(model: CVAE, schema: dict[str, Any], class_counts: Sequence[int] | dict[int, int], seed: int,
             batch_size: int = 8192, stats: GenStats | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic rows `(X, y)`: `class_counts[c]` rows of class c, in class order. Deterministic given `seed`.
    `stats` (from `fit_gen_stats` on the train split) enables residual noise and snapping to the seen support."""
    model.eval()
    counts = _to_counts(class_counts, model.n_classes)
    y = np.repeat(np.arange(model.n_classes), counts).astype(np.int64)
    rng = np.random.default_rng(seed)
    out = np.empty((len(y), model.layout.D), dtype=np.float32)
    for i in range(0, len(y), batch_size):
        yb = torch.as_tensor(y[i:i + batch_size])
        z = torch.as_tensor(rng.standard_normal((len(yb), model.latent_dim)), dtype=torch.float32)
        out[i:i + batch_size] = model.decode(z, one_hot(yb, model.n_classes)).numpy()
    return postprocess(out, schema, rng, stats, y), y
