"""Optimisation O3: a second generator, federated DP marginals with a Chow-Liu tree (`MG`), written for this study (no external package).

Attributes (from `feature_schema.json`): every numeric column is discretised into `FINE` equal bins on the encoded range [-clip, clip]
(the encoded values are clipped at `clip_sigma`, so the range is public); binary and is_na flags have 2 values; a categorical column has
one value per one-hot slot. The class label y conditions every table.

Three releases, each the SUM over clients of count tables plus Gaussian noise (one record adds 1 to one cell of every table of a release,
so the L2 sensitivity of a release with m tables is sqrt(m)); a sum of counts is what secure aggregation computes, so the noise may be
split over the clients (distributed DP, valid here because each release is a single additive mechanism):

1. `fine`: per-class 1-way histograms of every attribute (numeric columns at FINE bins). Post-processing gives, per numeric column,
   `bins` coarse bins at the quantiles of the pooled noisy histogram, and the within-bin distribution used when sampling.
2. `pairs`: 2-way tables of every pair of attributes on the coarse values (pooled over classes); their noisy mutual information gives the
   maximum spanning tree (Chow-Liu).
3. `edges`: for the tree root, its (root, y) table; for every edge, the (parent, child, y) table on coarse values.

Sampling, per class: root ~ P(root | y), then child ~ P(child | parent, y) along the tree (noisy counts clipped at 0 plus `alpha`); a
numeric coarse value is refined to a fine bin with the class's fine histogram, then uniformly inside it; the row goes through the same
`postprocess` as the CVAE output (integer rounding, is_na consistency, dead categories).

Privacy: zCDP, rho = sensitivity^2 / (2 sigma^2) per release, summed, converted to (epsilon, delta) with eps = rho + 2 sqrt(rho log(1/delta)).
The total rho is split over the releases by `split`. Distributed (`noise_share` = 1/K): each client adds N(0, sigma^2 / K) per cell; with a
single honest client the effective rho is K times larger (reported). Local (`noise_share` = 1): each client adds the full noise, so each
client's release alone is (epsilon, delta)-DP and the sum carries K times the variance. Limitation (as for the CVAE route): the encoding
(means, stds, categories, dead slots) comes from the non-private preprocessing of the pooled train split; the SecAgg quantisation of the
noisy counts is treated as for DP-FedSGD (`fl/dpfedsgd.py`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

FINE = 64
SPLIT = (0.3, 0.1, 0.6)          # share of rho for the fine, pairs and edges releases


# --------------------------------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------------------------------
def rho_from_eps(eps: float, delta: float) -> float:
    """Largest rho with rho + 2 sqrt(rho log(1/delta)) <= eps."""
    L = math.log(1.0 / delta)
    r = (-math.sqrt(L) + math.sqrt(L + eps)) ** 2
    return float(r)


def eps_from_rho(rho: float, delta: float) -> float:
    return float(rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta)))


def sigma_for(rho: float, n_tables: int) -> float:
    """Noise std per cell so that a release of `n_tables` tables (sensitivity sqrt(n_tables)) costs `rho`."""
    return float(math.sqrt(n_tables / (2.0 * rho)))


# Skellam (optimisation O2): integer noise Sk = Poisson(mu/2) - Poisson(mu/2), variance mu per cell. A sum of independent Skellams is a
# Skellam, so K clients adding variance mu/K each give exactly Sk with variance mu, and integer counts plus integer noise go through the
# modular arithmetic of secure aggregation without rounding. RDP (Agarwal, Kairouz, Liu, NeurIPS 2021, Corollary 3.6), integer alpha > 1:
#   eps(alpha) <= alpha D2^2 / (2 mu) + min(((2 alpha - 1) D2^2 + 6 D1) / (4 mu^2), 3 D1 / (2 mu))
# with D1, D2 the L1, L2 sensitivities (a release of m tables: D1 = m, D2 = sqrt(m)). Composition adds RDP; conversion to (eps, delta) as in
# Opacus' RDP accountant (Balle et al. 2020).
SK_ALPHAS = list(range(2, 64)) + list(range(64, 1025, 8))


def skellam_rdp(alpha: int, m: int, mu: float) -> float:
    d1, d2sq = float(m), float(m)
    return alpha * d2sq / (2 * mu) + min(((2 * alpha - 1) * d2sq + 6 * d1) / (4 * mu ** 2), 3 * d1 / (2 * mu))


def rdp_to_eps(rdp: Sequence[float], alphas: Sequence[int], delta: float) -> float:
    return float(min(r - (math.log(delta) + math.log(a)) / (a - 1) + math.log((a - 1) / a) for r, a in zip(rdp, alphas)))


def skellam_eps(releases: Sequence[tuple[int, float]], delta: float) -> float:
    """(eps, delta) of the composition of Skellam releases [(number of tables m, variance mu), ...]."""
    return rdp_to_eps([sum(skellam_rdp(a, m, mu) for m, mu in releases) for a in SK_ALPHAS], SK_ALPHAS, delta)


def skellam_mus(eps: float, delta: float, sizes: Sequence[int], split: Sequence[float]) -> list[float]:
    """Variance per cell of every release: mu_r = m_r / (2 t s_r) (the Gaussian split of rho), with the largest t (least noise) whose
    composed Skellam epsilon is <= eps."""
    mus = lambda t: [m / (2.0 * t * s) for m, s in zip(sizes, split)]          # noqa: E731
    lo, hi = 1e-8, rho_from_eps(eps, delta)
    while skellam_eps(list(zip(sizes, mus(hi))), delta) <= eps:
        lo, hi = hi, hi * 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if skellam_eps(list(zip(sizes, mus(mid))), delta) <= eps:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6 * hi:
            break
    return mus(lo)


# --------------------------------------------------------------------------------------------------
# Attributes
# --------------------------------------------------------------------------------------------------
@dataclass
class Attr:
    name: str
    kind: str                     # numeric | flag | categorical
    start: int
    width: int                    # one-hot width (categorical), 1 otherwise
    n_fine: int
    dead: np.ndarray | None = None


def attributes(schema: dict[str, Any]) -> list[Attr]:
    from ppfeddata.models.generate import dead_category_mask

    dead = dead_category_mask(schema)
    out = []
    for b in schema["blocks"]:
        if b["type"] == "numeric":
            out.append(Attr(b["name"], "numeric", b["start"], 1, FINE))
        elif b["type"] in ("binary", "na_flag"):
            out.append(Attr(b["name"], "flag", b["start"], 1, 2))
        else:
            out.append(Attr(b["name"], "categorical", b["start"], b["width"], b["width"], dead[b["start"]:b["start"] + b["width"]].copy()))
    return out


def discretise_fine(X: np.ndarray, attrs: Sequence[Attr], clip: float) -> np.ndarray:
    """[n, A] fine values."""
    F = np.zeros((len(X), len(attrs)), dtype=np.int64)
    for j, a in enumerate(attrs):
        if a.kind == "numeric":
            F[:, j] = np.clip(np.floor((X[:, a.start] + clip) / (2 * clip) * FINE), 0, FINE - 1)
        elif a.kind == "flag":
            F[:, j] = (X[:, a.start] >= 0.5).astype(np.int64)
        else:
            F[:, j] = X[:, a.start:a.start + a.width].argmax(1)
    return F


def coarse_maps(fine_hist: list[np.ndarray], attrs: Sequence[Attr], bins: int) -> list[np.ndarray]:
    """Per attribute, fine value -> coarse value. Numeric: `bins` bins at the quantiles of the pooled noisy fine histogram (edges on fine
    bins; empty bins merged); flags and categories unchanged."""
    out = []
    for a, h in zip(attrs, fine_hist):
        if a.kind != "numeric":
            out.append(np.arange(a.n_fine))
            continue
        p = np.maximum(h.sum(0), 0.0) + 1e-9                    # pooled over classes
        cdf = np.cumsum(p) / p.sum()
        cut = np.searchsorted(cdf, np.arange(1, bins) / bins, side="left")
        m = np.zeros(a.n_fine, dtype=np.int64)
        for c in np.unique(cut):
            m[c + 1:] += 1
        out.append(np.unique(m, return_inverse=True)[1].astype(np.int64))
    return out


# --------------------------------------------------------------------------------------------------
# Releases (federated sums with Gaussian noise)
# --------------------------------------------------------------------------------------------------
def client_noise(shape: tuple[int, ...], scale: float, noise_share: float, rng: np.random.Generator, mechanism: str = "gaussian") -> np.ndarray:
    """One client's noise for one table: Gaussian with std `scale` x sqrt(share), or Skellam with variance `scale` x share (integers)."""
    if mechanism == "gaussian":
        return rng.normal(0.0, scale * math.sqrt(noise_share), size=shape)
    lam = scale * noise_share / 2.0
    return (rng.poisson(lam, size=shape) - rng.poisson(lam, size=shape)).astype(np.float64)


def _noisy_sum(tables_per_client: list[list[np.ndarray]], sigma: float, noise_share: float, rng: np.random.Generator,
               mechanism: str = "gaussian") -> list[np.ndarray]:
    """Sum over clients of their tables, each client adding its share of the noise per cell (what SecAgg returns). `sigma` is the std
    (Gaussian) or the variance mu (Skellam) of the summed noise."""
    out = [np.zeros_like(t, dtype=np.float64) for t in tables_per_client[0]]
    for tabs in tables_per_client:
        for o, t in zip(out, tabs):
            o += t + client_noise(t.shape, sigma, noise_share, rng, mechanism)
    return out


def _count(idx: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.bincount(np.ravel_multi_index(idx, shape), minlength=int(np.prod(shape))).reshape(shape).astype(np.float64)


def max_spanning_tree(w: np.ndarray) -> list[tuple[int, int]]:
    """Prim on a dense symmetric weight matrix; edges (parent, child) from node 0."""
    n = len(w)
    seen, edges = {0}, []
    best = w[0].copy()
    parent = np.zeros(n, dtype=np.int64)
    best[0] = -np.inf
    for _ in range(n - 1):
        cand = np.where(np.isin(np.arange(n), list(seen)), -np.inf, best)
        j = int(np.argmax(cand))
        edges.append((int(parent[j]), j))
        seen.add(j)
        upd = w[j] > best
        parent[upd] = j
        best = np.maximum(best, w[j])
    return edges


def mutual_information(t: np.ndarray) -> float:
    p = np.maximum(t, 0.0) + 1e-3
    p /= p.sum()
    pi, pj = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    return float((p * np.log(p / (pi * pj))).sum())


@dataclass
class MarginalModel:
    attrs: list[Attr]
    n_classes: int
    clip: float
    fine: list[np.ndarray]                      # [A] of [k, n_fine]
    maps: list[np.ndarray]                      # [A] fine -> coarse
    edges: list[tuple[int, int]]
    root_table: np.ndarray                      # [n_coarse(root), k]
    edge_tables: list[np.ndarray]               # [parent coarse, child coarse, k]
    info: dict[str, Any] = field(default_factory=dict)


# ---- stages shared by the in-process simulation (`fit`) and the Flower app (`fl/mg_app.py`)
STAGES = ("fine", "pairs", "edges")


def pair_list(A: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(A) for j in range(i + 1, A)]


def stage_sizes(A: int) -> dict[str, int]:
    """Number of tables of each release (= L1 sensitivity, and the squared L2 sensitivity)."""
    return {"fine": A, "pairs": len(pair_list(A)), "edges": A}            # edges: the root table + A - 1 tree edges


def noise_scales(eps: float, delta: float, A: int, split: Sequence[float], mechanism: str = "gaussian") -> dict[str, float]:
    """Per release: the std (Gaussian) or variance (Skellam) of the summed noise per cell, for the target (eps, delta)."""
    sizes = stage_sizes(A)
    if mechanism == "gaussian":
        rho = rho_from_eps(eps, delta)
        return {s: sigma_for(rho * f, sizes[s]) for s, f in zip(STAGES, split)}
    return dict(zip(STAGES, skellam_mus(eps, delta, [sizes[s] for s in STAGES], split)))


def privacy_info(scales: dict[str, float], A: int, delta: float, share: float, mechanism: str) -> dict[str, float]:
    """eps of the summed release, and with a single honest client (its share of the noise only)."""
    sizes = stage_sizes(A)
    if mechanism == "gaussian":
        rho = sum(sizes[s] / (2 * scales[s] ** 2) for s in STAGES)
        return {"rho": rho, "eps": eps_from_rho(rho, delta), "eps_one_honest": eps_from_rho(rho / share, delta) if share < 1 else eps_from_rho(rho, delta)}
    e = skellam_eps([(sizes[s], scales[s]) for s in STAGES], delta)
    e1 = skellam_eps([(sizes[s], scales[s] * share) for s in STAGES], delta) if share < 1 else e
    return {"eps": e, "eps_one_honest": e1}


def eps_honest_curve(scales: dict[str, float], A: int, delta: float, K: int, mechanism: str) -> dict[int, float]:
    """eps when only h of the K clients add their share of the noise (the others collude and remove theirs)."""
    return {h: privacy_info(scales, A, delta, h / K, mechanism)["eps_one_honest"] for h in range(1, K + 1)}


def client_tables(stage: str, F: np.ndarray, y: np.ndarray, attrs: Sequence[Attr], k: int, maps: Sequence[np.ndarray] | None = None,
                  edges: Sequence[tuple[int, int]] | None = None) -> list[np.ndarray]:
    """One client's count tables of a release (no noise)."""
    A = len(attrs)
    if stage == "fine":
        return [_count((y, F[:, j]), (k, a.n_fine)) for j, a in enumerate(attrs)]
    nc = [int(m.max()) + 1 for m in maps]
    C = np.stack([maps[j][F[:, j]] for j in range(A)], 1)
    if stage == "pairs":
        return [_count((C[:, i], C[:, j]), (nc[i], nc[j])) for i, j in pair_list(A)]
    return [_count((C[:, 0], y), (nc[0], k))] + [_count((C[:, a], C[:, b], y), (nc[a], nc[b], k)) for a, b in edges]


def tree_from_pairs(pt: Sequence[np.ndarray], A: int) -> list[tuple[int, int]]:
    W = np.zeros((A, A))
    for (i, j), t in zip(pair_list(A), pt):
        W[i, j] = W[j, i] = mutual_information(t)
    return max_spanning_tree(W)


def fit(X: np.ndarray, y: np.ndarray, parts: Sequence[np.ndarray], schema: dict[str, Any], eps: float, delta: float, bins: int = 8,
        split: Sequence[float] = SPLIT, noise_share: float | None = None, seed: int = 0, mechanism: str = "gaussian") -> MarginalModel:
    """The three federated releases, simulated in-process (SecAgg returns the exact sum). `noise_share` None -> 1 / number of clients
    (distributed); 1.0 -> local DP. `mechanism`: "gaussian" (O3.1) or "skellam" (O2: integer noise, exact under SecAgg)."""
    K = len(parts)
    share = 1.0 / K if noise_share is None else float(noise_share)
    attrs, k, clip = attributes(schema), len(schema["label_map"]), float(schema["settings"]["clip_sigma"])
    A = len(attrs)
    sc = noise_scales(eps, delta, A, split, mechanism)
    rng = np.random.default_rng([int(seed), 4241])
    Fc = [discretise_fine(X[p], attrs, clip) for p in parts]
    yc = [np.asarray(y[p], dtype=np.int64) for p in parts]
    fine = _noisy_sum([client_tables("fine", F, yy, attrs, k) for F, yy in zip(Fc, yc)], sc["fine"], share, rng, mechanism)
    maps = coarse_maps(fine, attrs, bins)
    pt = _noisy_sum([client_tables("pairs", F, yy, attrs, k, maps) for F, yy in zip(Fc, yc)], sc["pairs"], share, rng, mechanism)
    edges = tree_from_pairs(pt, A)
    tabs = _noisy_sum([client_tables("edges", F, yy, attrs, k, maps, edges) for F, yy in zip(Fc, yc)], sc["edges"], share, rng, mechanism)
    return assemble(attrs, k, clip, fine, maps, edges, tabs, sc, eps, delta, share, mechanism, bins, split, K,
                    n_cells=int(sum(t.size for t in fine) + sum(t.size for t in pt) + sum(t.size for t in tabs)))


def assemble(attrs, k, clip, fine, maps, edges, tabs, scales, eps, delta, share, mechanism, bins, split, K, n_cells) -> MarginalModel:
    info = {"eps_target": float(eps), "delta": float(delta), "noise_share": share, "mechanism": mechanism, **privacy_info(scales, len(attrs), delta, share, mechanism),
            "scales": dict(scales), "bins": int(bins), "split": list(split), "n_cells": int(n_cells), "tree": [list(e) for e in edges], "num_clients": K}
    if mechanism == "gaussian":
        info["sigmas"] = dict(scales)
    return MarginalModel(list(attrs), k, clip, list(fine), list(maps), [tuple(e) for e in edges], tabs[0], list(tabs[1:]), info)


# --------------------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------------------
def _probs(t: np.ndarray, alpha: float, axis: int = 0) -> np.ndarray:
    p = np.maximum(t, 0.0) + alpha
    return p / p.sum(axis=axis, keepdims=True)


def _draw(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One draw per row of a [n, m] probability matrix."""
    return (p.cumsum(1) > rng.random((len(p), 1))).argmax(1)


def sample(model: MarginalModel, schema: dict[str, Any], class_counts: Sequence[int], seed: int, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    from ppfeddata.models.generate import postprocess

    rng = np.random.default_rng(seed)
    k, A = model.n_classes, len(model.attrs)
    y = np.repeat(np.arange(k), np.asarray(class_counts, dtype=np.int64))
    n = len(y)
    C = np.zeros((n, A), dtype=np.int64)
    C[:, 0] = _draw(_probs(model.root_table, alpha, 0)[:, y].T, rng)
    for (a, b), t in zip(model.edges, model.edge_tables):
        p = _probs(t, alpha, 1)                              # [parent, child, k], normalised over the child
        C[:, b] = _draw(p[C[:, a], :, y], rng)
    raw = np.zeros((n, schema["n_features"]), dtype=np.float32)
    for j, at in enumerate(model.attrs):
        m = model.maps[j]
        if at.kind == "numeric":
            fine = np.zeros(n, dtype=np.int64)
            for c in range(k):
                for v in np.unique(C[y == c, j]):
                    rows = np.flatnonzero((y == c) & (C[:, j] == v))
                    cand = np.flatnonzero(m == v)
                    w = np.maximum(model.fine[j][c, cand], 0.0) + 1e-3
                    fine[rows] = cand[_draw(np.tile(w / w.sum(), (len(rows), 1)), rng)]
            width = 2 * model.clip / FINE
            raw[:, at.start] = -model.clip + (fine + rng.random(n)) * width
        elif at.kind == "flag":
            raw[:, at.start] = np.where(C[:, j] == 1, 30.0, -30.0)
        else:
            lg = np.full((n, at.width), -30.0, dtype=np.float32)
            lg[np.arange(n), C[:, j]] = 30.0
            raw[:, at.start:at.start + at.width] = lg
    # like the CVAE decoder output, the numeric columns are encoded values; postprocess decodes and re-encodes them (integer rounding,
    # raw bounds, is_na consistency)
    return postprocess(raw, schema, rng, None, y), y
