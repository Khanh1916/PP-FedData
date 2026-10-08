"""Optimisation O3(b): FedDP-Marginal generalised to a per-class Bayesian network (the tree of `marginal.py` is the case degree 1, shared
structure, shared bins).

Options (each one keeps the three releases and their number of tables per record, so the accounting of `marginal.py` is unchanged:
`fine` A tables, `pairs` P = A(A-1)/2 tables, `cond` A tables; one record adds 1 to one cell of each table of its own class):

- `degree` 1 (tree, Chow-Liu) or 2 (PrivBayes-style: every attribute gets up to 2 parents among the attributes already placed, chosen by
  the noisy mutual information of the `pairs` release, i.e. by post-processing; the conditional tables are then (parent 1, parent 2, child)
  per class);
- `class_trees`: the `pairs` tables are released per class (a record is in one class, so it still adds to P tables) and every class gets its
  own structure, instead of one structure from the pooled pairs;
- `class_maps`: the coarse bins of a numeric attribute come from the class's own noisy fine histogram, with `bins_rare` bins for the rare
  classes (noisy class size <= `rare_limit`, read from the `fine` release: post-processing) and `bins` for the others. Fewer, larger cells
  give a better signal-to-noise ratio to the small classes. A smaller noise for the rare classes would NOT be valid: their records would get a
  weaker guarantee than the announced epsilon;
- `fine`: number of fine bins of a numeric attribute.

The release sizes are those of `marginal.stage_sizes`, so `marginal.noise_scales` and `marginal.privacy_info` apply as they are.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ppfeddata.models import marginal as mg


@dataclass
class BNModel:
    attrs: list[mg.Attr]
    n_classes: int
    clip: float
    fine: list[np.ndarray]                       # [A] of [k, n_fine]
    maps: list[list[np.ndarray]]                 # [k][A] fine -> coarse
    order: list[list[int]]                       # [k] attributes in sampling order
    parents: list[dict[int, tuple[int, ...]]]    # [k] attribute -> parents
    tables: list[dict[int, np.ndarray]]          # [k] attribute -> noisy counts [parents..., child]
    info: dict[str, Any] = field(default_factory=dict)


def structure(W: np.ndarray, degree: int) -> tuple[list[int], dict[int, tuple[int, ...]]]:
    """Greedy order from attribute 0: next = the attribute with the largest MI to one already placed; its parents = the `degree` placed
    attributes with the largest MI to it (degree 1 = Prim's maximum spanning tree)."""
    A = len(W)
    order, parents = [0], {0: ()}
    while len(order) < A:
        rest = [j for j in range(A) if j not in parents]
        j = max(rest, key=lambda r: max(W[r, i] for i in order))
        parents[j] = tuple(sorted(order, key=lambda i: -W[j, i])[:degree])
        order.append(j)
    return order, parents


def _coarse(F: np.ndarray, maps: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([maps[j][F[:, j]] for j in range(F.shape[1])], 1)


def fit(X: np.ndarray, y: np.ndarray, parts: Sequence[np.ndarray], schema: dict[str, Any], eps: float, delta: float, bins: int = 8,
        split: Sequence[float] = mg.SPLIT, noise_share: float | None = None, seed: int = 0, mechanism: str = "skellam", degree: int = 1,
        class_trees: bool = False, class_maps: bool = False, bins_rare: int | None = None, fine: int = mg.FINE, rare_limit: int = 5000) -> BNModel:
    K = len(parts)
    share = 1.0 / K if noise_share is None else float(noise_share)
    attrs, k, clip = mg.attributes(schema, fine), len(schema["label_map"]), float(schema["settings"]["clip_sigma"])
    A = len(attrs)
    sc = mg.noise_scales(eps, delta, A, split, mechanism)
    rng = np.random.default_rng([int(seed), 5113])
    Fc = [mg.discretise_fine(X[p], attrs, clip) for p in parts]
    yc = [np.asarray(y[p], dtype=np.int64) for p in parts]
    pairs = mg.pair_list(A)

    # 1. per-class fine histograms
    fine_t = mg._noisy_sum([mg.client_tables("fine", F, yy, attrs, k) for F, yy in zip(Fc, yc)], sc["fine"], share, rng, mechanism)
    n_class = np.maximum(fine_t[0].sum(1), 0.0)                         # noisy class sizes (post-processing)
    pooled = mg.coarse_maps(fine_t, attrs, bins)
    if class_maps:
        nb = [int(bins_rare or bins) if n_class[c] <= rare_limit else int(bins) for c in range(k)]
        maps = [mg.coarse_maps([h[c:c + 1] for h in fine_t], attrs, nb[c]) for c in range(k)]
    else:
        maps = [pooled] * k

    # 2. pairs: pooled (one structure) or per class (one structure per class); same P tables per record either way
    if class_trees:
        def ptabs(F, yy):
            out = []
            for c in range(k):
                C, m = _coarse(F[yy == c], maps[c]), maps[c]
                nc = [int(x.max()) + 1 for x in m]
                out += [mg._count((C[:, i], C[:, j]), (nc[i], nc[j])) for i, j in pairs]
            return out
        pt = mg._noisy_sum([ptabs(F, yy) for F, yy in zip(Fc, yc)], sc["pairs"], share, rng, mechanism)
        Ws = []
        for c in range(k):
            W = np.zeros((A, A))
            for (i, j), t in zip(pairs, pt[c * len(pairs):(c + 1) * len(pairs)]):
                W[i, j] = W[j, i] = mg.mutual_information(t)
            Ws.append(W)
    else:
        pt = mg._noisy_sum([mg.client_tables("pairs", F, yy, attrs, k, pooled) for F, yy in zip(Fc, yc)], sc["pairs"], share, rng, mechanism)
        W = np.zeros((A, A))
        for (i, j), t in zip(pairs, pt):
            W[i, j] = W[j, i] = mg.mutual_information(t)
        Ws = [W] * k
    struct = [structure(W, degree) for W in Ws]

    # 3. conditional tables per class: one table per attribute
    def ctabs(F, yy):
        out = []
        for c in range(k):
            C, m = _coarse(F[yy == c], maps[c]), maps[c]
            nc = [int(x.max()) + 1 for x in m]
            order, par = struct[c]
            out += [mg._count(tuple(C[:, q] for q in par[a]) + (C[:, a],), tuple(nc[q] for q in par[a]) + (nc[a],)) for a in order]
        return out
    ct = mg._noisy_sum([ctabs(F, yy) for F, yy in zip(Fc, yc)], sc["edges"], share, rng, mechanism)
    tables = []
    for c in range(k):
        order = struct[c][0]
        tables.append({a: t for a, t in zip(order, ct[c * A:(c + 1) * A])})
    info = {"eps_target": float(eps), "delta": float(delta), "noise_share": share, "mechanism": mechanism,
            **mg.privacy_info(sc, A, delta, share, mechanism), "scales": dict(sc), "bins": int(bins), "bins_rare": bins_rare, "split": list(split),
            "degree": int(degree), "class_trees": bool(class_trees), "class_maps": bool(class_maps), "fine": int(fine), "num_clients": K,
            "n_cells": int(sum(t.size for t in fine_t) + sum(t.size for t in pt) + sum(t.size for t in ct)),
            "noisy_class_sizes": n_class.round(1).tolist()}
    return BNModel(attrs, k, clip, fine_t, maps, [s[0] for s in struct], [s[1] for s in struct], tables, info)


def loglik(model: BNModel, X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """log P(record | class) under the released model (the probabilities the sampler draws from): the membership score of O3(a)."""
    F = mg.discretise_fine(X, model.attrs, model.clip)
    y = np.asarray(y, dtype=np.int64)
    out = np.zeros(len(X))
    for c in range(model.n_classes):
        rows = np.flatnonzero(y == c)
        if not len(rows):
            continue
        C = _coarse(F[rows], model.maps[c])
        for a in model.order[c]:
            t = np.maximum(model.tables[c][a], 0.0) + alpha
            p = t / t.sum(-1, keepdims=True)
            idx = tuple(C[:, q] for q in model.parents[c][a]) + (C[:, a],)
            out[rows] += np.log(p[idx])
        out[rows] += _fine_loglik(model.fine, model.maps[c], model.attrs, c, F[rows], C)
    return out


def _fine_loglik(fine, maps, attrs, c: int, F: np.ndarray, C: np.ndarray) -> np.ndarray:
    """log P(fine bin | coarse bin, class) of the numeric attributes (uniform inside a fine bin adds a constant, left out)."""
    out = np.zeros(len(F))
    for j, at in enumerate(attrs):
        if at.kind != "numeric":
            continue
        w = np.maximum(fine[j][c], 0.0) + 1e-3
        Z = np.bincount(maps[j], weights=w)
        out += np.log(w[F[:, j]] / Z[C[:, j]])
    return out


def loglik_tree(model: mg.MarginalModel, X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """The same score for the tree model of `marginal.py` (shared bins, tables with the class as last axis)."""
    F = mg.discretise_fine(X, model.attrs, model.clip)
    y = np.asarray(y, dtype=np.int64)
    C = _coarse(F, model.maps)
    out = np.log(mg._probs(model.root_table, alpha, 0)[C[:, model.info.get("root", 0)], y])
    for (a, b), t in zip(model.edges, model.edge_tables):
        out += np.log(mg._probs(t, alpha, 1)[C[:, a], C[:, b], y])
    for c in range(model.n_classes):
        rows = np.flatnonzero(y == c)
        if len(rows):
            out[rows] += _fine_loglik(model.fine, model.maps, model.attrs, c, F[rows], C[rows])
    return out


def sample(model: BNModel, schema: dict[str, Any], class_counts: Sequence[int], seed: int, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    from ppfeddata.models.generate import postprocess

    rng = np.random.default_rng(seed)
    k, A = model.n_classes, len(model.attrs)
    y = np.repeat(np.arange(k), np.asarray(class_counts, dtype=np.int64))
    n = len(y)
    raw = np.zeros((n, schema["n_features"]), dtype=np.float32)
    for c in range(k):
        rows = np.flatnonzero(y == c)
        if not len(rows):
            continue
        C = np.zeros((len(rows), A), dtype=np.int64)
        for a in model.order[c]:
            t = np.maximum(model.tables[c][a], 0.0) + alpha
            p = t / t.sum(-1, keepdims=True)
            par = model.parents[c][a]
            pr = p[tuple(C[:, q] for q in par)] if par else np.tile(p, (len(rows), 1))
            C[:, a] = mg._draw(pr, rng)
        for j, at in enumerate(model.attrs):
            m = model.maps[c][j]
            if at.kind == "numeric":
                fine = np.zeros(len(rows), dtype=np.int64)
                for v in np.unique(C[:, j]):
                    sel = np.flatnonzero(C[:, j] == v)
                    cand = np.flatnonzero(m == v)
                    w = np.maximum(model.fine[j][c, cand], 0.0) + 1e-3
                    fine[sel] = cand[mg._draw(np.tile(w / w.sum(), (len(sel), 1)), rng)]
                raw[rows, at.start] = -model.clip + (fine + rng.random(len(rows))) * (2 * model.clip / at.n_fine)
            elif at.kind == "flag":
                raw[rows, at.start] = np.where(C[:, j] == 1, 30.0, -30.0)
            else:
                lg = np.full((len(rows), at.width), -30.0, dtype=np.float32)
                lg[np.arange(len(rows)), C[:, j]] = 30.0
                raw[rows, at.start:at.start + at.width] = lg
    return postprocess(raw, schema, rng, None, y), y
