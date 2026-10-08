"""Optimisation O3: choice of the marginal generator's settings on validation, then the final runs (`MGd-eps<e>`, `MGl-eps<e>`).

The generator (`models/marginal.py`) is cheap (seconds per fit), so a small grid replaces the Optuna search: coarse bins x split of the
privacy budget, at each epsilon, with the distributed noise (seed 0, the same partition as the FL runs). Fitness as in O1/O2: validation
macro-F1 of a RF on `tune.syn_per_class` rows per class. The chosen setting of an epsilon is then trained with every seed in two
variants: `MGd` (noise split over the clients, the sum via secure aggregation; epsilon also for a single honest client) and `MGl` (every
client adds its full noise: local DP, no trust in the other clients), and `MGs` (O2: the distributed variant with Skellam noise, run through
Flower's real SecAgg+, `fl/mg_app.py`; bytes measured, the secure sum checked against the exact one). Limitation (as before): chosen on
non-private validation data; the settings of `MGs` are those chosen for the Gaussian noise.
"""
from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ppfeddata.eval.baselines import load_data
from ppfeddata.models import marginal as mg
from ppfeddata.tune_dp_full import rare_labels, val_scores_xy

logger = logging.getLogger("ppfeddata.tune_marginal")

BEST_PATH = Path("./configs/best_marginal.yaml")
GRID = {"bins": [4, 8, 16, 32], "split": [[0.3, 0.1, 0.6], [0.5, 0.1, 0.4], [0.15, 0.05, 0.8]]}


def _parts(cfg: dict[str, Any], y: np.ndarray, schema: dict[str, Any], seed: int) -> list[np.ndarray]:
    from ppfeddata.partition import make_partition

    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    return make_partition(cfg, y, classes, float(cfg["fl"]["dirichlet_alpha"]), seed, int(cfg["fl"]["num_clients"]))[0]


def search(cfg: dict[str, Any], eps_list: list[float] | None = None, seed: int = 0, n_jobs: int = -1) -> dict[str, Any]:
    data, schema = load_data(cfg)
    X, y = data["train"]["X"], np.asarray(data["train"]["y"], dtype=np.int64)
    parts, rare, delta = _parts(cfg, y, schema, seed), rare_labels(cfg, schema), float(cfg["dp"]["delta"])
    k, spc = len(schema["label_map"]), int(cfg["tune"]["syn_per_class"])
    out = yaml.safe_load(BEST_PATH.read_text(encoding="utf-8")) if BEST_PATH.exists() else {}
    for eps in [float(e) for e in (eps_list or cfg["dp"]["epsilons"])]:
        rows = []
        for bins, split in itertools.product(GRID["bins"], GRID["split"]):
            m = mg.fit(X, y, parts, schema, eps, delta, bins=bins, split=split, seed=seed)
            Xs, ys = mg.sample(m, schema, [spc] * k, seed)
            r = val_scores_xy(cfg, Xs, ys, schema, data, seed, rare, n_jobs=n_jobs)
            rows.append({"bins": bins, "split": list(split), **r, "eps": m.info["eps"], "eps_one_honest": m.info["eps_one_honest"]})
            logger.info("MG eps %g bins %d split %s -> val macro-F1 %.4f (binary %.3f, rare recall %.3f)", eps, bins, split, r["macro_f1"], r["bin_f1"],
                        r["rare_recall"])
        best = max(rows, key=lambda r: r["macro_f1"])
        out.setdefault("searches", {})[f"eps{eps:g}"] = {
            "epsilon": eps, "best": {"bins": best["bins"], "split": best["split"]}, "val_macro_f1": best["macro_f1"],
            "val": {k2: float(best[k2]) for k2 in ("macro_f1", "bin_f1", "rare_recall")}, "grid": rows,
            "fitness": "val macro-F1 of RF on syn_per_class rows per class; distributed noise, seed 0",
            "limitation": "tuned on real non-private validation data, one seed"}
        BEST_PATH.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    return out


BN_PATH = Path("./configs/best_marginal_bn.yaml")
# O3(b): options of models/marginal_bn.py tried one at a time from the current best setting of each epsilon
OPTIONS = {"class_trees": {"class_trees": True}, "degree2": {"degree": 2}, "degree2_bins4": {"degree": 2, "bins": 4},
           "class_maps_rare2": {"class_maps": True, "bins_rare": 2}, "class_maps_rare4": {"class_maps": True, "bins_rare": 4},
           "fine32": {"fine": 32}, "fine128": {"fine": 128}, "split_pairs20": {"split": [0.2, 0.2, 0.6]}, "split_cond80": {"split": [0.1, 0.1, 0.8]}}


def search_bn(cfg: dict[str, Any], eps_list: list[float] | None = None, seeds: tuple[int, ...] = (0, 1, 2), n_jobs: int = -1) -> dict[str, Any]:
    """O3(b): every option alone, then the combinations of the options that beat the reference by more than the seed std; each candidate
    scored by the MEAN over `seeds` (partition, noise and sampling seed) of the validation scores, with Skellam noise split over the clients.
    The reference is the tree of O3.1/O2 (`marginal.fit`) and its re-implementation in `marginal_bn` (degree 1, shared structure)."""
    from ppfeddata.models import marginal_bn as bn

    data, schema = load_data(cfg)
    X, y = data["train"]["X"], np.asarray(data["train"]["y"], dtype=np.int64)
    rare, delta, k, spc = rare_labels(cfg, schema), float(cfg["dp"]["delta"]), len(schema["label_map"]), int(cfg["tune"]["syn_per_class"])
    parts = {s: _parts(cfg, y, schema, s) for s in seeds}
    best0 = yaml.safe_load(BEST_PATH.read_text(encoding="utf-8"))["searches"]
    out = yaml.safe_load(BN_PATH.read_text(encoding="utf-8")) if BN_PATH.exists() else {}

    def score(name: str, eps: float, kw: dict[str, Any] | None) -> dict[str, Any]:
        rows = []
        for s in seeds:
            if kw is None:
                m = mg.fit(X, y, parts[s], schema, eps, delta, bins=base["bins"], split=base["split"], seed=s, mechanism="skellam")
                Xs, ys = mg.sample(m, schema, [spc] * k, s)
            else:
                m = bn.fit(X, y, parts[s], schema, eps, delta, seed=s, mechanism="skellam", **{**base, **kw})
                Xs, ys = bn.sample(m, schema, [spc] * k, s)
            rows.append(val_scores_xy(cfg, Xs, ys, schema, data, s, rare, n_jobs=n_jobs))
        r = {m_: (float(np.mean([x[m_] for x in rows])), float(np.std([x[m_] for x in rows]))) for m_ in ("macro_f1", "bin_f1", "rare_recall")}
        logger.info("MG-bn eps %g %-28s val macro-F1 %.4f +- %.4f, rare %.3f +- %.3f, binary %.3f", eps, name, *r["macro_f1"], *r["rare_recall"], r["bin_f1"][0])
        return {"name": name, "options": kw, **{f"{m_}_mean": v[0] for m_, v in r.items()}, **{f"{m_}_std": v[1] for m_, v in r.items()}}

    for eps in [float(e) for e in (eps_list or cfg["dp"]["epsilons"])]:
        base = {"bins": int(best0[f"eps{eps:g}"]["best"]["bins"]), "split": list(best0[f"eps{eps:g}"]["best"]["split"])}
        ref = score("tree (O2 reference)", eps, None)
        rows = [ref, score("bn degree 1 (same model)", eps, {})] + [score(n, eps, kw) for n, kw in OPTIONS.items()]
        tol = lambda r: max(r["macro_f1_std"], ref["macro_f1_std"])                       # noqa: E731
        wins = [r for r in rows[2:] if r["macro_f1_mean"] > ref["macro_f1_mean"] + tol(r)]
        # combinations of compatible winners (one value per option key)
        combos = []
        for i in range(len(wins)):
            for j in range(i + 1, len(wins)):
                a, b = wins[i]["options"], wins[j]["options"]
                if not set(a) & set(b):
                    combos.append(score(f"{wins[i]['name']}+{wins[j]['name']}", eps, {**a, **b}))
        rows += combos
        best = max(rows, key=lambda r: r["macro_f1_mean"])
        if best["macro_f1_mean"] <= ref["macro_f1_mean"] + tol(best):
            best = ref                                    # no option beats the reference beyond the seed std: keep the tree
        out.setdefault("searches", {})[f"eps{eps:g}"] = {"epsilon": eps, "base": base, "seeds": list(seeds), "chosen": best["name"],
                                                         "chosen_options": best["options"], "rows": rows,
                                                         "rule": "mean of 3 seeds on validation; an option is kept only when it beats the tree by more than the seed std"}
        BN_PATH.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    return out


def final_name(variant: str, eps: float) -> str:
    return f"MG{variant}-eps{eps:g}"


def run_final(cfg: dict[str, Any], eps_list: list[float] | None = None, seeds: list[int] | None = None, variants: tuple[str, ...] = ("d", "l", "s", "b"),
              resume: bool = True) -> list[str]:
    from ppfeddata.eval.overhead import Timer
    from ppfeddata.eval.runs import RunLedger, runs_csv_path
    from ppfeddata.fl import mg_app
    from ppfeddata.fl.dpfedsgd import bytes_per_param_m2
    from ppfeddata.fl.m1 import _all_done
    from ppfeddata.models.b2 import evaluate_generator

    from ppfeddata.models import marginal_bn as bn

    best = yaml.safe_load(BEST_PATH.read_text(encoding="utf-8"))["searches"]
    bn_best = yaml.safe_load(BN_PATH.read_text(encoding="utf-8"))["searches"] if BN_PATH.exists() else {}
    data, schema = load_data(cfg)
    X, y = data["train"]["X"], np.asarray(data["train"]["y"], dtype=np.int64)
    ledger, mode, delta, K = RunLedger(runs_csv_path(cfg)), cfg["label_mode"], float(cfg["dp"]["delta"]), int(cfg["fl"]["num_clients"])
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    bpp = bytes_per_param_m2(cfg)
    names = []
    for eps in [float(e) for e in (eps_list or cfg["dp"]["epsilons"])]:
        s = best[f"eps{eps:g}"]["best"]
        for v in variants:
            opts = (bn_best.get(f"eps{eps:g}") or {}).get("chosen_options")
            if v == "b" and opts is None:                 # O3(b) kept the tree at this epsilon: nothing new to run
                continue
            name = final_name(v, eps)
            names.append(name)
            for seed in seeds:
                if resume and _all_done(ledger, [name], seed, mode):
                    logger.info("skip %s seed %d", name, seed)
                    continue
                fl = None
                with Timer() as t:
                    if v == "b":                      # O3(b): per-class Bayesian network chosen on 3 seeds, Skellam split over the clients (simulated sum)
                        m = bn.fit(X, y, _parts(cfg, y, schema, seed), schema, eps, delta, seed=seed, mechanism="skellam",
                                   **{"bins": int(s["bins"]), "split": s["split"], **opts})
                    elif v == "s":                    # O2: Skellam noise, the releases through Flower's real SecAgg+ (fl/mg_app.py)
                        fl = mg_app.run(cfg, seed, name, eps, int(s["bins"]), s["split"], "skellam", resume=resume)
                        m = fl["model"]
                    else:
                        m = mg.fit(X, y, _parts(cfg, y, schema, seed), schema, eps, delta, bins=int(s["bins"]), split=s["split"],
                                   noise_share=None if v == "d" else 1.0, seed=seed)
                i = m.info
                cells = int(i["n_cells"])
                if fl is not None:
                    cost = {"bytes_per_round": fl["bytes_total"] / 3, "sa_bytes_per_round": fl["bytes_total"] / 3, "bytes_estimated": False,
                            "fl_total_s": fl["seconds"], "cvae_train_s": fl["seconds"], "sa_max_abs_diff": fl["max_abs_diff"], "dp_mechanism": "skellam",
                            **{f"dp_eps_honest_{h}": e for h, e in i["eps_by_honest_clients"].items()}}
                elif v == "b":                        # bytes = cells x the bytes per released value measured through SecAgg+ for MGs
                    bpv = bytes_per_value_mgs(ledger)
                    cost = {"bytes_per_round": (bpv * cells / 3) if not math.isnan(bpv) else None, "bytes_estimated": True, "fl_total_s": t.seconds,
                            "cvae_train_s": t.seconds, "dp_mechanism": "skellam", "mg_options": str(opts),
                            **{f"dp_eps_honest_{h}": e for h, e in mg.eps_honest_curve(i["scales"], len(m.attrs), delta, K, "skellam").items()}}
                else:
                    cost = {"bytes_per_round": (bpp * cells / 3) if not math.isnan(bpp) else None, "bytes_estimated": True, "fl_total_s": t.seconds,
                            "cvae_train_s": t.seconds, "dp_mechanism": "gaussian"}
                extra = {"alpha": float(cfg["fl"]["dirichlet_alpha"]), "num_clients": K, "rounds": 3, "local_epochs": 0, **cost,
                         "dp_mode": "local" if v == "l" else "distributed", "dp_target_eps": eps, "dp_eps_max": i["eps"], "dp_eps_median": i["eps"],
                         "dp_eps_one_honest": i["eps_one_honest"], "dp_delta": delta, "dp_ratio_to_target": i["eps"] / eps, "mg_bins": int(s["bins"]),
                         "mg_split": str(s["split"]), "mg_cells": cells, "mg_rho": i.get("rho")}
                logger.info("%s seed %d: eps %.3f (one honest %.1f), %d cells, %.1fs%s", name, seed, i["eps"], i["eps_one_honest"], cells, t.seconds,
                            f", {fl['bytes_total'] / 1e6:.2f} MB through SecAgg+, max |secure - exact| {fl['max_abs_diff']:g}" if fl else "")
                smp = bn.sample if v == "b" else mg.sample
                evaluate_generator(cfg, None, name, seed, data, schema, ledger, extra, use_stats=False,
                                   sample_fn=lambda counts, sd, m=m, smp=smp: smp(m, schema, counts, sd))
    return names


def bytes_per_value_mgs(ledger) -> float:
    """Bytes per released value through Flower SecAgg+, measured on the MGs runs (all three rounds / cells)."""
    df = ledger.frame()
    if not len(df) or "mg_cells" not in df:
        return math.nan
    r = df[df["config"].astype(str).str.match(r"^MGs-eps") & df["protocol"].eq("TSTR")]
    return float((r["bytes_per_round"] * 3 / r["mg_cells"]).mean()) if len(r) else math.nan
