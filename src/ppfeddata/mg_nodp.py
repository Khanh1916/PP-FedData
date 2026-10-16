"""Round Q1: FedDP-Marginal without DP, through Flower SecAgg+ (`MGs-nodp`) and without secure aggregation (`MGp-nodp`).

The two empty cells of the generator's protection matrix, as B3 and M2 are for the CVAE route:

|        | no SecAgg              | SecAgg+                    |
|--------|------------------------|----------------------------|
| no DP  | MGp-nodp (this round)  | MGs-nodp (this round)      |
| DP     | MGl-eps<e> (local DP)  | MGs-eps<e>, MGr-eps<e>     |

`MGs-nodp` is the epsilon = infinity reference of FedDP-Marginal itself (its DP cost was measured against B3, the CVAE without DP, so far);
`MGp-nodp` gives what SecAgg+ costs its releases (bytes and seconds against the same three rounds in plain Flower, measured on the same grid
tap). Without noise the secure sum equals the exact sum, so both runs release the same tables and the same synthetic data: only the overhead
differs, and the report checks that.

Stages: `search` (coarse bins chosen on validation, mean of 3 seeds, as every choice of the generator since spec v1.5; in-process, where the
simulated sum is the exact one; the budget split has no effect without noise) -> configs/best_marginal_nodp.yaml; `final` (3 seeds through
Flower, both modes) -> results/runs.csv; `mia` (the membership attack of `mia_marginal.py` on the released tables) -> results/mia_marginal.json
(merged); `report` -> results/mg_nodp.json, results/reports/mg_nodp.md.
"""
from __future__ import annotations

import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger("ppfeddata.mg_nodp")

BINS = [4, 8, 16, 32, 64]                 # 64 = the fine bins: no coarsening at all
BEST_PATH = Path("./configs/best_marginal_nodp.yaml")
NAMES = {"s": "MGs-nodp", "p": "MGp-nodp"}  # through Flower SecAgg+ / plain Flower rounds


def chosen_bins() -> int:
    return int(yaml.safe_load(BEST_PATH.read_text(encoding="utf-8"))["nodp"]["chosen_bins"])


def nodp_settings() -> dict[str, dict[str, Any]]:
    """Settings of the membership attack (`mia_marginal.py`); both names release the same tables, so the attack gives the same numbers."""
    from ppfeddata.models import marginal as mg

    if not BEST_PATH.exists():
        return {}
    s = {"eps": None, "bins": chosen_bins(), "split": list(mg.SPLIT), "options": None}
    return {nm: dict(s) for nm in NAMES.values()}


def search(cfg: dict[str, Any], seeds: tuple[int, ...] = (0, 1, 2), n_jobs: int = -1) -> dict[str, Any]:
    from ppfeddata.eval.baselines import load_data
    from ppfeddata.models import marginal as mg
    from ppfeddata.tune_dp_full import rare_labels, val_scores_xy
    from ppfeddata.tune_marginal import _parts

    data, schema = load_data(cfg)
    X, y = data["train"]["X"], np.asarray(data["train"]["y"], dtype=np.int64)
    rare, delta, k, spc = rare_labels(cfg, schema), float(cfg["dp"]["delta"]), len(schema["label_map"]), int(cfg["tune"]["syn_per_class"])
    rows = []
    for b in BINS:
        res = []
        for s in seeds:
            m = mg.fit(X, y, _parts(cfg, y, schema, s), schema, None, delta, bins=b, seed=s)
            Xs, ys = mg.sample(m, schema, [spc] * k, s)
            res.append(val_scores_xy(cfg, Xs, ys, schema, data, s, rare, n_jobs=n_jobs))
        r = {"bins": b, "cells": int(m.info["n_cells"])}
        for key in ("macro_f1", "bin_f1", "rare_recall"):
            r[f"{key}_mean"], r[f"{key}_std"] = float(np.mean([x[key] for x in res])), float(np.std([x[key] for x in res]))
        rows.append(r)
        logger.info("MG without DP, bins %d -> val macro-F1 %.4f +- %.4f, binary %.3f, rare %.3f, %d cells", b, r["macro_f1_mean"], r["macro_f1_std"],
                    r["bin_f1_mean"], r["rare_recall_mean"], r["cells"])
    best = max(rows, key=lambda r: r["macro_f1_mean"])
    out = {"nodp": {"chosen_bins": best["bins"], "rows": rows, "rule": "mean of 3 seeds on validation (spec v1.5)",
                    "split": "not used: without noise the budget split has no effect", "seeds": list(seeds)}}
    BEST_PATH.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    return out


def final(cfg: dict[str, Any], seeds: list[int] | None = None, resume: bool = True) -> list[str]:
    from ppfeddata.eval.baselines import load_data
    from ppfeddata.eval.runs import RunLedger, runs_csv_path
    from ppfeddata.fl import mg_app
    from ppfeddata.fl.m1 import _all_done
    from ppfeddata.models import marginal as mg
    from ppfeddata.models.b2 import evaluate_generator

    data, schema = load_data(cfg)
    ledger, mode, K = RunLedger(runs_csv_path(cfg)), cfg["label_mode"], int(cfg["fl"]["num_clients"])
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    bins = chosen_bins()
    for v, nm in NAMES.items():
        for seed in seeds:
            if resume and _all_done(ledger, [nm], seed, mode):
                logger.info("skip %s seed %d", nm, seed)
                continue
            fl = mg_app.run(cfg, seed, nm, None, bins, list(mg.SPLIT), "none", resume=resume, secure=v == "s")
            model = fl["model"]
            extra = {"alpha": float(cfg["fl"]["dirichlet_alpha"]), "num_clients": K, "rounds": 3, "local_epochs": 0,
                     "bytes_per_round": fl["bytes_total"] / 3, "sa_bytes_per_round": fl["bytes_total"] / 3 if v == "s" else None, "bytes_estimated": False,
                     "fl_total_s": fl["seconds"], "cvae_train_s": fl["seconds"], "sa_max_abs_diff": fl["max_abs_diff"], "dp_mechanism": "none",
                     "secure_aggregation": v == "s", "mg_bins": bins, "mg_cells": int(model.info["n_cells"])}
            logger.info("%s seed %d: %.2f MB, %.1f s, max |sum - exact| %g", nm, seed, fl["bytes_total"] / 1e6, fl["seconds"], fl["max_abs_diff"])
            evaluate_generator(cfg, None, nm, seed, data, schema, ledger, extra, use_stats=False,
                               sample_fn=lambda counts, sd, m_=model: mg.sample(m_, schema, counts, sd))
    return list(NAMES.values())


def mia(cfg: dict[str, Any], seeds: list[int] | None = None) -> dict[str, Any]:
    from ppfeddata import mia_marginal as mm

    return mm.run_all(cfg, settings=nodp_settings(), seeds=seeds, controls=False, merge=True)


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def _same_release(cfg: dict[str, Any], seed: int) -> dict[str, bool] | None:
    """Do the two modes release the same tables and the same synthetic data at this seed?"""
    from ppfeddata.eval.runs import run_id

    art = Path(cfg["compute"]["artifacts_dir"])
    rd = {v: art / run_id(nm, seed, cfg["label_mode"]) for v, nm in NAMES.items()}
    if not all((d / "mg_model.pkl").exists() and (d / "synthetic.npz").exists() for d in rd.values()):
        return None
    ms = {}
    for v, d in rd.items():
        with open(d / "mg_model.pkl", "rb") as fh:
            ms[v] = pickle.load(fh)
    a, b = ms["s"], ms["p"]
    tabs = lambda m: list(m.fine) + [m.root_table] + list(m.edge_tables)      # noqa: E731
    same_tables = a.edges == b.edges and all(np.array_equal(x, z) for x, z in zip(tabs(a), tabs(b)))
    sa, sb = np.load(rd["s"] / "synthetic.npz"), np.load(rd["p"] / "synthetic.npz")
    return {"tables": bool(same_tables), "synthetic": bool(np.array_equal(sa["X"], sb["X"]) and np.array_equal(sa["y"], sb["y"]))}


def _timing(d6, prefix: str) -> dict[str, float]:
    import pandas as pd

    g = d6[d6["config"] == f"{prefix}-TSTR-rf"].drop_duplicates("seed")
    num = lambda c: float(pd.to_numeric(g[c], errors="coerce").mean()) if c in g and len(g) else float("nan")       # noqa: E731
    mx = lambda c: float(pd.to_numeric(g[c], errors="coerce").max()) if c in g and len(g) else float("nan")         # noqa: E731
    est = "bytes_estimated" in g and len(g) and g["bytes_estimated"].astype(str).str.lower().eq("true").any()
    flower = not math.isnan(mx("sa_max_abs_diff"))            # ran through Flower (SecAgg+ or plain); the local variant runs in-process
    return {"seconds": num("fl_total_s") if flower else float("nan"), "mb_per_round": num("bytes_per_round") / 1e6, "max_abs_diff": mx("sa_max_abs_diff"),
            "bytes_estimated": bool(est), "flower": flower}


def _full_rule(cfg: dict[str, Any], pairs: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    """The rule of R1-R6 for each (label, a, b) of ledger configurations: seed-averaged difference a - b of the test macro-F1 (and of the rare-class
    recall), 95 % paired bootstrap over the test rows and, as a sensitivity, over whole TCP streams; an effect when the interval excludes 0 and
    |difference| exceeds the seed std of both (`eval/compare.py`)."""
    from ppfeddata.eval.compare import MACRO_F1, PairedBootstrap, compare_runs, recall_of
    from ppfeddata.eval.runs import load_predictions, run_id
    from ppfeddata.interpret import BOOT_SEED, load_test_split

    y, streams, _, classes = load_test_split(cfg)
    mode, seeds = cfg["label_mode"], [int(s) for s in cfg["seeds"]]
    preds = {}
    for n in sorted({n for _, a, b in pairs for n in (a, b)}):
        for s in seeds:
            try:
                preds[(n, s)] = load_predictions(cfg, run_id(n, s, mode))["y_pred"]
            except FileNotFoundError:
                continue
    if not preds:
        return []
    nb = int(cfg["eval"]["bootstrap"])
    boots = {"rows": PairedBootstrap(y, len(classes), nb, BOOT_SEED, "rows")}
    if streams is not None:
        boots["streams"] = PairedBootstrap(y, len(classes), nb, BOOT_SEED, "streams", streams)
    for b in boots.values():
        b.add(preds)
    rare = recall_of(*[classes.index(c) for c, v in cfg.get("quota", {}).items() if v[0] <= 5000 and c in classes])
    out = []
    for label, a, b in pairs:
        ss = [s for s in seeds if (a, s) in preds and (b, s) in preds]
        if not ss:
            continue
        ka, kb = [(a, s) for s in ss], [(b, s) for s in ss]
        c = compare_runs(boots["rows"], ka, kb, MACRO_F1)
        st = compare_runs(boots["streams"], ka, kb, MACRO_F1) if "streams" in boots else c
        rr = compare_runs(boots["rows"], ka, kb, rare)
        out.append({"label": label, "a": a, "b": b, "seeds": ss, **{k: c[k] for k in ("delta", "lo", "hi", "sigma", "effect")},
                    "stream_effect": st["effect"], "rare_delta": rr["delta"], "rare_effect": rr["effect"]})
    return out


def _mia_auc(mia: dict[str, Any], label: str) -> float | None:
    try:
        return float(mia["configs"][label]["calibrated"]["auc_mean"]["mean"])
    except (KeyError, TypeError):
        return None


def report(cfg: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    from ppfeddata.interpret import to_jsonable
    from ppfeddata.privacy_units import _setting
    from ppfeddata.tune_marginal import framework_name
    from ppfeddata.robustness import _rows

    res = Path(cfg["compute"]["runs_csv"]).parent
    df = pd.read_csv(Path(cfg["compute"]["runs_csv"]), low_memory=False)
    d6 = df[df["label_mode"] == cfg["label_mode"]]
    rare = [c for c, v in cfg.get("quota", {}).items() if v[0] <= 5000]
    mp = res / "mia_marginal.json"
    mia = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    with_cost = lambda r: None if r is None else {**r, **_timing(d6, r["config"]), "mia_auc": _mia_auc(mia, r["config"])}       # noqa: E731
    out: dict[str, Any] = {"bins": chosen_bins() if BEST_PATH.exists() else None, "matrix": {}, "dp_cost": [], "secagg_utility": [], "vs_b3": [],
                           "secagg_cost": None, "same_release": {}, "mia": {}, "r6_aside": None}
    out["matrix"]["plain"] = with_cost(_rows(d6, NAMES["p"], rare))
    out["matrix"]["secagg"] = with_cost(_rows(d6, NAMES["s"], rare))
    out["matrix"]["dp_local"], out["matrix"]["dp_secagg"] = [], []
    ref, p = out["matrix"]["secagg"], out["matrix"]["plain"]
    dp_pairs = []
    for e in [float(x) for x in cfg["dp"]["epsilons"]]:
        _, _, v = _setting(e)
        loc, dist = with_cost(_rows(d6, f"MGl-eps{e:g}", rare)), with_cost(_rows(d6, f"MG{v}-eps{e:g}", rare))
        if loc:
            out["matrix"]["dp_local"].append({**loc, "eps_target": e})
        if dist:
            out["matrix"]["dp_secagg"].append({**dist, "eps_target": e})
            # the DP cost of the framework's configuration against its own epsilon = infinity, both classifiers (as R4)
            dp_pairs += [(f"{e:g}|{clf}", f"MG{v}-eps{e:g}-TSTR-{clf}", f"{NAMES['s']}-TSTR-{clf}") for clf in ("rf", "mlp")]
    full = {r["label"]: r for r in _full_rule(cfg, dp_pairs
                                                 + [(f"secagg|{clf}", f"{NAMES['s']}-TSTR-{clf}", f"{NAMES['p']}-TSTR-{clf}") for clf in ("rf", "mlp")]
                                                 + [(f"b3|{clf}", f"{NAMES['s']}-TSTR-{clf}", f"B3-TSTR-{clf}") for clf in ("rf", "mlp")]
                                                 + [("r6|rf", f"{NAMES['s']}-TAug-rf", f"{framework_name(5.0)}-TAug-rf")])}
    for key, r in full.items():
        kind, tag = key.split("|")
        if kind == "secagg":
            out["secagg_utility"].append({**r, "classifier": tag})
        elif kind == "b3":
            out["vs_b3"].append({**r, "classifier": tag})
        elif kind == "r6":
            g = lambda n: d6[d6["config"] == n].drop_duplicates("seed")        # noqa: E731
            out["r6_aside"] = {**r, "taug_nodp": float(g(r["a"])["macro_f1"].mean()), "taug_mg_eps5": float(g(r["b"])["macro_f1"].mean()),
                               "synthetic_mia_nodp": float(pd.to_numeric(g(f"{NAMES['s']}-TSTR-rf")["mia_auc_mean"], errors="coerce").mean())}
        else:
            out["dp_cost"].append({**r, "eps": float(kind), "classifier": tag, "config": r["a"].rsplit("-TSTR-", 1)[0]})
    if p and ref:
        out["secagg_cost"] = {"bytes_ratio": ref["mb_per_round"] / p["mb_per_round"] if p["mb_per_round"] else None,
                              "time_ratio": ref["seconds"] / p["seconds"] if p["seconds"] else None,
                              "mb_per_round": [ref["mb_per_round"], p["mb_per_round"]], "seconds": [ref["seconds"], p["seconds"]]}
    for s in cfg["seeds"]:
        out["same_release"][int(s)] = _same_release(cfg, int(s))
    cal = lambda c: (c or {}).get("calibrated", {})                                                      # noqa: E731
    nd = cal(mia.get("configs", {}).get(NAMES["s"]))
    if nd:
        out["mia"] = {"nodp": nd["auc_mean"]["mean"], "nodp_rare": (nd.get("auc_rare_mean") or {}).get("mean"),
                      "nodp_tpr_1pct": (nd.get("tpr_at_1pct_fpr") or {}).get("mean"),
                      "control_nonoise_half": (cal(mia.get("controls", {}).get("nonoise_half")).get("auc_mean") or {}).get("mean"),
                      "dp": {r["config"]: r["mia_auc"] for r in out["matrix"]["dp_secagg"] if r.get("mia_auc") is not None},
                      "threshold": float(cfg["thresholds"]["mia_auc_max"])}
    (res / "mg_nodp.json").write_text(json.dumps(to_jsonable(out), indent=1), encoding="utf-8")
    (res / "reports" / "mg_nodp.md").write_text(render(out), encoding="utf-8")
    return out


def _f(x: Any, d: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{v:.{d}f}"


def _line(r: dict[str, Any], cell: str) -> str:
    eps = _f(r.get("eps"), 2) if r.get("eps") == r.get("eps") and r.get("eps") is not None else "-"
    mb = _f(r["mb_per_round"], 3) + (" (est.)" if r.get("bytes_estimated") else "")
    secs = _f(r["seconds"], 1) if r.get("flower", True) else "n/a (in-process)"
    return (f"| {cell} | {r['config']} | {_f(r['tstr_f1'])} ± {_f(r['tstr_f1_std'])} | {_f(r['bin_f1'])} | {_f(r['rare_recall'])} | {eps} | "
            f"{mb} | {secs} | {_f(r.get('mia_auc'))} |")


def render(out: dict[str, Any]) -> str:
    L = ["# FedDP-Marginal without DP: plain and through SecAgg+ (round Q1)", "",
         "Generated by `ppfeddata mg-nodp` from `results/runs.csv` and `results/mia_marginal.json`; do not edit. TSTR on the real test split, "
         "classifier chosen on validation, mean ± std over 3 seeds. MIA = calibrated AUC of the membership attack with access to the released "
         "tables (`mia_marginal.py`, positive control passed); not run for the local variant. MB / round: measured on the Flower grid, or "
         "(est.) = cells x the bytes per value measured for M2, for the local variant, which runs in-process.", "",
         f"Coarse bins without noise, chosen on validation (mean of 3 seeds): {out.get('bins')}.", "",
         "## The protection matrix of FedDP-Marginal", "",
         "| cell | configuration | TSTR macro-F1 | binary F1 | rare recall | ε | MB / round | FL seconds | MIA AUC |",
         "|---|---|---|---|---|---|---|---|---|"]
    m = out["matrix"]
    if m.get("plain"):
        L.append(_line(m["plain"], "no DP, no SecAgg"))
    if m.get("secagg"):
        L.append(_line(m["secagg"], "no DP, SecAgg+"))
    L += [_line(r, "DP (local), no SecAgg") for r in m.get("dp_local", [])]
    L += [_line(r, "DP (distributed), SecAgg+") for r in m.get("dp_secagg", [])]
    rule = ("Rule of R1-R6: difference of test macro-F1 averaged over the seeds, 95 % paired bootstrap over the test rows; an effect when the interval "
            "excludes 0 and |Δ| exceeds the seed std of both configurations; † = the verdict changes with the bootstrap over whole TCP streams.")
    L += ["", "## What DP costs FedDP-Marginal (against its own ε = ∞ reference, MGs-nodp)", "", rule, "",
          "| ε | configuration | classifier | Δ TSTR macro-F1 | 95 % interval | seed std | verdict | Δ rare recall |", "|---|---|---|---|---|---|---|---|"]
    L += [_cmp_line(r, f"{r['eps']:g} | {r['config']} | {r['classifier']}") for r in out["dp_cost"]]
    sc = out.get("secagg_cost")
    if sc or out["secagg_utility"]:
        L += ["", "## What SecAgg+ costs FedDP-Marginal (MGs-nodp against MGp-nodp)", ""]
        if sc:
            L += [f"Bytes per round {_f(sc['mb_per_round'][0])} against {_f(sc['mb_per_round'][1])} MB ({_f(sc['bytes_ratio'], 2)}x); FL seconds "
                  f"{_f(sc['seconds'][0], 1)} against {_f(sc['seconds'][1], 1)} ({_f(sc['time_ratio'], 2)}x).", ""]
        L += ["| classifier | Δ TSTR macro-F1 | 95 % interval | seed std | verdict | Δ rare recall |", "|---|---|---|---|---|---|"]
        L += [_cmp_line(r, r["classifier"]) for r in out["secagg_utility"]]
    L += ["", "## Same release with and without SecAgg+", "", "| seed | same tables | same synthetic data |", "|---|---|---|"]
    for s, r in out["same_release"].items():
        L.append(f"| {s} | {'n/a' if r is None else ('yes' if r['tables'] else 'NO')} | {'n/a' if r is None else ('yes' if r['synthetic'] else 'NO')} |")
    if out["vs_b3"]:
        L += ["", "## FedDP-Marginal without DP against the CVAE without DP (MGs-nodp - B3, TSTR)", "",
              "| classifier | Δ TSTR macro-F1 | 95 % interval | seed std | verdict | Δ rare recall |", "|---|---|---|---|---|---|"]
        L += [_cmp_line(r, r["classifier"]) for r in out["vs_b3"]]
    mi = out.get("mia")
    if mi:
        L += ["", "## Membership attack on the released tables", "",
              f"- Without DP: calibrated AUC {_f(mi['nodp'])} (rare classes {_f(mi.get('nodp_rare'))}); threshold {mi['threshold']:g}; positive control without noise "
              f"on half the data: {_f(mi.get('control_nonoise_half'))}.",
              "- With DP: " + ", ".join(f"{k} {_f(v)}" for k, v in mi["dp"].items()) + "."]
    r6 = out.get("r6_aside")
    if r6:
        L += ["", "## Under the rule of R6 (TAug-RF)", "",
              f"MGs-nodp {_f(r6['taug_nodp'], 4)} against MG-eps5 {_f(r6['taug_mg_eps5'], 4)}: Δ {r6['delta']:+.4f} [{r6['lo']:+.4f}, {r6['hi']:+.4f}], "
              f"seed std {_f(r6['sigma'], 4)}, {VERDICT[r6['effect']]}; the attack on the synthetic data that R6 filters on gives {_f(r6['synthetic_mia_nodp'])}."]
    return "\n".join(L) + "\n"


VERDICT = {"better": "better", "worse": "worse", "none": "no effect"}


def _cmp_line(r: dict[str, Any], head: str) -> str:
    dag = " †" if r.get("stream_effect") and r["stream_effect"] != r["effect"] else ""
    return (f"| {head} | {r['delta']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | {_f(r['sigma'])} | {VERDICT[r['effect']]}{dag} | {r['rare_delta']:+.3f} |")
