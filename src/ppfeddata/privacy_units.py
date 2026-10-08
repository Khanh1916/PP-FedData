"""Follow-up round, Gate P1: group-level DP (item 1) and the honest-client threshold (item 2) for FedDP-Marginal.

- Item 1 (`group_dp.py`): the protected unit is a TCP stream or a capture; every unit is held by one client and capped at m records. m is
  chosen on validation (mean of 3 seeds) among `CAPS`, per unit and epsilon; then 3 seeds through Flower SecAgg+ -> `MG<v>-strm<m>-eps<e>`,
  `MG<v>-cap<m>-eps<e>` in the run ledger.
- Item 2: each client adds 1/t of the calibrated variance, so the target epsilon holds as long as t clients are honest (t = K is MGs, t = 1 is
  the local variant); 3 seeds through Flower for t in `THRESHOLDS` -> `MG<v>-t<t>-eps<e>`; the epsilon with every client honest and with
  one honest client is recorded.
Both use the framework's configuration of each epsilon (bins and budget split of configs/best_marginal.yaml, the post-processing of O3.3 at
epsilon 1: v = r there, v = s otherwise). Report: results/privacy_units.json, results/reports/privacy_units.md.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger("ppfeddata.privacy_units")

CAPS = {"stream": [1, 2, 4], "capture": [25, 50, 100]}
THRESHOLDS = [3, 2]
BEST_PATH = Path("./configs/best_group_dp.yaml")
TAG = {"stream": "strm", "capture": "cap"}


def _setting(eps: float) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    """(bins and split, post-processing or None, variant letter) of the framework's configuration at `eps`."""
    from ppfeddata.tune_marginal import BEST_PATH as MB, REFINE_PATH

    b = yaml.safe_load(MB.read_text(encoding="utf-8"))["searches"][f"eps{eps:g}"]["best"]
    ref = None
    if eps == 1.0 and REFINE_PATH.exists():                  # the framework uses MGr at epsilon 1 only (O3.3 / spec v1.5)
        ref = (yaml.safe_load(REFINE_PATH.read_text(encoding="utf-8"))["searches"].get("eps1") or {}).get("chosen")
    return {"bins": int(b["bins"]), "split": list(b["split"])}, ref, ("r" if ref else "s")


def search(cfg: dict[str, Any], eps_list: list[float] | None = None, seeds: tuple[int, ...] = (0, 1, 2), n_jobs: int = -1) -> dict[str, Any]:
    from ppfeddata.eval.baselines import load_data
    from ppfeddata.group_dp import capped_parts
    from ppfeddata.models import marginal as mg
    from ppfeddata.tune_dp_full import rare_labels, val_scores_xy

    data, schema = load_data(cfg)
    X, y = data["train"]["X"], np.asarray(data["train"]["y"], dtype=np.int64)
    rare, delta, k, spc = rare_labels(cfg, schema), float(cfg["dp"]["delta"]), len(schema["label_map"]), int(cfg["tune"]["syn_per_class"])
    out = yaml.safe_load(BEST_PATH.read_text(encoding="utf-8")) if BEST_PATH.exists() else {}
    for eps in [float(e) for e in (eps_list or cfg["dp"]["epsilons"])]:
        st, ref, _ = _setting(eps)
        for unit, caps in CAPS.items():
            rows = []
            for m in caps:
                res, kept = [], []
                for s in seeds:
                    parts, _ = capped_parts(cfg, unit, m, s)
                    kept.append(sum(len(p) for p in parts))
                    model = mg.fit(X, y, parts, schema, eps, delta, seed=s, mechanism="skellam", rows=m, **st)
                    model = mg.refine(model, **ref) if ref else model
                    Xs, ys = mg.sample(model, schema, [spc] * k, s)
                    res.append(val_scores_xy(cfg, Xs, ys, schema, data, s, rare, n_jobs=n_jobs))
                r = {"m": m, "rows_kept": float(np.mean(kept))}
                for key in ("macro_f1", "bin_f1", "rare_recall"):
                    r[f"{key}_mean"], r[f"{key}_std"] = float(np.mean([x[key] for x in res])), float(np.std([x[key] for x in res]))
                rows.append(r)
                logger.info("group DP eps %g %s m %d -> val macro-F1 %.4f +- %.4f, rare %.3f, rows kept %.0f", eps, unit, m, r["macro_f1_mean"],
                            r["macro_f1_std"], r["rare_recall_mean"], r["rows_kept"])
            best = max(rows, key=lambda r: r["macro_f1_mean"])
            out.setdefault("searches", {}).setdefault(f"eps{eps:g}", {})[unit] = {"chosen_m": best["m"], "rows": rows,
                                                                                   "rule": "mean of 3 seeds on validation"}
            BEST_PATH.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    return out


def name(v: str, eps: float, unit: str | None = None, m: int | None = None, t: int | None = None) -> str:
    tag = f"{TAG[unit]}{m}" if unit else f"t{t}"
    return f"MG{v}-{tag}-eps{eps:g}"


def final(cfg: dict[str, Any], eps_list: list[float] | None = None, seeds: list[int] | None = None, resume: bool = True) -> list[str]:
    from ppfeddata.eval.baselines import load_data
    from ppfeddata.eval.runs import RunLedger, runs_csv_path
    from ppfeddata.fl import mg_app
    from ppfeddata.fl.m1 import _all_done
    from ppfeddata.group_dp import capped_parts
    from ppfeddata.models import marginal as mg
    from ppfeddata.models.b2 import evaluate_generator

    data, schema = load_data(cfg)
    ledger, mode, K = RunLedger(runs_csv_path(cfg)), cfg["label_mode"], int(cfg["fl"]["num_clients"])
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    chosen = yaml.safe_load(BEST_PATH.read_text(encoding="utf-8"))["searches"]
    names = []
    for eps in [float(e) for e in (eps_list or cfg["dp"]["epsilons"])]:
        st, ref, v = _setting(eps)
        jobs = [(unit, int(chosen[f"eps{eps:g}"][unit]["chosen_m"]), None) for unit in CAPS] + [(None, 1, t) for t in THRESHOLDS]
        for unit, m, t in jobs:
            nm = name(v, eps, unit, m, t)
            names.append(nm)
            for seed in seeds:
                if resume and _all_done(ledger, [nm], seed, mode):
                    continue
                pp = capped_parts(cfg, unit, m, seed)[1] if unit else None
                fl = mg_app.run(cfg, seed, nm, eps, st["bins"], st["split"], "skellam", resume=resume, rows=m, honest_t=t, parts_path=pp, unit=unit)
                model = mg.refine(fl["model"], **ref) if ref else fl["model"]
                i = model.info
                extra = {"alpha": float(cfg["fl"]["dirichlet_alpha"]), "num_clients": K, "rounds": 3, "local_epochs": 0,
                         "bytes_per_round": fl["bytes_total"] / 3, "sa_bytes_per_round": fl["bytes_total"] / 3, "bytes_estimated": False,
                         "fl_total_s": fl["seconds"], "cvae_train_s": fl["seconds"], "sa_max_abs_diff": fl["max_abs_diff"], "dp_mechanism": "skellam",
                         "dp_mode": "distributed", "dp_target_eps": eps, "dp_eps_max": i["eps"], "dp_eps_median": i["eps"], "dp_eps_one_honest": i["eps_one_honest"],
                         "dp_eps_all_honest": i.get("eps_all_honest"), "dp_delta": float(cfg["dp"]["delta"]), "dp_ratio_to_target": i["eps"] / eps,
                         "dp_unit": unit or "packet", "dp_unit_rows": m, "dp_honest_t": t or K, "mg_cells": int(i["n_cells"]), "mg_refine": str(ref),
                         "rows_kept": int(sum(len(p) for p in json.loads(Path(pp).read_text(encoding="utf-8"))["parts"])) if pp else None,
                         **{f"dp_eps_honest_{h}": e for h, e in i["eps_by_honest_clients"].items()}}
                logger.info("%s seed %d: eps %.3f (all honest %s, one honest %.1f), %.2f MB, max |secure - exact| %g", nm, seed, i["eps"],
                            i.get("eps_all_honest"), i["eps_one_honest"], fl["bytes_total"] / 1e6, fl["max_abs_diff"])
                evaluate_generator(cfg, None, nm, seed, data, schema, ledger, extra, use_stats=False,
                                   sample_fn=lambda counts, sd, m_=model: mg.sample(m_, schema, counts, sd))
    return names


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def report(cfg: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    from ppfeddata.interpret import to_jsonable
    from ppfeddata.robustness import _rows

    df = pd.read_csv(Path(cfg["compute"]["runs_csv"]), low_memory=False)
    d6 = df[df["label_mode"] == "6class"]
    rare = [c for c, v in cfg.get("quota", {}).items() if v[0] <= 5000]
    out: dict[str, Any] = {"units": [], "thresholds": []}
    for eps in [float(e) for e in cfg["dp"]["epsilons"]]:
        _, _, v = _setting(eps)
        base = _rows(d6, f"MG{v}-eps{eps:g}", rare)
        if base:
            out["units"].append({**base, "unit": "packet", "m": 1})
            out["thresholds"].append({**base, "t": int(cfg["fl"]["num_clients"])})
        for unit in CAPS:
            for nm in sorted({c.rsplit("-TSTR-", 1)[0] for c in d6["config"] if c.startswith(f"MG{v}-{TAG[unit]}") and c.endswith(("-TSTR-rf",)) and f"-eps{eps:g}-" in c}):
                r = _rows(d6, nm, rare)
                if r:
                    g = d6[d6["config"] == f"{nm}-TSTR-rf"]
                    out["units"].append({**r, "unit": unit, "m": int(g["dp_unit_rows"].iloc[0]), "rows_kept": float(g["rows_kept"].mean())})
        for t in THRESHOLDS:
            nm = name(v, eps, t=t)
            r = _rows(d6, nm, rare)
            if r:
                g = d6[d6["config"] == f"{nm}-TSTR-rf"]
                out["thresholds"].append({**r, "t": t, "eps_all_honest": float(g["dp_eps_all_honest"].mean())})
    res = Path(cfg["compute"]["runs_csv"]).parent
    (res / "privacy_units.json").write_text(json.dumps(to_jsonable(out), indent=1), encoding="utf-8")
    (res / "reports" / "privacy_units.md").write_text(render(out), encoding="utf-8")
    return out


def _f(x: Any, d: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{v:.{d}f}"


def render(out: dict[str, Any]) -> str:
    L = ["# Group-level DP and honest-client threshold of FedDP-Marginal (follow-up round, Gate P1)", "",
         "Generated by `ppfeddata privacy-units` from `results/runs.csv`; do not edit. TSTR on the real test split, classifier chosen on validation, "
         "mean ± std over 3 seeds; Skellam noise split over the clients, summed through Flower SecAgg+.", "",
         "## Protected unit: packet (record-level, as before), TCP stream, capture", "",
         "Each unit is held by one client and capped at m records; epsilon then holds for the whole unit.", "",
         "| configuration | unit | m | rows kept | TSTR macro-F1 | binary F1 | rare recall | ε (unit) |", "|---|---|---|---|---|---|---|---|"]
    for r in out["units"]:
        L.append(f"| {r['config']} | {r['unit']} | {r['m']} | {_f(r.get('rows_kept'), 0)} | {_f(r['tstr_f1'])} ± {_f(r['tstr_f1_std'])} | {_f(r['bin_f1'])} | "
                 f"{_f(r['rare_recall'])} | {_f(r['eps'], 2)} |")
    L += ["", "## Honest-client threshold t (5 clients)", "",
          "Each client adds 1/t of the calibrated noise: the target ε holds with t honest clients (t = 5: the distributed variant as before).", "",
          "| configuration | t | TSTR macro-F1 | binary F1 | rare recall | ε (t honest) | ε (all honest) | ε (one honest) |", "|---|---|---|---|---|---|---|---|"]
    for r in out["thresholds"]:
        L.append(f"| {r['config']} | {r['t']} | {_f(r['tstr_f1'])} ± {_f(r['tstr_f1_std'])} | {_f(r['bin_f1'])} | {_f(r['rare_recall'])} | {_f(r['eps'], 2)} | "
                 f"{_f(r.get('eps_all_honest', r['eps']), 2)} | {_f(r.get('eps_one_honest'), 1)} |")
    return "\n".join(L) + "\n"
