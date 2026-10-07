"""Optimisation round O0: one scorecard for the protected configurations (B3, M1, M2, M3) and their Pareto front.

The user asked to optimise M1, M2 and M3 on every metric at once, so no single number ranks them. Each configuration gets:

- utility, all on the REAL test split, the classifier (RF or MLP) chosen per configuration and protocol by its VALIDATION macro-F1:
  `tstr_f1` (macro-F1, IDS trained on synthetic data only), `bin_f1` (attack vs normal, same run), `rare_recall` (mean recall of the rare classes,
  same run) and `taug_gain` (macro-F1 of real + synthetic minus B0 with the same classifier);
- privacy: `eps` (epsilon reached, max over clients; infinite without DP), `secagg` (the server sees only the sum of the updates), `mia_auc`
  (calibrated attack with access to the released model, `results/mia_model.json`; missing when that configuration was not attacked);
- cost: `bytes_per_round` and `fl_total_s`.

A row is `valid` when its numbers mean what its name says. The DP rows with residual noise (variant `standard`) are not: the per-class residual std
is computed from the train data without DP, so they are listed but kept out of the front until that statistic is private (stage O1).

Dominance uses the seed rule of the study: for a utility metric, a is better than b only when the difference is larger than the std over seeds of
both; within that band they are equal. Epsilon is compared with a 2 % tolerance (the accountant reaches 0.996-1.000 of the target), costs with 5 %.
A configuration is on the front when no valid configuration is at least as good on every axis and better on one.

The directly federated classifier of `fed_classifier.json` (`protected`) is listed as a reference: it detects as well but releases no data, so it
answers a different requirement and is never on the front.

`freeze_baseline` keeps the first scorecard as `results/scorecard_baseline.json`; later stages compare against it with the same seed rule.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppfeddata import aggregate as ag

logger = logging.getLogger("ppfeddata.scorecard")

UTILITY = ("tstr_f1", "bin_f1", "rare_recall", "taug_gain")
EPS_TOL = 0.02
COST_TOL = 0.05
COST = ("bytes_per_round", "fl_total_s")
PROTECTED = ("B3", "M1", "M2", "M3")
DIRECT = {"FedMLPcwT": (math.inf, False), "FedMLPcwT-sa": (math.inf, True), "FedMLPcwT-dp1": (1.0, False), "FedMLPcwT-dp5": (5.0, False),
          "FedMLPcwT-dp10": (10.0, False), "FedMLPcwT-dp5-sa": (5.0, True)}


# --------------------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------------------
def _row(summ: pd.DataFrame, name: str) -> pd.Series | None:
    r = summ[summ["config"] == name]
    return r.iloc[0] if len(r) else None


def _val(r: pd.Series | None, metric: str) -> tuple[float, float]:
    if r is None:
        return math.nan, math.nan
    m, s = r.get(f"{metric}_mean", math.nan), r.get(f"{metric}_std", math.nan)
    return float(m), (0.0 if pd.isna(s) else float(s))


def _pick(summ: pd.DataFrame, cfgs: dict[tuple[str, str], str], proto: str) -> tuple[str, str] | None:
    """(classifier, config) of `proto` with the higher validation macro-F1 (ties: RF)."""
    best = None
    for clf in ("rf", "mlp"):
        r = _row(summ, cfgs.get((proto, clf), ""))
        if r is None or pd.isna(r.get("val_macro_f1_mean", math.nan)):
            continue
        v = float(r["val_macro_f1_mean"])
        if best is None or v > best[0]:
            best = (v, clf, r["config"])
    return None if best is None else (best[1], best[2])


def _rare_recall(summ: pd.DataFrame, df: pd.DataFrame | None, name: str, rare: list[str]) -> tuple[float, float]:
    """Mean recall of the rare classes: per seed from the ledger when given (so the std is over seeds), else from the summary means."""
    if df is not None and len(df[df["config"] == name]):
        g = df[df["config"] == name].drop_duplicates("seed")
        per_seed = g[[f"recall_{c}" for c in rare]].astype(float).mean(axis=1)
        return float(per_seed.mean()), float(per_seed.std(ddof=0))
    r = _row(summ, name)
    if r is None:
        return math.nan, math.nan
    ms = [_val(r, f"recall_{c}") for c in rare]
    return float(np.mean([m for m, _ in ms])), float(math.sqrt(np.mean([s * s for _, s in ms])))


def _mia(mia: dict[str, Any] | None, label: str) -> float:
    c = (mia or {}).get("configs", {}).get(label)
    try:
        return float(c["calibrated"]["auc_mean"]["mean"])
    except (TypeError, KeyError):
        return math.nan


def candidate_rows(summ: pd.DataFrame, df: pd.DataFrame | None = None, mia: dict[str, Any] | None = None, rare: list[str] | None = None) -> list[dict[str, Any]]:
    """One scorecard row per protected entry of the matrix, plus the residual-noise twin of each DP entry (not valid, see the module doc)."""
    rare = rare or [c[len("recall_"):-len("_mean")] for c in summ.columns if c.startswith("recall_") and c.endswith("_mean")][-4:]
    ents = [e for e in ag.entries(summ) if e["label"].split("-")[0] in PROTECTED]
    twins = []
    for e in ents:
        if e.get("eps"):
            twins.append(dict(e, label=f"{e['label']} (residual noise)", twin=True, cfgs={k: v.replace("-plain-", "-") for k, v in e["cfgs"].items()}))
    rows = []
    for e in ents + twins:
        method = e["label"].split("-")[0]
        tstr, taug = _pick(summ, e["cfgs"], "TSTR"), _pick(summ, e["cfgs"], "TAug")
        if tstr is None:
            continue
        rt = _row(summ, tstr[1])
        r = {"label": e["label"], "method": method, "variant": "standard" if e.get("twin") else ("plain" if e.get("eps") else "standard"),
             "dp": bool(e.get("eps")), "secagg": method in ("M2", "M3"), "dp_mode": "local" if e.get("eps") else None,
             "eps_target": float(e["eps"]) if e.get("eps") else math.inf,
             "eps": float(rt.get("dp_eps_max_mean", math.nan)) if e.get("eps") else math.inf,
             "tstr_config": tstr[1], "tstr_classifier": tstr[0], "n_seeds": int(rt["n_seeds"])}
        r["tstr_f1"], r["tstr_f1_std"] = _val(rt, "macro_f1")
        r["bin_f1"], r["bin_f1_std"] = _val(rt, "bin_f1")
        r["rare_recall"], r["rare_recall_std"] = _rare_recall(summ, df, tstr[1], rare)
        if taug is not None:
            ra, b0 = _row(summ, taug[1]), _row(summ, f"B0-{taug[0]}")
            (m, s), (bm, bs) = _val(ra, "macro_f1"), _val(b0, "macro_f1")
            r.update(taug_config=taug[1], taug_classifier=taug[0], taug_gain=m - bm, taug_gain_std=max(s, bs))
        else:
            r.update(taug_config=None, taug_classifier=None, taug_gain=math.nan, taug_gain_std=math.nan)
        r["mia_auc"] = math.nan if e.get("twin") else _mia(mia, e["label"])
        for c in COST:
            r[c] = _val(rt, c)[0]
        r["valid"] = not (e.get("twin") and r["dp"])
        r["invalid_reason"] = "per-class residual std from train data without DP" if not r["valid"] else None
        rows.append(r)
    return rows


def direct_rows(fed: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The directly federated classifier at the same protection levels (reference only, releases no data)."""
    cls = ((fed or {}).get("protected") or {}).get("classifiers", {})
    out = []
    for name, (eps, sa) in DIRECT.items():
        c = cls.get(name)
        if not c:
            continue
        out.append({"label": name, "method": "direct", "dp": math.isfinite(eps), "secagg": sa, "eps_target": eps,
                    "eps": float(c.get("dp", {}).get("eps_max_over_seeds", eps)) if math.isfinite(eps) else math.inf,
                    "tstr_f1": float(c["macro_f1_mean"]), "tstr_f1_std": float(c["macro_f1_std"]),
                    "rare_recall": float(c.get("rare_recall_mean", math.nan)), "n_seeds": len(c.get("seeds", [])), "valid": True, "reference": True})
    return out


# --------------------------------------------------------------------------------------------------
# Dominance and the front
# --------------------------------------------------------------------------------------------------
def _cmp_util(a: dict[str, Any], b: dict[str, Any], m: str) -> int:
    """+1 a better, -1 b better, 0 equal within the seed std (or a value missing)."""
    x, y = a.get(m, math.nan), b.get(m, math.nan)
    if pd.isna(x) or pd.isna(y):
        return 0
    tol = max(a.get(f"{m}_std", 0.0) or 0.0, b.get(f"{m}_std", 0.0) or 0.0)
    return 1 if x > y + tol else (-1 if y > x + tol else 0)


def _cmp_low(x: float, y: float, rel: float) -> int:
    """+1 when x is lower (better) than y beyond a relative tolerance."""
    if pd.isna(x) or pd.isna(y) or (math.isinf(x) and math.isinf(y)):
        return 0
    if math.isinf(x) or math.isinf(y):
        return -1 if math.isinf(x) else 1
    return 1 if x < y * (1 - rel) else (-1 if y < x * (1 - rel) else 0)


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, int]:
    """Per axis: +1 a better, -1 b better, 0 equal."""
    out = {m: _cmp_util(a, b, m) for m in UTILITY}
    out["eps"] = _cmp_low(a["eps"], b["eps"], EPS_TOL)
    out["secagg"] = int(a["secagg"]) - int(b["secagg"])
    for c in COST:
        out[c] = _cmp_low(a.get(c, math.nan), b.get(c, math.nan), COST_TOL)
    return out


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    c = compare(a, b).values()
    return all(v >= 0 for v in c) and any(v > 0 for v in c)


def pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark `pareto` (and `dominated_by`) on every row; only valid, non-reference rows compete."""
    pool = [r for r in rows if r.get("valid") and not r.get("reference")]
    for r in rows:
        by = [o["label"] for o in pool if o is not r and dominates(o, r)] if r in pool else []
        r["dominated_by"] = by
        r["pareto"] = r in pool and not by
    return rows


# --------------------------------------------------------------------------------------------------
# Baseline and deltas
# --------------------------------------------------------------------------------------------------
def deltas(cur: list[dict[str, Any]], base: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per label present in both: the change of every utility metric and whether it passes the seed rule."""
    b = {r["label"]: r for r in base}
    out = []
    for r in cur:
        o = b.get(r["label"])
        if o is None:
            continue
        d = {"label": r["label"]}
        for m in UTILITY:
            x, y = r.get(m, math.nan), o.get(m, math.nan)
            d[m] = None if pd.isna(x) or pd.isna(y) else float(x - y)
            d[f"{m}_effect"] = {1: "better", -1: "worse", 0: "no change"}[_cmp_util(r, o, m)]
        out.append(d)
    return out


def build(summ: pd.DataFrame, df: pd.DataFrame | None = None, mia: dict[str, Any] | None = None, fed: dict[str, Any] | None = None,
          rare: list[str] | None = None, base: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = pareto(candidate_rows(summ, df, mia, rare) + direct_rows(fed))
    out = {"axes": {"utility": list(UTILITY), "privacy": ["eps", "secagg"], "cost": list(COST)}, "rule": {"utility": "difference larger than the std over seeds of both",
           "eps_rel_tol": EPS_TOL, "cost_rel_tol": COST_TOL}, "rows": rows, "front": [r["label"] for r in rows if r["pareto"]]}
    if base:
        out["vs_baseline"] = deltas([r for r in rows if not r.get("reference")], [r for r in base["rows"] if not r.get("reference")])
    return out


# --------------------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------------------
def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) else ("inf" if math.isinf(x) else float(x))
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _from_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _from_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_from_json(v) for v in x]
    if x == "inf":
        return math.inf
    return math.nan if x is None else x


def _fmt(x: Any, s: Any = None, d: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return f"{x:.{d}f}" + (f" ± {s:.{d}f}" if s is not None and not (isinstance(s, float) and math.isnan(s)) else "")


def render(sc: dict[str, Any]) -> str:
    L = ["# Scorecard of the protected configurations (optimisation round O0)", "",
         "Generated by `ppfeddata scorecard` from `results/summary.csv`, `results/runs.csv`, `results/mia_model.json` and `results/fed_classifier.json`; do not edit.",
         "Utility on the real test split, mean ± std over seeds; the classifier of each protocol is the one with the higher validation macro-F1. "
         "`taug_gain` = macro-F1 of real + synthetic minus B0 with the same classifier. ε = max over clients (∞ = no DP). "
         "Front = not dominated by a valid configuration on any axis (utility within the seed std counts as equal; ε within 2 %, cost within 5 %).", "",
         "| configuration | valid | front | TSTR macro-F1 | binary F1 | rare recall | TAug gain | ε | SecAgg | MIA AUC (model) | MB / round | FL time (s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sc["rows"]:
        if r.get("reference"):
            continue
        mb = r.get("bytes_per_round")
        L.append(f"| {r['label']} ({r['tstr_classifier']}) | {'yes' if r['valid'] else 'no'} | {'**yes**' if r['pareto'] else 'no'} | {_fmt(r['tstr_f1'], r['tstr_f1_std'])} | "
                 f"{_fmt(r['bin_f1'], r['bin_f1_std'])} | {_fmt(r['rare_recall'], r['rare_recall_std'])} | {_fmt(r['taug_gain'], r['taug_gain_std'])} | {_fmt(r['eps'], d=2)} | "
                 f"{'yes' if r['secagg'] else 'no'} | {_fmt(r['mia_auc'])} | {_fmt(mb / 1e6 if mb is not None and not (isinstance(mb, float) and math.isnan(mb)) else None, d=2)} | {_fmt(r['fl_total_s'], d=0)} |")
    bad = [r for r in sc["rows"] if not r["valid"]]
    if bad:
        L += ["", "Not valid (kept out of the front): " + "; ".join(f"{r['label']}: {r['invalid_reason']}" for r in bad) + "."]
    dom = [r for r in sc["rows"] if r.get("dominated_by")]
    if dom:
        L += ["", "Dominated: " + "; ".join(f"{r['label']} by {', '.join(r['dominated_by'])}" for r in dom) + "."]
    ref = [r for r in sc["rows"] if r.get("reference")]
    if ref:
        L += ["", "## Reference: the classifier trained directly by FL (releases no data)", "", "| configuration | macro-F1 | rare recall | ε | SecAgg |", "|---|---|---|---|---|"]
        L += [f"| {r['label']} | {_fmt(r['tstr_f1'], r['tstr_f1_std'])} | {_fmt(r['rare_recall'])} | {_fmt(r['eps'], d=2)} | {'yes' if r['secagg'] else 'no'} |" for r in ref]
    if sc.get("vs_baseline"):
        L += ["", "## Change against the baseline scorecard (`results/scorecard_baseline.json`)", "", "| configuration | " + " | ".join(UTILITY) + " |", "|---|" + "---|" * len(UTILITY)]
        for d in sc["vs_baseline"]:
            L.append(f"| {d['label']} | " + " | ".join("n/a" if d[m] is None else f"{d[m]:+.3f} ({d[m + '_effect']})" for m in UTILITY) + " |")
    return "\n".join(L) + "\n"


def run(cfg: dict[str, Any], freeze_baseline: bool = False, out_dir: str | Path | None = None) -> dict[str, Any]:
    res = Path(out_dir) if out_dir else Path(cfg["compute"]["runs_csv"]).parent
    summ = pd.read_csv(res / "summary.csv")
    try:
        df = ag.load_ledger(cfg)
    except FileNotFoundError:
        df = None
    read = lambda n: json.loads((res / n).read_text(encoding="utf-8")) if (res / n).exists() else None      # noqa: E731
    mia, fed = read("mia_model.json"), read("fed_classifier.json")
    rare = (mia or {}).get("rare") or ag.rare_classes(cfg, summ)
    base_p = res / "scorecard_baseline.json"
    if freeze_baseline and base_p.exists():
        raise FileExistsError(f"{base_p} exists: the baseline is frozen once (delete it by hand to start the comparison again)")
    base = None if freeze_baseline or not base_p.exists() else _from_json(json.loads(base_p.read_text(encoding="utf-8")))
    sc = build(summ, df, mia, fed, rare, base)
    sc["rare_classes"] = list(rare)
    text = json.dumps(_jsonable(sc), indent=1, ensure_ascii=False)
    (res / "scorecard.json").write_text(text, encoding="utf-8")
    if freeze_baseline:
        base_p.write_text(text, encoding="utf-8")
    (res / "reports").mkdir(parents=True, exist_ok=True)
    (res / "reports" / "scorecard.md").write_text(render(sc), encoding="utf-8")
    logger.info("scorecard: %d rows, front %s", len(sc["rows"]), sc["front"])
    return sc
