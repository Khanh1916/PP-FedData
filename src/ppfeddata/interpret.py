"""Phase 12: interpretation rules R1-R6, the red flags and the recommendation rule, computed from the run ledger and the saved test predictions.

Nothing here trains a model. Every number comes from `results/runs.csv`, the predictions in `artifacts/<run>/preds/test.npz`, the run specs and the
round logs; the thresholds come from `thresholds` in the config. `interpret()` returns a JSON-able dict, `render()` turns it into the Markdown of
section 9 of `final_report.md`.

Rule of the spec (R1, R2): a difference "has an effect" when the paired-bootstrap interval of the difference excludes 0 AND |difference| is larger
than the std over seeds. See `ppfeddata.eval.compare` for how the intervals are built.
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
from ppfeddata.data.preprocess import processed_dir
from ppfeddata.eval.compare import MACRO_F1, Metric, PairedBootstrap, compare_runs, recall_of
from ppfeddata.eval.dp_check import recompute_run
from ppfeddata.eval.runs import artifacts_dir, load_predictions, run_id

logger = logging.getLogger("ppfeddata.interpret")

BOOT_SEED = 0
PRIMARY = "rf"                       # the classifier whose TAug macro-F1 ranks the configurations in R6
CLFS = ("rf", "mlp")
PROTOS = ("TSTR", "TAug")
MIA_CHANCE_BAND = 0.05               # an MIA AUC within 0.5 +/- this is "no detectable signal" (the config's mia_auc_max is 0.55)
RARE_CELL = 10                       # descriptive only (not a decision threshold): a (client, class) cell with at most this many records is called "thin"


# --------------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------------
def load_test_split(cfg: dict[str, Any]):
    d = processed_dir(cfg)
    z = np.load(d / "test.npz", allow_pickle=True)
    schema = json.loads((d / "feature_schema.json").read_text(encoding="utf-8"))
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    streams = z["stream_id"] if "stream_id" in z.files else None
    groups = z["group_id"] if "group_id" in z.files else None
    return z["y"].astype(np.int64), streams, groups, classes


def summary_value(summ: pd.DataFrame, config: str, metric: str) -> float:
    """Mean over seeds of `metric` for a ledger configuration (NaN when absent)."""
    r = summ[summ["config"] == config]
    return float(r.iloc[0].get(f"{metric}_mean", np.nan)) if len(r) else float("nan")


class Context:
    """The ledger, the summary, the bootstraps with every needed run loaded, and a cache of comparisons."""

    def __init__(self, cfg, summ, df, ents, refs, boots, seeds_of, classes, rare, meta):
        self.cfg, self.summ, self.df, self.ents, self.refs, self.boots = cfg, summ, df, ents, refs, boots
        self.seeds_of, self.classes, self.meta = seeds_of, classes, meta
        self.rare = {c: classes.index(c) for c in rare}                      # rare class name -> label
        self.rare_metric = recall_of(*self.rare.values())
        self._cache: dict[tuple, dict[str, Any]] = {}

    def has(self, name: str) -> bool:
        return bool(self.seeds_of.get(name))

    def cmp(self, a: str, b: str, metric: Metric = MACRO_F1) -> dict[str, Any]:
        """Seed-averaged difference metric(a) - metric(b) with the row bootstrap (the spec's) and the stream bootstrap (sensitivity)."""
        key = (a, b, metric)
        if key in self._cache:
            return self._cache[key]
        seeds = sorted(set(self.seeds_of.get(a, [])) & set(self.seeds_of.get(b, [])))
        if not seeds:
            raise KeyError(f"no seed in common for {a} and {b}")
        ka, kb = [(a, s) for s in seeds], [(b, s) for s in seeds]
        c = compare_runs(self.boots["rows"], ka, kb, metric)
        st = compare_runs(self.boots["streams"], ka, kb, metric) if "streams" in self.boots else c
        c["streams"] = {x: st[x] for x in ("lo", "hi", "ci_excludes_zero", "effect")}
        c.update(a=a, b=b, seeds=seeds, metric=metric.kind, classes=list(metric.classes))
        self._cache[key] = c
        return c

    def summary_value(self, config: str, metric: str) -> float:
        return summary_value(self.summ, config, metric)


def build_context(cfg: dict[str, Any], summ: pd.DataFrame, df: pd.DataFrame, n_boot: int | None = None, boot_seed: int = BOOT_SEED) -> Context:
    fam = ag.dp_families(set(summ["config"]))
    ents, refs = ag.entries(summ, fam), ag.reference_entries(summ)
    y, streams, groups, classes = load_test_split(cfg)
    k, mode = len(classes), cfg["label_mode"]
    names = sorted({n for e in ents + refs for n in e["cfgs"].values()})
    led = df.drop_duplicates(["config", "seed"]).set_index(["config", "seed"])["macro_f1"]
    preds, seeds_of = {}, {}
    for n in names:
        for s in sorted(int(x) for x in df.loc[df["config"] == n, "seed"].unique()):
            try:
                preds[(n, s)] = load_predictions(cfg, run_id(n, s, mode))["y_pred"]
            except FileNotFoundError:
                logger.warning("no saved predictions for %s seed %d: left out of the comparisons", n, s)
                continue
            seeds_of.setdefault(n, []).append(s)
    if not preds:
        raise FileNotFoundError("no saved test predictions found (artifacts/<run_id>/preds/test.npz)")
    nb = int(n_boot or cfg["eval"]["bootstrap"])
    boots = {"rows": PairedBootstrap(y, k, nb, boot_seed, "rows")}
    boots["rows"].add(preds)
    if streams is not None:
        boots["streams"] = PairedBootstrap(y, k, nb, boot_seed, "streams", streams)
        boots["streams"].add(preds)
    worst = max(abs(float(MACRO_F1(boots["rows"].point[key])) - float(led.loc[key])) for key in preds)
    meta = {"label_mode": mode, "n_boot": nb, "boot_seed": boot_seed, "bootstrap_modes": list(boots), "n_test_rows": int(len(y)),
            "n_streams": int(len(np.unique(streams))) if streams is not None else None, "n_groups": int(len(np.unique(groups))) if groups is not None else None,
            "seeds": sorted({s for v in seeds_of.values() for s in v}), "primary_classifier": PRIMARY, "n_runs": len(preds),
            "integrity_max_abs_diff_vs_ledger": float(worst), "thresholds": dict(cfg["thresholds"])}
    rare = ag.rare_classes(cfg, summ)
    meta["rare_classes"] = [c for c in rare if c in classes]
    return Context(cfg, summ, df, ents, refs, boots, seeds_of, classes, meta["rare_classes"], meta)


# --------------------------------------------------------------------------------------------------
# R1 - R3
# --------------------------------------------------------------------------------------------------
def _entry(ents, label):
    return next((e for e in ents if e["label"] == label), None)


def r1(ctx: Context) -> dict[str, Any]:
    """R1: does synthetic data improve the IDS? Delta = TAug minus B0 (macro-F1 and recall of the rare classes). B1a / B1b are non-generative references."""
    rows = []
    for e in ctx.ents:
        if e["label"] == "B0":
            continue
        base = e["color"] == "baseline"
        row: dict[str, Any] = {"label": e["label"], "reference": base}
        for clf in CLFS:
            a, b = e["cfgs"].get(("TRTR", clf) if base else ("TAug", clf)), f"B0-{clf}"
            if not a or not ctx.has(a) or not ctx.has(b):
                continue
            row[clf] = {"macro_f1": ctx.cmp(a, b), "rare_mean": ctx.cmp(a, b, ctx.rare_metric),
                        "per_class": {c: ctx.cmp(a, b, recall_of(i)) for c, i in ctx.rare.items()}}
        rows.append(row)
    return {"rows": rows}


def r2(ctx: Context) -> dict[str, Any]:
    """R2: is the CVAE better than the simple methods? TAug of B2 (and of B3) against B1a (class weights, RF only) and B1b (SMOTE)."""
    rows = []
    for gen in ("B2", "B3"):
        e = _entry(ctx.ents, gen)
        if not e:
            continue
        for ref, clf in (("B1a", "rf"), ("B1b", "rf"), ("B1b", "mlp")):
            a, b = e["cfgs"].get(("TAug", clf)), f"{ref}-{clf}"
            if a and ctx.has(a) and ctx.has(b):
                rows.append({"generator": gen, "reference": ref, "classifier": clf, "macro_f1": ctx.cmp(a, b), "rare_mean": ctx.cmp(a, b, ctx.rare_metric)})
    return {"rows": rows}


def r3(ctx: Context) -> dict[str, Any]:
    """R3: what does non-IID FL cost? B2 minus B3 (reported, no pass / fail threshold)."""
    b2, b3 = _entry(ctx.ents, "B2"), _entry(ctx.ents, "B3")
    rows = []
    if b2 and b3:
        for proto in PROTOS:
            for clf in CLFS:
                a, b = b2["cfgs"][(proto, clf)], b3["cfgs"][(proto, clf)]
                if ctx.has(a) and ctx.has(b):
                    c = ctx.cmp(a, b)
                    rows.append({"protocol": proto, "classifier": clf, "macro_f1": c, "rare_mean": ctx.cmp(a, b, ctx.rare_metric),
                                 "relative_loss": float(c["delta"] / c["mean_a"]) if c["mean_a"] else float("nan")})
    return {"rows": rows}


# --------------------------------------------------------------------------------------------------
# R4 - R5
# --------------------------------------------------------------------------------------------------
def _positive_control(ctx: Context) -> dict[str, Any]:
    p = artifacts_dir(ctx.cfg) / f"B2_positive_control_{ctx.cfg['label_mode']}.json"
    if not p.exists():
        return {"available": False}
    d = json.loads(p.read_text(encoding="utf-8"))
    thr = float(ctx.cfg["thresholds"]["mia_auc_max"])
    cop = {float(k): float(v) for k, v in d.get("copier_sensitivity", {}).items()}
    return {"available": True, "overfit_cvae_auc": float(d["mia_auc_mean"]), "overfit_cvae_detected": bool(d["mia_auc_mean"] > thr), "copier_auc": cop,
            "copier_detected_at_zero_noise": bool(cop.get(0.0, 0.0) > thr), "threshold": thr}


def _curve_checks(values: dict[float, tuple[float, float]], higher_is_better: bool = True) -> dict[str, Any]:
    """`values` = {eps: (mean, std)}. A step to a LARGER epsilon that makes the metric worse by more than the larger std of the two points is a
    violation of monotonicity (more privacy budget should not hurt); `flat` = the whole range is inside the largest std."""
    eps = sorted(values)
    viol, sig = [], max(v[1] for v in values.values())
    for lo, hi in zip(eps, eps[1:]):
        step = (values[hi][0] - values[lo][0]) * (1 if higher_is_better else -1)
        if -step > max(values[lo][1], values[hi][1]):
            viol.append({"from_eps": lo, "to_eps": hi, "change": values[hi][0] - values[lo][0]})
    rng = max(v[0] for v in values.values()) - min(v[0] for v in values.values())
    return {"values": {str(k): list(v) for k, v in values.items()}, "violations": viol, "monotone_within_noise": not viol, "range": float(rng), "max_std": float(sig), "flat": bool(rng <= sig)}


def r4(ctx: Context) -> dict[str, Any]:
    """R4: what does DP cost and what does it bring? M1(eps) minus B3-plain (the eps = infinity point, same plain decoder) and the empirical privacy columns."""
    m1 = sorted((e for e in ctx.ents if e["color"] == "dp"), key=lambda e: e["eps"])
    ref = next((e for e in ctx.refs if e["label"].startswith("B3-plain")), None)
    out: dict[str, Any] = {"cost": [], "privacy": [], "curves": {}, "reference_available": ref is not None}
    if not m1:
        return out
    for e in m1:
        for proto in PROTOS:
            for clf in CLFS:
                a = e["cfgs"][(proto, clf)]
                b = ref["cfgs"][(proto, clf)] if ref else None
                if b and ctx.has(a) and ctx.has(b):
                    out["cost"].append({"label": e["label"], "eps": e["eps"], "protocol": proto, "classifier": clf, "macro_f1": ctx.cmp(a, b)})
    pts = ([("inf", ref, None)] if ref else []) + [(f"{e['eps']:g}", e, e["eps"]) for e in reversed(m1)]
    for lab, e, eps in pts:
        n = e["cfgs"][("TSTR", PRIMARY)]
        g = ctx.df[ctx.df["config"] == n]
        out["privacy"].append({"eps": lab, "label": e["label"], "mia_auc": ag.mean_std(ctx.summ, n, "mia_auc_mean"), "dcr_ratio": ag.mean_std(ctx.summ, n, "dcr_ratio_mean"),
                               "dup_rate": ag.mean_std(ctx.summ, n, "dup_rate"), "val_elbo": ag.mean_std(ctx.summ, n, "fl_final_val_loss"),
                               "eps_max": float(g["dp_eps_max"].max()) if "dp_eps_max" in g and g["dp_eps_max"].notna().any() else None})
    for clf in CLFS:
        vals = {e["eps"]: ag.mean_std(ctx.summ, e["cfgs"][("TSTR", clf)], "macro_f1") for e in m1}
        if all(v is not None for v in vals.values()):
            out["curves"][f"TSTR-{clf}"] = _curve_checks(vals)
    elbo = {e["eps"]: ag.mean_std(ctx.summ, e["cfgs"][("TSTR", PRIMARY)], "fl_final_val_loss") for e in m1}
    if all(v is not None for v in elbo.values()):
        out["curves"]["validation ELBO (lower is better)"] = _curve_checks(elbo, higher_is_better=False)
    mia = [p["mia_auc"] for p in out["privacy"] if p["mia_auc"]]
    out["mia"] = {"values": [m[0] for m in mia], "near_chance": bool(mia) and all(abs(m[0] - 0.5) <= MIA_CHANCE_BAND for m in mia), "chance_band": MIA_CHANCE_BAND,
                  "spread": float(max(m[0] for m in mia) - min(m[0] for m in mia)) if mia else None}
    if len(mia) >= 2:                      # does the AUC approach 0.5 as epsilon shrinks? Compare the smallest epsilon with epsilon = infinity, against the seed std
        gap_inf, gap_min = abs(mia[0][0] - 0.5), abs(mia[-1][0] - 0.5)
        sd = max(mia[0][1], mia[-1][1])
        out["mia"].update(gap_at_inf=float(gap_inf), gap_at_smallest_eps=float(gap_min), seed_std=float(sd), approaches_chance=bool(gap_inf - gap_min > sd))
    out["positive_control"] = _positive_control(ctx)
    return out


def r5(ctx: Context, cost: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """R5: what does SecAgg cost? |M2 - B3| and M3 - M1(eps = 5) in utility, plus the overhead."""
    out: dict[str, Any] = {"pairs": [], "overhead": []}
    m2, b3, m3 = _entry(ctx.ents, "M2"), _entry(ctx.ents, "B3"), _entry(ctx.ents, "M3-eps5")
    m1_5 = next((e for e in ctx.ents if e["color"] == "dp" and e.get("eps") == 5), None)
    for name, x, y in (("M2 - B3", m2, b3), ("M3-eps5 - M1-eps5", m3, m1_5)):
        if not (x and y):
            continue
        for proto in PROTOS:
            for clf in CLFS:
                a, b = x["cfgs"][(proto, clf)], y["cfgs"][(proto, clf)]
                if ctx.has(a) and ctx.has(b):
                    out["pairs"].append({"pair": name, "protocol": proto, "classifier": clf, "macro_f1": ctx.cmp(a, b)})
    for name, x, y in (("M2 / B3", "M2", "B3"), ("M3-eps5 / M1-eps5", "M3-eps5", "M1-eps5")):
        cx, cy = cost.get(x), cost.get(y)
        if cx and cy:
            out["overhead"].append({"pair": name, "time_ratio": float(cx["s_round"] / cy["s_round"]), "bytes_per_param_ratio": float(cx["bytes_per_param"] / cy["bytes_per_param"]),
                                    "s_round": [float(cx["s_round"]), float(cy["s_round"])], "bytes_per_param": [float(cx["bytes_per_param"]), float(cy["bytes_per_param"])]})
    return out


# --------------------------------------------------------------------------------------------------
# R6 - recommendation
# --------------------------------------------------------------------------------------------------
def _protection(r: dict[str, Any]) -> tuple:
    """Order used only to break ties between configurations that are equal in utility: formal DP (smaller epsilon first), then SecAgg."""
    return (r["kind"] in ("dp", "dpsa"), -(r["eps_max"] or 0.0), r["kind"] in ("secagg", "dpsa"))


def r6(ctx: Context, cost: dict[str, dict[str, Any]]) -> dict[str, Any]:
    th = ctx.cfg["thresholds"]
    rows = []
    for e in ctx.ents:
        if e["color"] == "baseline":
            continue
        tstr, taug = e["cfgs"][("TSTR", PRIMARY)], e["cfgs"][("TAug", PRIMARY)]
        if not (ctx.has(tstr) and ctx.has(taug)):
            continue
        g = ctx.df[ctx.df["config"] == tstr]
        eps = float(g["dp_eps_max"].max()) if "dp_eps_max" in g and g["dp_eps_max"].notna().any() else None
        mia = ctx.summary_value(tstr, "mia_auc_mean")
        c = cost.get(e["label"])
        ratio = float(c["ratio"]) if c and not math.isnan(c["ratio"]) else None
        r = {"label": e["label"], "kind": e["color"], "federated": e["label"] != "B2", "taug_cfg": taug, "tstr_cfg": tstr, "taug_f1": ctx.summary_value(taug, "macro_f1"),
             "tstr_f1": ctx.summary_value(tstr, "macro_f1"), "mia_auc": mia, "eps_max": eps, "time_ratio": ratio,
             "ok_mia": bool(mia <= th["mia_auc_max"]), "ok_eps": eps is None or eps <= th["eps_max_recommend"],
             "ok_overhead": ratio is None or ratio <= th["overhead_ratio_max"]}
        r["eligible"] = bool(r["federated"] and r["ok_mia"] and r["ok_eps"] and r["ok_overhead"])
        rows.append(r)
    out: dict[str, Any] = {"candidates": rows, "rule": {"mia_auc_max": th["mia_auc_max"], "eps_max": th["eps_max_recommend"], "overhead_ratio_max": th["overhead_ratio_max"],
                                                       "ranking": f"TAug macro-F1 of the {PRIMARY.upper()} classifier, mean over seeds"}}
    elig = [r for r in rows if r["eligible"]]
    if elig:
        best = max(elig, key=lambda r: r["taug_f1"])
        tied = [r for r in elig if r is best or ctx.cmp(r["taug_cfg"], best["taug_cfg"])["effect"] != "worse"]
        chosen = max(tied, key=_protection)
        out.update(literal=best["label"], tied=[r["label"] for r in tied], recommended=chosen["label"], tie_break_used=chosen is not best)
        # the same rule ranked by the synthetic-only macro-F1 (TSTR): the metric that counts when synthetic data replace real data
        bt = max(elig, key=lambda r: r["tstr_f1"])
        tied_t = [r for r in elig if r is bt or ctx.cmp(r["tstr_cfg"], bt["tstr_cfg"])["effect"] != "worse"]
        out["tstr_ranking"] = {"literal": bt["label"], "tied": [r["label"] for r in tied_t], "recommended": max(tied_t, key=_protection)["label"]}
    else:
        out.update(literal=None, tied=[], recommended=None, tie_break_used=False)
    dp = [r for r in rows if r["kind"] in ("dp", "dpsa") and r["ok_mia"] and r["ok_eps"] and r["federated"]]
    if dp:
        best_dp = max(dp, key=lambda r: r["taug_f1"])
        tied_dp = [r for r in dp if r is best_dp or ctx.cmp(r["taug_cfg"], best_dp["taug_cfg"])["effect"] != "worse"]
        out["dp_alternative"] = {"literal": best_dp["label"], "tied": [r["label"] for r in tied_dp], "recommended": max(tied_dp, key=_protection)["label"]}
    return out


# --------------------------------------------------------------------------------------------------
# Why a CVAE at all (the numbers behind section 9.9)
# --------------------------------------------------------------------------------------------------
def why(ctx: Context) -> dict[str, Any]:
    """What each option reaches when the real data can be pooled and when they cannot, what the generators cost and how easy their rows are to tell from real ones."""
    sv, by = ctx.summary_value, {e["label"]: e for e in ctx.ents}

    def macro(label: str, proto: str) -> dict[str, float]:
        e = by.get(label)
        return {clf: sv(e["cfgs"][(proto, clf)], "macro_f1") for clf in CLFS if e and (proto, clf) in e["cfgs"]}

    rows = []
    for label, proto, name in (("B1b", "TRTR", "B1b (SMOTE)"), ("B1a", "TRTR", "B1a (class weights)"), ("B0", "TRTR", "B0 (real data only)"),
                               ("B2", "TAug", "B2 + real data (TAug)"), ("B3", "TAug", "B3 + real data (TAug)")):
        if label in by:
            rows.append({"id": f"{label}:{proto}", "setting": "real data can be pooled", "option": name, "needs_pooled_real_data": True, "macro_f1": macro(label, proto)})
    fed = [e["label"] for e in ctx.ents if e["color"] != "baseline" and e["label"] != "B2"]
    for label in fed:
        rows.append({"id": f"{label}:TSTR", "setting": "raw data stays at the clients", "option": f"{label} (synthetic data only, TSTR)", "needs_pooled_real_data": False,
                     "macro_f1": macro(label, "TSTR")})
    b0 = macro("B0", "TRTR").get(PRIMARY)
    first = lambda label: by[label]["cfgs"][("TSTR", PRIMARY)]                                  # noqa: E731
    out: dict[str, Any] = {"rows": rows, "b0": macro("B0", "TRTR"), "retention_primary": {}, "c2st": {}, "mia": {}, "train_seconds": {}}
    for label in ["B2"] + fed:
        if label not in by:
            continue
        t = macro(label, "TSTR").get(PRIMARY)
        if label in fed and t is not None and b0:
            out["retention_primary"][label] = float(t / b0 - 1)
        out["c2st"][label] = sv(first(label), "c2st_auc_mean")
        out["mia"][label] = sv(first(label), "mia_auc_mean")
        out["train_seconds"][label] = sv(first(label), "cvae_train_s")
    ref = next((e for e in ctx.refs if e["label"].startswith("B3-plain")), None)               # relative TSTR loss of DP against epsilon = infinity (same decoder)
    loss = []
    if ref:
        for e in ctx.ents:
            if e["color"] == "dp":
                for clf in CLFS:
                    base = sv(ref["cfgs"][("TSTR", clf)], "macro_f1")
                    if base:
                        loss.append(float(sv(e["cfgs"][("TSTR", clf)], "macro_f1") / base - 1))
    out["dp_tstr_relative_loss"] = {"min": min(loss), "max": max(loss), "n": len(loss)} if loss else None
    return out


# --------------------------------------------------------------------------------------------------
# Which configuration for which requirement (M1, M2, M3) and what the IoT / MQTT data change about it (section 9.10)
# --------------------------------------------------------------------------------------------------
def _recall_stats(ctx: Context, name: str | None) -> dict[str, Any] | None:
    """Recall by class (mean over seeds) of a ledger configuration, the mean over the rare classes and the macro-F1."""
    r = ctx.summ[ctx.summ["config"] == name] if name else ctx.summ.iloc[0:0]
    if not len(r):
        return None
    r = r.iloc[0]
    rec = {c: float(r[f"recall_{c}_mean"]) for c in ctx.classes if f"recall_{c}_mean" in r.index and not pd.isna(r[f"recall_{c}_mean"])}
    if not rec:
        return None
    rare = [rec[c] for c in ctx.rare if c in rec]
    return {"recall": rec, "rare_mean": float(np.mean(rare)) if rare else None, "macro_f1": float(r["macro_f1_mean"])}


def _majority_class(ctx: Context) -> str:
    q = ctx.cfg.get("quota", {})
    known = [c for c in ctx.classes if c in q]
    return max(known, key=lambda c: q[c][0]) if known else ctx.classes[0]


def _by_class(ctx: Context) -> dict[str, Any]:
    """Recall by class of the synthetic-only classifier (TSTR) for the options that need no pooled data, with the real-data-only reference.
    The DP rows use the plain decoder (the one inside epsilon) and are compared with B3-plain, the epsilon = infinity point with the same decoder."""
    ref = next((e for e in ctx.refs if e["label"].startswith("B3-plain")), None)
    rows: list[dict[str, Any]] = []

    def add(label: str, decoder: str, e: dict[str, Any], kind: str, eps: float | None = None) -> None:
        row: dict[str, Any] = {"label": label, "decoder": decoder, "kind": kind, "eps": eps}
        for clf in CLFS:
            st = _recall_stats(ctx, e["cfgs"].get(("TSTR", clf)))
            if st:
                row[clf] = st
        if any(clf in row for clf in CLFS):
            rows.append(row)

    if ref:
        add("B3-plain", "plain", ref, "ref")
    for e in ctx.ents:
        if e["color"] != "baseline" and e["label"] != "B2":
            add(e["label"], "plain" if e["color"] in ("dp", "dpsa") else "residual noise", e, e["color"], e.get("eps"))
    b0 = _entry(ctx.ents, "B0")
    real = {clf: st for clf in CLFS if b0 and (st := _recall_stats(ctx, b0["cfgs"].get(("TRTR", clf))))}
    major, base = _majority_class(ctx), next((r for r in rows if r["label"] == "B3-plain"), None)
    change = []
    for r in rows:
        if r["kind"] not in ("dp", "dpsa") or not base:
            continue
        for clf in CLFS:
            if clf in r and clf in base and base[clf]["rare_mean"]:
                change.append({"label": r["label"], "classifier": clf, "rare_mean": r[clf]["rare_mean"], "rare_mean_inf": base[clf]["rare_mean"],
                               "rare_relative": float(r[clf]["rare_mean"] / base[clf]["rare_mean"] - 1), "major_recall": r[clf]["recall"].get(major),
                               "major_recall_inf": base[clf]["recall"].get(major)})
    return {"classes": list(ctx.classes), "rare": list(ctx.rare), "majority": major, "rows": rows, "real_only": real, "dp_change": change}


def _client_rarity(ctx: Context) -> dict[str, Any] | None:
    """How thin the rare classes are at the clients: the (client, rare class) cells of the Dirichlet partition of each seed."""
    from ppfeddata.partition import partition_path
    alpha = ctx.cfg.get("fl", {}).get("dirichlet_alpha")
    out: dict[str, Any] = {"alpha": alpha, "cutoff": RARE_CELL, "seeds": {}}
    for s in ctx.meta["seeds"]:
        try:
            p = partition_path(ctx.cfg, alpha, s)
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (KeyError, TypeError, FileNotFoundError, OSError, ValueError):
            continue
        table, classes = np.asarray(meta["table"]), meta["classes"]
        idx = [classes.index(c) for c in ctx.rare if c in classes]
        if not idx:
            continue
        cells = table[:, idx]
        out["seeds"][str(s)] = {"clients": int(table.shape[0]), "cells": int(cells.size), "thin": int((cells <= RARE_CELL).sum()), "empty": int((cells == 0).sum()),
                                "min_by_class": {classes[i]: int(table[:, i].min()) for i in idx}}
    return out if out["seeds"] else None


def _privacy_unit(ctx: Context) -> dict[str, Any] | None:
    """How many rows one TCP stream and one capture group contribute to each class of the train split: the size of what record-level epsilon does not cover."""
    try:
        z = np.load(processed_dir(ctx.cfg) / "train.npz", allow_pickle=True)
        y, g, s = z["y"], z["group_id"], z["stream_id"]
    except (KeyError, TypeError, FileNotFoundError, OSError):
        return None
    out: dict[str, Any] = {"split": "train", "n_groups": int(len(np.unique(g))), "classes": {}}
    for i, c in enumerate(ctx.classes):
        m = y == i
        if not m.any():
            continue
        gc, sc = np.unique(g[m], return_counts=True)[1], np.unique(s[m], return_counts=True)[1]
        out["classes"][c] = {"rows": int(m.sum()), "groups": int(len(gc)), "rows_per_group_median": float(np.median(gc)), "rows_per_group_max": int(gc.max()),
                             "rows_per_stream_mean": float(sc.mean()), "rows_per_stream_max": int(sc.max())}
    return out


def _scenarios(R4: dict[str, Any], R6: dict[str, Any]) -> list[dict[str, Any]]:
    """Which labelled configurations answer which requirement. S1: only the aggregation server must not see the updates; S2: a formal guarantee on the released
    model / synthetic data; S3: both; S4: clients that cannot afford DP-SGD's extra compute (the overhead filter of R6)."""
    cand = R6.get("candidates", [])
    dp = sorted((r for r in cand if r["kind"] == "dp"), key=lambda r: r["eps_max"] if r["eps_max"] is not None else 1e9)
    flat = bool((R4.get("curves", {}).get(f"TSTR-{PRIMARY}") or {}).get("flat"))
    return [{"id": "S1", "choose": [r["label"] for r in cand if r["kind"] == "secagg"]},
            {"id": "S2", "choose": [dp[0]["label"]] if dp and flat else [r["label"] for r in dp], "eps_curve_flat": flat},
            {"id": "S3", "choose": [r["label"] for r in cand if r["kind"] == "dpsa"]},
            {"id": "S4", "choose": [r["label"] for r in cand if r["federated"] and r["kind"] == "secagg" and r["ok_overhead"]]}]


def guide(ctx: Context, R4: dict[str, Any], R6: dict[str, Any]) -> dict[str, Any]:
    return {"by_class": _by_class(ctx), "client_rarity": _client_rarity(ctx), "privacy_unit": _privacy_unit(ctx), "scenarios": _scenarios(R4, R6)}


# --------------------------------------------------------------------------------------------------
# Red flags
# --------------------------------------------------------------------------------------------------
def _flag(fid: str, title: str, rule: str, triggered: bool, status: str, evidence: Any = None, notes: list[str] | None = None) -> dict[str, Any]:
    return {"id": fid, "title": title, "rule": rule, "triggered": bool(triggered), "status": status, "evidence": evidence, "notes": notes or []}


def flag_near_one(ctx: Context) -> dict[str, Any]:
    th = ctx.cfg["thresholds"]["f1_near_one"]
    b0 = {clf: summary_value(ctx.summ, f"B0-{clf}", "macro_f1") for clf in CLFS}
    top = ctx.summ["macro_f1_mean"].max()
    trig = bool(b0.get(PRIMARY, 0) >= th)
    return _flag("F1", "macro-F1 near 1.00, B0 included", f"B0-{PRIMARY} macro-F1 >= {th}: leakage / artefact, go back to Phase 5", trig,
                 "TRIGGERED - investigate" if trig else "not triggered", {"B0": b0, "highest_macro_f1_of_any_configuration": float(top), "threshold": th})


def flag_tstr_over_trtr(ctx: Context) -> dict[str, Any]:
    """TSTR above TRTR (B0) in the spec's sense: the 'better' verdict of TSTR - B0. Investigation: the same TSTR against the class-balanced real reference."""
    rows = []
    for e in ctx.ents:
        if e["color"] == "baseline":
            continue
        for clf in CLFS:
            a, b0 = e["cfgs"][("TSTR", clf)], f"B0-{clf}"
            if not (ctx.has(a) and ctx.has(b0)):
                continue
            c = ctx.cmp(a, b0)
            row = {"label": e["label"], "classifier": clf, "vs_b0": c}
            if c["effect"] == "better":
                refs = [("B1b", f"B1b-{clf}")] + ([("B1a", "B1a-rf")] if clf == "rf" else [])
                row["vs_balanced_real"] = {name: ctx.cmp(a, n) for name, n in refs if ctx.has(n)}
                row["recall_normal"] = ctx.cmp(a, b0, recall_of(0))
                row["recall_rare"] = ctx.cmp(a, b0, ctx.rare_metric)
            rows.append(row)
    hit = [r for r in rows if r["vs_b0"]["effect"] == "better"]
    open_ = [r for r in hit if any(v["effect"] == "better" for v in r.get("vs_balanced_real", {}).values()) or not r.get("vs_balanced_real")]
    if not hit:
        status = "not triggered"
    elif not open_:
        status = "TRIGGERED, investigated: not above the class-balanced real reference"
    else:
        status = "TRIGGERED - OPEN: above even the class-balanced real reference"
    qa = ctx.cfg.get("tune", {}).get("syn_per_class")
    notes = [f"TSTR trains on {qa} synthetic rows per class (balanced by construction); B0 trains on the imbalanced real train pool (quota per class: "
             + ", ".join(f"{c} {v[0]}" for c, v in ctx.cfg.get("quota", {}).items()) + ")."] if qa else []
    rf = [r for r in rows if r["classifier"] == "rf"]
    summary = {"n_pairs": len(rows), "n_hit": len(hit), "hit_classifiers": sorted({r["classifier"] for r in hit}),
               "below_balanced_in_all_hits": bool(hit) and not open_,
               "recall_signature_in_all_hits": bool(hit) and all(r["recall_normal"]["delta"] < 0 < r["recall_rare"]["delta"] for r in hit),
               "n_rf_pairs": len(rf), "n_rf_below_b0": sum(r["vs_b0"]["effect"] == "worse" for r in rf)}
    return _flag("F2", "TSTR above TRTR", "TSTR minus B0 has an effect (interval excludes 0 and abs(delta) > seed std) in favour of TSTR: suspect label leakage through the synthetic data or a misused test set",
                 bool(hit), status, {"rows": rows, "summary": summary}, notes)


def flag_copying(ctx: Context) -> dict[str, Any]:
    th = ctx.cfg["thresholds"]
    g = ctx.df.dropna(subset=["dup_rate", "dcr_ratio_mean"]).drop_duplicates(["fl_run", "prefix", "seed"]) if "prefix" in ctx.df else ctx.df.dropna(subset=["dup_rate"])
    worst_dup = g.sort_values("dup_rate", ascending=False).head(3)
    worst_dcr = g.sort_values("dcr_ratio_mean").head(3)
    trig = bool((g["dup_rate"] > th["dup_rate_max"]).any() or (g["dcr_ratio_mean"] < th["dcr_ratio_min"]).any())
    ev = {"max_dup_rate": [{"config": r["config"], "seed": int(r["seed"]), "value": float(r["dup_rate"])} for _, r in worst_dup.iterrows()],
          "min_dcr_ratio": [{"config": r["config"], "seed": int(r["seed"]), "value": float(r["dcr_ratio_mean"])} for _, r in worst_dcr.iterrows()],
          "dup_rate_max": th["dup_rate_max"], "dcr_ratio_min": th["dcr_ratio_min"], "n_generators_checked": int(len(g))}
    return _flag("F3", "model copies training data", f"duplicate rate > {th['dup_rate_max']} or DCR ratio < {th['dcr_ratio_min']} in any generator and seed", trig,
                 "TRIGGERED - investigate" if trig else "not triggered", ev)


def _count_by(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def flag_seed_spread(ctx: Context) -> dict[str, Any]:
    th = ctx.cfg["thresholds"]
    s = ctx.summ[(ctx.summ["n_seeds"] >= 2) & ctx.summ["macro_f1_std"].notna()].sort_values("macro_f1_std", ascending=False)
    red, soft = s[s["macro_f1_std"] > th["seed_std_redflag"]], s[s["macro_f1_std"] > th["seed_std_max"]]
    row = lambda r: {"config": r["config"], "std": float(r["macro_f1_std"]), "mean": float(r["macro_f1_mean"]), "in_matrix": bool(r.get("in_matrix", False))}      # noqa: E731
    inm = s[s["in_matrix"].astype(bool)] if "in_matrix" in s else s
    parsed = [ag.parse_config(n) for n in soft["config"]]
    breakdown = {"by_method": _count_by([p["method"] or "other" for p in parsed]), "by_protocol": _count_by([p["protocol"] or "other" for p in parsed])}
    ev = {"largest": [row(r) for _, r in s.head(6).iterrows()], "above_redflag": [row(r) for _, r in red.iterrows()], "soft_breakdown": breakdown,
          "n_above_redflag": int(len(red)), "n_above_redflag_in_matrix": int(red["in_matrix"].astype(bool).sum()) if "in_matrix" in red else int(len(red)),
          "n_above_seed_std_max": int(len(soft)), "n_configurations": int(len(s)), "n_in_matrix": int(len(inm)), "max_std_in_matrix": float(inm["macro_f1_std"].max()) if len(inm) else None,
          "redflag": th["seed_std_redflag"], "seed_std_max": th["seed_std_max"], "above_seed_std_max": [r["config"] for _, r in soft.iterrows()][:40]}
    trig = bool(len(red))
    where = "only in rows outside the spec matrix" if trig and ev["n_above_redflag_in_matrix"] == 0 else "including rows of the spec matrix"
    status = "not triggered" if not trig else f"TRIGGERED ({where}) - more seeds or a stability check"
    return _flag("F4", "large spread between seeds", f"std of macro-F1 over seeds > {th['seed_std_redflag']} for any configuration", trig, status, ev,
                 [f"{len(soft)} of {len(s)} configurations are above the softer level seed_std_max = {th['seed_std_max']} (a note, not a red flag)."] if len(soft) else [])


def flag_epsilon(ctx: Context) -> dict[str, Any]:
    th = float(ctx.cfg["thresholds"]["eps_recompute_rtol"])
    d = ctx.df
    if "dp_eps_max" not in d or "fl_run" not in d:
        return _flag("F5", "reported epsilon far from an independent recomputation", "", False, "not checked (no DP runs in the ledger)")
    runs = d[d["dp_eps_max"].notna() & d["fl_run"].notna()].drop_duplicates(["fl_run", "seed"])
    rows, missing = [], []
    for _, r in runs.iterrows():
        rid = run_id(r["fl_run"], int(r["seed"]), ctx.cfg["label_mode"])
        sp, rl = artifacts_dir(ctx.cfg) / rid / "spec.json", artifacts_dir(ctx.cfg) / rid / "rounds.jsonl"
        if not (sp.exists() and rl.exists()):
            missing.append(rid)
            continue
        spec = json.loads(sp.read_text(encoding="utf-8"))
        if not spec.get("dp"):
            continue
        last = [json.loads(x) for x in rl.read_text(encoding="utf-8").splitlines() if x.strip()][-1]
        res = recompute_run(spec, last.get("client_dp_steps"), float(r["dp_eps_max"]))
        rows.append({"run": rid, "target": res["target_eps"], "reported": res["eps_reported"], "independent": res["eps_independent"], "rel_diff": res["rel_diff"],
                     "steps_match_plan": res["steps_match_plan"], "steps_match_loader": res["steps_match_loader"]})
    if not rows:
        return _flag("F5", "reported epsilon far from an independent recomputation", "", False, "not checked (no run specs found)", {"missing": missing})
    far = [x for x in rows if not (abs(x["rel_diff"]) <= th)]
    bad_steps = [x for x in rows if not (x["steps_match_plan"] or x["steps_match_loader"])]
    plan_gap = [x for x in rows if not x["steps_match_plan"] and x["steps_match_loader"]]
    trig = bool(far or bad_steps)
    notes = []
    if plan_gap:
        notes.append(f"{len(plan_gap)} runs have fewer counted DP steps than rounds x epochs x ceil(n/B) for one client: Opacus' Poisson loader has int(1 / sample_rate) batches per epoch and "
                     "1 / (1 / 99) = 98.99999999999999 truncates to 98. The report uses the counted steps (and the true sampling rate), so epsilon is right; only the noise "
                     "calibration was one step per epoch more conservative than needed for that client.")
    ev = {"runs": rows, "max_abs_rel_diff": float(max(abs(x["rel_diff"]) for x in rows)), "rtol": th, "n_runs": len(rows), "n_far": len(far), "n_counters_unexplained": len(bad_steps),
          "signed_rel_diff_range": [float(min(x["rel_diff"] for x in rows)), float(max(x["rel_diff"] for x in rows))], "missing_specs": missing}
    return _flag("F5", "reported epsilon far from an independent recomputation",
                 f"relative difference above {th:g} (either sign) between the reported epsilon and the one recomputed with own RDP code, or step counters that match neither the plan nor the loader",
                 trig, "TRIGGERED - investigate" if trig else "not triggered", ev, notes)


def flags(ctx: Context) -> list[dict[str, Any]]:
    return [flag_near_one(ctx), flag_tstr_over_trtr(ctx), flag_copying(ctx), flag_seed_spread(ctx), flag_epsilon(ctx)]


# --------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------
def read_extra(cfg: dict[str, Any], name: str) -> dict[str, Any] | None:
    """A result file written by a follow-up command next to results/runs.csv (e.g. fed_classifier.json); read as it is, None when absent."""
    try:
        p = Path(cfg["compute"]["runs_csv"]).parent / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (KeyError, TypeError, ValueError, OSError):
        return None


def interpret(cfg: dict[str, Any], summ: pd.DataFrame, df: pd.DataFrame, n_boot: int | None = None, ctx: Context | None = None) -> dict[str, Any]:
    ctx = ctx or build_context(cfg, summ, df, n_boot)
    cost = {r["label"]: r for r in ag.cost_rows(summ, ctx.ents)}
    R = {"meta": ctx.meta, "R1": r1(ctx), "R2": r2(ctx), "R3": r3(ctx), "R4": r4(ctx), "R5": r5(ctx, cost), "R6": r6(ctx, cost), "flags": flags(ctx), "why": why(ctx)}
    R["guide"] = guide(ctx, R["R4"], R["R6"])
    R["fed_classifier"] = read_extra(cfg, "fed_classifier.json")         # follow-ups run after the main study (`fed-baseline`, `sensitivity`); None when not run
    R["sensitivity"] = read_extra(cfg, "sensitivity.json")
    R["mia_model"] = read_extra(cfg, "mia_model.json")
    R["cost"] = {k: {x: v[x] for x in ("s_round", "ratio", "bytes", "bytes_per_param")} for k, v in cost.items()}
    return R


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else round(v, 8)                     # strict JSON has no NaN / Infinity
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def write_json(R: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(R), indent=1), encoding="utf-8")
    return p
