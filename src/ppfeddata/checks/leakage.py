"""Phase 5: leakage and label-reliability checks (C1-C5).

Small Random Forests (100 trees), fixed seeds, macro-F1 on the validation split (groups never seen in training).
The module only measures and flags; every threshold is a heuristic from `thresholds` in the config and the user
decides at G2 what to drop.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ppfeddata.data.preprocess import processed_dir
from ppfeddata.utils import config_hash

logger = logging.getLogger("ppfeddata.checks.leakage")

# Time columns of the core feature set (Phase 2 time-feature audit). Any column that is still `diagnostic` is handled
# separately (since G2 there is none, but the machinery stays for other configurations).
TIME_COLUMNS = ["irtt", "time_delta_from_previous_displayed_frame", "time_since_first_frame_in_this_tcp_stream"]


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------
def _rf(seed: int, n_trees: int, balanced: bool = False, min_leaf: int = 1) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=seed, min_samples_leaf=min_leaf,
                                  class_weight="balanced" if balanced else None)


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _ms(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std())


def load_processed(cfg: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    d = processed_dir(cfg)
    data = {s: dict(np.load(d / f"{s}.npz")) for s in ("train", "val", "test")}
    with open(d / "feature_schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    return data, schema


def feature_groups(schema: dict[str, Any]) -> dict[str, list[int]]:
    """Core feature = one raw column: its value block, its is_na flag, or its one-hot group."""
    groups: dict[str, list[int]] = {}
    for b in schema["blocks"]:
        groups.setdefault(b["column"], []).extend(range(b["start"], b["start"] + b["width"]))
    return groups


def na_flag_columns(schema: dict[str, Any]) -> dict[str, int]:
    return {b["column"]: b["start"] for b in schema["blocks"] if b["type"] == "na_flag"}


# --------------------------------------------------------------------------------------------------
# C1: presence matrix
# --------------------------------------------------------------------------------------------------
def c1_presence(presence_csv: Path, absent_max: float, present_min: float) -> pd.DataFrame:
    """Columns that are (almost) empty in some sub-classes and populated in others."""
    pm = pd.read_csv(presence_csv, index_col=0)
    rows = []
    for col in pm.columns:
        s = pm[col]
        empty = [k for k, v in s.items() if v <= absent_max]
        full = [k for k, v in s.items() if v >= present_min]
        rows.append({"column": col, "n_empty_classes": len(empty), "n_populated_classes": len(full),
                     "min_presence": float(s.min()), "max_presence": float(s.max()),
                     "empty_in": ",".join(empty), "populated_in": ",".join(full),
                     "mixed": bool(empty and full)})
    return pd.DataFrame(rows).sort_values(["mixed", "max_presence"], ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# Reference model and C3
# --------------------------------------------------------------------------------------------------
def reference_model(data, seeds, n_trees, cols=None, balanced: bool = False) -> dict[str, Any]:
    cols = slice(None) if cols is None else cols
    f1s, recalls = [], []
    for s in seeds:
        rf = _rf(s, n_trees, balanced).fit(data["train"]["X"][:, cols], data["train"]["y"])
        pred = rf.predict(data["val"]["X"][:, cols])
        f1s.append(macro_f1(data["val"]["y"], pred))
        recalls.append(f1_score(data["val"]["y"], pred, average=None, labels=np.unique(data["train"]["y"]), zero_division=0))
    return {"macro_f1": _ms(f1s), "per_class_f1": np.mean(recalls, axis=0).tolist()}


def c3_single_feature(data, groups, seeds, n_trees, flag_thr, diag_cols: dict[str, list[int]] | None = None,
                      class_names: dict[int, str] | None = None) -> pd.DataFrame:
    """Val scores of a RF trained on ONE raw column (value + its flag / one-hot group).

    The RF is class-balanced with min_samples_leaf = 5: with the heavy train imbalance an unweighted RF collapses to
    the majority class and every single column would look harmless. Reported: macro-F1 (the spec criterion) and the
    best per-class F1, because at packet level a column can identify one class on a subset of its rows only.
    A column is flagged if either exceeds `flag_thr`.
    """
    labels = np.unique(data["train"]["y"])
    rows = []
    items = [(name, "core", data["train"]["X"], data["val"]["X"], cols) for name, cols in groups.items()]
    for name, cols in (diag_cols or {}).items():
        items.append((name, "diagnostic", data["train"]["X_diag"], data["val"]["X_diag"], cols))
    for name, kind, Xtr, Xva, cols in items:
        f1s, pcs = [], []
        for s in seeds:
            pred = _rf(s, n_trees, balanced=True, min_leaf=5).fit(Xtr[:, cols], data["train"]["y"]).predict(Xva[:, cols])
            f1s.append(macro_f1(data["val"]["y"], pred))
            pcs.append(f1_score(data["val"]["y"], pred, average=None, labels=labels, zero_division=0))
        m, sd = _ms(f1s)
        pc = np.mean(pcs, axis=0)
        best = int(np.argmax(pc))
        rows.append({"feature": name, "kind": kind, "macro_f1": m, "std": sd,
                     "best_class": (class_names or {}).get(int(labels[best]), int(labels[best])),
                     "best_class_f1": float(pc[best]), "flag": bool(m > flag_thr or pc[best] > flag_thr)})
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def c3_leave_one_out(data, groups, seeds, n_trees, base_f1: float) -> pd.DataFrame:
    """Macro-F1 on val when one raw column is removed from the full model."""
    n = data["train"]["X"].shape[1]
    rows = []
    for name, cols in groups.items():
        keep = np.setdiff1d(np.arange(n), cols)
        f1s = [macro_f1(data["val"]["y"], _rf(s, n_trees).fit(data["train"]["X"][:, keep], data["train"]["y"])
                        .predict(data["val"]["X"][:, keep])) for s in seeds]
        m, sd = _ms(f1s)
        rows.append({"removed": name, "macro_f1": m, "std": sd, "delta_vs_full": m - base_f1})
    return pd.DataFrame(rows).sort_values("delta_vs_full").reset_index(drop=True)


def ablations(data, groups, seeds, n_trees, time_cols) -> dict[str, Any]:
    """Full-model variants that matter for the G2 feature decision (all class-balanced, so they are comparable).

    Each entry: {"macro_f1": (mean, std), "per_class_f1": [...]}.
    """
    n = data["train"]["X"].shape[1]
    ytr, yva = data["train"]["y"], data["val"]["y"]
    labels = np.unique(ytr)

    def score(Xtr, Xva):
        f1s, pcs = [], []
        for s in seeds:
            pred = _rf(s, n_trees, balanced=True).fit(Xtr, ytr).predict(Xva)
            f1s.append(macro_f1(yva, pred))
            pcs.append(f1_score(yva, pred, average=None, labels=labels, zero_division=0))
        return {"macro_f1": _ms(f1s), "per_class_f1": np.mean(pcs, axis=0).tolist()}

    out = {"full": score(data["train"]["X"], data["val"]["X"])}
    if data["train"]["X_diag"].shape[1] > 0:
        out["plus_diagnostic"] = score(np.hstack([data["train"]["X"], data["train"]["X_diag"]]),
                                       np.hstack([data["val"]["X"], data["val"]["X_diag"]]))
    t = [c for name in time_cols if name in groups for c in groups[name]]
    keep = np.setdiff1d(np.arange(n), t)
    out["without_time_columns"] = score(data["train"]["X"][:, keep], data["val"]["X"][:, keep])
    out["time_columns"] = [name for name in time_cols if name in groups]
    return out


# --------------------------------------------------------------------------------------------------
# C2: only the is_na flags
# --------------------------------------------------------------------------------------------------
def c2_na_flags(data, flag_cols: dict[str, int], n_classes: int, seeds, n_trees, flag_thr) -> dict[str, Any]:
    cols = list(flag_cols.values())
    if not cols:
        return {"n_flags": 0}
    f1s, imps, recalls = [], [], []
    for s in seeds:
        rf = _rf(s, n_trees).fit(data["train"]["X"][:, cols], data["train"]["y"])
        pred = rf.predict(data["val"]["X"][:, cols])
        f1s.append(macro_f1(data["val"]["y"], pred))
        imps.append(rf.feature_importances_)
        recalls.append(f1_score(data["val"]["y"], pred, average=None, labels=np.unique(data["train"]["y"]), zero_division=0))
    m, sd = _ms(f1s)
    imp = pd.Series(np.mean(imps, axis=0), index=list(flag_cols)).sort_values(ascending=False)
    return {"n_flags": len(cols), "macro_f1": m, "std": sd, "chance": 1.0 / n_classes, "flag": m > flag_thr,
            "importance": imp.to_dict(), "per_class_f1": np.mean(recalls, axis=0).tolist()}


# --------------------------------------------------------------------------------------------------
# C4: group split vs random split
# --------------------------------------------------------------------------------------------------
def c4_split_gap(Xp, yp, group_p, stream_p, Xv, yv, seeds, n_trees, folds, gap_thr) -> dict[str, Any]:
    """Cross-validated macro-F1 on the train pool under three ways of cutting it, plus the true val score.

    random : stratified K-fold over rows (packets of one stream / block land on both sides)
    stream : K-fold with whole TCP streams kept together
    group  : K-fold with whole groups (captures / blocks) kept together
    val    : model fit on the whole pool, evaluated on the validation split (different groups)
    """
    out: dict[str, list[float]] = {"random": [], "stream": [], "group": [], "val": []}
    for s in seeds:
        splitters = {
            "random": StratifiedKFold(folds, shuffle=True, random_state=s).split(Xp, yp),
            "stream": StratifiedGroupKFold(folds, shuffle=True, random_state=s).split(Xp, yp, stream_p),
            "group": StratifiedGroupKFold(folds, shuffle=True, random_state=s).split(Xp, yp, group_p),
        }
        for name, it in splitters.items():
            f1s = []
            for tr, te in it:
                f1s.append(macro_f1(yp[te], _rf(s, n_trees).fit(Xp[tr], yp[tr]).predict(Xp[te])))
            out[name].append(float(np.mean(f1s)))
        out["val"].append(macro_f1(yv, _rf(s, n_trees).fit(Xp, yp).predict(Xv)))
    res = {k: _ms(v) for k, v in out.items()}
    res["gap_random_minus_group"] = res["random"][0] - res["group"][0]
    res["gap_random_minus_stream"] = res["random"][0] - res["stream"][0]
    # Informational only: the CV folds keep the (imbalanced) class mix of the train pool while val is balanced,
    # so macro-F1 on the two is not directly comparable and must not drive the flag.
    res["gap_random_minus_val"] = res["random"][0] - res["val"][0]
    res["flag"] = bool(res["gap_random_minus_group"] > gap_thr or res["gap_random_minus_stream"] > gap_thr)
    res["folds"] = folds
    return res


# --------------------------------------------------------------------------------------------------
# C5: DoS vs DDoS inside one scenario (11-class data)
# --------------------------------------------------------------------------------------------------
def c5_dos_vs_ddos(data11, label_map11: dict[str, int], seeds, n_trees, flag_thr) -> pd.DataFrame:
    scen = sorted({k[:-len("_DoS")] for k in label_map11 if k.endswith("_DoS")})
    rows = []
    for sc in scen:
        a, b = label_map11[f"{sc}_DoS"], label_map11[f"{sc}_DDoS"]
        tr, va = data11["train"], data11["val"]
        mt, mv = np.isin(tr["y"], [a, b]), np.isin(va["y"], [a, b])
        ytr, yva = (tr["y"][mt] == b).astype(int), (va["y"][mv] == b).astype(int)
        f1s, bas = [], []
        for s in seeds:
            pred = _rf(s, n_trees).fit(tr["X"][mt], ytr).predict(va["X"][mv])
            f1s.append(macro_f1(yva, pred))
            bas.append(float(balanced_accuracy_score(yva, pred)))
        m, sd = _ms(f1s)
        rows.append({"scenario": sc, "n_train": int(mt.sum()), "n_val": int(mv.sum()), "macro_f1": m, "std": sd,
                     "balanced_acc": float(np.mean(bas)), "chance": 0.5, "flag_indistinguishable": m < flag_thr})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------
def run_leakage(cfg: dict[str, Any], out_dir: str | Path | None = None, make_report: bool = True) -> dict[str, Any]:
    th = cfg["thresholds"]
    lk = cfg.get("leakage", {})
    n_trees, folds, seeds = int(lk.get("rf_trees", 100)), int(lk.get("cv_folds", 5)), list(cfg["seeds"])
    hcfg = cfg.get("harmonize", {})
    out = Path(out_dir or lk.get("reports_dir", "./results/reports/leakage"))
    out.mkdir(parents=True, exist_ok=True)

    data, schema = load_processed(cfg)
    groups = feature_groups(schema)
    diag_cols = {b["column"]: list(range(b["start"], b["start"] + b["width"]))
                 for b in schema["diagnostic"]["blocks"] if b["type"] != "na_flag"}
    n_classes = len(schema["label_map"])
    res: dict[str, Any] = {"label_mode": cfg["label_mode"], "seeds": seeds, "n_trees": n_trees,
                           "config_hash": config_hash(cfg), "D": schema["n_features"]}

    logger.info("Reference model ...")
    res["reference"] = reference_model(data, seeds, n_trees)
    res["reference_balanced"] = reference_model(data, seeds, n_trees, balanced=True)
    logger.info("C1 ...")
    pm = Path(cfg["paths"]["work_dir"]) / "inventory" / "presence_matrix.csv"
    c1 = c1_presence(pm, hcfg.get("absent_presence_max", 0.001), hcfg.get("present_presence_min", 0.01))
    logger.info("C2 ...")
    res["c2"] = c2_na_flags(data, na_flag_columns(schema), n_classes, seeds, n_trees, th["na_only_f1_flag"])
    logger.info("C3 single feature ...")
    c3 = c3_single_feature(data, groups, seeds, n_trees, th["single_feature_f1_flag"], diag_cols,
                           {v: k for k, v in schema["label_map"].items()})
    logger.info("C3 leave-one-out ...")
    c3b = c3_leave_one_out(data, groups, seeds, n_trees, res["reference"]["macro_f1"][0])
    logger.info("Ablations ...")
    res["ablations"] = ablations(data, groups, seeds, n_trees, TIME_COLUMNS)
    logger.info("C4 ...")
    tr = data["train"]
    res["c4"] = c4_split_gap(tr["X"], tr["y"], tr["group_id"], tr["stream_id"], data["val"]["X"], data["val"]["y"],
                             seeds, n_trees, folds, th["leakage_gap_flag"])

    c5 = None
    cfg11 = {**cfg, "label_mode": "11class"}
    if (processed_dir(cfg11) / "feature_schema.json").exists():
        logger.info("C5 ...")
        d11, s11 = load_processed(cfg11)
        c5 = c5_dos_vs_ddos(d11, s11["label_map"], seeds, n_trees, th["dos_ddos_f1_flag"])

    c1.to_csv(out / "c1_presence.csv", index=False)
    c3.to_csv(out / "c3_single_feature.csv", index=False)
    c3b.to_csv(out / "c3_leave_one_out.csv", index=False)
    if c5 is not None:
        c5.to_csv(out / "c5_dos_vs_ddos.csv", index=False)
    with open(out / "leakage_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    if make_report:
        from ppfeddata.checks.leakage_report import write_report
        write_report(cfg, res, c1, c3, c3b, c5, schema, out)
    return {"results": res, "c1": c1, "c3": c3, "c3b": c3b, "c5": c5}
