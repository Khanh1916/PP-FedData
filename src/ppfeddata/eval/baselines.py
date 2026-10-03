"""Phase 6: baselines B0 (real only), B1a (class_weight balanced RF), B1b (SMOTE in the encoded space).

Every run is trained on the train split only and evaluated on the REAL test split (validation is only reported as a
sanity number). Predictions are saved for paired bootstrap; one row per run goes to results/runs.csv.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppfeddata.data.preprocess import processed_dir
from ppfeddata.eval.overhead import Timer, peak_rss_gb
from ppfeddata.eval.runs import RunLedger, load_predictions, run_id, runs_csv_path, save_predictions
from ppfeddata.eval.stats import bootstrap_macro_f1, paired_bootstrap_diff
from ppfeddata.eval.utility import (build_train_set, compute_metrics, full_proba, make_classifier,
                                    smote_oversample)
from ppfeddata.utils import config_hash, git_commit

logger = logging.getLogger("ppfeddata.eval.baselines")

# name -> (classifier, class_weight balanced, SMOTE)
BASELINES: dict[str, tuple[str, bool, bool]] = {
    "B0-rf": ("rf", False, False),
    "B0-mlp": ("mlp", False, False),
    "B1a-rf": ("rf", True, False),
    "B1b-rf": ("rf", False, True),
    "B1b-mlp": ("mlp", False, True),
}


def load_data(cfg: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    import json
    d = processed_dir(cfg)
    data = {s: dict(np.load(d / f"{s}.npz")) for s in ("train", "val", "test")}
    schema = json.loads((d / "feature_schema.json").read_text(encoding="utf-8"))
    check_fit_on_train(schema, data)
    return data, schema


def check_fit_on_train(schema: dict[str, Any], data: dict[str, dict[str, np.ndarray]]) -> None:
    """The Preprocessor must have been fitted on the training split only (recorded by Phase 4)."""
    fit = schema.get("fit")
    if not fit:
        raise RuntimeError("feature_schema.json has no `fit` record; re-run `preprocess`")
    if fit.get("split") != "train" or int(fit["n_rows"]) != len(data["train"]["y"]):
        raise RuntimeError(f"Preprocessor was not fitted on the train split alone: {fit}, train rows {len(data['train']['y'])}")


def _flat_metrics(m: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {f"{prefix}macro_f1": m["macro_f1"], f"{prefix}balanced_acc": m["balanced_acc"]}
    if "pr_auc_macro" in m:
        out[f"{prefix}pr_auc_macro"] = m["pr_auc_macro"]
    for k, v in m["binary"].items():
        out[f"{prefix}bin_{k}"] = v
    for cls, d in m["per_class"].items():
        for kk, vv in d.items():
            out[f"{prefix}{kk}_{cls}"] = vv
    return out


def run_one(cfg, name: str, seed: int, data, schema, ledger: RunLedger) -> dict[str, Any]:
    clf_name, balanced, smote = BASELINES[name]
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    k = len(classes)
    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    with Timer() as t_prep:
        if smote:
            Xtr, ytr = smote_oversample(Xtr, ytr, schema, int(cfg["generate"]["target_per_class"]), seed,
                                        int(cfg["eval"].get("smote", {}).get("k_neighbors", 5)))
    clf = make_classifier(clf_name, cfg, seed, balanced)
    with warnings.catch_warnings(), Timer() as t_fit:
        warnings.simplefilter("ignore")      # MLP ConvergenceWarning is recorded through n_iter below
        clf.fit(Xtr, ytr)
    peak = peak_rss_gb()
    rid = run_id(name, seed, cfg["label_mode"])
    res: dict[str, Any] = {}
    for split in ("test", "val"):
        X, y = data[split]["X"], data[split]["y"]
        pred, proba = clf.predict(X), full_proba(clf, X, k)
        res[split] = compute_metrics(y, pred, proba, classes)
        save_predictions(cfg, rid, split, pred, proba if split == "test" else None,
                         meta={"run_id": rid, "config": name, "seed": seed, "n_train": int(len(ytr))} if split == "test" else None)
    row = {"run_id": rid, "config": name, "classifier": clf_name, "protocol": "TRTR" if not smote else "TRTR+SMOTE",
           "seed": seed, "label_mode": cfg["label_mode"], "config_hash": config_hash(cfg), "git_commit": git_commit(),
           "n_train": int(len(ytr)), "n_synthetic": int(len(ytr) - len(data["train"]["y"])) if smote else 0,
           "prep_time_s": t_prep.seconds, "fit_time_s": t_fit.seconds, "peak_rss_gb": peak,
           "n_iter": int(getattr(clf, "n_iter_", 0)) if clf_name == "mlp" else 0,
           **_flat_metrics(res["test"]), **_flat_metrics(res["val"], "val_")}
    ledger.append(row)
    logger.info("%s: test macro-F1 %.4f (val %.4f), fit %.1fs", rid, row["macro_f1"], row["val_macro_f1"], t_fit.seconds)
    return row


def run_baselines(cfg: dict[str, Any], names: list[str] | None = None, seeds: list[int] | None = None,
                  resume: bool = True) -> pd.DataFrame:
    names = names or list(BASELINES)
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    data, schema = load_data(cfg)
    ledger = RunLedger(runs_csv_path(cfg))
    for name in names:
        for seed in seeds:
            rid = run_id(name, seed, cfg["label_mode"])
            if resume and ledger.done(rid):
                logger.info("skip %s (already in the ledger)", rid)
                continue
            run_one(cfg, name, seed, data, schema, ledger)
    return ledger.frame()


# --------------------------------------------------------------------------------------------------
# G3 report
# --------------------------------------------------------------------------------------------------
def _ms(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.4f} ± {x.std():.4f}"


def write_g3_report(cfg: dict[str, Any], out: str | Path = "./results/reports/g3_baseline.md") -> dict[str, Any]:
    ledger = RunLedger(runs_csv_path(cfg)).frame()
    mode = cfg["label_mode"]
    df = ledger[ledger["label_mode"] == mode]
    data, schema = load_data(cfg)
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    k, nb = len(classes), int(cfg["eval"]["bootstrap"])
    thr = cfg["thresholds"]["seed_std_max"]
    y = data["test"]["y"]
    names = [n for n in BASELINES if (df["config"] == n).any()]
    seeds = [int(x) for x in sorted(df["seed"].unique())]
    L = [f"# G3 - Baseline report ({mode})", "",
         f"Auto-generated by `ppfeddata baseline`. Train on the train split, evaluate on the REAL test split "
         f"({len(y)} rows, class-balanced). Seeds {seeds}; RF {cfg['eval']['rf']['n_estimators']} trees; "
         f"MLP {cfg['eval']['mlp']['hidden']}; SMOTE target {cfg['generate']['target_per_class']} rows/class; "
         f"bootstrap {nb} stratified resamples. All numbers come from `results/runs.csv` and saved predictions.", ""]
    gate = {"std_ok": {}, "b0_not_perfect": {}}
    rows = []
    for n in names:
        s = df[df["config"] == n].sort_values("seed")
        rows.append({"config": n, "runs": len(s), "macro-F1 (test)": _ms(s["macro_f1"]), "balanced acc": _ms(s["balanced_acc"]),
                     "PR-AUC macro": _ms(s["pr_auc_macro"]), "macro-F1 (val)": _ms(s["val_macro_f1"]),
                     "fit s": f"{s['fit_time_s'].mean():.1f}"})
        gate["std_ok"][n] = bool(s["macro_f1"].std(ddof=0) < thr and len(s) == len(seeds))
        if n.startswith("B0"):
            gate["b0_not_perfect"][n] = bool(s["macro_f1"].max() < 0.99)
    L += ["## Utility (mean ± std over seeds)", "", _table(rows), ""]

    rows = []
    for n in names:
        s = df[df["config"] == n]
        rows.append({"config": n, **{c: f"{s[f'recall_{c}'].mean():.3f}" for c in classes}})
    L += ["## Recall per class (test, mean over seeds)", "", _table(rows), ""]
    rows = []
    for n in names:
        s = df[df["config"] == n]
        rows.append({"config": n, **{c: f"{s[f'f1_{c}'].mean():.3f}" for c in classes}})
    L += ["## F1 per class (test, mean over seeds)", "", _table(rows), ""]

    rows = []
    for n in names:
        s = df[df["config"] == n]
        rows.append({"config": n, "precision": f"{s['bin_precision'].mean():.3f}", "recall": f"{s['bin_recall'].mean():.3f}",
                     "F1": f"{s['bin_f1'].mean():.3f}", "accuracy": f"{s['bin_accuracy'].mean():.3f}"})
    L += ["## Binary Normal vs Attack (collapsed from the multi-class prediction, test)", "", _table(rows), ""]

    L += ["## Bootstrap CI of macro-F1 (test, per seed)", ""]
    rows = []
    for n in names:
        for sd in seeds:
            rid = run_id(n, sd, mode)
            if not (df["run_id"] == rid).any():
                continue
            b = bootstrap_macro_f1(y, load_predictions(cfg, rid)["y_pred"], k, nb, seed=sd)
            rows.append({"config": n, "seed": sd, "macro-F1": f"{b['macro_f1']:.4f}", "95% CI": f"[{b['lo']:.4f}, {b['hi']:.4f}]"})
    L += [_table(rows), ""]

    L += ["## Paired difference vs B0 (same classifier, test, macro-F1)", "",
          "R1 criterion (spec): the paired CI excludes 0 AND |diff| > std of macro-F1 over seeds.", ""]
    rows = []
    for n in names:
        if n.startswith("B0"):
            continue
        base = "B0-" + BASELINES[n][0]
        if base not in names:
            continue
        for sd in seeds:
            ra, rb = run_id(n, sd, mode), run_id(base, sd, mode)
            if not ((df["run_id"] == ra).any() and (df["run_id"] == rb).any()):
                continue
            d = paired_bootstrap_diff(y, load_predictions(cfg, ra)["y_pred"], load_predictions(cfg, rb)["y_pred"], k, nb, seed=sd)
            std_seed = float(df[df["config"] == n]["macro_f1"].std(ddof=0))
            rows.append({"config": f"{n} - {base}", "seed": sd, "diff": f"{d['diff']:+.4f}", "95% CI": f"[{d['lo']:+.4f}, {d['hi']:+.4f}]",
                         "excludes 0": d["excludes_zero"], "|diff| > seed std": abs(d["diff"]) > std_seed})
    L += [_table(rows), ""]

    L += ["## Gate G3 checks", "",
          f"- std of macro-F1 over seeds < {thr} (and all seeds present): " + ", ".join(f"{n} {'OK' if v else 'FAIL'}" for n, v in gate["std_ok"].items()),
          "- B0 macro-F1 not ~1.000 (all runs < 0.99): " + ", ".join(f"{n} {'OK' if v else 'FAIL'}" for n, v in gate["b0_not_perfect"].items()), ""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(L), encoding="utf-8")
    return gate


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_no rows_"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)
