"""Extensions A4 and A5 of the spec (Phase 11): do the conclusions depend on WHICH capture groups form the validation / test split (A4), and on how many
packets of one TCP stream may enter the train pool (A5)?

Each scenario is a world of its own, built from the same inputs as the main study with one setting changed:
- `A4-s1`, `A4-s2`: `split.split_seed` 1 and 2 (another choice of the groups / blocks that go to val and test, and of the rows drawn), then Phase 3, 4, B0 and B3;
- `A5-cap20`: `train_sampling.max_rows_per_stream: 20`, then Phase 3, 4, B0, B3 and M1-eps5 (the DP-tuned family).
A world has its own data (`data/_sens/<tag>`), runs and predictions (`artifacts/_sens/<tag>`) and manifests, so nothing of the main study is touched. The raw-data scan of Phase 3
and the inventory are shared (copied), so no raw file is read again.

`report()` puts the main study next to every world: macro-F1 of B0 and B3 (TSTR, TAug), the verdict of R1 (TAug minus B0), of the synthetic-only classifier against B0, and (A5) the DP cost,
each with the paired bootstrap of that world's own test split and the rule of the spec (interval excludes 0 and |delta| above the seed std). It writes `results/sensitivity.json` and
`results/reports/sensitivity.md`.
"""
from __future__ import annotations

import copy
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ppfeddata.data.preprocess import processed_dir
from ppfeddata.eval.compare import MACRO_F1, PairedBootstrap, compare_runs, recall_of
from ppfeddata.eval.runs import RunLedger, load_predictions, run_id, runs_csv_path

logger = logging.getLogger("ppfeddata.sensitivity")

SCENARIOS: dict[str, dict[str, Any]] = {
    "A4-s1": {"extension": "A4", "set": {"split": {"split_seed": 1}}, "runs": ["B0", "B3"], "what": "split_seed 1: another choice of the test / validation groups"},
    "A4-s2": {"extension": "A4", "set": {"split": {"split_seed": 2}}, "runs": ["B0", "B3"], "what": "split_seed 2: another choice of the test / validation groups"},
    "A5-cap20": {"extension": "A5", "set": {"train_sampling": {"max_rows_per_stream": 20}}, "runs": ["B0", "B3", "M1-eps5"], "what": "at most 20 packets of one TCP stream in the train pool"},
}
MAIN = "main"
RF, MLP = "rf", "mlp"


def world_cfg(cfg: dict[str, Any], tag: str) -> dict[str, Any]:
    """The config of a world: the study's config with one setting changed and its own data, artifacts and ledger."""
    if tag == MAIN:
        return cfg
    sc = SCENARIOS[tag]
    c = copy.deepcopy(cfg)
    for section, kv in sc["set"].items():
        c.setdefault(section, {}).update(kv)
    work = Path(cfg["paths"]["work_dir"]) / "_sens" / tag
    root = Path(cfg["compute"]["artifacts_dir"]) / "_sens" / tag
    c["paths"] = {**cfg["paths"], "work_dir": str(work), "shared_manifest_dir": str(work / "manifests")}
    c["compute"] = {**cfg["compute"], "artifacts_dir": str(root), "runs_csv": str(root / "runs.csv")}
    return c


def prepare(cfg: dict[str, Any], tag: str) -> dict[str, Any]:
    """Phase 3 and 4 for a world; the inventory and the raw-data scan are copied from the main study (nothing is scanned again)."""
    from ppfeddata.data.preprocess import run_preprocess
    from ppfeddata.data.split_sample import run_sample

    w = world_cfg(cfg, tag)
    main_work, work = Path(cfg["paths"]["work_dir"]), Path(w["paths"]["work_dir"])
    for sub in ("inventory", Path("interim") / "_scan"):
        if not (work / sub).exists():
            shutil.copytree(main_work / sub, work / sub)
    if not (processed_dir(w) / "feature_schema.json").exists():
        run_sample(w)
        run_preprocess(w)
    return w


def run_world(cfg: dict[str, Any], tag: str, seeds: list[int] | None = None) -> None:
    """Data and runs of one world. Resumable: a finished run is skipped."""
    from ppfeddata.eval.baselines import run_baselines
    from ppfeddata.fl.b3 import run_b3
    from ppfeddata.fl.m1 import run_m1

    sc, w = SCENARIOS[tag], prepare(cfg, tag)
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    run_baselines(w, ["B0-rf", "B0-mlp"], seeds)
    run_b3(w, seeds)
    if "M1-eps5" in sc["runs"]:
        run_m1(w, [5.0], seeds, tuned=True)


# --------------------------------------------------------------------------------------------------
# Summary of a world
# --------------------------------------------------------------------------------------------------
def _names(cfg: dict[str, Any]) -> dict[str, str]:
    """Ledger names of the runs the report compares."""
    try:
        from ppfeddata.fl.m1 import m1_name, tuned_setup
        fam = m1_name(5.0, tuned_setup(cfg)[0])
    except Exception:                                     # no tuned DP family on file
        fam = None
    out = {"B0-rf": "B0-rf", "B0-mlp": "B0-mlp"}
    for p in ("TSTR", "TAug"):
        for c in (RF, MLP):
            out[f"B3-{p}-{c}"] = f"B3-{p}-{c}"
            if fam:
                out[f"M1-eps5-{p}-{c}"] = f"{fam}-plain-{p}-{c}"
    out.update({f"B3-plain-TSTR-{c}": f"B3-plain-TSTR-{c}" for c in (RF, MLP)})
    return out


def _preds(w: dict[str, Any], names: dict[str, str], seeds: list[int]) -> dict[tuple[str, int], np.ndarray]:
    got: dict[tuple[str, int], np.ndarray] = {}
    for key, name in names.items():
        for s in seeds:
            try:
                got[(key, s)] = load_predictions(w, run_id(name, s, w["label_mode"]))["y_pred"]
            except FileNotFoundError:
                break
    return {k: v for k, v in got.items() if all((k[0], s) in got for s in seeds)}


def summarise_world(cfg: dict[str, Any], tag: str, n_boot: int | None = None) -> dict[str, Any] | None:
    """Test-split facts, macro-F1 of the runs and the paired comparisons of one world; None when the world has no runs yet."""
    w = world_cfg(cfg, tag)
    if not runs_csv_path(w).exists():
        return None
    seeds = [int(s) for s in cfg["seeds"]]
    z = np.load(processed_dir(w) / "test.npz", allow_pickle=True)
    schema = json.loads((processed_dir(w) / "feature_schema.json").read_text(encoding="utf-8"))
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    y = z["y"].astype(np.int64)
    k = len(classes)
    names = _names(w)
    preds = _preds(w, names, seeds)
    have = sorted({key for key, _ in preds})
    boot = PairedBootstrap(y, k, int(n_boot or cfg["eval"]["bootstrap"]), 0, "rows")
    boot.add(preds)
    q = cfg.get("quota", {})
    rare = [c for c in classes if c in q and q[c][0] <= 5000] or classes[-4:]
    rare_metric = recall_of(*[classes.index(c) for c in rare])
    f1 = {key: [float(MACRO_F1(boot.point[(key, s)])) for s in seeds] for key in have}
    out: dict[str, Any] = {"tag": tag, "what": SCENARIOS[tag]["what"] if tag != MAIN else "the main study", "seeds": seeds, "n_test_rows": int(len(y)),
                           "n_test_groups": int(len(np.unique(z["group_id"]))) if "group_id" in z.files else None,
                           "test_group_ids": sorted(str(g) for g in np.unique(z["group_id"])) if "group_id" in z.files else [],
                           "n_test_streams": int(len(np.unique(z["stream_id"]))) if "stream_id" in z.files else None,
                           "macro_f1": {key: {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v} for key, v in f1.items()}, "comparisons": {}}
    tr = np.load(processed_dir(w) / "train.npz", allow_pickle=True)
    if "stream_id" in tr.files:
        _, cnt = np.unique(tr["stream_id"], return_counts=True)
        out["train_rows_per_stream"] = {"mean": float(cnt.mean()), "max": int(cnt.max()), "n_train_rows": int(len(tr["y"]))}

    def cmp(label: str, a: str, b: str, what: str) -> None:
        if a in have and b in have:
            ka, kb = [(a, s) for s in seeds], [(b, s) for s in seeds]
            out["comparisons"][label] = {"a": a, "b": b, "what": what, "macro_f1": compare_runs(boot, ka, kb), "rare_recall": compare_runs(boot, ka, kb, rare_metric)}

    for c in (RF, MLP):
        cmp(f"R1-{c}", f"B3-TAug-{c}", f"B0-{c}", f"real + synthetic (B3) minus real only ({c.upper()}): R1")
        cmp(f"TSTR-vs-B0-{c}", f"B3-TSTR-{c}", f"B0-{c}", f"synthetic only (B3) minus real only ({c.upper()})")
        cmp(f"DP-cost-{c}", f"M1-eps5-TSTR-{c}", f"B3-plain-TSTR-{c}", f"DP (eps 5, plain decoder) minus eps = infinity, synthetic only ({c.upper()}): R4")
    out["rare_classes"] = rare
    return out


def report(cfg: dict[str, Any], tags: list[str] | None = None, n_boot: int | None = None, out_dir: str | Path | None = None) -> dict[str, Any]:
    """Write results/sensitivity.json and results/reports/sensitivity.md from the worlds that have runs (the main study first)."""
    root = Path(out_dir) if out_dir else Path(cfg["compute"]["runs_csv"]).parent
    worlds = [summarise_world(cfg, t, n_boot) for t in [MAIN] + [t for t in (tags or list(SCENARIOS)) if t != MAIN]]
    worlds = [w for w in worlds if w]
    main_ids = set(worlds[0]["test_group_ids"]) if worlds and worlds[0]["tag"] == MAIN else set()
    for w in worlds:
        w["test_groups_shared_with_main"] = len(main_ids & set(w["test_group_ids"])) if main_ids and w["tag"] != MAIN else None
    res = {"worlds": worlds, "label_mode": cfg["label_mode"]}
    from ppfeddata.interpret import to_jsonable
    root.mkdir(parents=True, exist_ok=True)
    (root / "sensitivity.json").write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    p = root / "reports" / "sensitivity.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(render(res)) + "\n", encoding="utf-8")
    return res


# --------------------------------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------------------------------
def _table(rows: list[dict[str, Any]]) -> str:
    from ppfeddata.aggregate import _table as t
    return t(rows)


def _f(x: float | None, nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def verdicts(res: dict[str, Any]) -> list[dict[str, Any]]:
    """For each comparison label: the verdict (better / worse / none) in every world, and whether the main study's verdict is the same in all of them."""
    worlds = res["worlds"]
    labels = [k for k in (worlds[0]["comparisons"] if worlds else {})]
    out = []
    for lab in labels:
        v = {w["tag"]: w["comparisons"][lab]["macro_f1"]["effect"] for w in worlds if lab in w["comparisons"]}
        out.append({"label": lab, "what": worlds[0]["comparisons"][lab]["what"], "verdicts": v, "same_everywhere": len(set(v.values())) == 1 and len(v) == len(worlds)})
    return out


def render(res: dict[str, Any]) -> list[str]:
    W = res["worlds"]
    L = ["# Sensitivity A4 and A5", "",
         "Auto-generated by `ppfeddata sensitivity` (module `sensitivity.py`). Each world is the main study with one setting changed (A4: `split.split_seed`, i.e. which capture groups form the validation and test splits; "
         "A5: `train_sampling.max_rows_per_stream`), re-run from Phase 3 on with its own data, runs and test split; the main study is the first row. Every number is read from the ledger and the saved predictions of "
         "that world; the verdict is the rule of the spec (paired-bootstrap interval of the difference excludes 0 and |delta| exceeds the seed std; 3 seeds).", ""]
    if not W:
        return L + ["_(no world has runs yet)_"]
    L += ["## Worlds", "", _table([{"world": w["tag"], "what": w["what"], "test rows": f"{w['n_test_rows']:,}", "test groups (shared with main)": ("-" if w["n_test_groups"] is None else f"{w['n_test_groups']}" + (f" ({w['test_groups_shared_with_main']})" if w.get("test_groups_shared_with_main") is not None else "")),
                                    "test streams": f"{w['n_test_streams']:,}" if w["n_test_streams"] else "-",
                                    "train rows / stream (mean, max)": (f"{w['train_rows_per_stream']['mean']:.2f}, {w['train_rows_per_stream']['max']}" if w.get("train_rows_per_stream") else "-")} for w in W]), ""]
    keys = [k for k in ("B0-rf", "B0-mlp", "B3-TSTR-rf", "B3-TSTR-mlp", "B3-TAug-rf", "B3-TAug-mlp", "M1-eps5-TSTR-rf", "M1-eps5-TSTR-mlp") if any(k in w["macro_f1"] for w in W)]
    L += ["## Test macro-F1 (mean ± std over seeds)", "", _table([{"world": w["tag"], **{k: (f"{w['macro_f1'][k]['mean']:.3f} ± {w['macro_f1'][k]['std']:.3f}" if k in w["macro_f1"] else "-") for k in keys}} for w in W]), ""]
    V = verdicts(res)
    cells = []
    for v in V:
        cells.append({"comparison": v["what"], **{w["tag"]: (f"{w['comparisons'][v['label']]['macro_f1']['delta']:+.4f} {w['comparisons'][v['label']]['macro_f1']['effect']}" if v["label"] in w["comparisons"] else "-") for w in W},
                      "same verdict in every world": "yes" if v["same_everywhere"] else "NO"})
    L += ["## Verdicts (delta of macro-F1 and the rule of the spec)", "", _table(cells), ""]
    changed = [v for v in V if not v["same_everywhere"]]
    n_w = len(W)
    L += [f"**Reading.** {len(V) - len(changed)} of {len(V)} comparisons have the same verdict in all {n_w} worlds"
          + ("." if not changed else "; those that change: " + "; ".join(f"{v['what']} ({', '.join(f'{t} {e}' for t, e in v['verdicts'].items())})" for v in changed) + "."), ""]
    L += ["What this does and does not cover: it covers the choice of the capture groups of the validation and test splits (A4) and the cap on packets per stream (A5), with the same hyper-parameters (tuned once, on the main "
          "study's validation data) and 3 seeds per world. It does not cover other datasets, other quotas (A2), the 11-class mode (A3), or a different number of clients. Most sub-classes come from one capture file "
          "(their test rows are a block of that file; see the limitation on block splits), so another `split_seed` mostly moves the block, not the capture.", ""]
    return L
