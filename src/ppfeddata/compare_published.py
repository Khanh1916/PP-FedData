"""Section 9.13 of the report: published results on the same dataset, and how much the evaluation protocol alone moves the score.

Nothing here trains a generator or writes to the ledger, and no number of the study changes. Two measurements, next to the published results typed
in configs/published_results.yaml:
- `protocol`: the real-data IDS of B0 (RF, 3 seeds, the same class sizes) under the study's split or a row-by-row random one, without or with the
  identifier columns the study drops (source IP; absolute time, stream index, time since the first frame of the capture);
- `closest`: the saved test predictions of the study, scored on the binary tasks of the published papers (every attack class counts as "attack").

Writes results/compare_published.json and results/reports/compare_published.md. `section` renders 9.13 from the JSON, so the report needs no data.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("ppfeddata.compare_published")

PUBLISHED = Path("./configs/published_results.yaml")
SPLITS = ("train", "val", "test")
ID_COLS = {"ip": "source", "epoch": "epoch_time", "stream": "stream_index", "rel": "time_since_reference_or_first_frame"}
ID_SETS = {"none": [], "ip": ["ip"], "time_stream": ["epoch", "stream", "rel"], "all": ["ip", "epoch", "stream", "rel"]}
CONDITIONS = (("capture", "none"), ("capture", "all"), ("random", "none"), ("random", "ip"), ("random", "time_stream"), ("random", "all"))
SPLIT_TEXT = {"capture": "by capture or block, as in the study", "random": "random, row by row"}
ID_TEXT = {"none": "none, as in the study", "ip": "source IP", "time_stream": "absolute time, stream index, time since capture start", "all": "all of them"}


def ip_to_int(s: Any) -> float:
    try:
        a, b, c, d = str(s).split(".")
        return float(int(a) * 256**3 + int(b) * 256**2 + int(c) * 256 + int(d))
    except ValueError:
        return 0.0


def _load(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from ppfeddata.data.preprocess import processed_dir
    from ppfeddata.data.split_sample import interim_dir

    out = {}
    for sp in SPLITS:
        z = np.load(processed_dir(cfg) / f"{sp}.npz", allow_pickle=True)
        d = pd.read_parquet(interim_dir(cfg) / f"{sp}.parquet", columns=list(ID_COLS.values()) + ["group_id"])
        if not (d["group_id"].astype(str).to_numpy() == z["group_id"]).all():
            raise ValueError(f"{sp}: the rows of the parquet file and of the npz file differ")
        ids = {"ip": d["source"].map(ip_to_int).to_numpy(float)}
        for k in ("epoch", "stream", "rel"):
            ids[k] = pd.to_numeric(d[ID_COLS[k]], errors="coerce").fillna(0.0).to_numpy(float)
        out[sp] = {"X": z["X"].astype(np.float64), "y": z["y"].astype(int), "ids": ids}
    return out


def redraw(data: dict[str, dict[str, Any]], seed: int) -> dict[str, dict[str, Any]]:
    """Pool the three splits and redraw them row by row, with the class sizes of the study's train and test splits (val rows are set aside as in the study)."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([data[s]["y"] for s in SPLITS])
    X = np.concatenate([data[s]["X"] for s in SPLITS])
    ids = {k: np.concatenate([data[s]["ids"][k] for s in SPLITS]) for k in data["train"]["ids"]}
    tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_tr, n_val, n_te = (int((data[s]["y"] == c).sum()) for s in SPLITS)
        te.append(idx[:n_te])
        tr.append(idx[n_te + n_val:n_te + n_val + n_tr])
    a, b = np.concatenate(tr), np.concatenate(te)
    return {"train": {"X": X[a], "y": y[a], "ids": {k: v[a] for k, v in ids.items()}},
            "test": {"X": X[b], "y": y[b], "ids": {k: v[b] for k, v in ids.items()}}}


def _stack(part: dict[str, Any], which: list[str]) -> np.ndarray:
    return np.column_stack([part["X"]] + [part["ids"][k] for k in which]) if which else part["X"]


def binary_f1(attack: np.ndarray, flagged: np.ndarray) -> float:
    tp, fp, fn = int((attack & flagged).sum()), int((~attack & flagged).sum()), int((attack & ~flagged).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {"macro_f1": float(f1_score(y, p, average="macro", labels=np.unique(y), zero_division=0)), "accuracy": float(accuracy_score(y, p)),
            "binary_f1": binary_f1(y > 0, p > 0)}


def closest_f1(y: np.ndarray, p: np.ndarray, attacks: list[int]) -> float:
    """F1 of "attack" on the rows of the normal class (0) and of `attacks`; a packet flagged as any attack counts, as in a binary detector."""
    keep = (y == 0) | np.isin(y, attacks)
    return binary_f1(y[keep] > 0, p[keep] > 0)


def protocol(cfg: dict[str, Any], data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    from ppfeddata.eval.utility import make_classifier

    rows = []
    for split, idset in CONDITIONS:
        runs = []
        for seed in [int(s) for s in cfg["seeds"]]:
            d = data if split == "capture" else redraw(data, seed)
            which = ID_SETS[idset]
            rf = make_classifier("rf", cfg, seed).fit(_stack(d["train"], which), d["train"]["y"])
            m = scores(d["test"]["y"], rf.predict(_stack(d["test"], which)))
            m.update(seed=seed, importance=float(rf.feature_importances_[d["train"]["X"].shape[1]:].sum()))
            runs.append(m)
            logger.info("%s / %s / seed %d: macro-F1 %.4f, binary F1 %.4f", split, idset, seed, m["macro_f1"], m["binary_f1"])
        rows.append({"split": split, "identifiers": idset, "runs": runs, **{k: float(np.mean([r[k] for r in runs])) for k in ("macro_f1", "accuracy", "binary_f1", "importance")},
                     "macro_f1_std": float(np.std([r["macro_f1"] for r in runs])), "binary_f1_std": float(np.std([r["binary_f1"] for r in runs]))})
    return rows


def _predictions(cfg: dict[str, Any], config: str, seed: int) -> np.ndarray:
    from ppfeddata.eval.runs import artifacts_dir, run_id

    return np.load(artifacts_dir(cfg) / run_id(config, seed, cfg["label_mode"]) / "preds" / "test.npz", allow_pickle=True)["y_pred"]


def closest(cfg: dict[str, Any], y: np.ndarray, label_map: dict[str, int], tasks: list[dict[str, Any]], configs: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for t in tasks:
        codes = [int(label_map[a]) for a in t["attacks"]]
        row = {k: t[k] for k in ("name", "study", "f1")}
        for col, config in configs.items():
            row[col] = float(np.mean([closest_f1(y, _predictions(cfg, config, int(s)), codes) for s in cfg["seeds"]]))
        rows.append(row)
    return rows


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    from ppfeddata.data.preprocess import processed_dir
    from ppfeddata.interpret import to_jsonable
    from ppfeddata.tune_marginal import framework_name

    res = Path(cfg["compute"]["runs_csv"]).parent
    pub = yaml.safe_load(PUBLISHED.read_text(encoding="utf-8"))
    data = _load(cfg)
    label_map = json.loads((processed_dir(cfg) / "label_map.json").read_text(encoding="utf-8"))
    rows = protocol(cfg, data)
    df = pd.read_csv(Path(cfg["compute"]["runs_csv"]), low_memory=False)
    b0 = df[(df["config"] == "B0-rf") & (df["label_mode"] == cfg["label_mode"])].drop_duplicates("seed").set_index("seed")["macro_f1"]
    base = next(r for r in rows if r["split"] == "capture" and r["identifiers"] == "none")
    card = {r["label"]: r for r in json.loads((res / "scorecard.json").read_text(encoding="utf-8"))["rows"]}
    mg = card[framework_name(10.0)]
    summ = pd.read_csv(res / "summary.csv", low_memory=False).set_index("config")
    study = {"B0": {"macro_f1": float(summ.loc["B0-rf", "macro_f1_mean"]), "binary_f1": float(summ.loc["B0-rf", "bin_f1_mean"])},
             "SMOTE": {"macro_f1": float(summ.loc["B1b-rf", "macro_f1_mean"]), "binary_f1": float(summ.loc["B1b-rf", "bin_f1_mean"])},
             "MG": {"label": mg["label"], "classifier": mg["tstr_classifier"], "macro_f1": float(mg["tstr_f1"]), "binary_f1": float(mg["bin_f1"])}}
    configs = {"real": "B1a-rf", "synthetic": mg["tstr_config"]}
    manifest = json.loads((res / "manifests" / f"split_manifest_{cfg['label_mode']}.json").read_text(encoding="utf-8"))
    out = {"seeds": [int(s) for s in cfg["seeds"]], "n_train": int(len(data["train"]["y"])), "n_test": int(len(data["test"]["y"])),
           "n_pooled": int(sum(len(data[s]["y"]) for s in SPLITS)), "n_features": int(data["train"]["X"].shape[1]),
           "max_diff_vs_ledger": max(abs(float(b0[r["seed"]]) - r["macro_f1"]) for r in base["runs"]),
           "subclasses": len(manifest["subclasses"]), "block_subclasses": sum(v["split_kind"] == "block" for v in manifest["subclasses"].values()),
           "study": study, "configs": configs, "protocol": rows, "closest": closest(cfg, data["test"]["y"], label_map, pub["closest_task"], configs)}
    out = to_jsonable(out)
    (res / "compare_published.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    write_report(cfg, out, pub)
    return out


def _size(delta: float) -> str:
    return "little" if abs(delta) < 0.05 else "much of the gap"


def _row(out: dict[str, Any], split: str, idset: str) -> dict[str, Any]:
    return next(r for r in out["protocol"] if r["split"] == split and r["identifiers"] == idset)


def section(out: dict[str, Any], pub: dict[str, Any]) -> list[str]:
    """The lines of section 9.13."""
    sn, st = pub["snapshot"], out["study"]
    base, rnd, ip, ts, both, cap = (_row(out, *c) for c in (("capture", "none"), ("random", "none"), ("random", "ip"), ("random", "time_stream"), ("random", "all"), ("capture", "all")))
    inc = pub["included"]
    d = out["max_diff_vs_ledger"]
    same = "The first row is exactly the ledger's B0." if d < 1e-9 else f"The first row differs from the ledger's B0 by at most {d:.1e} per seed."
    L = ["### 9.13 Against published results on the same dataset", "",
         f"Most published scores on this dataset are far above the macro-F1 of this study, but they answer other questions. This section sets them side by side and then measures how much the evaluation protocol alone moves the score. No number above changes (`compare-published`, SPEC_DEVIATIONS R.11; details in `results/reports/compare_published.md`).", "",
         "| Study | Task | Input | Split | Reported result |", "|---|---|---|---|---|"]
    L += [f"| [{p['name']}]({p['url']}) | {p['task']} | {p['input']} | {p['split']} | {p['result']} |" for p in inc]
    L += [f"| **This study** | 6 classes, one packet at a time (and the binary tasks below) | packets; {out['n_features']} encoded features, without source IP, absolute time or stream index | by capture, or by block of a capture for {out['block_subclasses']} of {out['subclasses']} sub-classes; {out['n_test']:,} real test packets, class-balanced | macro-F1: real data {st['B0']['macro_f1']:.3f}, SMOTE {st['SMOTE']['macro_f1']:.3f}, {st['MG']['label']} (synthetic data only, ε 10) {st['MG']['macro_f1']:.3f}; binary F1 {st['B0']['binary_f1']:.3f}, {st['SMOTE']['binary_f1']:.3f}, {st['MG']['binary_f1']:.3f} |", ""]
    roles = "; ".join(f"{p['name']}, {p['role']}" for p in inc)
    L += [f"**Why only these three.** A score can be read only with its task, input and split: the measurement below moves the macro-F1 of one classifier on one dataset from {base['macro_f1']:.3f} to {max(ts['macro_f1'], both['macro_f1']):.3f} by changing the protocol alone. So a study enters the table only if (1) it trains or tests on this dataset and (2) its full text was read at the source, so that those facts come from the paper itself. "
          f"Exactly {len(inc)} studies meet both, and they cover the three reference points a comparison needs: {roles}. "
          f"The other {len(pub['uses_dataset_not_read'])} studies found to use the dataset fail (2), so their scores are left out rather than quoted without a protocol. This is a limit of access to the papers, not a judgement of their quality:", ""]
    L += [f"- [{p['name']}]({p['url']}), {p['citation']}: {p['why']}." for p in pub["uses_dataset_not_read"]]
    L += ["", f"The search is not systematic: {sn['citing']} works cite the dataset paper ({sn['index']}, {sn['date']}), {sn['no_abstract']} of them have no abstract there, {sn['mqtt_and_detection']} mention MQTT and detection, and only the most likely ones were opened. {len(pub['checked_other_data'])} that were opened evaluate on other data (`results/reports/compare_published.md`). None of the studies read combines federated learning, differential privacy and synthetic data on this dataset, so the protected results have no published reference.", "",
          f"**On the closest binary tasks.** The saved test predictions are scored as a binary detector: every attack class counts as \"attack\", with no retraining. Real data = class weights, RF (B1a); synthetic data = {st['MG']['label']}, {st['MG']['classifier'].upper()} (the classifier chosen on validation). Mean of {len(out['seeds'])} seeds.", "",
          "| Task | Published | Study | Real data | Synthetic data (ε 10) |", "|---|---|---|---|---|"]
    L += [f"| {r['name']} | {r['f1']:.3f} | {r['study']} | {r['real']:.3f} | {r['synthetic']:.3f} |" for r in out["closest"]]
    L += ["", "The tasks are close, not identical: the papers train a binary model, this study reuses its 6-class models; the first row holds 2 attack packets for 1 normal one, the others 1 for 1; BCF here has both the DoS and the DDoS variant, BF in Zhang et al. the DDoS one.", "",
          f"**How much the protocol alone moves the score.** The IDS of B0 (RF, {len(out['seeds'])} seeds, the same class sizes) on the {out['n_pooled']:,} sampled packets of the study; only the split and the identifier columns change. {same}", "",
          "| Split | Identifier columns added | Macro-F1 | Binary F1 |", "|---|---|---|---|"]
    L += [f"| {SPLIT_TEXT[r['split']]} | {ID_TEXT[r['identifiers']]} | {r['macro_f1']:.3f} ± {r['macro_f1_std']:.3f} | {r['binary_f1']:.3f} ± {r['binary_f1_std']:.3f} |" for r in out["protocol"]]
    L += ["", f"- **The split alone explains {_size(rnd['macro_f1'] - base['macro_f1'])}:** row by row instead of by capture, macro-F1 moves by {rnd['macro_f1'] - base['macro_f1']:+.3f}.",
          f"- **Source IP alone explains {_size(ip['macro_f1'] - rnd['macro_f1'])}:** {ip['macro_f1'] - rnd['macro_f1']:+.3f} on top of the random split.",
          f"- **The time and stream columns explain {_size(ts['macro_f1'] - rnd['macro_f1'])}:** {ts['macro_f1'] - rnd['macro_f1']:+.3f}, to {ts['macro_f1']:.3f}. Each capture was recorded at its own time, so absolute time can identify the capture, hence the class; a deployed detector cannot use that. This is why the study drops these columns.",
          f"- **A split by capture does not stop this leak** ({cap['macro_f1']:.3f} with all identifiers). The likely reason: the test rows of {out['block_subclasses']} of {out['subclasses']} sub-classes are a block of the same capture as their train rows.",
          "- **What this does not show:** that any published score is inflated. Alatram et al. list these columns but do not say which ones the classifiers received; Zhang et al. use the payload, which this measurement does not touch; Parra & Bayas give their classifier four traffic statistics, none of these columns. It covers one classifier and the sampled packets of the study.", ""]
    return L


def section_for(cfg: dict[str, Any]) -> list[str]:
    p = Path(cfg["compute"]["runs_csv"]).parent / "compare_published.json"
    if not p.exists() or not PUBLISHED.exists():
        return []
    return section(json.loads(p.read_text(encoding="utf-8")), yaml.safe_load(PUBLISHED.read_text(encoding="utf-8")))


def write_report(cfg: dict[str, Any], out: dict[str, Any], pub: dict[str, Any]) -> None:
    L = ["# Comparison with published results on DoS/DDoS-MQTT-IoT", "", "Written by `compare-published`; do not edit. The published values are typed in `configs/published_results.yaml`.", ""]
    L += [ln.replace("### 9.13 ", "## ") for ln in section(out, pub)]
    L += ["## Scores per seed", "", "| Split | Identifiers | Seed | Macro-F1 | Accuracy | Binary F1 | Share of RF importance of the added columns |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['split']} | {r['identifiers']} | {x['seed']} | {x['macro_f1']:.4f} | {x['accuracy']:.4f} | {x['binary_f1']:.4f} | {x['importance']:.3f} |" for r in out["protocol"] for x in r["runs"]]
    L += ["", "## How each study was read", "", "| Study | Citation | How it was read |", "|---|---|---|"]
    L += [f"| {p['name']} | {p['citation']} | {p['read']} |" for p in pub["included"]]
    L += ["", "## Opened, and evaluated on other data", "", "| Work | Data |", "|---|---|"]
    L += [f"| {p['name']} | {p['data']} |" for p in pub["checked_other_data"]]
    res = Path(cfg["compute"]["runs_csv"]).parent
    (res / "reports" / "compare_published.md").write_text("\n".join(L) + "\n", encoding="utf-8")
