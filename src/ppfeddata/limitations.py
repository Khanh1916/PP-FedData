"""Limitations of the study (spec Phase 13, Definition of Done): one list, written once, used by the final report (section 10), the README and the demo.

Every number that can be recomputed from a file of the repository is read from it (split manifest, feature schema, config, interpretation.json). A number that exists only
in a documented measurement of an earlier phase is cited by its SPEC_DEVIATIONS row instead of being re-typed as a fresh result. When a source file is missing the item stays
in the list and loses its number.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# The Definition of Done of the spec lists the limitations that must be stated: item -> a phrase the generated item must contain (checked by `acceptance.py` and the tests)
SPEC_LIST = {"packet-level data": "packet-level", "epsilon per record, correlated packets": "group privacy", "labels not protected by DP": "label",
             "time features / independent packets": "independently", "non-private tuning": "tuned on real", "single machine, no network latency": "latency",
             "normalisation with central statistics": "centrally", "test split from few groups": "capture groups", "thresholds are heuristics": "heuristics", "one dataset": "one dataset",
             "classes hard to tell apart per packet": "separability", "8 of 11 sub-classes have one capture file": "single capture file", "protocol filter": "RIPv2",
             "first element of multi-valued cells": "first element", "time_delta tail clipped at 5 sigma": "clipped at 5 sigma", "very sparse columns": "very sparse",
             "DoS / DDoS merged in 6-class mode": "merged"}


def manifest_dir(cfg: dict[str, Any]) -> Path:
    return Path((cfg.get("paths") or {}).get("shared_manifest_dir", "./results/manifests"))


def manifest_path(cfg: dict[str, Any]) -> Path:
    return manifest_dir(cfg) / f"split_manifest_{cfg['label_mode']}.json"


def schema_paths(cfg: dict[str, Any]) -> list[Path]:
    """Where the feature schema may be: the processed data (not committed) or the committed copy next to the split manifest."""
    return [Path((cfg.get("paths") or {}).get("work_dir", "./data")) / "processed" / cfg["label_mode"] / "feature_schema.json",
            manifest_dir(cfg) / f"feature_schema_{cfg['label_mode']}.json"]


def _read(p: Path) -> dict[str, Any] | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_manifest(cfg: dict[str, Any]) -> dict[str, Any] | None:
    return _read(manifest_path(cfg))


def load_schema(cfg: dict[str, Any]) -> dict[str, Any] | None:
    for p in schema_paths(cfg):
        if p.exists():
            return _read(p)
    return None


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f} %"


def block_split_shares(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    """Sub-classes split by consecutive blocks (one capture file) and the share of their eligible rows lost by dropping the streams that straddle blocks."""
    if not manifest or "subclasses" not in manifest:
        return None
    subs = manifest["subclasses"]
    block = {k: v for k, v in subs.items() if v.get("split_kind") == "block"}
    share = {k: v["cross_block"]["rows_removed"] / v["cross_block"]["eligible_before"] for k, v in block.items()
             if v.get("cross_block") and v["cross_block"].get("eligible_before")}
    return {"n_block": len(block), "n_sub": len(subs), "share": share,
            "min": min(share.values()) if share else None, "max": max(share.values()) if share else None}


def time_delta_clip(schema: dict[str, Any] | None, col: str = "time_delta_from_previous_displayed_frame") -> dict[str, Any] | None:
    """Rows of the train split clipped at the sigma bound for the time-gap column (Phase 4 audit)."""
    if not schema or "audit" not in schema or "fit" not in schema:
        return None
    n = schema["audit"].get("train", {}).get("clipped_sigma", {}).get(col, 0)
    rows = int(schema["fit"].get("n_rows", 0))
    return {"column": col, "rows": int(n), "n_rows": rows, "share": n / rows if rows else None, "sigma": schema.get("settings", {}).get("clip_sigma"),
            "scale": schema.get("settings", {}).get("numeric_scale", {}).get(col)}


def ran_a4(summ: pd.DataFrame | None) -> bool:
    """A4 (sensitivity to the choice of test groups) leaves configurations named `A4...` in the ledger."""
    return bool(summ is not None and len(summ) and summ["config"].astype(str).str.startswith("A4").any())


def limitations(cfg: dict[str, Any], interp: dict[str, Any] | None = None, summ: pd.DataFrame | None = None) -> list[dict[str, str]]:
    """The list, in the order of the spec's Definition of Done; each item is {id, topic, text, source}."""
    man, sch = load_manifest(cfg), load_schema(cfg)
    th = cfg.get("thresholds", {})
    meta = (interp or {}).get("meta", {})
    why = (interp or {}).get("why", {})
    out: list[dict[str, str]] = []

    def add(topic: str, text: str, source: str) -> None:
        out.append({"id": f"L{len(out) + 1:02d}", "topic": topic, "text": text, "source": source})

    add("Packet-level data, record-level epsilon",
        "The data are packet-level and epsilon is per record (one packet). Packets of one TCP stream are strongly correlated, so what DP protects about a whole attack session "
        "is much weaker than epsilon suggests (group privacy).", "spec Definition of Done; section 9.8")
    add("Labels are not protected",
        "As in the spec, the claim of epsilon is scoped to the features of a record: the class label is treated as known side information. Epsilon is not claimed to hide labels, class "
        "counts or which classes a client holds.", "spec Definition of Done")
    add("Time features and independent packets",
        "The time features (gap to the previous frame, time since the first frame of the stream, round-trip time) depend on how the capture was exported, and the CVAE generates "
        "every packet independently: a generated packet has no stream context, so the temporal structure of a session is not reproduced.", "spec Definition of Done; leakage_report.md (ablation without the time columns)")
    add("Non-private tuning",
        "The hyper-parameters of the CVAE (and of the DP variant) were tuned on real, non-private validation data; tuning is not covered by epsilon.", "SPEC_DEVIATIONS 7.3, 9.5, 9.14")
    add("Single-machine simulation",
        "Federated training is simulated on one machine (Flower simulation, the clients are Ray actors sharing the same CPU): no network latency is measured, so the times per round "
        "are compute time on this machine, neither the sum of the clients' times nor what a deployment would see. Ray on Windows was unstable (SPEC_DEVIATIONS 9.8).",
        "compute_budget.md; SPEC_DEVIATIONS 8.10, 9.8")
    fit_rows = f" ({int(sch['fit']['n_rows']):,} rows)" if sch and sch.get("fit", {}).get("n_rows") else ""
    add("Normalisation uses central statistics",
        f"Standardisation, the log1p choices, the category lists and the mask of unused categories were computed centrally on the pooled train split{fit_rows}, as if every client had shared its "
        "data once. A real federation would have to compute them securely or fix them beforehand. They are not covered by epsilon, and neither are the per-class scales of the "
        "residual-noise variant (the rows without the `-plain` suffix).", "spec Definition of Done; SPEC_DEVIATIONS 7.1, 8.8")
    groups = f"{meta['n_groups']} capture groups and {meta['n_streams']:,} TCP streams" if meta.get("n_groups") and meta.get("n_streams") else "few capture groups"
    a4 = ("Sensitivity A4 (other split seeds) was run: see the ledger." if ran_a4(summ)
          else "Sensitivity A4 (other choices of test groups) was not run, so how far the conclusions depend on these groups is untested.")
    add("Test split from few capture groups",
        f"The real test split comes from {groups}. {a4}", "leakage_report.md; SPEC_DEVIATIONS 3.3-3.5")
    thr = ", ".join(f"{k} = {th[k]:g}" for k in ("mia_auc_max", "eps_max_recommend", "overhead_ratio_max", "seed_std_max", "seed_std_redflag") if k in th)
    add("Thresholds are heuristics",
        "The thresholds of the evaluation (`thresholds` in configs/default.yaml" + (f"; {thr}" if thr else "") + ") are heuristics fixed by the user, not derived from data. Verdicts such as "
        "\"eligible\" in R6 and the red flags depend on them; four were added in Phase 12 (SPEC_DEVIATIONS 12.1).", "configs/default.yaml; SPEC_DEVIATIONS 12.1")
    add("One dataset", "All results come from one dataset (the MQTT-IoT-IDS DoS/DDoS capture); nothing here shows that they carry over to other traffic or other networks.", "spec Definition of Done")
    b0 = (why.get("b0") or {})
    hard = (f"the real-data-only classifiers reach macro-F1 {b0['rf']:.3f} (RF) and {b0['mlp']:.3f} (MLP)" if "rf" in b0 and "mlp" in b0
            else "the real-data-only classifiers stay far below 1 (B0 about 0.45-0.50)")
    add("Attack classes are hard to tell apart packet by packet",
        f"{hard[0].upper() + hard[1:]}. A generator of independent packets cannot create separability that the features do not contain.", "g3_baseline.md; section 9.1")
    bs = block_split_shares(man)
    if bs and bs["share"]:
        text = (f"{bs['n_block']} of the {bs['n_sub']} sub-classes come from a single capture file, so their validation and test rows are consecutive blocks of the same capture, not other "
                f"captures. Dropping the TCP streams that straddle two blocks removes {pct(bs['min'])} to {pct(bs['max'])} of the eligible rows of each such sub-class (a bias toward short connections).")
    else:
        text = ("Most sub-classes come from a single capture file, so their validation and test rows are consecutive blocks of the same capture. Dropping the TCP streams that straddle "
                "two blocks removes a share of the eligible rows (a bias toward short connections); the split manifest was not found, so the shares are not shown.")
    add("Block splits inside one capture", text, "results/manifests/split_manifest_<mode>.json; SPEC_DEVIATIONS 3.3, 3.5")
    add("Protocol filter",
        "Only rows with protocol TCP or MQTT are kept. The filter removes about 0.5 % of the Normal rows and the 128 RIPv2 rows of the attack data.", "SPEC_DEVIATIONS 2.9, 3.5")
    pol = (sch or {}).get("settings", {}).get("multi_policy", "first_only")
    add("Multi-valued cells",
        f"Only the first element of a multi-valued cell is kept (policy `{pol}`). The information of several MQTT messages in one packet is lost; part of it remains in `tcp_segment_len`.",
        "SPEC_DEVIATIONS 2.4, 4.3")
    tc = time_delta_clip(sch)
    if tc and tc["share"] is not None:
        sc = f" after the x{tc['scale']:g} scaling" if tc.get("scale") else ""
        add("Tail of the time-gap feature",
            f"The tail of `{tc['column']}` is still clipped at {tc['sigma']:g} sigma: {tc['rows']} of {tc['n_rows']:,} train rows ({pct(tc['share'], 2)}){sc}; without the scaling it was 1.04 %. "
            "Extreme gaps are therefore merged in the features and in the generated rows.", "feature_schema.json audit; SPEC_DEVIATIONS 4.10")
    else:
        add("Tail of the time-gap feature",
            "The tail of the time-gap feature is still clipped at 5 sigma (a small share of the train rows); extreme gaps are merged in the features and in the generated rows.",
            "feature_schema.json audit; SPEC_DEVIATIONS 4.10")
    add("Very sparse columns",
        "Columns that are empty in almost every row (but not all) use the statistics of the rows where they apply, with an `_is_na` flag (`ultra_sparse_fix`); this departs from the rule of spec v1.1.",
        "SPEC_DEVIATIONS 4.5")
    add("DoS and DDoS are merged",
        "In the 6-class mode used for every experiment the DoS / DDoS sub-classes of an attack family are one class, so neither the classifiers nor the generator separate them (the 11-class "
        "data were prepared, but no run of the experiment matrix uses them).", "spec Definition of Done; leakage_report.md C5; configs/default.yaml g2_log")
    pc = ((interp or {}).get("R4") or {}).get("positive_control") or {}
    c2st = [v for v in (why.get("c2st") or {}).values() if isinstance(v, (int, float))]
    c2st_txt = f"C2ST AUC is {min(c2st):.4f}-{max(c2st):.4f}" if c2st else "C2ST is close to 1"
    mia = (f" The attack did not detect an over-fitted CVAE (AUC {pc['overfit_cvae_auc']:.3f}), so the positive control the spec asks for is not met for the CVAE: an AUC near 0.5 does not show privacy."
           if pc.get("available") and not pc.get("overfit_cvae_detected") else " The membership-inference attack is weak, so an AUC near 0.5 does not show privacy by itself.")
    add("Weak fidelity and privacy diagnostics",
        f"{c2st_txt} for the generators (a classifier tells synthetic rows from real ones almost perfectly).{mia}", "b2_cvae.md; section 4 and 9.4")
    add("Seed-to-seed spread",
        "Seed-to-seed spread includes the sensitivity of FL training to tiny perturbations: two runs that differ only by noise of 1e-5 differ by about 0.03 macro-F1 in TSTR. "
        "A standard deviation over three seeds is a rough estimate.", "SPEC_DEVIATIONS 10.4")
    add("Intervals and verdicts",
        "The intervals and verdicts of section 9 reflect the sampling of test rows (or of whole streams) and three training seeds only. They do not cover the choice of capture groups, "
        "the hyper-parameters or the data sampling.", "section 9")
    add("What was not compared",
        "The CVAE is the premise of the spec, not the result of a comparison of generators. Not tested: federated training of the classifier itself, federated class weights or SMOTE, other "
        "generators, other ways to use the synthetic data (see section 9.9).", "section 9.9; SPEC_DEVIATIONS 12.10")
    return out


def render_md(items: list[dict[str, str]], with_source: bool = True) -> list[str]:
    return [f"- **{i['topic']}.** {i['text']}" + (f" *({i['source']})*" if with_source and i.get("source") else "") for i in items]
