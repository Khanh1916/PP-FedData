"""PP-FedData demo (Streamlit): data overview, results, trade-offs and recommendation, sample generation, threat model and limitations.

Run from the repository root:   python -m ppfeddata.cli demo        (or: streamlit run demo/app.py -- --config configs/default.yaml)

Everything shown is read from `results/` and `artifacts/`, which the experiments wrote; the only computation here is sampling from the trained generators,
FedDP-Marginal (the framework's generator) or a CVAE (the spec's baseline), through `ppfeddata.demo_lib.sample`, the code path of the evaluation.
The logic lives in `src/ppfeddata/demo_lib.py` and `limitations.py`, where it is tested. The interface is in English, like the report it quotes (SPEC_DEVIATIONS R.16).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                                                  # the paths of the config are relative to the repository root
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                             # noqa: E402
import streamlit as st                                          # noqa: E402

from ppfeddata import demo_lib as dl                            # noqa: E402
from ppfeddata import limitations                               # noqa: E402
from ppfeddata.utils import load_config                         # noqa: E402

PAGES = ["1. Data", "2. Results", "3. Trade-offs and recommendation", "4. Generate samples", "5. Threat model and limitations"]
VARIANT_LABELS = {"residual": "Decoder + per-class residual noise (the variant of B2, B3, M2 in the report)", "plain": "Plain decoder"}
FIG_TABS = [("Macro-F1 by configuration", "f1_by_config.png", "Macro-F1 on the real test set (error bars = standard deviation across seeds)."),
            ("Rare-class recall", "recall_rare_classes.png", "Recall of the rare classes (DELAYED, SYN, INVALID, WILL), RF classifier."),
            ("Utility - privacy", "utility_privacy.png", "Macro-F1, MIA AUC and DCR against ε for FedDP-Marginal (MG, MGl) and the CVAE (M1, M3). "
             "ε = ∞ is each generator's release without DP: MGs-nodp for FedDP-Marginal, B3-plain for the CVAE."),
            ("Fidelity", "fidelity_vs_eps.png", "Wasserstein distance and C2ST of the synthetic data against ε, FedDP-Marginal and the CVAE."),
            ("Cost", "overhead.png", "Time and bytes per FL round of the CVAE route. FedDP-Marginal trains no model over rounds; it sends three releases (sections 9.9b, 9.12)."),
            ("Pareto", "pareto.png", "Macro-F1 against MIA AUC; marker size follows the overhead. FedDP-Marginal is a square of fixed size (no training rounds).")]


# --------------------------------------------------------------------------------------------------
# Cached loaders (keyed on the file modification time so a new `ppfeddata aggregate` shows up without a restart)
# --------------------------------------------------------------------------------------------------
def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--config", default=None)
    return ap.parse_known_args(sys.argv[1:])[0]


@st.cache_resource(show_spinner=False)
def get_cfg(path: str | None) -> dict:
    return load_config(path)


def _stamp(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def get_summary(path: str | None, stamp: float):
    return dl.load_summary(get_cfg(path))


@st.cache_data(show_spinner=False)
def get_interpretation(path: str | None, stamp: float):
    return dl.load_interpretation(get_cfg(path))


@st.cache_data(show_spinner=False)
def get_report(path: str | None, stamp: float):
    return dl.read_report(get_cfg(path))


@st.cache_resource(show_spinner=False)
def get_assets(path: str | None):
    return dl.load_assets(get_cfg(path))


@st.cache_resource(show_spinner=False)
def get_model(path: str | None, label: str, seed: int):
    cfg = get_cfg(path)
    g = next(x for x in dl.generators(cfg) if x.label == label)
    return dl.load_model(cfg, g, seed, get_assets(path).schema)


def show_md(md: str, base: Path) -> None:
    """Markdown with the image links of the report turned into st.image (the report links the figures relatively)."""
    for kind, val, alt in dl.md_parts(md, base):
        if kind == "md":
            st.markdown(val)
        elif Path(val).exists():
            st.image(str(val), caption=alt or None, width="stretch")


def protections_text(cfg: dict, g: dl.Generator) -> str:
    fl, dp = cfg.get("fl", {}), cfg.get("dp", {})
    if g.kind == "mg":
        return (f"federated ({fl.get('num_clients', '?')} non-IID clients); each client sends its count tables with its share of Skellam noise added, and the server only "
                f"receives the sum through SecAgg+ (distributed DP, target ε {g.eps_target:g}, holds while the clients are honest)"
                + ("; plus post-processing chosen on validation, at no extra privacy cost" if g.run_name.startswith("MGr") else ""))
    parts = [f"federated training (FedAvg, {fl.get('num_clients', '?')} non-IID clients)" if g.federated else "centralised training (no FL)"]
    if g.dp:
        parts.append(f"client-side DP-SGD (target ε {g.eps_target:g}, δ = {dp.get('delta', '?'):g})")
    if g.secagg:
        parts.append("secure aggregation SecAgg+")
    return ", ".join(parts)


def fmt_ms(x) -> str:
    return "-" if not x else (f"{x[0]:.3f} ± {x[1]:.3f}" if x[1] is not None else f"{x[0]:.3f}")


# --------------------------------------------------------------------------------------------------
# Page 1: data
# --------------------------------------------------------------------------------------------------
def page_data(cfg: dict) -> None:
    st.header("Data overview")
    ov = dl.data_overview(cfg)
    if not ov["manifest"]:
        st.warning("No `results/manifests/split_manifest_<mode>.json` found. Run `python -m ppfeddata.cli sample` (Phase 3) to create it.")
        return
    st.markdown(
        "Each row is **one network packet** of an IoT system that uses the MQTT protocol, with the features extracted from it. "
        f"Labels in mode **{cfg['label_mode']}**: normal traffic (NORMAL) and the DoS/DDoS attack types (BCF - Basic Connect Flooding, DELAYED, SYN, INVALID, WILL). "
        "The data were sampled with a quota per class and split by **capture group**: no group is in two splits (see `leakage_report.md`).")
    rows = ov["rows"] or {}
    c = st.columns(4)
    for col, split in zip(c[:3], ("train", "val", "test")):
        col.metric(f"{split} rows", f"{rows.get(split, 0):,}")
    if ov["features"]:
        c[3].metric("Features after encoding", ov["features"]["n_features"])
    cls = ov["classes"]
    st.subheader("Class distribution")
    st.dataframe(cls, width="stretch")
    st.bar_chart(cls["train"], y_label="train rows", x_label="class", sort=False)
    ratio = dl.imbalance_ratio(cls)
    notes = []
    if ratio:
        notes.append(f"In the train split the largest class ({ratio[0]}) is **{ratio[2]:.0f} times** the smallest ({ratio[1]}): this is the imbalance the synthetic data are meant to offset.")
    if cls["val"].nunique() == 1 and cls["test"].nunique() == 1:
        notes.append(f"Val and test take {int(cls['val'].iloc[0]):,} and {int(cls['test'].iloc[0]):,} rows of every class, so they do not reflect the real proportions of the traffic.")
    for n in notes:
        st.markdown("- " + n)
    with st.expander("Sub-classes (DoS / DDoS) and how train/val/test were split"):
        sub = ov["subclasses"].copy()
        sub["rows dropped (streams across blocks)"] = sub["rows dropped (streams across blocks)"].map(lambda v: "-" if pd.isna(v) else f"{100 * v:.1f} %")
        st.dataframe(sub, hide_index=True, width="stretch")
        st.caption("`group`: split by capture file. `block`: the sub-class has a single capture file, split into consecutive blocks; TCP streams that cross blocks are dropped (last column).")
    if ov["features"]:
        with st.expander("Features"):
            names = {"numeric": "numeric", "binary": "binary", "na_flag": "empty flag (_is_na)", "categorical": "categorical (one-hot)"}
            for t, cols in ov["features"]["blocks"].items():
                st.markdown(f"**{names.get(t, t)}** ({len(cols)}): " + ", ".join(f"`{x}`" for x in cols))
    f1, f2 = dl.figure(cfg, "partition_alpha0.5_seed0.png"), dl.figure(cfg, "presence_heatmap.png")
    if f1:
        st.image(str(f1), caption=f"Train split given to the clients by a Dirichlet draw (figure `{f1.name}`): each client holds very different class proportions (non-IID).", width="stretch")
    if f2:
        st.image(str(f2), caption="Share of non-empty cells of each column per sub-class (Phase 2 audit).", width="stretch")


# --------------------------------------------------------------------------------------------------
# Page 2: results
# --------------------------------------------------------------------------------------------------
def page_results(cfg: dict, path: str | None) -> None:
    st.header("Results of the experiments")
    summ = get_summary(path, _stamp(dl.results_dir(cfg) / "summary.csv"))
    if summ is None:
        st.warning("No `results/summary.csv` yet. Run `python -m ppfeddata.cli aggregate`.")
        return
    seeds = sorted({s for v in summ["seeds"].astype(str) for s in v.split(",")})
    st.caption(f"{len(summ)} configurations in the run ledger, seeds {', '.join(seeds)}. Macro-F1 is measured on the **real test set**: mean ± standard deviation across seeds (`results/summary.csv`).")
    c1, c2, c3 = st.columns(3)
    only = c1.toggle("Only the configurations of the spec matrix", value=True)
    clf = c2.radio("Classifier", ["All", "rf", "mlp"], horizontal=True)
    proto = c3.radio("Evaluation protocol", ["All", "TRTR", "TSTR", "TAug"], horizontal=True,
                     help="TRTR: train and test on real data (B0, B1). TSTR: train on synthetic data only. TAug: real data + synthetic data.")
    d = summ
    if only:
        d = d[d["in_matrix"]]
    if clf != "All":
        d = d[d["classifier"] == clf]
    if proto != "All":
        d = d[d["protocol"] == proto]
    cols = {"config": "configuration", "protocol": "protocol", "classifier": "classifier", "n_seeds": "seeds", "macro_f1_mean": "macro-F1", "macro_f1_std": "± std",
            "balanced_acc_mean": "balanced acc", "c2st_auc_mean_mean": "C2ST AUC", "mia_auc_mean_mean": "MIA AUC", "dp_eps_max_mean": "ε reached (max)", "round_s_median_mean": "s per FL round"}
    show = d[[k for k in cols if k in d.columns]].rename(columns=cols)
    st.dataframe(show.style.format(na_rep="-", precision=4), hide_index=True, width="stretch")
    st.download_button("Download summary.csv", summ.to_csv(index=False).encode("utf-8"), file_name="summary.csv", mime="text/csv")
    st.subheader("Figures")
    tabs = st.tabs([t for t, _, _ in FIG_TABS])
    for tab, (_, fname, cap) in zip(tabs, FIG_TABS):
        with tab:
            p = dl.figure(cfg, fname)
            if p:
                st.image(str(p), caption=cap, width="stretch")
            else:
                st.info(f"No `results/figures/{fname}` yet.")
    md = get_report(path, _stamp(dl.report_path(cfg)))
    if md:
        with st.expander("Final report (generated): tables 2a-8c"):
            secs = dl.split_markdown(md, 2)
            for k, v in secs.items():
                if k[:2] in ("1.", "2a", "2b", "3.", "4.", "5.", "6.", "7.", "8.", "8b", "8c"):
                    st.markdown(f"#### {k}")
                    show_md(v, dl.report_path(cfg).parent)


# --------------------------------------------------------------------------------------------------
# Page 3: trade-offs and recommendation
# --------------------------------------------------------------------------------------------------
def page_tradeoff(cfg: dict, path: str | None) -> None:
    st.header("Trade-offs (utility - privacy - overhead) and recommendation")
    R = get_interpretation(path, _stamp(dl.results_dir(cfg) / "interpretation.json"))
    if R is None:
        st.warning("No `results/interpretation.json` yet. Run `python -m ppfeddata.cli aggregate` (Phase 12).")
        return
    md = get_report(path, _stamp(dl.report_path(cfg)))
    secs = dl.split_markdown(md, 3) if md else {}
    base = dl.report_path(cfg).parent
    why_mg = dl.why_mg_note(R, cfg)
    if why_mg:
        st.success("**The framework uses FedDP-Marginal**: a federated data generator built on DP count tables, with the noise split over the clients and summed through SecAgg+. "
                   "The spec's CVAE (M1, M2, M3) is the baseline it is compared with. The text below is generated from section 9.0 of the report:\n\n" + why_mg)
    page_requirement(cfg)
    guide = next((k for k in secs if k.split()[0] == "9.10"), None)
    mg_part, cvae_part = dl.split_guide(secs[guide]) if guide else ("", "")
    if mg_part:
        st.subheader("FedDP-Marginal per deployment requirement")
        st.caption("Section 9.10b of `final_report.md`, read from `results/recommend.json`.")
        show_md(mg_part, base)
    c = st.columns(2)
    for col, name, cap in ((c[0], "utility_privacy.png", "Utility - privacy against ε, FedDP-Marginal and the CVAE."),
                           (c[1], "overhead.png", "Overhead per round of the CVAE route (FedDP-Marginal only sends three releases).")):
        p = dl.figure(cfg, name)
        if p:
            col.image(str(p), caption=cap, width="stretch")
    p = dl.figure(cfg, "pareto.png")
    if p:
        st.image(str(p), caption="Pareto: macro-F1 against MIA AUC, marker size by overhead; FedDP-Marginal is a square of fixed size.", width="stretch")

    st.subheader("The spec's original design: rule R6 and the CVAE route")
    st.caption("Kept for comparison with the spec. Rule R6 ranks by TAug macro-F1, a number that barely separates the configurations; the framework does not choose a configuration by this rule.")
    rec = dl.recommendation(R)
    if rec:
        with st.expander("The spec's rule R6 and the configuration it picks"):
            rule = rec["rule"]
            st.markdown(f"Rule (thresholds under `thresholds` in the config): MIA AUC ≤ {rule.get('mia_auc_max')}, ε ≤ {rule.get('eps_max')} with DP, time per round ≤ "
                        f"{rule.get('overhead_ratio_max')} × plain FL (B3); among the configurations that pass, pick the highest TAug (RF) macro-F1.")
            if rec["recommended"]:
                extra = f" Read literally, the rule picks **{rec['literal']}**; configurations tied within the noise: {', '.join(rec['tied'])}; the tie is broken by the level of protection." if rec["tie_break_used"] else ""
                st.info(f"Rule R6 picks: **{rec['recommended']}**." + extra)
            else:
                st.info("No configuration meets all three conditions; see the table and the Pareto figure.")
            if rec["tstr_recommended"]:
                st.markdown(f"Ranked by TSTR (synthetic data in place of real data), the same rule picks **{rec['tstr_recommended']}**.")
            if rec["dp_alternative"]:
                st.markdown(f"If a formal DP guarantee is required, the DP option the rule would pick without the overhead condition is **{rec['dp_alternative']}**.")
            st.dataframe(rec["table"], hide_index=True, width="stretch")
            st.caption("R6 ranks the generators (CVAE and FedDP-Marginal) against each other; it does not say that synthetic data beat real data when the real data can be pooled (sections 9.6, 9.9 of the report).")
    if cvae_part:
        with st.expander("The spec's CVAE route: which configuration for which requirement (M1, M2, M3)"):
            st.caption("Section 9.10 of `final_report.md`, the part before 9.10b.")
            show_md(cvae_part, base)
    note = dl.why_cvae_note(R, cfg)
    if note:
        with st.expander("Why the spec chose the CVAE, and what R1, R2 say about that choice"):
            st.markdown(note)
    st.subheader("Sections of the report")
    for key in [k for k in secs if k.split()[0] in ("9.0", "9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7", "9.8", "9.9", "9.9b")]:
        with st.expander(key, expanded=key.split()[0] == "9.9b"):
            show_md(secs[key], base)


def page_requirement(cfg: dict) -> None:
    """Optimisation round O4: the configuration for a deployment requirement, with the same rules as `ppfeddata recommend` (results/scorecard.json)."""
    import json

    from ppfeddata import recommend as rc
    from ppfeddata import scorecard as scm

    p = dl.results_dir(cfg) / "scorecard.json"
    if not p.exists():
        return
    st.subheader("Configuration for a deployment requirement (`ppfeddata recommend`)")
    card = scm._from_json(json.loads(p.read_text(encoding="utf-8")))
    names = [s.name for s in rc.SCENARIOS]
    pick = st.selectbox("Example scenario", ["(custom)"] + names, key="rq_scn")
    base = rc.SCENARIOS[names.index(pick)] if pick in names else rc.Requirement()
    c = st.columns(4)
    ts = c[0].selectbox("Trust the server?", ["yes", "no"], index=0 if base.trust_server else 1, key=f"rq_ts_{pick}") == "yes"
    hon = c[1].selectbox("Minimum honest clients", ["all", "4", "3", "2", "1"],
                         index=0 if (base.trust_clients and not base.honest_clients) else (4 if not base.trust_clients else ["all", "4", "3", "2", "1"].index(str(base.honest_clients))),
                         key=f"rq_h_{pick}")
    eps = c[2].selectbox("Maximum ε", ["no DP needed", "1", "5", "10"], index=["no DP needed", "1", "5", "10"].index(f"{base.max_eps:g}" if base.max_eps else "no DP needed"),
                         key=f"rq_e_{pick}")
    pri = c[3].selectbox("Priority", ["macro", "binary", "rare"], index=["macro", "binary", "rare"].index(base.priority), key=f"rq_p_{pick}")
    c2 = st.columns(3)
    mbt = c2[0].number_input("Maximum total MB (0 = no limit)", min_value=0.0, value=float(base.max_mb_total or 0.0), key=f"rq_mb_{pick}")
    mia = c2[1].number_input("Maximum MIA AUC (0 = no limit)", min_value=0.0, max_value=1.0, value=float(base.max_mia or 0.0), key=f"rq_mia_{pick}")
    rr = c2[2].number_input("Minimum rare-class recall", min_value=0.0, max_value=1.0, value=float(base.min_rare_recall or 0.0), key=f"rq_rr_{pick}")
    req = rc.Requirement("demo", ts, hon != "1", None if eps == "no DP needed" else float(eps), None, mbt or None, None, pri, rr or None,
                         honest_clients=None if hon in ("all", "1") else int(hon), max_mia=mia or None)
    res = rc.recommend(card, req)
    if res["best"]:
        b = res["best"]
        st.success(f"Recommendation: **{b['label']}** · {res['priority_metric']} {b.get(res['priority_metric']):.3f} · ε under the trust assumption {b['eps_eff']:.2f}"
                   + (f" · tied within the seed spread: {', '.join(res['ties_with_best'])}" if res["ties_with_best"] else "") + f"  \nCommand: `{res['command']}`")
    else:
        st.info("No valid configuration meets the requirement.")
    st.dataframe([{k: r.get(k) for k in rc.SLIM_KEYS} for r in res["front"]], hide_index=True, width="stretch")
    if res["rejected"]:
        with st.expander(f"Excluded ({len(res['rejected'])})"):
            st.dataframe([{"configuration": r["label"], "reason": "; ".join(r["rejected_because"])} for r in res["rejected"]], hide_index=True, width="stretch")
    st.caption("Pareto front on the deployment axes: TSTR macro-F1, binary F1, rare-class recall (equal within the seed spread), ε under the trust assumption, SecAgg, MB per round, MB in total. "
               "Same rules as `python -m ppfeddata.cli recommend` and `results/reports/recommend.html`.")


# --------------------------------------------------------------------------------------------------
# Page 4: sample generation
# --------------------------------------------------------------------------------------------------
def page_generate(cfg: dict, path: str | None) -> None:
    st.header("Generate samples from a trained generator (FedDP-Marginal or CVAE)")
    avail = dl.available_generators(cfg)
    if not avail:
        st.warning("No trained model found in `artifacts/` (or the processed data in `data/processed/` are missing). Run Phases 7-10 first (see the README).")
        return
    try:
        assets = get_assets(path)
    except Exception as e:                                                     # missing processed data / preprocessor
        st.error(f"Could not load the processed data: {e}")
        return
    by = {g.label: (g, seeds) for g, seeds in dl.framework_first(avail)}
    c1, c2 = st.columns(2)
    label = c1.selectbox("Configuration", list(by), format_func=lambda k: f"{k} - {by[k][0].title}" + (" (framework)" if by[k][0].kind == "mg" else " (CVAE, baseline)"))
    g, seeds = by[label]
    if g.kind == "mg":
        st.markdown("Choose a configuration, a class and a number of rows; the table holds **synthetic packets** sampled from the DP-noised count tables that FedDP-Marginal "
                    "released (Chow-Liu tree), converted back to the original units (empty cell = feature does not apply). The same configuration, seed, class and number always give the same table.")
    else:
        st.markdown("Choose a configuration, a class and a number of rows; the table holds **synthetic packets** generated by the CVAE decoder (the spec's baseline), converted back to the "
                    "original units (empty cell = feature does not apply). The same configuration, seed, class and number always give the same table.")
    train_seed = c2.selectbox("Training seed of the model", seeds)
    st.caption(f"Training and protection: {protections_text(cfg, g)}.")
    c3, c4, c5 = st.columns(3)
    cls_opt = c3.selectbox("Class", ["All classes"] + assets.classes)
    n = c4.number_input("Rows per class", min_value=1, max_value=dl.MAX_PER_CLASS, value=100, step=50)
    sample_seed = c5.number_input("Sampling seed", min_value=0, value=0, step=1)
    if g.kind == "mg":
        variant = "plain"
        st.info("FedDP-Marginal has no decoder: rows are sampled from the count tables released under DP, so every row is post-processing and stays within ε.")
    elif g.dp:
        variant = "plain"
        st.info("DP configurations of the CVAE generate with the **plain decoder** only: the per-class residual noise is estimated on the whole train split, so it lies outside ε. "
                "The rows are post-processing of the model trained with DP.")
    else:
        variant = st.radio("Generation", ["residual", "plain"], horizontal=True, format_func=lambda v: VARIANT_LABELS[v],
                           help="Residual noise: Gaussian noise with the standard deviation of each class's residuals (estimated on train) is added to the decoder's mean output; "
                                "this is the variant used for TAug/TSTR of B2, B3, M2.")
    form = st.radio("Table to download", ["Original units", f"Encoded features ({assets.schema['n_features']} columns)"], horizontal=True)
    if st.button("Generate samples", type="primary"):
        try:
            with st.spinner("Loading the model and generating samples..."):
                st.session_state["samples"] = dl.sample(cfg, g, int(train_seed), None if cls_opt == "All classes" else [cls_opt], int(n), int(sample_seed), variant,
                                                         assets=assets, model=get_model(path, g.label, int(train_seed)))
        except Exception as e:
            st.session_state.pop("samples", None)
            st.error(f"Could not generate samples: {e}")
    s = st.session_state.get("samples")
    if s is not None:
        st.subheader(f"{s.generator.label}, training seed {s.train_seed}, sampling seed {s.sample_seed}: {len(s.table):,} rows ({', '.join(s.classes)})")
        for note in s.notes:
            st.caption(note)
        st.dataframe(s.table.head(500), hide_index=True, width="stretch")
        if len(s.table) > 500:
            st.caption(f"Showing the first 500 rows; the CSV file has all {len(s.table):,} rows.")
        enc = form.startswith("Encoded")
        st.download_button("Download CSV", lambda: s.csv(encoded=enc), file_name=f"{s.generator.label}_seed{s.train_seed}_{'-'.join(s.classes) if len(s.classes) < 6 else 'all'}_n{s.n_per_class}.csv", mime="text/csv")
    st.subheader("Measured quality of this configuration")
    q = dl.quality(get_summary(path, _stamp(dl.results_dir(cfg) / "summary.csv")), g, "plain" if g.dp else variant)
    if q is None:
        st.info("No measurements of this configuration in `results/summary.csv` yet.")
    else:
        m = st.columns(4)
        m[0].metric("TSTR macro-F1 (RF)", fmt_ms(q["tstr_rf"]), help="Classifier trained on synthetic data only, tested on real data.")
        m[1].metric("Real data only (B0, RF)", fmt_ms(q["real_only_rf"]))
        m[2].metric("C2ST AUC", "-" if q["c2st_auc"] is None else f"{q['c2st_auc']:.4f}", help="Close to 1: a classifier tells the synthetic data from the real data almost perfectly.")
        m[3].metric("MIA AUC", "-" if q["mia_auc"] is None else f"{q['mia_auc']:.3f}",
                    help=("Membership inference attack on the synthetic data; about 0.5 is chance. For FedDP-Marginal, the attack on the released tables (section 9.12) passes its "
                          "positive control and separates the release without DP from the releases with DP." if g.kind == "mg"
                          else "Membership inference attack on the synthetic data; about 0.5 is chance. This attack is weak and has not passed a positive control with the CVAE."))
        st.caption(f"Measurements of `{q['config']}` (mean ± std over seeds). " + (f"ε reached (max over clients, δ = {cfg['dp']['delta']:g}): {q['eps_max']:.3f}." if q["eps_max"] else "No DP: no ε guarantee."))
    if not g.dp:
        st.warning("This configuration has no formal privacy guarantee. The synthetic rows may carry information about the training data; the empirical measurements (MIA, match rate) "
                   "are weak evidence only.")
    st.caption("The samples are for illustration only. Compare TSTR with \"real data only\" above to see how far the synthetic data can replace the real data; the conclusions about the IDS are on page 3.")


# --------------------------------------------------------------------------------------------------
# Page 5: threat model and limitations
# --------------------------------------------------------------------------------------------------
def page_threat(cfg: dict, path: str | None) -> None:
    st.header("Threat model and limitations")
    sa = cfg.get("secagg", {})
    st.markdown(
        "**Threat model (spec).** The FL server is *honest-but-curious*: it runs the protocol correctly but tries to infer what it can from what it receives. "
        "The clients are honest and do **not collude** with each other or with the server. The simulation runs on one machine, with no real network.")
    st.markdown("**FedDP-Marginal (the framework's generator).**")
    st.markdown("| Mechanism | Protects against | Does not protect against |\n|---|---|---|\n"
                "| MG: distributed DP on count tables (each client adds its share of Skellam noise, the server only receives the sum through SecAgg+) | The server sees no client's tables, "
                "only the noisy sum; the synthetic data are sampled from the released tables, so they stay within ε at record level | ε holds only while the clients are honest (ε with fewer "
                "honest clients in section 9.10b); class labels; a whole session or capture; hyper-parameter tuning |\n"
                "| MGl: local DP (each client adds the full noise itself) | No need to trust the other clients | The noise adds up K times, so utility is lower; the server sees each client's noisy tables |")
    st.markdown("**The spec's CVAE route (baseline).**")
    st.markdown("| Mechanism | Protects against | Does not protect against |\n|---|---|---|\n"
                "| DP-SGD at the client | Leakage about each training **record** through the global model and the synthetic data drawn from it (post-processing). ε at record level | Class labels; "
                "a whole attack session of many correlated packets; normalisation statistics computed centrally; hyper-parameter tuning |\n"
                f"| SecAgg (Flower's SecAgg+; {sa.get('num_shares', '?')} shares, reconstruction threshold {sa.get('reconstruction_threshold', '?')}) | The server seeing **each client's update**: "
                "it only receives the masked sum | The final aggregated model and the synthetic data drawn from it can still leak; no ε guarantee |\n"
                "| DP + SecAgg | Both: the server does not see the clients' updates, and the model and synthetic data have an ε guarantee | The limits of DP above |")
    st.caption("Per the spec: SecAgg protects each client's update from the server, and DP protects against leakage from the model and the synthetic data. Section 9.8 of the report (below) describes both generators.")
    R = get_interpretation(path, _stamp(dl.results_dir(cfg) / "interpretation.json"))
    md = get_report(path, _stamp(dl.report_path(cfg)))
    if md:
        secs = dl.split_markdown(md, 3)
        k = next((x for x in secs if x.startswith("9.8")), None)
        if k:
            with st.expander(k + " (report)"):
                st.markdown(secs[k])
    st.subheader("Limitations that apply to every number")
    summ = get_summary(path, _stamp(dl.results_dir(cfg) / "summary.csv"))
    items = limitations.limitations(cfg, R, summ)
    st.caption(f"{len(items)} items, generated by `limitations.py` (the same list as section 10 of the report and the README). Numbers come from the manifest, schema, config and interpretation.json.")
    for it in items:
        st.markdown(f"**{it['id']} {it['topic']}.** {it['text']}  \n*Source: {it['source']}*")
    md = get_report(path, _stamp(dl.report_path(cfg)))
    sec = next((v for k, v in (dl.split_markdown(md, 3) if md else {}).items() if k.split()[0] == "9.9b"), None)
    if sec:
        st.subheader("Why did the framework move from the CVAE to FedDP-Marginal?")
        st.caption("Section 9.9b of the report, generated from `results/scorecard.json`.")
        show_md(sec, dl.report_path(cfg).parent)
    note = dl.why_cvae_note(R, cfg)
    if note:
        st.subheader("Why did the spec first use the CVAE (now the baseline)?")
        st.markdown(note + "\n\nDetails: section 9.9 of the report (page 3).")


# --------------------------------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="PP-FedData demo", page_icon=":material/shield:", layout="wide")
    path = _args().config
    cfg = get_cfg(path)
    st.title("PP-FedData")
    st.caption("Privacy-preserving federated data generation to support the detection of MQTT DoS/DDoS attacks on IoT. The demo runs locally; numbers are read from `results/` and `artifacts/`.")
    page = st.sidebar.radio("Page", PAGES, key="page")
    R = get_interpretation(path, _stamp(dl.results_dir(cfg) / "interpretation.json"))
    pm = dl.premise(R)
    mg = bool(dl.why_mg_note(R, cfg))
    head = ("**Main conclusion (page 3):** the framework uses FedDP-Marginal, a federated generator built on DP count tables; the spec's CVAE is the baseline. "
            if mg else "**Main conclusion (page 3):** ")
    if pm and pm["unsupported"]:
        st.sidebar.markdown(head + "By R1 and R2, synthetic data do not improve the IDS when the real data can be pooled; the value of the framework is training an IDS when the raw data must stay at the clients.")
    elif pm and all(v["best"] < (pm["smote"].get(clf) or float("inf")) for clf, v in pm["per_clf"].items()):
        st.sidebar.markdown(head + "When the real data can be pooled, synthetic data help the IDS less than SMOTE (R1, R2); the value of the framework is training an IDS when the raw data must stay at the clients.")
    elif pm:
        st.sidebar.markdown(head + "See R1, R2 and section 9.9b on page 3 for how far synthetic data help the IDS.")
    {PAGES[0]: lambda: page_data(cfg), PAGES[1]: lambda: page_results(cfg, path), PAGES[2]: lambda: page_tradeoff(cfg, path),
     PAGES[3]: lambda: page_generate(cfg, path), PAGES[4]: lambda: page_threat(cfg, path)}[page]()


main()
