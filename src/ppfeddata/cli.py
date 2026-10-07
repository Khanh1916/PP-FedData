"""Unified CLI entry point for PP-FedData."""
from __future__ import annotations

import argparse
import sys

from ppfeddata.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="ppfeddata",
        description="PP-FedData: Privacy-Preserving Federated Data Generation for IoT IDS",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Logging level"
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # Phase 1
    sub.add_parser("inventory", help="Scan and inventory raw data files")
    # Phase 2
    p_h = sub.add_parser("harmonize", help="Harmonize schema and audit formats")
    p_h.add_argument("--recompute", action="store_true",
                     help="Redo all streaming statistics instead of using cached CSVs")
    p_h.add_argument("--decisions-only", action="store_true",
                     help="Skip streaming; rebuild feature_decisions.yaml and heatmap from cached CSVs")
    # Phase 3
    p_s = sub.add_parser("sample", help="Group-split and sample data")
    p_s.add_argument("--label-mode", choices=["6class", "11class"], default=None,
                     help="Override label_mode from the config (output goes to data/interim/<mode>/)")
    p_s.add_argument("--no-cache", action="store_true", help="Ignore the pass-A scan cache")
    # Phase 4
    p_p = sub.add_parser("preprocess", help="Parse multi-value cells and preprocess features")
    p_p.add_argument("--label-mode", choices=["6class", "11class"], default=None,
                     help="Override label_mode from the config (reads data/interim/<mode>/)")
    # Phase 5
    p_c = sub.add_parser("check", help="Leakage and label reliability checks")
    p_c.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    # Phase 6
    p_b = sub.add_parser("baseline", help="Run baseline evaluations (B0, B1a, B1b) and write the G3 report")
    p_b.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_b.add_argument("--configs", nargs="+", default=None, help="subset of B0-rf B0-mlp B1a-rf B1b-rf B1b-mlp")
    p_b.add_argument("--seeds", nargs="+", type=int, default=None)
    p_b.add_argument("--no-resume", action="store_true", help="re-run runs that are already in results/runs.csv")
    # Phase 7
    p_t = sub.add_parser("tune", help="Optuna hyperparameter tuning for CVAE (resumes from artifacts/optuna_cvae_*.db)")
    p_t.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_t.add_argument("--n-trials", type=int, default=None, help="target total number of trials (default tune.n_trials)")
    p_b2 = sub.add_parser("b2", help="Centralised CVAE (B2): train, generate, evaluate TSTR/TAug, fidelity, privacy")
    p_b2.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_b2.add_argument("--seeds", nargs="+", type=int, default=None)
    p_b2.add_argument("--no-resume", action="store_true")
    p_bench = sub.add_parser("benchmark", help="7.4 compute benchmark (plain vs DP-SGD epoch time, RAM) -> compute_budget.md")
    p_bench.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    # Phase 8
    p_b3 = sub.add_parser("b3", help="CVAE + FedAvg over non-IID clients (B3): runs, sanity checks and report")
    p_b3.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_b3.add_argument("--seeds", nargs="+", type=int, default=None)
    p_b3.add_argument("--no-resume", action="store_true")
    p_b3.add_argument("--skip-sanity", action="store_true", help="skip the single-client / IID / resume sanity runs")
    # Phase 9
    p_m1 = sub.add_parser("m1", help="CVAE + FedAvg + client-side DP-SGD (M1): runs, sanity checks and report")
    p_m1.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_m1.add_argument("--eps", nargs="+", type=float, default=None, help="target epsilons (default dp.epsilons)")
    p_m1.add_argument("--seeds", nargs="+", type=int, default=None)
    p_m1.add_argument("--no-resume", action="store_true")
    p_m1.add_argument("--skip-sanity", action="store_true")
    p_m1.add_argument("--tuned", action="store_true", help="use the DP-specific hyper-parameters selected in configs/best_cvae_dp.yaml")
    p_td = sub.add_parser("tune-dp", help="Optuna search of the CVAE hyper-parameters under client-side DP-SGD (single-client proxy)")
    p_td.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_td.add_argument("--eps", type=float, default=5.0)
    p_td.add_argument("--n-trials", type=int, default=24, help="target total number of trials")
    p_vd = sub.add_parser("verify-dp", help="Real FL runs (one seed) of the best DP-search trials; selects the winner on validation")
    p_vd.add_argument("--trials", nargs="+", type=int, required=True)
    p_vd.add_argument("--eps", type=float, default=5.0)
    p_vd.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    # Phase 10
    p_m2 = sub.add_parser("m2", help="CVAE + FedAvg + secure aggregation (M2): runs and ledger rows")
    p_m2.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_m2.add_argument("--seeds", nargs="+", type=int, default=None)
    p_m2.add_argument("--no-resume", action="store_true")
    p_m3 = sub.add_parser("m3", help="CVAE + FedAvg + client-side DP + secure aggregation (M3): runs and ledger rows")
    p_m3.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_m3.add_argument("--eps", type=float, default=5.0)
    p_m3.add_argument("--seeds", nargs="+", type=int, default=None)
    p_m3.add_argument("--no-resume", action="store_true")
    p_m3.add_argument("--untuned", action="store_true", help="use the Phase 7 hyper-parameters instead of the DP-specific ones")
    p_sc = sub.add_parser("secagg-check", help="T-SA1/T-SA2/T-SA3 on real data (seed 0): secure vs plain per round, masked vectors, a dropping client")
    p_sc.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_sc.add_argument("--seed", type=int, default=0)
    p_sc.add_argument("--rounds", type=int, default=None)
    p_sc.add_argument("--no-resume", action="store_true")
    p_sr = sub.add_parser("secagg-report", help="Write results/reports/m2_m3_secagg.md from the ledger and the check results")
    p_sr.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_sr.add_argument("--eps", type=float, default=5.0)
    p_sr.add_argument("--untuned", action="store_true")
    # Phase 11
    p_run = sub.add_parser("run", help="Run the experiment matrix of configs/exp/*.yaml. stage trial = one seed in an isolated workspace (Gate G4); "
                                       "stage full = all seeds in the main workspace, needs --approve-g4")
    p_run.add_argument("--stage", choices=["trial", "full"], default="trial")
    p_run.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_run.add_argument("--seeds", nargs="+", type=int, default=None, help="default: config seeds (trial: the first one)")
    p_run.add_argument("--only", nargs="+", default=None, help="configuration ids, e.g. M1-eps5 M3")
    p_run.add_argument("--groups", nargs="+", choices=["matrix", "reference", "extension"], default=["matrix"])
    p_run.add_argument("--workspace", choices=["main", "trial"], default=None, help="default: trial for stage trial, main for stage full")
    p_run.add_argument("--dry-run", action="store_true", help="print the plan only")
    p_run.add_argument("--no-resume", action="store_true")
    p_run.add_argument("--approve-g4", action="store_true", help="the user has read results/reports/g4_trial.md and agrees to the full run")
    p_run.add_argument("--report", action="store_true", help="only rewrite results/reports/g4_trial.md from the saved trial session (nothing is run)")
    p_ag = sub.add_parser("aggregate", help="Summarise the run ledger: results/summary.csv, the six figures and results/reports/final_report.md")
    p_ag.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_ag.add_argument("--runs-csv", default=None, help="ledger to summarise (default compute.runs_csv)")
    p_ag.add_argument("--artifacts-dir", default=None, help="artifacts of that ledger (default compute.artifacts_dir)")
    p_ag.add_argument("--out-dir", default=None, help="write summary.csv, figures/ and reports/ here instead of results/ (e.g. for a preview)")
    p_ag.add_argument("--no-interpret", action="store_true", help="skip the Phase 12 interpretation (R1-R6, red flags, recommendation); it re-reads every prediction file and bootstraps, ~30 s")
    p_ag.add_argument("--n-boot", type=int, default=None, help="bootstrap resamples of the interpretation (default eval.bootstrap)")
    # Phase 13
    p_demo = sub.add_parser("demo", help="Launch the Streamlit demo (demo/app.py); needs results/ and artifacts/ from the experiments")
    p_demo.add_argument("--port", type=int, default=None, help="server port (default: Streamlit's, 8501)")
    p_demo.add_argument("--headless", action="store_true", help="do not open a browser window")
    p_ac = sub.add_parser("accept", help="Check the Definition of Done of the spec against the files of the repository -> results/reports/dod_checklist.md (exit 1 if a row FAILS)")
    p_ac.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_ac.add_argument("--no-regenerate", action="store_true", help="do not re-run `aggregate` into a scratch folder to compare the final report with what the code produces (about 40 s)")
    p_fc = sub.add_parser("fed-baseline", help="Train the IDS classifier itself by FedAvg on the same non-IID clients as B3 (and the pooled controls) -> results/fed_classifier.json; run `aggregate` after it")
    p_fc.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_fc.add_argument("--seeds", nargs="+", type=int, default=None, help="default: config seeds")
    p_fc.add_argument("--only", nargs="+", default=None, help="subset of FedMLP FedMLPcw CentMLP CentMLPcw")
    p_fc.add_argument("--protected", action="store_true", help="instead: tune the class-weighted FedAvg MLP on validation and run it with SecAgg numerics, DP (eps 1, 5, 10) and both -> adds `tuning` and `protected` to results/fed_classifier.json (run after the plain baseline)")
    p_mi = sub.add_parser("mia", help="Membership inference with access to the released model (loss-based, calibrated by a reference model) with positive controls: B3 and M1 on random halves of the train pool -> results/mia_model.json")
    p_mi.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_mi.add_argument("--seeds", nargs="+", type=int, default=None)
    p_mi.add_argument("--configs", nargs="+", default=None, help="subset of B3 M1-eps1 M1-eps5 M1-eps10")
    p_mi.add_argument("--controls-only", action="store_true", help="only the centralised positive controls")
    p_sc = sub.add_parser("scorecard", help="Optimisation round O0: scorecard of B3, M1, M2, M3 on every metric and their Pareto front -> results/scorecard.json, results/reports/scorecard.md (run after `aggregate`)")
    p_sc.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_sc.add_argument("--freeze-baseline", action="store_true", help="also save it as results/scorecard_baseline.json, the reference of the later stages (refused when it exists)")
    p_tf = sub.add_parser("tune-dp-full", help="Optimisation O1: DP search at full scale (real FL runs, seed 0), one Optuna study per epsilon, also over rounds, local epochs, class weights and the DP residual statistics; validation only -> configs/best_cvae_dp_full.yaml")
    p_tf.add_argument("--eps", nargs="+", type=float, default=None, help="default dp.epsilons")
    p_tf.add_argument("--n-trials", type=int, default=30, help="trials per epsilon (resumable)")
    p_tf.add_argument("--final", action="store_true", help="instead: train the best trial of each epsilon with every seed as M1o-t<n>-eps<e> and M3o-... (with SecAgg) into the run ledger; run `aggregate` and `scorecard` after it")
    p_tf.add_argument("--seeds", nargs="+", type=int, default=None)
    p_se = sub.add_parser("sensitivity", help="Extensions A4 (other split_seed) and A5 (max_rows_per_stream): re-run Phase 3-4, B0, B3 (and M1-eps5 for A5) in separate worlds, then write results/sensitivity.json and results/reports/sensitivity.md")
    p_se.add_argument("--tags", nargs="+", default=None, help="worlds to run, default A4-s1 A4-s2 A5-cap20")
    p_se.add_argument("--report-only", action="store_true", help="do not run anything, only rewrite the report from the worlds that exist")
    p_se.add_argument("--seeds", nargs="+", type=int, default=None)
    p_pk = sub.add_parser("package", help="Reproducibility bundle: results/repro/ (pip freeze, configs, environment, checksums), the feature schema next to the split manifests, requirements-lock.txt")
    p_pk.add_argument("--label-mode", choices=["6class", "11class"], default=None)
    p_pk.add_argument("--out-dir", default=None, help="default results/repro")
    p_pk.add_argument("--no-update-lock", action="store_true", help="do not rewrite requirements-lock.txt")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_logging(args.log_level)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    logger.info("Command: %s", args.command)

    from ppfeddata.utils import load_config

    def cmd_inventory(a):
        from ppfeddata.data.inventory import run_inventory
        c = load_config(a.config)
        result = run_inventory(c)
        if result["verification_errors"]:
            logger.error("Verification FAILED")
            sys.exit(1)
        logger.info("Inventory complete. Groups per subclass:")
        print(result["group_counts_df"].to_string(index=False))
        if result["low_group_warnings"]:
            for w in result["low_group_warnings"]:
                logger.warning(w)

    def cmd_harmonize(a):
        from ppfeddata.data.harmonize import run_harmonize
        c = load_config(a.config)
        result = run_harmonize(c, recompute=a.recompute, decisions_only=a.decisions_only)
        logger.info("Harmonize complete.")
        if result["suspects"]:
            logger.warning("SUSPECT columns found:")
            for col, reason in result["suspects"]:
                logger.warning("  %s: %s", col, reason)

    def cmd_sample(a):
        from ppfeddata.data.split_sample import QuotaError, run_sample
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        try:
            m = run_sample(c, use_cache=not a.no_cache)
        except QuotaError as e:
            logger.error("STOP: %s", e)
            raise SystemExit(2)
        logger.info("rows=%s peak_rss_gb=%s", m["rows"], m["peak_rss_gb"])

    def cmd_preprocess(a):
        from ppfeddata.data.preprocess import run_preprocess
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        s = run_preprocess(c)
        logger.info("D=%d, is_na flags=%d, diagnostic features=%d", s["n_features"], s["n_na_flags"],
                    s["diagnostic"]["n_features"])
        for split, dist in s["class_distribution"].items():
            logger.info("%s: %s", split, dist)

    def cmd_check(a):
        from ppfeddata.checks.leakage import run_leakage
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        r = run_leakage(c)
        logger.info("reference macro-F1 on val: %.4f", r["results"]["reference"]["macro_f1"][0])

    def cmd_baseline(a):
        from ppfeddata.eval.baselines import run_baselines, write_g3_report
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        run_baselines(c, a.configs, a.seeds, resume=not a.no_resume)
        gate = write_g3_report(c)
        logger.info("G3 checks: %s", gate)

    def cmd_tune(a):
        from ppfeddata.tune import run_tune
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        r = run_tune(c, a.n_trials)
        logger.info("best: %s", r)

    def cmd_b2(a):
        from ppfeddata.models.b2 import run_b2, write_b2_report
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        run_b2(c, a.seeds, resume=not a.no_resume)
        logger.info("B2 gate: %s", write_b2_report(c))

    def cmd_b3(a):
        from ppfeddata.fl.b3 import run_b3, run_sanity, write_b3_report
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        run_b3(c, a.seeds, resume=not a.no_resume)
        if not a.skip_sanity:
            run_sanity(c, resume=not a.no_resume)
        logger.info("B3 gate: %s", write_b3_report(c))

    def cmd_m1(a):
        from ppfeddata.fl.m1 import run_dp_sanity, run_m1, write_m1_report
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        run_m1(c, a.eps, a.seeds, resume=not a.no_resume, tuned=a.tuned)
        if not a.skip_sanity and not a.tuned:
            run_dp_sanity(c, resume=not a.no_resume)
        logger.info("M1 gate: %s", write_m1_report(c, tuned=a.tuned))

    def cmd_tune_dp(a):
        from ppfeddata.tune_dp import run_tune_dp
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        run_tune_dp(c, a.eps, a.n_trials)

    def cmd_verify_dp(a):
        from ppfeddata.fl.m1 import choose_dp_candidate, verify_dp_candidates
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        verify_dp_candidates(c, a.trials, a.eps)
        logger.info("selected: %s", choose_dp_candidate(c, a.trials, a.eps))

    def cmd_m2(a):
        from ppfeddata.fl.m2 import run_m2
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        run_m2(c, a.seeds, resume=not a.no_resume)

    def cmd_m3(a):
        from ppfeddata.fl.m2 import run_m3
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        run_m3(c, a.eps, a.seeds, resume=not a.no_resume, tuned=not a.untuned)

    def cmd_secagg_check(a):
        from ppfeddata.fl.m2 import run_checks
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        r = run_checks(c, a.seed, resume=not a.no_resume, rounds=a.rounds)
        logger.info("T-SA1 largest aggregation error %.2e (bound %.2e); T-SA2 largest |rho| %.4f; T-SA3 clients per round %s", r["tsa1"]["agg_err_max"],
                    r["tsa1"]["error_bound"], max(abs(x["rho"]) for x in r["tsa2"]), r["tsa3"]["n_clients_per_round"])

    def cmd_secagg_report(a):
        from ppfeddata.fl.m2 import write_secagg_report
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        logger.info("Phase 10 gate: %s", write_secagg_report(c, eps=a.eps, tuned=not a.untuned))

    def cmd_run(a):
        from ppfeddata.run_experiment import execute, load_matrix, write_trial_report
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        exps = load_matrix()
        seeds = a.seeds if a.seeds is not None else list(c["seeds"])
        if a.report:
            from ppfeddata.run_experiment import status_path, workspace_cfg
            import json as _json
            session = _json.loads(status_path(workspace_cfg(c, "trial")).read_text(encoding="utf-8"))
            logger.info("trial report: %s", write_trial_report(c, session, exps))
            return
        try:
            session = execute(c, exps, seeds, stage=a.stage, workspace=a.workspace, approved=a.approve_g4, groups=tuple(a.groups), only=a.only,
                              resume=not a.no_resume, dry_run=a.dry_run)
        except PermissionError as e:
            logger.error("%s", e)
            raise SystemExit(3)
        if a.stage == "trial" and not a.dry_run and session["workspace"] == "trial":
            logger.info("trial report: %s", write_trial_report(c, session, exps))
        if any(r["status"] in ("failed", "incomplete") for r in session["items"]):
            raise SystemExit(1)

    def cmd_aggregate(a):
        from ppfeddata.aggregate import aggregate
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        if a.runs_csv:
            c["compute"]["runs_csv"] = a.runs_csv
        if a.artifacts_dir:
            c["compute"]["artifacts_dir"] = a.artifacts_dir
        out = aggregate(c, out_dir=a.out_dir, with_interpretation=not a.no_interpret, n_boot=a.n_boot)
        logger.info("aggregate: %s", out)

    def cmd_demo(a):
        import subprocess
        from pathlib import Path

        try:
            import streamlit  # noqa: F401
        except ImportError:
            logger.error("streamlit is not installed: pip install -r requirements.txt")
            raise SystemExit(2)
        root = Path(__file__).resolve().parents[2]
        cmd = [sys.executable, "-m", "streamlit", "run", str(root / "demo" / "app.py")]
        if a.port:
            cmd += ["--server.port", str(a.port)]
        if a.headless:
            cmd += ["--server.headless", "true"]
        if a.config:
            cmd += ["--", "--config", str(Path(a.config).resolve())]
        logger.info("launching: %s", " ".join(cmd))
        raise SystemExit(subprocess.run(cmd, cwd=root).returncode)

    def cmd_accept(a):
        from ppfeddata.acceptance import write_report
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        path, items = write_report(c, regenerate=not a.no_regenerate)
        for i in items:
            logger.info("%-4s %-8s %s", i.id, i.status, i.criterion)
        logger.info("written: %s", path)
        if any(i.status == "FAIL" for i in items):
            raise SystemExit(1)

    def cmd_fed_baseline(a):
        from ppfeddata.fed_classifier import NAMES, run_all
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        if a.protected:
            from ppfeddata.fed_protected import run_all as run_protected
            r = run_protected(c, seeds=a.seeds)
            for n, v in r["protected"]["classifiers"].items():
                logger.info("%s: macro-F1 %.4f +- %.4f, recall of the rare classes %.3f%s", n, v["macro_f1_mean"], v["macro_f1_std"], v["rare_recall_mean"],
                            f", epsilon max {v['dp']['eps_max_over_seeds']:.3f}" if "dp" in v else "")
            return
        r = run_all(c, seeds=a.seeds, names=tuple(a.only) if a.only else NAMES)
        for n, v in r["classifiers"].items():
            logger.info("%s: macro-F1 %.4f +- %.4f, recall of the rare classes %.3f", n, v["macro_f1_mean"], v["macro_f1_std"], v["rare_recall_mean"])

    def cmd_mia(a):
        from ppfeddata.mia_model import CONFIGS, run_all
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        r = run_all(c, seeds=a.seeds, configs=() if a.controls_only else tuple(a.configs or CONFIGS))
        for n, v in {**r["controls"], **r["configs"]}.items():
            logger.info("%s: calibrated AUC %.3f (rare %.3f), plain %.3f", n, v["calibrated"]["auc_mean"]["mean"], v["calibrated"].get("auc_rare_mean", {}).get("mean", float("nan")), v["plain"]["auc_mean"]["mean"])

    def cmd_scorecard(a):
        from ppfeddata.scorecard import run
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        sc = run(c, freeze_baseline=a.freeze_baseline)
        logger.info("scorecard: front %s", ", ".join(sc["front"]))

    def cmd_tune_dp_full(a):
        from ppfeddata.tune_dp_full import run_final, run_search
        c = load_config(a.config)
        if a.final:
            logger.info("O1 final runs: %s", run_final(c, a.eps, a.seeds))
            return
        for e in (a.eps or c["dp"]["epsilons"]):
            y = run_search(c, float(e), n_trials=a.n_trials)
            s = y.get("searches", {}).get(f"eps{float(e):g}", {})
            logger.info("eps %g: best trial %s, val macro-F1 %s (%s; anchor %s)", e, s.get("best_trial"), s.get("val_macro_f1"), s.get("variant"), s.get("anchor_val_macro_f1"))

    def cmd_sensitivity(a):
        from ppfeddata.sensitivity import SCENARIOS, report, run_world
        c = load_config(a.config)
        tags = a.tags or list(SCENARIOS)
        if not a.report_only:
            for t in tags:
                logger.info("sensitivity world %s", t)
                run_world(c, t, a.seeds)
        r = report(c, tags)
        logger.info("sensitivity: %d worlds in the report", len(r["worlds"]))

    def cmd_package(a):
        from ppfeddata.package import package
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        r = package(c, out_dir=a.out_dir, update_lock=not a.no_update_lock)
        logger.info("package: %s", r)
        if r["missing"]:
            logger.warning("not found (not hashed): %s", r["missing"])

    def cmd_benchmark(a):
        from ppfeddata.models.benchmark import run_benchmark
        c = load_config(a.config)
        if a.label_mode:
            c["label_mode"] = a.label_mode
        run_benchmark(c)

    # Dispatch to subcommands
    dispatch = {
        "inventory": cmd_inventory,
        "harmonize": cmd_harmonize,
        "sample": cmd_sample,
        "preprocess": cmd_preprocess,
        "check": cmd_check,
        "baseline": cmd_baseline,
        "tune": cmd_tune,
        "b2": cmd_b2,
        "b3": cmd_b3,
        "m1": cmd_m1,
        "tune-dp": cmd_tune_dp,
        "verify-dp": cmd_verify_dp,
        "m2": cmd_m2,
        "m3": cmd_m3,
        "secagg-check": cmd_secagg_check,
        "secagg-report": cmd_secagg_report,
        "run": cmd_run,
        "aggregate": cmd_aggregate,
        "benchmark": cmd_benchmark,
        "demo": cmd_demo,
        "package": cmd_package,
        "fed-baseline": cmd_fed_baseline,
        "sensitivity": cmd_sensitivity,
        "mia": cmd_mia,
        "scorecard": cmd_scorecard,
        "tune-dp-full": cmd_tune_dp_full,
        "accept": cmd_accept,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        logger.info("Command '%s' is not yet implemented.", args.command)
        sys.exit(0)
    handler(args)


if __name__ == "__main__":
    main()
