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
    sub.add_parser("run", help="Run experiment configurations")
    # Phase 11
    sub.add_parser("aggregate", help="Aggregate results and generate figures")
    # Phase 13
    sub.add_parser("demo", help="Launch Streamlit demo")

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
        "benchmark": cmd_benchmark,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        logger.info("Command '%s' is not yet implemented.", args.command)
        sys.exit(0)
    handler(args)


if __name__ == "__main__":
    main()
