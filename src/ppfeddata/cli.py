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
    sub.add_parser("tune", help="Optuna hyperparameter tuning for CVAE")
    # Phase 8-10
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

    # Dispatch to subcommands
    dispatch = {
        "inventory": cmd_inventory,
        "harmonize": cmd_harmonize,
        "sample": cmd_sample,
        "preprocess": cmd_preprocess,
        "check": cmd_check,
        "baseline": cmd_baseline,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        logger.info("Command '%s' is not yet implemented.", args.command)
        sys.exit(0)
    handler(args)


if __name__ == "__main__":
    main()
