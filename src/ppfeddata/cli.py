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
    sub.add_parser("harmonize", help="Harmonize schema and audit formats")
    # Phase 3
    sub.add_parser("sample", help="Group-split and sample data")
    # Phase 4
    sub.add_parser("preprocess", help="Parse multi-value cells and preprocess features")
    # Phase 5
    sub.add_parser("check", help="Leakage and label reliability checks")
    # Phase 6
    sub.add_parser("baseline", help="Run baseline evaluations")
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

    # Dispatch to subcommands
    dispatch = {
        "inventory": cmd_inventory,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        logger.info("Command '%s' is not yet implemented.", args.command)
        sys.exit(0)
    handler(args)


if __name__ == "__main__":
    main()
