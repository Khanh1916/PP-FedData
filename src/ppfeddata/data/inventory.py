"""Inventory raw data files: scan, count rows by chunk, assign group_id."""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ppfeddata.utils import load_config

logger = logging.getLogger("ppfeddata.data.inventory")

# Chunk size for counting rows without loading everything into RAM
_CHUNK = 100_000


def _load_label_map(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load label_map.yaml from the configs directory."""
    label_map_path = Path(__file__).resolve().parents[3] / "configs" / "label_map.yaml"
    with open(label_map_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _count_rows_chunked(path: Path) -> int:
    """Count data rows in a CSV file using chunked reading. Returns 0 for empty files."""
    if path.stat().st_size == 0:
        return 0
    count = 0
    for chunk in pd.read_csv(path, chunksize=_CHUNK, usecols=[0], dtype=str,
                             low_memory=False, on_bad_lines="skip"):
        count += len(chunk)
    return count


def _count_columns(path: Path) -> int:
    """Count columns from the header of a CSV file."""
    if path.stat().st_size == 0:
        return 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        return len(header) if header else 0


def _extract_capture_id_from_filename(filename: str) -> str:
    """Extract the capture ID number from a Normal data part filename.

    E.g. 'Pcap_Files__NormalData30_part005.csv' → 'NormalData30'
    """
    m = re.match(r"Pcap_Files__(NormalData\d+)_part\d+\.csv", filename)
    if m:
        return m.group(1)
    return filename


def _extract_attack_group_id(filename: str) -> str:
    """Extract group_id from an attack file name.

    BCF files are slices of captures: BF1_DoS_AD_12.csv → group is the file itself.
    Other scenarios have single files per (scenario, attack_type) so file = group.
    """
    # For BCF files: BF1_DoS_AD_XX.csv or BF1_DDoS_AD_XX.csv
    # Each file is a separate capture, so group_id = filename (stem)
    return Path(filename).stem


def scan_attack_files(
    raw_root: Path,
    label_map: dict[str, Any],
) -> list[dict[str, Any]]:
    """Scan attack data directories and collect file metadata."""
    records = []
    scenarios = label_map["scenarios"]
    for dir_name, scenario_code in scenarios.items():
        scenario_dir = raw_root / dir_name
        if not scenario_dir.exists():
            logger.warning("Scenario dir not found: %s", scenario_dir)
            continue
        for attack_type in label_map["attack_types"]:
            csv_dir = scenario_dir / attack_type / "CSV Files"
            if not csv_dir.exists():
                logger.warning("CSV dir not found: %s", csv_dir)
                continue
            for csv_file in sorted(csv_dir.glob("*.csv")):
                file_size = csv_file.stat().st_size
                n_rows = _count_rows_chunked(csv_file)
                n_cols = _count_columns(csv_file)
                group_id = _extract_attack_group_id(csv_file.name)
                class11 = f"{scenario_code}_{attack_type}"
                class6 = scenario_code

                records.append({
                    "path": str(csv_file),
                    "filename": csv_file.name,
                    "scenario": scenario_code,
                    "attack_type": attack_type,
                    "class6": class6,
                    "class11": class11,
                    "group_id": group_id,
                    "rows": n_rows,
                    "cols": n_cols,
                    "bytes": file_size,
                    "source": "attack",
                    "empty": n_rows == 0,
                })
    return records


def scan_normal_files(
    raw_root: Path,
    normal_csv_dir: str,
) -> list[dict[str, Any]]:
    """Scan Normal data directory and collect file metadata."""
    records = []
    normal_dir = raw_root / normal_csv_dir
    if not normal_dir.exists():
        raise FileNotFoundError(f"Normal data dir not found: {normal_dir}")

    for csv_file in sorted(normal_dir.glob("*.csv")):
        # Skip summary files
        if "summary" in csv_file.name.lower():
            continue
        file_size = csv_file.stat().st_size
        n_rows = _count_rows_chunked(csv_file)
        n_cols = _count_columns(csv_file)
        capture_id = _extract_capture_id_from_filename(csv_file.name)

        records.append({
            "path": str(csv_file),
            "filename": csv_file.name,
            "scenario": "NORMAL",
            "attack_type": "NONE",
            "class6": "NORMAL",
            "class11": "NORMAL",
            "group_id": capture_id,
            "rows": n_rows,
            "cols": n_cols,
            "bytes": file_size,
            "source": "normal",
            "empty": n_rows == 0,
        })
    return records


def build_class_counts(files_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row counts by class11."""
    counts = (
        files_df[~files_df["empty"]]
        .groupby("class11")["rows"]
        .sum()
        .reset_index()
        .rename(columns={"rows": "total_rows"})
    )
    return counts


def verify_counts(
    class_counts_df: pd.DataFrame,
    label_map: dict[str, Any],
) -> list[str]:
    """Verify counts against expected_counts. Returns list of error messages."""
    expected = label_map["expected_counts"]
    errors = []

    # Check per-class
    for _, row in class_counts_df.iterrows():
        cls = row["class11"]
        actual = int(row["total_rows"])
        if cls in expected:
            exp = expected[cls]
            if actual != exp:
                errors.append(
                    f"Class {cls}: expected {exp:,}, got {actual:,} "
                    f"(diff={actual - exp:+,})"
                )

    # Check total
    actual_total = int(class_counts_df["total_rows"].sum())
    exp_total = expected["total"]
    if actual_total != exp_total:
        errors.append(
            f"Total: expected {exp_total:,}, got {actual_total:,} "
            f"(diff={actual_total - exp_total:+,})"
        )

    return errors


def count_groups_per_subclass(files_df: pd.DataFrame) -> pd.DataFrame:
    """Count distinct groups per (scenario, attack_type)."""
    return (
        files_df[~files_df["empty"]]
        .groupby(["scenario", "attack_type"])["group_id"]
        .nunique()
        .reset_index()
        .rename(columns={"group_id": "n_groups"})
    )


def run_inventory(cfg: dict[str, Any]) -> dict[str, Any]:
    """Main inventory pipeline. Returns dict with DataFrames and verification results."""
    raw_root = Path(cfg["paths"]["raw_root"])
    normal_csv_dir = cfg["paths"]["normal_csv_dir"]
    label_map = _load_label_map(cfg)

    logger.info("Scanning attack files under %s ...", raw_root)
    attack_records = scan_attack_files(raw_root, label_map)
    logger.info("Found %d attack files.", len(attack_records))

    logger.info("Scanning normal files under %s ...", raw_root / normal_csv_dir)
    normal_records = scan_normal_files(raw_root, normal_csv_dir)
    logger.info("Found %d normal files.", len(normal_records))

    all_records = attack_records + normal_records
    files_df = pd.DataFrame(all_records)

    # Class counts
    class_counts_df = build_class_counts(files_df)

    # Verify
    errors = verify_counts(class_counts_df, label_map)
    if errors:
        for e in errors:
            logger.error("VERIFICATION FAILED: %s", e)
    else:
        logger.info("All counts verified OK.")

    # Group counts
    group_counts_df = count_groups_per_subclass(files_df)

    # Warnings for subclasses with < 3 groups
    low_group_warnings = []
    for _, row in group_counts_df.iterrows():
        if row["n_groups"] < 3:
            msg = (
                f"{row['scenario']}_{row['attack_type']}: "
                f"only {row['n_groups']} group(s) — will use fallback block-split"
            )
            low_group_warnings.append(msg)
            logger.warning(msg)

    # Save outputs
    work_dir = Path(cfg["paths"]["work_dir"])
    inv_dir = work_dir / "inventory"
    inv_dir.mkdir(parents=True, exist_ok=True)

    files_path = inv_dir / "files.csv"
    files_df.to_csv(files_path, index=False)
    logger.info("Saved %s (%d files)", files_path, len(files_df))

    counts_path = inv_dir / "class_counts.csv"
    class_counts_df.to_csv(counts_path, index=False)
    logger.info("Saved %s", counts_path)

    groups_path = inv_dir / "group_counts.csv"
    group_counts_df.to_csv(groups_path, index=False)
    logger.info("Saved %s", groups_path)

    return {
        "files_df": files_df,
        "class_counts_df": class_counts_df,
        "group_counts_df": group_counts_df,
        "verification_errors": errors,
        "low_group_warnings": low_group_warnings,
    }
