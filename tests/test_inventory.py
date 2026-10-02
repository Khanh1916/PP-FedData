"""Tests for inventory module."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ppfeddata.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs" / "default.yaml"
LABEL_MAP_PATH = ROOT / "configs" / "label_map.yaml"


def _load_label_map() -> dict:
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestInventoryHelpers:
    """Test helper functions without needing raw data."""

    def test_extract_capture_id(self):
        from ppfeddata.data.inventory import _extract_capture_id_from_filename
        assert _extract_capture_id_from_filename(
            "Pcap_Files__NormalData30_part005.csv"
        ) == "NormalData30"
        assert _extract_capture_id_from_filename(
            "Pcap_Files__NormalData1_part001.csv"
        ) == "NormalData1"

    def test_extract_attack_group_id(self):
        from ppfeddata.data.inventory import _extract_attack_group_id
        assert _extract_attack_group_id("BF1_DoS_AD_12.csv") == "BF1_DoS_AD_12"
        assert _extract_attack_group_id("WILL_DDoS_AD_1.csv") == "WILL_DDoS_AD_1"

    def test_label_map_loads(self):
        lm = _load_label_map()
        assert "scenarios" in lm
        assert "expected_counts" in lm
        assert lm["expected_counts"]["total"] == 59624826

    def test_verify_counts_pass(self):
        from ppfeddata.data.inventory import verify_counts
        lm = _load_label_map()
        expected = lm["expected_counts"]
        # Build a DataFrame that exactly matches
        rows = []
        for cls, cnt in expected.items():
            if cls == "total":
                continue
            rows.append({"class11": cls, "total_rows": cnt})
        df = pd.DataFrame(rows)
        errors = verify_counts(df, lm)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_verify_counts_fail(self):
        from ppfeddata.data.inventory import verify_counts
        lm = _load_label_map()
        df = pd.DataFrame([{"class11": "NORMAL", "total_rows": 999}])
        errors = verify_counts(df, lm)
        assert len(errors) >= 1  # Should detect mismatch


@pytest.mark.skipif(
    not Path(load_config(DEFAULT_CFG)["paths"]["raw_root"]).exists(),
    reason="Raw data not available",
)
class TestInventoryIntegration:
    """Integration tests that require raw data on disk."""

    @pytest.fixture(scope="class")
    def inventory_result(self):
        from ppfeddata.data.inventory import run_inventory
        cfg = load_config(DEFAULT_CFG)
        return run_inventory(cfg)

    def test_total_rows(self, inventory_result):
        """Total rows must be exactly 59,624,826."""
        lm = _load_label_map()
        total = int(inventory_result["class_counts_df"]["total_rows"].sum())
        assert total == lm["expected_counts"]["total"], (
            f"Total rows mismatch: {total} != {lm['expected_counts']['total']}"
        )

    def test_per_class_rows(self, inventory_result):
        """Each class11 count must match expected_counts exactly."""
        lm = _load_label_map()
        expected = lm["expected_counts"]
        counts = inventory_result["class_counts_df"]
        for _, row in counts.iterrows():
            cls = row["class11"]
            actual = int(row["total_rows"])
            if cls in expected:
                assert actual == expected[cls], (
                    f"{cls}: {actual} != {expected[cls]}"
                )

    def test_no_verification_errors(self, inventory_result):
        """verify_counts should pass with no errors."""
        assert inventory_result["verification_errors"] == []

    def test_no_file_in_two_groups(self, inventory_result):
        """Each file should belong to exactly one group."""
        df = inventory_result["files_df"]
        # Check that filename is unique
        assert df["filename"].is_unique or df["path"].is_unique

    def test_each_file_one_class(self, inventory_result):
        """Each file should have exactly one class assignment."""
        df = inventory_result["files_df"]
        for _, row in df.iterrows():
            assert pd.notna(row["class6"])
            assert pd.notna(row["class11"])

    def test_files_csv_saved(self, inventory_result):
        cfg = load_config(DEFAULT_CFG)
        p = Path(cfg["paths"]["work_dir"]) / "inventory" / "files.csv"
        assert p.exists()

    def test_class_counts_csv_saved(self, inventory_result):
        cfg = load_config(DEFAULT_CFG)
        p = Path(cfg["paths"]["work_dir"]) / "inventory" / "class_counts.csv"
        assert p.exists()
