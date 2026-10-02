"""Test config loading and hashing."""
from pathlib import Path

import pytest

from ppfeddata.utils import config_hash, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs" / "default.yaml"


class TestLoadConfig:
    """Tests for load_config."""

    def test_load_default(self):
        """Default config loads and has expected top-level keys."""
        cfg = load_config(DEFAULT_CFG)
        assert isinstance(cfg, dict)
        for key in ["seeds", "paths", "label_mode", "quota", "cvae", "fl", "dp"]:
            assert key in cfg, f"Missing key: {key}"

    def test_seeds_list(self):
        """Seeds should be [0, 1, 2]."""
        cfg = load_config(DEFAULT_CFG)
        assert cfg["seeds"] == [0, 1, 2]

    def test_missing_file_raises(self):
        """Non-existent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_label_mode(self):
        """Default label_mode is 6class."""
        cfg = load_config(DEFAULT_CFG)
        assert cfg["label_mode"] == "6class"


class TestConfigHash:
    """Tests for config_hash stability."""

    def test_deterministic(self):
        """Same dict produces same hash."""
        d = {"a": 1, "b": [2, 3], "c": {"d": 4}}
        h1 = config_hash(d)
        h2 = config_hash(d)
        assert h1 == h2
        assert len(h1) == 12

    def test_order_invariant(self):
        """Key order does not affect hash (sort_keys=True)."""
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert config_hash(d1) == config_hash(d2)

    def test_different_dicts(self):
        """Different dicts produce different hashes."""
        d1 = {"a": 1}
        d2 = {"a": 2}
        assert config_hash(d1) != config_hash(d2)

    def test_real_config(self):
        """Hash of default config is stable across calls."""
        cfg = load_config(DEFAULT_CFG)
        h1 = config_hash(cfg)
        h2 = config_hash(cfg)
        assert h1 == h2
        assert isinstance(h1, str) and len(h1) == 12
