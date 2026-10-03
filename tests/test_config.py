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


class TestMachineSpecificConfig:
    def _write(self, d):
        (d / "default.yaml").write_text("paths:\n  raw_root: ./raw\nseeds: [0]\nx: {a: 1, b: 2}\n", encoding="utf-8")

    def test_local_yaml_deep_merges(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PPFEDDATA_RAW_ROOT", raising=False)
        self._write(tmp_path)
        (tmp_path / "local.yaml").write_text("paths:\n  raw_root: /data/x\nx: {b: 3}\n", encoding="utf-8")
        cfg = load_config(tmp_path / "default.yaml")
        assert cfg["paths"]["raw_root"] == "/data/x" and cfg["x"] == {"a": 1, "b": 3} and cfg["seeds"] == [0]

    def test_env_overrides_raw_root(self, tmp_path, monkeypatch):
        self._write(tmp_path)
        monkeypatch.setenv("PPFEDDATA_RAW_ROOT", "/env/root")
        assert load_config(tmp_path / "default.yaml")["paths"]["raw_root"] == "/env/root"

    def test_hash_ignores_paths_but_not_settings(self):
        a = {"paths": {"raw_root": "C:/a"}, "seeds": [0]}
        b = {"paths": {"raw_root": "/b"}, "seeds": [0]}
        c = {"paths": {"raw_root": "/b"}, "seeds": [1]}
        assert config_hash(a) == config_hash(b) != config_hash(c)

    def test_default_yaml_has_no_personal_path(self):
        text = DEFAULT_CFG.read_text(encoding="utf-8")
        assert "Users" not in text and "C:/" not in text
