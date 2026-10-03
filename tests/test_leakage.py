"""Tests for Phase 5 checks, using data with leakage planted on purpose (known answers)."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from ppfeddata.checks.leakage import (  # noqa: E402
    ablations, c1_presence, c2_na_flags, c3_leave_one_out, c3_single_feature, c4_split_gap, c5_dos_vs_ddos,
    feature_groups, na_flag_columns, reference_model, run_leakage,
)

SEEDS = [0, 1]
TREES = 25


def make_data(n_per_class=300, n_classes=3, seed=0, leak_col=None, flag_leak=False):
    """Noise features; optionally column `leak_col` equals the label, and column 5 (an is_na flag) follows the label."""
    rng = np.random.default_rng(seed)
    out = {}
    for split in ("train", "val", "test"):
        y = np.repeat(np.arange(n_classes), n_per_class)
        X = rng.normal(size=(len(y), 6)).astype(np.float32)
        X[:, 5] = (rng.random(len(y)) < 0.5)
        if leak_col is not None:
            X[:, leak_col] = y
        if flag_leak:
            X[:, 5] = (y == 1).astype(np.float32)
        out[split] = {"X": X, "X_diag": rng.normal(size=(len(y), 1)).astype(np.float32), "y": y,
                      "group_id": np.array([f"{split}_g{i % 6}" for i in range(len(y))]),
                      "stream_id": np.array([f"{split}_s{i % 40}" for i in range(len(y))])}
    return out


GROUPS = {"f0": [0], "f1": [1], "f2": [2], "f3": [3], "f4": [4], "f5": [5]}


class TestC3:
    def test_label_leaking_column_is_flagged_and_noise_is_not(self):
        d = make_data(leak_col=2)
        r = c3_single_feature(d, GROUPS, SEEDS, TREES, 0.9)
        assert r.iloc[0]["feature"] == "f2" and r.iloc[0]["macro_f1"] > 0.99 and bool(r.iloc[0]["flag"])
        assert r.iloc[0]["best_class_f1"] > 0.99
        assert not r[r["feature"] != "f2"]["flag"].any()

    def test_column_that_identifies_one_class_only_is_flagged_by_best_class(self):
        """A column that is perfect for ONE class on a subset of rows has a low macro-F1 but a high per-class F1."""
        d = make_data(n_per_class=400)
        for sp in ("train", "val"):
            d[sp]["X"][:, 3] = np.where(d[sp]["y"] == 1, 7.0, np.random.default_rng(1).normal(size=len(d[sp]["y"])))
        r = c3_single_feature(d, GROUPS, SEEDS, TREES, 0.9, class_names={0: "A", 1: "B", 2: "C"})
        row = r[r["feature"] == "f3"].iloc[0]
        assert row["best_class"] == "B" and row["best_class_f1"] > 0.9 and bool(row["flag"])

    def test_diagnostic_column_is_scored_separately(self):
        d = make_data()
        r = c3_single_feature(d, GROUPS, SEEDS, TREES, 0.9, {"time_diag": [0]})
        assert r[r["feature"] == "time_diag"]["kind"].iloc[0] == "diagnostic"

    def test_leave_one_out_drop_of_leaking_column_hurts_most(self):
        d = make_data(leak_col=2)
        base = reference_model(d, SEEDS, TREES)["macro_f1"][0]
        r = c3_leave_one_out(d, GROUPS, SEEDS, TREES, base)
        assert r.iloc[0]["removed"] == "f2" and r.iloc[0]["delta_vs_full"] < -0.3


class TestAblations:
    def test_removing_time_columns_hurts_when_they_carry_the_signal(self):
        d = make_data(leak_col=0)
        r = ablations(d, GROUPS, SEEDS, TREES, ["f0"])
        assert r["full"]["macro_f1"][0] > 0.99 and r["without_time_columns"]["macro_f1"][0] < 0.6
        assert r["time_columns"] == ["f0"] and len(r["full"]["per_class_f1"]) == 3
        assert "plus_diagnostic" in r and set(r) == {"full", "plus_diagnostic", "without_time_columns", "time_columns"}

    def test_no_diagnostic_columns_skips_that_variant(self):
        d = make_data(leak_col=0)
        for sp in d.values():
            sp["X_diag"] = sp["X_diag"][:, :0]
        assert "plus_diagnostic" not in ablations(d, GROUPS, SEEDS, TREES, ["f0"])


class TestC2:
    def test_predictive_flag_detected(self):
        d = make_data(flag_leak=True)
        r = c2_na_flags(d, {"col_a": 5}, 3, SEEDS, TREES, 0.5)
        assert r["flag"] and r["macro_f1"] > 0.5 and r["chance"] == pytest.approx(1 / 3)

    def test_random_flag_not_flagged(self):
        r = c2_na_flags(make_data(), {"col_a": 5}, 3, SEEDS, TREES, 0.5)
        assert not r["flag"] and r["macro_f1"] < 0.5

    def test_no_flags(self):
        assert c2_na_flags(make_data(), {}, 3, SEEDS, TREES, 0.5) == {"n_flags": 0}


def _fingerprint_pool(n_groups_per_class=8, rows=40, fingerprint=True, seed=0):
    """Each group has a constant 'fingerprint' column; rows of a group share it, classes differ by nothing else."""
    rng = np.random.default_rng(seed)
    X, y, g, s = [], [], [], []
    for c in range(3):
        for k in range(n_groups_per_class):
            fp = rng.normal() * 10
            for r in range(rows):
                X.append([fp if fingerprint else rng.normal(), rng.normal(), rng.normal()])
                y.append(c)
                g.append(f"c{c}g{k}")
                s.append(f"c{c}g{k}s{r % 4}")
    return np.array(X, dtype=np.float32), np.array(y), np.array(g), np.array(s)


class TestC4:
    def test_group_fingerprint_is_detected_as_leakage(self):
        X, y, g, s = _fingerprint_pool(fingerprint=True)
        Xv, yv, _, _ = _fingerprint_pool(fingerprint=True, seed=99)      # new groups, new fingerprints
        r = c4_split_gap(X, y, g, s, Xv, yv, SEEDS, TREES, 4, 0.05)
        assert r["random"][0] > 0.8 and r["group"][0] < 0.6
        assert r["flag"] and r["gap_random_minus_group"] > 0.25 and r["gap_random_minus_val"] > 0.25
        assert "gap_random_minus_stream" in r

    def test_no_fingerprint_no_gap(self):
        X, y, g, s = _fingerprint_pool(fingerprint=False)
        Xv, yv, _, _ = _fingerprint_pool(fingerprint=False, seed=99)
        r = c4_split_gap(X, y, g, s, Xv, yv, SEEDS, TREES, 4, 0.05)
        assert not r["flag"] and abs(r["gap_random_minus_group"]) < 0.15

    def test_val_gap_alone_never_raises_the_flag(self):
        """Random-vs-group gap ~0 but val differs from CV (distribution shift): must not be flagged."""
        X, y, g, s = _fingerprint_pool(fingerprint=False)
        rng = np.random.default_rng(5)
        Xv, yv = rng.normal(size=(300, 3)).astype(np.float32) + 3.0, rng.integers(0, 3, 300)
        r = c4_split_gap(X, y, g, s, Xv, yv, SEEDS, TREES, 4, 0.05)
        assert not r["flag"]

    def test_group_folds_are_disjoint(self):
        from sklearn.model_selection import StratifiedGroupKFold
        X, y, g, s = _fingerprint_pool()
        for tr, te in StratifiedGroupKFold(4, shuffle=True, random_state=0).split(X, y, g):
            assert not set(g[tr]) & set(g[te])


class TestC5:
    def _d11(self, separable):
        rng = np.random.default_rng(0)
        out = {}
        for split in ("train", "val"):
            y = np.repeat([0, 1, 2, 3], 200)           # BCF_DoS, BCF_DDoS, SYN_DoS, SYN_DDoS
            X = rng.normal(size=(len(y), 4)).astype(np.float32)
            if separable:
                X[:, 0] = (y == 1) * 5.0 + (y == 3) * 5.0    # DDoS shifted in both scenarios
            out[split] = {"X": X, "y": y}
        return out

    LM = {"BCF_DoS": 0, "BCF_DDoS": 1, "SYN_DoS": 2, "SYN_DDoS": 3}

    def test_separable_pairs(self):
        r = c5_dos_vs_ddos(self._d11(True), self.LM, SEEDS, TREES, 0.6)
        assert set(r["scenario"]) == {"BCF", "SYN"} and (r["macro_f1"] > 0.95).all() and not r["flag_indistinguishable"].any()

    def test_indistinguishable_pairs_flagged(self):
        r = c5_dos_vs_ddos(self._d11(False), self.LM, SEEDS, TREES, 0.6)
        assert r["flag_indistinguishable"].all() and (r["macro_f1"] < 0.6).all()


class TestC1:
    def test_mixed_columns(self, tmp_path):
        pm = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [0.0, 0.0, 0.2], "c": [0.0, 0.0, 0.0]},
                          index=["NORMAL", "X_DoS", "WILL_DoS"])
        p = tmp_path / "pm.csv"
        pm.to_csv(p)
        r = c1_presence(p, 0.001, 0.01).set_index("column")
        assert bool(r.loc["b", "mixed"]) and not r.loc["a", "mixed"] and not r.loc["c", "mixed"]
        assert r.loc["b", "populated_in"] == "WILL_DoS"


class TestHelpers:
    def test_feature_groups_merge_value_flag_and_onehot(self):
        schema = {"blocks": [
            {"column": "irtt", "type": "numeric", "start": 0, "width": 1},
            {"column": "irtt", "type": "na_flag", "start": 1, "width": 1},
            {"column": "protocol", "type": "categorical", "start": 2, "width": 4}]}
        assert feature_groups(schema) == {"irtt": [0, 1], "protocol": [2, 3, 4, 5]}
        assert na_flag_columns(schema) == {"irtt": 1}

    def test_reference_is_reproducible(self):
        d = make_data(leak_col=2)
        assert reference_model(d, SEEDS, TREES) == reference_model(d, SEEDS, TREES)


def test_run_leakage_smoke(tmp_path):
    """End-to-end on tiny processed data: files, JSON and the markdown report are produced."""
    work = tmp_path / "work"
    for mode, ncls in (("6class", 3), ("11class", 4)):
        d = work / "processed" / mode
        d.mkdir(parents=True)
        data = make_data(80, ncls)
        for split, v in data.items():
            np.savez(d / f"{split}.npz", **v)
        names = ["BCF_DoS", "BCF_DDoS", "SYN_DoS", "SYN_DDoS"] if mode == "11class" else ["NORMAL", "BCF", "SYN"]
        schema = {"n_features": 6, "n_na_flags": 1, "label_map": {n: i for i, n in enumerate(names)},
                  "blocks": [{"column": f"f{i}", "type": "numeric", "start": i, "width": 1} for i in range(5)]
                  + [{"column": "f5", "type": "na_flag", "start": 5, "width": 1}],
                  "diagnostic": {"n_features": 1, "blocks": [{"column": "tdiag", "type": "numeric", "start": 0, "width": 1}]}}
        (d / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    inv = work / "inventory"
    inv.mkdir()
    pd.DataFrame({"a": [1.0, 0.0]}, index=["NORMAL", "X"]).to_csv(inv / "presence_matrix.csv")
    cfg = {"label_mode": "6class", "seeds": [0], "paths": {"work_dir": str(work)},
           "thresholds": {"single_feature_f1_flag": 0.9, "na_only_f1_flag": 0.5, "dos_ddos_f1_flag": 0.6,
                          "leakage_gap_flag": 0.05},
           "leakage": {"rf_trees": 10, "cv_folds": 3, "reports_dir": str(tmp_path / "rep"),
                       "report_path": str(tmp_path / "rep" / "leakage_report.md"), "figures_dir": str(tmp_path / "fig")},
           "harmonize": {}}
    r = run_leakage(cfg)
    assert (tmp_path / "rep" / "leakage_report.md").exists() and (tmp_path / "rep" / "leakage_results.json").exists()
    assert (tmp_path / "fig" / "c3_single_feature.png").exists()
    text = (tmp_path / "rep" / "leakage_report.md").read_text(encoding="utf-8")
    assert "C4" in text and "C5" in text and "Flags raised" in text and "class-balanced RF" in text
    assert r["c5"] is not None
