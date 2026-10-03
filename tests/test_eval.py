"""Tests for the Phase 6 evaluation framework: metrics, protocols, bootstrap, fidelity, privacy, overhead, ledger."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("imblearn")

from ppfeddata.eval.baselines import BASELINES, check_fit_on_train, run_baselines, write_g3_report  # noqa: E402
from ppfeddata.eval.fidelity import (c2st_auc, correlation_distance, fidelity_report, js_discrete,  # noqa: E402
                                     wasserstein_numeric)
from ppfeddata.eval.overhead import RoundMeter, Timer, bytes_per_round, model_bytes, overhead_ratio  # noqa: E402
from ppfeddata.eval.privacy import (dcr, dcr_ratio, duplicate_rate, mia_auc, positive_control,  # noqa: E402
                                    privacy_report)
from ppfeddata.eval.runs import RunLedger, load_predictions, run_id  # noqa: E402
from ppfeddata.eval.stats import bootstrap_macro_f1, paired_bootstrap_diff  # noqa: E402
from ppfeddata.eval.utility import (build_train_set, compute_metrics, confusion, macro_f1_from_confusion,  # noqa: E402
                                    make_classifier, smote_oversample, snap_encoded)

# A tiny schema: 2 numeric, 1 binary, 1 na_flag, one categorical group of 3 (D = 8)
SCHEMA = {"blocks": [
    {"name": "n0", "type": "numeric", "column": "n0", "start": 0, "width": 1},
    {"name": "n1", "type": "numeric", "column": "n1", "start": 1, "width": 1},
    {"name": "b0", "type": "binary", "column": "b0", "start": 2, "width": 1},
    {"name": "n1_is_na", "type": "na_flag", "column": "n1", "start": 3, "width": 1},
    {"name": "cat", "type": "categorical", "column": "cat", "start": 4, "width": 3, "categories": ["a", "b", "OTHER"]},
    {"name": "n2", "type": "numeric", "column": "n2", "start": 7, "width": 1},
], "label_map": {"A": 0, "B": 1, "C": 2}}
CLASSES = ["A", "B", "C"]
CFG = {"eval": {"rf": {"n_estimators": 20}, "mlp": {"hidden": [8], "max_iter": 30, "early_stopping": True},
                "bootstrap": 50, "smote": {"k_neighbors": 3}},
       "generate": {"target_per_class": 120}, "label_mode": "6class", "seeds": [0, 1],
       "thresholds": {"seed_std_max": 0.02}}


def toy(n_per=(200, 60, 30), seed=0, shift=2.0):
    """Valid encoded rows; the class shifts the first numeric column so a classifier can learn something."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c, n in enumerate(n_per):
        x = np.zeros((n, 8), dtype=np.float32)
        x[:, 0] = rng.normal(c * shift, 1, n)
        x[:, 1] = rng.normal(0, 1, n)
        x[:, 2] = rng.integers(0, 2, n)
        x[:, 3] = rng.integers(0, 2, n)
        x[np.arange(n), 4 + rng.integers(0, 3, n)] = 1
        x[:, 7] = rng.normal(0, 1, n)
        X.append(x)
        y += [c] * n
    return np.vstack(X), np.array(y)


class TestMetrics:
    def test_perfect_prediction(self):
        y = np.repeat([0, 1, 2], 10)
        proba = np.eye(3)[y]
        m = compute_metrics(y, y, proba, CLASSES)
        assert m["macro_f1"] == 1.0 and m["balanced_acc"] == 1.0 and m["pr_auc_macro"] == pytest.approx(1.0)
        assert m["binary"]["f1"] == 1.0 and np.array(m["confusion"]).trace() == 30

    def test_random_prediction_is_near_chance(self):
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1, 2], 3000)
        m = compute_metrics(y, rng.integers(0, 3, len(y)), None, CLASSES)
        assert abs(m["macro_f1"] - 1 / 3) < 0.02 and abs(m["balanced_acc"] - 1 / 3) < 0.02

    def test_constant_prediction_has_zero_f1_for_other_classes(self):
        y = np.repeat([0, 1, 2], 10)
        m = compute_metrics(y, np.zeros_like(y), None, CLASSES)
        assert m["per_class"]["A"]["recall"] == 1.0 and m["per_class"]["B"]["f1"] == 0.0
        assert m["macro_f1"] == pytest.approx(0.5 / 3)          # class A: precision 1/3, recall 1 -> F1 0.5

    def test_binary_table_collapses_multiclass(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        p = np.array([0, 1, 1, 0, 2, 2])         # normal=0: one normal called attack, one attack called normal
        b = compute_metrics(y, p, None, CLASSES)["binary"]
        assert b["precision"] == pytest.approx(3 / 4) and b["recall"] == pytest.approx(3 / 4)
        assert b["accuracy"] == pytest.approx(4 / 6)

    def test_macro_f1_matches_sklearn(self):
        from sklearn.metrics import f1_score
        rng = np.random.default_rng(1)
        y, p = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
        assert macro_f1_from_confusion(confusion(y, p, 4)) == pytest.approx(f1_score(y, p, average="macro"))


class TestProtocols:
    def test_trtr_is_real_only(self):
        X, y = toy()
        Xo, yo, info = build_train_set("TRTR", X, y, None, None, 3, 50, 100, 0)
        assert Xo is X and info["n_synthetic"] == {}

    def test_tstr_is_synthetic_and_balanced(self):
        X, y = toy()
        Xs, ys = toy((100, 100, 100), seed=5)
        Xo, yo, info = build_train_set("TSTR", X, y, Xs, ys, 3, 40, 100, 0)
        assert np.bincount(yo).tolist() == [40, 40, 40] and len(Xo) == 120

    def test_taug_reaches_target_and_never_reduces(self):
        X, y = toy((200, 60, 30))
        Xs, ys = toy((300, 300, 300), seed=5)
        Xo, yo, info = build_train_set("TAug", X, y, Xs, ys, 3, 40, 100, 0)
        counts = np.bincount(yo)
        assert counts.tolist() == [200, 100, 100]       # class A already above 100 -> untouched
        assert info["n_synthetic"] == {1: 40, 2: 70} and info["shortage"] == {}

    def test_taug_reports_shortage(self):
        X, y = toy((200, 60, 30))
        Xs, ys = toy((300, 10, 300), seed=5)
        _, yo, info = build_train_set("TAug", X, y, Xs, ys, 3, 40, 100, 0)
        assert info["shortage"] == {1: 30} and np.bincount(yo)[1] == 70

    def test_protocols_are_reproducible(self):
        X, y = toy()
        Xs, ys = toy((300, 300, 300), seed=5)
        a = build_train_set("TAug", X, y, Xs, ys, 3, 40, 100, 3)
        b = build_train_set("TAug", X, y, Xs, ys, 3, 40, 100, 3)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_unknown_protocol_and_missing_synthetic(self):
        X, y = toy()
        with pytest.raises(ValueError):
            build_train_set("XXX", X, y, None, None, 3, 1, 1, 0)
        with pytest.raises(ValueError):
            build_train_set("TSTR", X, y, None, None, 3, 1, 1, 0)


class TestSmote:
    def test_snap_makes_rows_valid(self):
        rng = np.random.default_rng(0)
        X = rng.random((50, 8)).astype(np.float32)
        S = snap_encoded(X, SCHEMA)
        assert set(np.unique(S[:, [2, 3]])) <= {0.0, 1.0}
        assert np.all(S[:, 4:7].sum(1) == 1.0) and set(np.unique(S[:, 4:7])) <= {0.0, 1.0}
        assert np.array_equal(S[:, [0, 1, 7]], X[:, [0, 1, 7]])        # numeric columns untouched

    def test_oversample_reaches_target_without_reducing(self):
        X, y = toy((200, 60, 30))
        Xo, yo = smote_oversample(X, y, SCHEMA, 120, seed=0, k_neighbors=3)
        assert np.bincount(yo).tolist() == [200, 120, 120]
        assert np.array_equal(Xo[:len(X)], X)                          # originals kept in place
        new = Xo[len(X):]
        assert np.all(new[:, 4:7].sum(1) == 1.0) and set(np.unique(new[:, [2, 3]])) <= {0.0, 1.0}

    def test_oversample_is_reproducible_and_noop_when_balanced(self):
        X, y = toy((50, 50, 50))
        assert smote_oversample(X, y, SCHEMA, 50, 0)[0] is X
        X, y = toy((200, 60, 30))
        a, b = smote_oversample(X, y, SCHEMA, 120, 7, 3), smote_oversample(X, y, SCHEMA, 120, 7, 3)
        assert np.array_equal(a[0], b[0])


class TestClassifiers:
    def test_rf_balanced_flag_and_mlp(self):
        assert make_classifier("rf", CFG, 0, balanced=True).class_weight == "balanced"
        assert make_classifier("rf", CFG, 0).class_weight is None
        assert make_classifier("mlp", CFG, 0).hidden_layer_sizes == (8,)
        with pytest.raises(ValueError):
            make_classifier("mlp", CFG, 0, balanced=True)

    def test_same_seed_same_predictions(self):
        X, y = toy()
        p1 = make_classifier("rf", CFG, 3).fit(X, y).predict(X)
        p2 = make_classifier("rf", CFG, 3).fit(X, y).predict(X)
        assert np.array_equal(p1, p2)


class TestBootstrap:
    def test_ci_contains_point_and_is_ordered(self):
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1, 2], 300)
        p = np.where(rng.random(len(y)) < 0.8, y, rng.integers(0, 3, len(y)))
        b = bootstrap_macro_f1(y, p, 3, 300, seed=1)
        assert b["lo"] <= b["macro_f1"] <= b["hi"] and b["hi"] - b["lo"] < 0.1

    def test_paired_identical_predictions_give_zero(self):
        y = np.repeat([0, 1, 2], 100)
        p = np.roll(y, 5)
        d = paired_bootstrap_diff(y, p, p, 3, 200, seed=0)
        assert d["diff"] == 0 and d["lo"] == 0 and d["hi"] == 0 and not d["excludes_zero"]

    def test_paired_detects_a_clearly_better_model(self):
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1, 2], 400)
        good = np.where(rng.random(len(y)) < 0.9, y, rng.integers(0, 3, len(y)))
        bad = np.where(rng.random(len(y)) < 0.5, y, rng.integers(0, 3, len(y)))
        d = paired_bootstrap_diff(y, good, bad, 3, 300, seed=0)
        assert d["diff"] > 0.2 and d["excludes_zero"] and d["p_positive"] == 1.0

    def test_resampling_is_stratified(self):
        from ppfeddata.eval.stats import _class_indices, _resample
        y = np.repeat([0, 1, 2], [50, 10, 5])
        ix = _resample(np.random.default_rng(0), _class_indices(y, 3))
        assert np.bincount(y[ix]).tolist() == [50, 10, 5]

    def test_same_seed_same_interval(self):
        y = np.repeat([0, 1], 100)
        p = np.r_[y[:150], 1 - y[150:]]
        assert bootstrap_macro_f1(y, p, 2, 100, seed=4) == bootstrap_macro_f1(y, p, 2, 100, seed=4)


class TestFidelity:
    def test_identical_distributions_score_zero(self):
        a, _ = toy((400, 400, 1), seed=1)
        b, _ = toy((400, 400, 1), seed=2)
        assert wasserstein_numeric(a, b, SCHEMA) < 0.2 and js_discrete(a, b, SCHEMA) < 0.01
        assert correlation_distance(a, b) < 0.15
        assert abs(c2st_auc(a, b, seed=0, n_trees=30) - 0.5) < 0.1

    def test_shifted_distribution_is_detected(self):
        a, _ = toy((400, 400, 1), seed=1)
        b = a.copy()
        b[:, 0] += 3.0
        b[:, 4:7] = np.eye(3)[np.zeros(len(b), int)]
        assert wasserstein_numeric(a, b, SCHEMA) > 0.5 and js_discrete(a, b, SCHEMA) > 0.05
        assert c2st_auc(a, b, seed=0, n_trees=30) > 0.95

    def test_report_per_class(self):
        X, y = toy((300, 300, 300))
        Xs, ys = toy((300, 300, 300), seed=9)
        r = fidelity_report(X, y, Xs, ys, SCHEMA, CLASSES, with_c2st=False)
        assert set(r) == {"A", "B", "C", "mean"} and "wasserstein" in r["mean"]


class TestPrivacy:
    def test_duplicate_rate(self):
        X, _ = toy((100, 1, 1))
        S = np.vstack([X[:30], X[:70] + 5.0])
        assert duplicate_rate(S, X) == pytest.approx(0.3)

    def test_copying_generator_has_small_dcr_ratio_and_high_mia(self):
        rng = np.random.default_rng(0)
        Xtr, _ = toy((400, 1, 1), seed=1)
        Xva, _ = toy((400, 1, 1), seed=2)
        copy = Xtr + rng.normal(0, 0.01, Xtr.shape).astype(np.float32)
        indep, _ = toy((400, 1, 1), seed=3)
        assert dcr_ratio(copy, Xtr, Xva)["ratio"] < 0.3
        assert 0.7 < dcr_ratio(indep, Xtr, Xva)["ratio"] < 1.4
        assert mia_auc(copy, Xtr[:300], Xva[:300]) > 0.9
        assert abs(mia_auc(indep, Xtr[:300], Xva[:300]) - 0.5) < 0.1

    def test_positive_control_detects_memorising_generator(self):
        Xtr, ytr = toy((600, 600, 600), seed=1)
        Xva, yva = toy((600, 600, 600), seed=2)

        def memoriser(X, y, seed):
            return X + np.random.default_rng(seed).normal(0, 0.01, X.shape).astype(np.float32), y

        def independent(X, y, seed):
            return toy((len(X) // 3,) * 3, seed=seed + 100)[0], np.repeat([0, 1, 2], len(X) // 3)

        pc = positive_control(memoriser, Xtr, ytr, Xva, yva, 3, n_members=300)
        assert pc["mia_auc_mean"] > 0.9
        weak = positive_control(independent, Xtr, ytr, Xva, yva, 3, n_members=300)
        assert abs(weak["mia_auc_mean"] - 0.5) < 0.15       # a MIA that cannot separate here would be flagged as too weak

    def test_privacy_report_shape(self):
        X, y = toy((300, 300, 300))
        S, ys = toy((300, 300, 300), seed=9)
        V, yv = toy((300, 300, 300), seed=10)
        r = privacy_report(S, ys, X, y, V, yv, CLASSES)
        assert {"duplicate_rate", "dcr_ratio_mean", "mia_auc_mean", "mia_auc_per_class"} <= set(r)
        assert set(r["per_class"]) == set(CLASSES)

    def test_dcr_is_distance_to_nearest(self):
        ref = np.array([[0.0, 0.0], [10.0, 0.0]])
        assert dcr(np.array([[1.0, 0.0], [9.0, 0.0]]), ref).tolist() == [1.0, 1.0]


class TestOverhead:
    def test_timer_and_meter(self):
        with Timer() as t:
            sum(range(10000))
        assert t.seconds >= 0
        m = RoundMeter()
        for s in (1.0, 3.0):
            m.record(s)
        sm = m.summary()
        assert sm["rounds"] == 2 and sm["mean_round_s"] == 2.0 and sm["total_s"] == 4.0 and sm["peak_rss_gb"] > 0

    def test_bytes(self):
        sizes = model_bytes([np.zeros((10, 10), np.float32), np.zeros(5, np.float32)])
        assert sizes == 420
        assert bytes_per_round(sizes, 5) == 420 * 5 * 2 and bytes_per_round(sizes, 5, 2.0) == 420 * 5 * 4
        assert overhead_ratio(6.0, 2.0) == 3.0 and overhead_ratio(1.0, 0.0) == float("inf")


class TestLedger:
    def test_run_id_and_resume(self, tmp_path):
        assert run_id("B0-rf", 1) == "B0-rf_1" and run_id("B0-rf", 1, "11class") == "B0-rf_11class_1"
        led = RunLedger(tmp_path / "runs.csv")
        assert not led.done("a_0")
        led.append({"run_id": "a_0", "macro_f1": 0.5})
        led.append({"run_id": "b_0", "macro_f1": 0.6})
        led.append({"run_id": "a_0", "macro_f1": 0.7})          # re-run replaces, does not duplicate
        df = led.frame()
        assert led.done("a_0") and len(df) == 2 and df.loc[df.run_id == "a_0", "macro_f1"].iloc[0] == 0.7


def _make_processed(tmp_path, fit_rows=None):
    """Minimal processed dir for a 3-class problem (so run_baselines / write_g3_report can be smoke-tested)."""
    work = tmp_path / "work"
    d = work / "processed" / "6class"
    d.mkdir(parents=True)
    n = {"train": (300, 90, 45), "val": (60, 60, 60), "test": (100, 100, 100)}
    for i, (s, per) in enumerate(n.items()):
        X, y = toy(per, seed=i + 1)
        np.savez(d / f"{s}.npz", X=X, y=y, group_id=np.array(["g"] * len(y)), stream_id=np.array(["s"] * len(y)))
    schema = {**SCHEMA, "fit": {"split": "train", "n_rows": sum(n["train"]) if fit_rows is None else fit_rows}}
    (d / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    cfg = json.loads(json.dumps(CFG))
    cfg["paths"] = {"work_dir": str(work)}
    cfg["compute"] = {"artifacts_dir": str(tmp_path / "art"), "runs_csv": str(tmp_path / "runs.csv")}
    cfg["eval"]["bootstrap"] = 30
    return cfg


class TestBaselines:
    def test_all_baselines_run_and_resume(self, tmp_path):
        cfg = _make_processed(tmp_path)
        df = run_baselines(cfg)
        assert len(df) == len(BASELINES) * 2 and set(df["config"]) == set(BASELINES)
        smote = df[df["config"] == "B1b-rf"].iloc[0]
        assert smote["n_train"] > 435 and smote["n_synthetic"] == smote["n_train"] - 435
        assert df[df["config"] == "B0-rf"].iloc[0]["n_synthetic"] == 0
        assert {"macro_f1", "balanced_acc", "pr_auc_macro", "val_macro_f1", "recall_A", "bin_f1", "git_commit"} <= set(df.columns)
        p = load_predictions(cfg, "B0-rf_0")
        assert len(p["y_pred"]) == 300 and p["proba"].shape == (300, 3)
        again = run_baselines(cfg)
        assert len(again) == len(df)                              # nothing was re-run or duplicated

    def test_same_seed_gives_same_metrics_across_ledgers(self, tmp_path):
        cfg = _make_processed(tmp_path)
        a = run_baselines(cfg, ["B0-rf", "B1b-rf"], [0])
        cfg["compute"]["runs_csv"] = str(tmp_path / "runs2.csv")
        b = run_baselines(cfg, ["B0-rf", "B1b-rf"], [0])
        assert a["macro_f1"].tolist() == b["macro_f1"].tolist()

    def test_refuses_preprocessor_not_fitted_on_train(self, tmp_path):
        cfg = _make_processed(tmp_path, fit_rows=999)
        with pytest.raises(RuntimeError, match="fitted on the train split"):
            run_baselines(cfg, ["B0-rf"], [0])
        with pytest.raises(RuntimeError):
            check_fit_on_train({}, {"train": {"y": np.zeros(3)}})

    def test_g3_report(self, tmp_path):
        cfg = _make_processed(tmp_path)
        run_baselines(cfg, ["B0-rf", "B1a-rf", "B1b-rf"], [0, 1])
        gate = write_g3_report(cfg, tmp_path / "g3.md")
        text = (tmp_path / "g3.md").read_text(encoding="utf-8")
        for needle in ("Utility", "Recall per class", "Binary Normal vs Attack", "Bootstrap CI", "Paired difference", "Gate G3"):
            assert needle in text
        assert set(gate["std_ok"]) == {"B0-rf", "B1a-rf", "B1b-rf"} and "B0-rf" in gate["b0_not_perfect"]
