"""Follow-up tests: the model-access membership-inference attack and the protected federated classifier (tiny synthetic data)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ppfeddata import fed_protected as fp  # noqa: E402
from ppfeddata import mia_model as mm  # noqa: E402
from ppfeddata.models.cvae import CVAE, build_layout  # noqa: E402
from ppfeddata.models.train import train_cvae  # noqa: E402
from tests.test_models import HP, make_schema, synthetic_data  # noqa: E402

CLASSES = ["a", "b", "c"]


def test_halves_are_disjoint_cover_the_pool_and_depend_only_on_the_seed():
    h1, h2 = mm.halves(1001, 3)
    assert len(h1) == 500 and len(h2) == 501 and not set(h1) & set(h2) and set(h1) | set(h2) == set(range(1001))
    assert np.array_equal(h1, mm.halves(1001, 3)[0]) and not np.array_equal(h1, mm.halves(1001, 4)[0])


def test_class_auc_is_one_for_separated_scores_half_for_noise_and_skips_one_sided_classes():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 200)
    member = np.tile([True, False], 300)
    sep = np.where(member, 1.0, 0.0) + 0.01 * rng.random(600)
    assert mm.class_auc(sep, member, y, CLASSES) == {"a": 1.0, "b": 1.0, "c": 1.0}
    noise = mm.class_auc(rng.random(600), member, y, CLASSES)
    assert all(abs(v - 0.5) < 0.12 for v in noise.values())
    member[y == 2] = True                                                            # class c has no non-member
    assert set(mm.class_auc(sep, member, y, CLASSES)) == {"a", "b"}


def test_the_calibrated_score_removes_the_difficulty_of_a_record_and_both_directions_are_averaged():
    rng = np.random.default_rng(1)
    n = 3000
    y = rng.integers(0, 3, n)
    in_a = rng.random(n) < 0.5
    difficulty = rng.normal(0, 3.0, n)                                                # records differ a lot in how hard they are, for every model
    gain = 0.8                                                                        # a model fits its own members better by this much
    La = difficulty + rng.normal(0, 0.3, n) - gain * in_a
    Lb = difficulty + rng.normal(0, 0.3, n) - gain * (~in_a)
    r = mm.attack(La, Lb, in_a, y, CLASSES, ["c"])
    assert r["plain"]["auc_mean"] < 0.62 and r["calibrated"]["auc_mean"] > 0.9                                        # difficulty hides the signal until it is subtracted
    assert set(r["calibrated"]["auc_by_class"]) == set(CLASSES) and r["calibrated"]["auc_rare_mean"] == r["calibrated"]["auc_by_class"]["c"]
    assert 0.0 <= r["calibrated"]["tpr_at_0_1pct_fpr"] <= r["calibrated"]["tpr_at_1pct_fpr"] <= 1.0
    none = mm.attack(difficulty + rng.normal(0, 0.3, n), difficulty + rng.normal(0, 0.3, n), in_a, y, CLASSES, ["c"])        # no model fits its members better
    assert abs(none["calibrated"]["auc_mean"] - 0.5) < 0.06


def test_negative_elbo_is_deterministic_per_seed_and_lower_for_trained_records():
    schema = make_schema()
    X, y = synthetic_data(600)
    m, _ = train_cvae(X[:300], y[:300], None, None, 3, build_layout(schema), {**HP, "batch_size": 32, "beta_warmup_epochs": 0}, seed=0, epochs=150, patience=None)
    a, b = mm.neg_elbo(m, X, y, seed=5), mm.neg_elbo(m, X, y, seed=5)
    assert np.array_equal(a, b) and a.shape == (600,) and np.isfinite(a).all() and not np.array_equal(a, mm.neg_elbo(m, X, y, seed=6))
    assert a[:300].mean() < a[300:].mean()                                                                         # an over-fitted model fits its training rows better
    assert isinstance(m, CVAE)


def test_the_attack_detects_an_overfitted_cvae_and_not_a_model_that_saw_both_halves():
    schema = make_schema()
    X, y = synthetic_data(1200, seed=2)
    lay, hp = build_layout(schema), {**HP, "batch_size": 32, "beta_warmup_epochs": 0}
    h = np.arange(1200)
    ma, _ = train_cvae(X[h < 120], y[h < 120], None, None, 3, lay, hp, seed=0, epochs=400, patience=None)
    mb, _ = train_cvae(X[(h >= 120) & (h < 240)], y[(h >= 120) & (h < 240)], None, None, 3, lay, hp, seed=1, epochs=400, patience=None)
    idx = h < 240
    in_a = h[idx] < 120
    La, Lb = mm.neg_elbo(ma, X[idx], y[idx]), mm.neg_elbo(mb, X[idx], y[idx])
    r = mm.attack(La, Lb, in_a, y[idx], CLASSES, ["c"])
    assert r["calibrated"]["auc_mean"] > mm.DETECT                                                                  # the positive control: it detects the over-fitted model
    same = mm.attack(La, La + np.random.default_rng(0).normal(0, 1e-3, len(La)), in_a, y[idx], CLASSES, ["c"])
    assert abs(same["calibrated"]["auc_mean"] - 0.5) < 0.1                                                          # two identical models: nothing to tell members apart


def test_mean_rows_averages_over_seeds_and_keeps_the_per_seed_rows():
    def row(a, c):
        return {"plain": {"auc_by_class": {"a": a}, "auc_mean": a, "auc_rare_mean": a}, "calibrated": {"auc_by_class": {"a": c}, "auc_mean": c, "auc_rare_mean": None,
                                                                                                        "tpr_at_1pct_fpr": 0.1, "tpr_at_0_1pct_fpr": 0.0}, "eps_max": 4.9}
    out = mm._mean_rows([row(0.5, 0.6), row(0.7, 0.8)])
    assert out["plain"]["auc_mean"]["mean"] == pytest.approx(0.6) and out["calibrated"]["auc_mean"]["std"] == pytest.approx(0.1) and "auc_rare_mean" not in out["calibrated"]
    assert out["eps_max"] == 4.9 and len(out["per_seed"]) == 2 and out["calibrated"]["auc_by_class"] == {"a": pytest.approx(0.7)}


# --------------------------------------------------------------------------------------------------
# protected direct classifier
# --------------------------------------------------------------------------------------------------
def test_quantised_average_is_the_weighted_average_up_to_the_quantisation_step():
    rng = np.random.default_rng(0)
    sizes = [500, 3000, 20000]
    states = [{"w": torch.tensor(rng.normal(0, 2, (4, 3)), dtype=torch.float32), "b": torch.tensor(rng.normal(0, 2, 5), dtype=torch.float32)} for _ in sizes]
    exact = {k: sum(s[k].double() * n for s, n in zip(states, sizes)) / sum(sizes) for k in states[0]}
    q = fp.quantised_average(states, sizes, clip=16.0, max_weight=100000.0, seed=0)
    bound = len(sizes) * 2 * 16.0 / 2 ** 22 / (sum(sizes) / 100000.0)
    for k in exact:
        assert (q[k].double() - exact[k]).abs().max() < bound + 1e-6 and q[k].dtype == torch.float32
    assert not torch.equal(q["w"], fp.quantised_average(states, sizes, seed=1)["w"])                         # stochastic rounding: another seed, another rounding
    assert torch.equal(q["w"], fp.quantised_average(states, sizes, clip=16.0, max_weight=100000.0, seed=0)["w"])
    big = [{"w": torch.full((2,), 50.0)}] * 2                                                                  # weights beyond clipping_range are clipped, as in Flower
    assert fp.quantised_average(big, [100000, 100000], clip=16.0)["w"].max() < 17.0


def test_local_update_dp_counts_steps_and_changes_the_weights():
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.normal(size=(300, 6)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, 3, 300), dtype=torch.long)
    st = {a: b.clone() for a, b in fp.make_mlp(6, 3, (8, 4)).state_dict().items()}
    w = torch.ones(3)
    plain, n1 = fp.local_update(st, X, y, 6, 3, (8, 4), 2, w, 1e-2, 0)
    dp, n2 = fp.local_update(st, X, y, 6, 3, (8, 4), 2, w, 1e-2, 0, sigma=1.0, clip=1.0)
    assert n1 == 2 * 2 and n2 == 2 * 2 and not torch.equal(plain["0.weight"], st["0.weight"]) and not torch.equal(dp["0.weight"], plain["0.weight"])    # ceil(300/256) = 2 steps per epoch
    again, _ = fp.local_update(st, X, y, 6, 3, (8, 4), 2, w, 1e-2, 0, sigma=1.0, clip=1.0)
    assert all(torch.equal(dp[k], again[k]) for k in dp)                                                          # same seed, same noise


# --------------------------------------------------------------------------------------------------
# text and acceptance built from the two result files
# --------------------------------------------------------------------------------------------------
def _stat(mean, std=0.01):
    return {"mean": mean, "std": std}


def _entry(cal, plain=0.5, rare=None, eps=None):
    e = {"plain": {"auc_mean": _stat(plain), "auc_by_class": {"a": plain}},
         "calibrated": {"auc_mean": _stat(cal), "auc_rare_mean": _stat(rare if rare is not None else cal), "tpr_at_1pct_fpr": _stat(0.02), "auc_by_class": {"a": cal, "b": cal}}}
    if eps is not None:
        e["eps_max"] = eps
    return e


def fake_mia(overfit=0.68, b3=0.52, dp=0.50):
    return {"classes": ["a", "b"], "rare": ["b"], "samples": 8, "detect_threshold": 0.55,
            "controls": {"overfit_500": {**_entry(overfit), "n_per_model": 500, "epochs": 1500}, "mild_5000": {**_entry(0.53), "n_per_model": 5000, "epochs": 100}},
            "configs": {"B3": _entry(b3), "M1-eps10": _entry(dp, eps=9.99), "M1-eps5": _entry(dp, eps=5.0), "M1-eps1": _entry(dp, eps=1.0)}}


def test_mia_text_follows_the_numbers_and_degrades_without_the_file():
    from ppfeddata import interpret_report as ir
    assert ir._mia_facts({}) is None and ir._mia_clause({}) is None and ir._mia_block({}) == []
    R = {"mia_model": fake_mia(), "R4": {"positive_control": {"overfit_cvae_auc": 0.519}}}
    c = ir._mia_clause(R)
    assert "passes its positive control (AUC 0.680 on the over-fitted CVAE against 0.519" in c and "B3 0.520" in c and "detects nothing" in c
    blk = "\n".join(ir._mia_block(R))
    assert "overfit_500: 500 rows per model, 1500 epochs" in blk and "| B3 | infinity |" in blk and "| M1-eps1 | 1.000 |" in blk and "no leakage is detected, including for the non-private model B3" in blk
    leak = "\n".join(ir._mia_block({"mia_model": fake_mia(b3=0.62, dp=0.51)}))
    assert "leakage is detected in B3" in leak and "decreases with epsilon, as DP predicts" in leak
    fail = ir._mia_clause({"mia_model": fake_mia(overfit=0.51)})
    assert "does not pass its positive control either" in fail
    assert ir._mia_clause({"mia_model": {**fake_mia(), "controls": {}}}) is None


def test_d6c_passes_through_the_model_access_attack_only_when_it_passes_its_own_control():
    from ppfeddata import acceptance as ac
    old = {"available": True, "overfit_cvae_auc": 0.519, "overfit_cvae_detected": False, "copier_auc": {"0.0": 0.977}, "copier_detected_at_zero_noise": True, "threshold": 0.55}
    assert ac.check_positive_control({"R4": {"positive_control": old}}).status == "FAIL"
    r = ac.check_positive_control({"R4": {"positive_control": old}, "mia_model": fake_mia()})
    assert r.status == "PASS" and "does NOT" in r.evidence and "0.680" in r.evidence and "no leakage detected in any" in r.evidence
    assert ac.check_positive_control({"R4": {"positive_control": old}, "mia_model": fake_mia(overfit=0.50)}).status == "FAIL"
    assert "leakage detected in B3" in ac.check_positive_control({"R4": {"positive_control": old}, "mia_model": fake_mia(b3=0.6)}).evidence


def test_protected_direct_classifier_table_and_reading():
    from ppfeddata import interpret_report as ir

    def cmp(d, e):
        return {"delta": d, "lo": d - 0.02, "hi": d + 0.02, "effect": e}
    labs = {"FedMLPcwT": ("B3-TSTR-mlp", "none", "none"), "FedMLPcwT-dp5": ("M1d-t21-eps5-plain-TSTR-mlp", "better", "DP eps 5")}
    P = {"delta": 1e-5, "protection": {"FedMLPcwT": "none", "FedMLPcwT-dp5": "DP eps 5"}, "pairs": {k: v[0] for k, v in labs.items()},
         "classifiers": {"FedMLPcwT": {"macro_f1_mean": 0.44, "macro_f1_std": 0.01, "rare_recall_mean": 0.3}, "FedMLPcwT-dp5": {"macro_f1_mean": 0.36, "macro_f1_std": 0.02, "rare_recall_mean": 0.2,
                                                                                                                           "dp": {"eps_max_over_seeds": 4.998}}},
         "references": {"B3-TSTR-mlp": {"macro_f1_mean": 0.42, "macro_f1_std": 0.01, "rare_recall_mean": 0.37}, "M1d-t21-eps5-plain-TSTR-mlp": {"macro_f1_mean": 0.26, "macro_f1_std": 0.02, "rare_recall_mean": 0.12}},
         "comparisons": {"FedMLPcwT": {"against": "B3-TSTR-mlp", "macro_f1": cmp(0.02, "none"), "rare_recall": cmp(-0.07, "worse")},
                         "FedMLPcwT-dp5": {"against": "M1d-t21-eps5-plain-TSTR-mlp", "macro_f1": cmp(0.10, "better"), "rare_recall": cmp(0.08, "better")}}}
    R = {"fed_classifier": {"protected": P, "tuning": {"non_dp": {"chosen": {"lr": 0.003, "hidden": [256, 128]}}, "dp": {"chosen": {"lr": 0.005, "clip": 2.0}}}}}
    txt = "\n".join(ir._fed_protected(R))
    assert "0.003, [256, 128]" in txt and "| DP eps 5 | 0.360 ± 0.020 | 4.998 |" in txt and "above the CVAE route in 1, not different in 1 and below in 0" in txt and "DP eps 5: +0.100" in txt
    assert "above the CVAE route with the same protection in 1 of 2 cases, not different in 1 and below in 0" in ir._fed_prot_clause(R)
    assert ir._fed_protected({}) == [] and ir._fed_prot_clause({"fed_classifier": {"protected": {"comparisons": {}}}}) is None
