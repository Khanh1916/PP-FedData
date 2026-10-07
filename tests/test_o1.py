"""Optimisation O1: DP residual statistics (accounting, sensitivity, aggregation), class weights, the full-scale search helpers."""
import warnings

import numpy as np
import pytest
import torch

from ppfeddata.eval.dp_check import epsilon_independent
from ppfeddata.fl import core, dp_stats, dp_utils

warnings.filterwarnings("ignore", message="Optimal order")


@pytest.mark.parametrize("eps,frac", [(1.0, 0.02), (5.0, 0.05), (10.0, 0.1)])
def test_statistics_release_alone_costs_its_share_of_epsilon(eps, frac):
    s = dp_utils.stat_noise_multiplier(eps, frac, 1e-5)
    alone = dp_utils.epsilon(0.0, 1.0, 0, 1e-5, s)
    assert alone <= frac * eps and alone == pytest.approx(frac * eps, rel=2e-3)


@pytest.mark.parametrize("eps", [1.0, 5.0, 10.0])
def test_training_noise_is_calibrated_for_the_composition(eps):
    n, bs, rounds, le = 4000, 512, 30, 2
    ss = dp_utils.stat_noise_multiplier(eps, 0.05, 1e-5)
    sig = dp_utils.calibrate_sigma(eps, 1e-5, n, bs, rounds, le, ss)
    q, T = dp_utils.sample_rate(n, bs), dp_utils.planned_steps(n, bs, rounds, le)
    total = dp_utils.epsilon(sig, q, T, 1e-5, ss)
    assert total <= eps and total == pytest.approx(eps, rel=5e-3)
    assert dp_utils.epsilon(sig, q, T, 1e-5) < total                  # the statistics are counted, not free
    # independent recomputation (own RDP of the sampled Gaussian plus a / (2 sigma^2), integer orders) agrees
    assert epsilon_independent(sig, q, T, 1e-5, orders=range(2, 1025), sigma_stat=ss) == pytest.approx(total, rel=0.01)


def test_without_statistics_the_accounting_is_unchanged():
    assert dp_utils.epsilon(1.2, 0.01, 3000, 1e-5) == dp_utils.epsilon(1.2, 0.01, 3000, 1e-5, None)
    assert dp_utils.calibrate_sigma(5.0, 1e-5, 4000, 512, 30, 2) == dp_utils.calibrate_sigma(5.0, 1e-5, 4000, 512, 30, 2, None)


def test_epsilon_table_includes_the_release():
    ss = 20.0
    t0 = dp_utils.epsilon_table([1000, 3000], [1.5, 1.5], {0: 240, 1: 720}, 256, 1e-5, 5.0)
    t1 = dp_utils.epsilon_table([1000, 3000], [1.5, 1.5], {0: 240, 1: 720}, 256, 1e-5, 5.0, ss)
    assert t1["eps_max"] > t0["eps_max"] and t1["sigma_stat"] == ss and "sigma_stat" not in t0


def test_one_record_changes_the_release_by_at_most_the_sensitivity():
    rng = np.random.default_rng(0)
    C = 1.5
    r = rng.normal(0, 3, (200, 7))
    r = r * np.minimum(1, C / np.linalg.norm(r, axis=1, keepdims=True))
    y = rng.integers(0, 3, 200)
    a = dp_stats.client_release(r, y, 3, 0.0, C, rng)
    b = dp_stats.client_release(r[1:], y[1:], 3, 0.0, C, rng)
    diff = np.sqrt(((a["S"] - b["S"]) ** 2).sum() + ((a["N"] - b["N"]) ** 2).sum())
    assert diff <= dp_stats.sensitivity(C) + 1e-12


def test_noise_free_aggregate_is_the_rms_of_the_clipped_residuals_over_clients():
    rng = np.random.default_rng(1)
    r1, r2 = rng.normal(0, 0.3, (300, 4)), rng.normal(0, 0.3, (100, 4))
    y1, y2 = rng.integers(0, 2, 300), rng.integers(0, 2, 100)
    std = dp_stats.aggregate_std([dp_stats.client_release(r1, y1, 2, 0.0, 10.0, rng), dp_stats.client_release(r2, y2, 2, 0.0, 10.0, rng)], 10.0)
    r, y = np.vstack([r1, r2]), np.concatenate([y1, y2])
    for c in range(2):
        assert np.allclose(std[c], np.sqrt((r[y == c] ** 2).mean(axis=0)))


def test_noisy_aggregate_is_capped_and_never_nan():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.1, (5, 3))
    y = np.zeros(5, dtype=int)
    std = dp_stats.aggregate_std([dp_stats.client_release(r, y, 2, 50.0, 1.0, rng)], 1.0)    # huge noise, class 1 empty
    assert np.isfinite(std).all() and (std >= 0).all() and (std <= 1.0).all()


def test_noise_share_scales_the_variance():
    rng = np.random.default_rng(3)
    r, y = np.zeros((10, 2)), np.zeros(10, dtype=int)
    full = np.std([dp_stats.client_release(r, y, 1, 2.0, 1.0, rng)["S"][0, 0] for _ in range(4000)])
    share = np.std([dp_stats.client_release(r, y, 1, 2.0, 1.0, rng, noise_share=0.25)["S"][0, 0] for _ in range(4000)])
    assert full == pytest.approx(2.0 * dp_stats.sensitivity(1.0), rel=0.05) and share == pytest.approx(full / 2, rel=0.06)


def test_clipped_residuals_have_bounded_norm():
    from ppfeddata.models.cvae import CVAE, Layout
    lay = Layout(D=5, n_num=5, n_bin=0, groups=())
    m = CVAE(lay, 3, 4, (16, 8))
    X = np.random.default_rng(0).normal(0, 5, (50, 5)).astype(np.float32)
    r = dp_stats.clipped_residuals(m, X, np.arange(50) % 3, 0.7)
    assert r.shape == (50, 5) and (np.linalg.norm(r, axis=1) <= 0.7 + 1e-9).all()


def test_class_weights():
    y = np.array([0] * 90 + [1] * 9 + [2])
    assert core.class_weights(y, 4, 0.0) is None
    w = core.class_weights(y, 4, 1.0).numpy()
    cnt = np.bincount(y, minlength=4)
    assert w[3] == 0 and (w * cnt).sum() / cnt.sum() == pytest.approx(1.0, rel=1e-6)
    assert w[1] / w[0] == pytest.approx(10.0, rel=1e-5)
    assert w[2] / w[0] == pytest.approx(core.CW_MAX / (100 / (3 * 90)), rel=1e-5)          # class 2 (1 record) hits the cap
    half = core.class_weights(y, 4, 0.5).numpy()
    assert half[1] / half[0] == pytest.approx(np.sqrt(10.0), rel=1e-5)


def test_weighted_loss_reduces_to_the_plain_loss_with_unit_weights():
    from ppfeddata.models.cvae import CVAE, Layout, loss_terms, one_hot
    lay = Layout(D=3, n_num=3, n_bin=0, groups=())
    m = CVAE(lay, 2, 2, (8, 4))
    x, y = torch.randn(6, 3), torch.tensor([0, 1, 0, 1, 0, 1])
    out, mu, lv = m(x, one_hot(y, 2))
    a = loss_terms(out, x, mu, lv, 0.5, lay)["loss"]
    b = loss_terms(out, x, mu, lv, 0.5, lay, torch.ones(6))["loss"]
    assert torch.allclose(a, b)


def test_search_helpers_map_parameters_to_run_arguments():
    from ppfeddata import tune_dp_full as t
    kw = t.run_kwargs({"latent_dim": 16, "hidden": [128, 64], "beta": 0.5, "lr": 1e-3, "batch_size": 256, "beta_warmup_epochs": 10}, t.ANCHOR, 5.0)
    assert kw["target_eps"] == 5.0 and kw["rounds"] == 30 and kw["local_epochs"] == 2 and kw["stat_frac"] == 0.05
    assert kw["hp"]["hidden"] == [128, 64] and kw["hp"]["cw_power"] == 0.0 and kw["hp"]["beta_warmup_epochs"] == 10
    assert set(t.ANCHOR) == set(t.SPACE)
    assert t.trial_name(10.0, 3) == "O1-eps10-t3"


def test_search_prunes_width_256_and_keeps_the_space():
    from ppfeddata import tune_dp_full as t
    assert t.skip_reason(t.ANCHOR) is None
    assert "width 256" in t.skip_reason({**t.ANCHOR, "width": 256})
    assert "batch 2048" in t.skip_reason({**t.ANCHOR, "width": 128, "batch_size": 2048})
    assert t.skip_reason({**t.ANCHOR, "width": 64, "batch_size": 2048}) is None
    assert 256 in t.SPACE["width"]  # unchanged: existing studies reject a different categorical distribution


def test_search_loop_counts_only_complete_trials():
    import optuna
    from ppfeddata import tune_dp_full as t
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))

    def objective(trial):
        w = trial.suggest_categorical("width", [64, 256])
        if w == 256:
            raise optuna.TrialPruned()
        return 1.0

    while t.n_complete(study) < 5 and len(study.trials) < 15:
        study.optimize(objective, n_trials=1)
    assert t.n_complete(study) == 5 and len(study.trials) >= 5
