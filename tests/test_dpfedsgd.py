"""Optimisation O2(a): DP-FedSGD with the noise split over the clients (M3-distributed): accounting, clipping, SecAgg quantisation, training."""
import math
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ppfeddata.fl import dp_utils, dpfedsgd as fs  # noqa: E402
from ppfeddata.models.cvae import Layout  # noqa: E402

warnings.filterwarnings("ignore", message="Optimal order")
LAYOUT = Layout(D=6, n_num=3, n_bin=1, groups=((4, 2),))
HP = {"latent_dim": 2, "hidden": [16, 8], "beta": 0.5, "beta_warmup_epochs": 1, "lr": 5e-3, "batch_size": 64, "cw_power": 0.0}


def _data(n=600, k=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    X = np.zeros((n, 6), np.float32)
    X[:, :3] = rng.normal(size=(n, 3)) + y[:, None]
    X[:, 3] = (y > 0)
    X[np.arange(n), 4 + (y % 2)] = 1.0
    return X, y.astype(np.int64)


@pytest.mark.parametrize("eps,frac", [(1.0, 0.0), (5.0, 0.05)])
def test_noise_is_calibrated_for_rounds_and_the_statistics(eps, frac):
    q, T, delta = 0.02, 500, 1e-5
    ss = dp_utils.stat_noise_multiplier(eps, frac, delta) if frac else None
    s = fs.calibrate(eps, delta, q, T, ss)
    r = fs.epsilon_report(s, q, T, delta, 5, ss)
    assert eps * 0.97 <= r["eps_all_honest"] <= eps
    assert r["eps_one_honest"] > 2 * eps                     # one honest client: only its sigma / sqrt(5) share protects it


def test_the_old_calibration_is_unchanged_by_the_refactoring():
    ss = dp_utils.stat_noise_multiplier(5.0, 0.05, 1e-5)
    a = dp_utils.calibrate_sigma(5.0, 1e-5, 4000, 512, 30, 2, ss)
    assert a == pytest.approx(2.7715, abs=2e-3)               # value recorded in SPEC_DEVIATIONS O1.1


def test_per_sample_gradients_are_clipped_before_the_sum():
    from ppfeddata.fl import core
    X, y = _data(64)
    model = core.make_model(LAYOUT, 3, HP)
    params = {k: v.detach() for k, v in model.named_parameters()}
    g, nc = fs.per_sample_clipped_sum(model, params, torch.as_tensor(X), torch.as_tensor(y), None, 0.5, LAYOUT, 3, clip=1e-3, chunk=16)
    norm = math.sqrt(sum(float(v.pow(2).sum()) for v in g.values()))
    assert nc == 64 and norm <= 64 * 1e-3 + 1e-9
    g2, nc2 = fs.per_sample_clipped_sum(model, params, torch.as_tensor(X), torch.as_tensor(y), None, 0.5, LAYOUT, 3, clip=1e6)
    assert nc2 == 0 and math.sqrt(sum(float(v.pow(2).sum()) for v in g2.values())) > norm


def test_quantisation_is_unbiased_bounded_and_counts_clipping():
    gen = torch.Generator().manual_seed(0)
    v = torch.full((20000,), 0.123456)
    out, over = fs.quantize(v, 1.0, 2 ** 8, gen)
    step = 2.0 / (2 ** 8 - 1)
    assert over == 0 and float((out - v).abs().max()) <= step + 1e-7 and abs(float(out.mean()) - 0.123456) < 2e-4
    out2, over2 = fs.quantize(torch.tensor([5.0, -5.0, 0.0]), 1.0, 2 ** 22, gen)
    assert over2 == 2 and float(out2.abs().max()) <= 1.0 + 1e-6


def test_m3f_family_is_parsed_and_scored_as_distributed_dp():
    import pandas as pd

    from ppfeddata import aggregate as ag
    from ppfeddata import scorecard as sc
    from tests.test_scorecard import RARE, _gen, _summary
    p = ag.parse_config("M3f-t4-eps5-plain-TSTR-mlp")
    assert p["method"] == "M3" and p["family"] == "o2 fedsgd t4" and p["eps"] == 5.0 and p["variant"] == "plain"
    rows = _gen("M3f-t4-eps5", 0.30, 0.30, eps=5.0, bytes_=1e5)
    for r in rows:
        r.update(rounds_mean=2000.0, dp_eps_one_honest_mean=60.0)
    by = {r["label"]: r for r in sc.build(pd.concat([_summary(), pd.DataFrame(rows)], ignore_index=True), rare=RARE)["rows"]}
    m = by["M3f-eps5"]
    assert m["secagg"] and m["dp_mode"] == "distributed" and m["eps_one_honest"] == 60.0 and m["bytes_total"] == 1e5 * 2000
    assert by["M3-eps5"]["dp_mode"] == "local" and by["M3-eps5"]["eps_one_honest"] == by["M3-eps5"]["eps"]
    assert "(60.0)" in sc.render(sc.build(pd.concat([_summary(), pd.DataFrame(rows)], ignore_index=True), rare=RARE))


def test_training_runs_and_learns_without_noise():
    from ppfeddata.fl import core
    X, y = _data(600)
    parts = [np.arange(0, 200), np.arange(200, 400), np.arange(400, 600)]
    torch.set_num_threads(1)
    out = fs.train(X, y, parts, LAYOUT, 3, HP, rounds=40, q=0.2, sigma=0.0, clip=10.0, seed=0, log_every=20, Xv=X[:100], yv=y[:100])
    first = core.val_loss(core.init_state(LAYOUT, 3, HP, 0), X[:100], y[:100], LAYOUT, 3, HP)["loss"]
    assert np.isfinite(out["log"][-1]["val_loss"]) and out["log"][-1]["val_loss"] < first
    assert 100 < out["rows_per_round"] < 140                  # Poisson sampling, q = 0.2 of 600 rows
    sa = {"params": {"clipping_range": 16.0, "quantization_range": 2 ** 22}}
    again = fs.train(X, y, parts, LAYOUT, 3, HP, rounds=3, q=0.2, sigma=1.0, clip=1.0, seed=0, secagg=sa, log_every=0)
    assert all(torch.isfinite(v).all() for v in again["state"].values()) and again["quant_clipped_frac"] == 0.0
