"""Phase 9 tests: privacy accounting, DP-SGD client training, DP checkpoint/resume through Flower."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("opacus")
pytest.importorskip("flwr")

from tests.test_fl import _toy_cfg_and_data, _toy_state_and_data  # noqa: E402
from tests.test_models import HP  # noqa: E402

from ppfeddata.fl import core, dp_utils  # noqa: E402


# ----------------------------------------------------------------------------- accounting
def test_sigma_grows_as_target_epsilon_shrinks_and_epsilon_matches_target():
    sig = {e: dp_utils.calibrate_sigma(e, 1e-5, 7000, 256, 30, 2) for e in (1, 5, 10)}
    assert sig[1] > sig[5] > sig[10] > 0
    q, steps = dp_utils.sample_rate(7000, 256), dp_utils.planned_steps(7000, 256, 30, 2)
    for e, s in sig.items():
        got = dp_utils.epsilon(s, q, steps, 1e-5)
        assert 0.97 * e <= got <= e + 1e-9, (e, got)                       # calibrated from below, within the search tolerance


def test_epsilon_properties():
    q = dp_utils.sample_rate(5000, 256)
    assert dp_utils.epsilon(1.0, q, 0, 1e-5) == 0.0
    assert dp_utils.epsilon(0.0, q, 100, 1e-5) == float("inf")
    e1, e2, e3 = (dp_utils.epsilon(1.2, q, t, 1e-5) for t in (100, 400, 1600))
    assert e1 < e2 < e3                                                    # more steps spend more budget
    assert dp_utils.epsilon(2.0, q, 400, 1e-5) < dp_utils.epsilon(1.0, q, 400, 1e-5)
    assert dp_utils.epsilon(1.2, 2 * q, 400, 1e-5) > dp_utils.epsilon(1.2, q, 400, 1e-5)


def test_delta_must_be_below_one_over_n():
    dp_utils.check_delta([500, 40000], 1e-5)
    with pytest.raises(ValueError):
        dp_utils.check_delta([500, 150000], 1e-5)


def test_sampling_rate_and_steps_match_a_real_opacus_loader():
    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader, TensorDataset
    for n, bs in ((1000, 256), (777, 128), (256, 256)):
        m = torch.nn.Linear(3, 1)
        loader = DataLoader(TensorDataset(torch.zeros(n, 3), torch.zeros(n)), batch_size=bs)
        _, _, dl = PrivacyEngine().make_private(module=m, optimizer=torch.optim.SGD(m.parameters(), lr=0.1), data_loader=loader,
                                                noise_multiplier=1.0, max_grad_norm=1.0, poisson_sampling=True)
        assert len(dl) == dp_utils.steps_per_epoch(n, bs)
        assert dl.sample_rate == pytest.approx(dp_utils.sample_rate(n, bs))


def test_epsilon_table_is_max_over_clients_and_flags_the_two_percent_rule():
    sizes, bs = [1000, 4000], 256
    sig = dp_utils.calibrate_clients(sizes, 5.0, 1e-5, bs, 10, 2)
    steps = {i: dp_utils.planned_steps(n, bs, 10, 2) for i, n in enumerate(sizes)}
    t = dp_utils.epsilon_table(sizes, sig, steps, bs, 1e-5, 5.0)
    assert t["eps_max"] == max(r["epsilon"] for r in t["rows"]) and t["eps_median"] <= t["eps_max"]
    assert t["within_2pct"] and t["max_ratio_to_target"] <= 1.0
    half = dp_utils.epsilon_table(sizes, sig, {i: s // 2 for i, s in steps.items()}, bs, 1e-5, 5.0)
    assert half["eps_max"] < t["eps_max"]                                  # fewer steps (an interrupted run) -> smaller epsilon
    string_keys = dp_utils.epsilon_table(sizes, sig, {str(i): s for i, s in steps.items()}, bs, 1e-5, 5.0)
    assert string_keys["eps_max"] == t["eps_max"]                         # the JSON round trip turns keys into strings


# ----------------------------------------------------------------------------- DP client training
def test_local_train_dp_is_deterministic_counts_steps_and_keeps_state_keys():
    lay, X, y, st = _toy_state_and_data()
    a = core.local_train_dp(st, X, y, lay, 3, HP, 2, 1, 0, 0, sigma=1.0, max_grad_norm=1.0)
    b = core.local_train_dp(st, X, y, lay, 3, HP, 2, 1, 0, 0, sigma=1.0, max_grad_norm=1.0)
    assert all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"])
    assert a["steps"] == 2 * dp_utils.steps_per_epoch(len(X), HP["batch_size"])
    assert list(a["state"]) == list(st) and not any(k.startswith("_module") for k in a["state"])
    c = core.local_train_dp(st, X, y, lay, 3, HP, 2, 2, 0, 0, sigma=1.0, max_grad_norm=1.0)
    assert not torch.equal(a["state"]["out.weight"], c["state"]["out.weight"])             # round changes the noise


def test_noise_hurts_and_zero_noise_matches_plain_training():
    lay, X, y, st = _toy_state_and_data()
    Xv, yv = X[:300], y[:300]
    plain = core.local_train(st, X, y, lay, 3, HP, 6, 1, 0, 0)
    quiet = core.local_train_dp(st, X, y, lay, 3, HP, 6, 1, 0, 0, sigma=0.0, max_grad_norm=1e6)
    loud = core.local_train_dp(st, X, y, lay, 3, HP, 6, 1, 0, 0, sigma=30.0, max_grad_norm=1.0)
    v = {n: core.val_loss(o["state"], Xv, yv, lay, 3, HP)["loss"] for n, o in (("plain", plain), ("quiet", quiet), ("loud", loud))}
    v0 = core.val_loss(st, Xv, yv, lay, 3, HP)["loss"]
    assert v["plain"] < v0 and v["quiet"] < v0
    assert abs(v["quiet"] - v["plain"]) / v["plain"] < 0.25                # sigma = 0, no clipping ~ ordinary training (Poisson batches)
    assert v["loud"] > v["quiet"] * 1.2                                    # heavy noise leaves a clearly worse model


def test_clipping_bounds_the_per_sample_gradient_norm():
    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader, TensorDataset

    from ppfeddata.models.cvae import loss_terms, one_hot
    lay, X, y, st = _toy_state_and_data()
    model = core.make_model(lay, 3, HP)
    model.load_state_dict(st)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=64)
    gs, dopt, dl = PrivacyEngine().make_private(module=model, optimizer=opt, data_loader=loader, noise_multiplier=0.0,
                                                max_grad_norm=0.05, poisson_sampling=False)
    xb, yb = next(iter(dl))
    out, mu, lv = gs(xb, one_hot(yb, 3))
    loss_terms(out, xb, mu, lv, 0.5, lay)["loss"].backward()
    raw = torch.stack([p.grad_sample.reshape(len(xb), -1) for p in gs.parameters()], dim=0) if False else None
    norms = torch.sqrt(sum((p.grad_sample.reshape(len(xb), -1) ** 2).sum(dim=1) for p in gs.parameters()))
    assert float(norms.max()) > 0.05 * 2                                   # unclipped norms are larger than the bound ...
    dopt.pre_step()
    clipped = torch.sqrt(sum((p.grad.reshape(-1) ** 2).sum() for p in gs.parameters())) * len(xb)   # ... sum of clipped grads
    assert float(clipped) <= 0.05 * len(xb) + 1e-4                         # <= batch size x C (triangle inequality)


# ----------------------------------------------------------------------------- Flower, DP, checkpoint / resume
def test_dp_flower_run_epsilon_and_resume(tmp_path):
    from ppfeddata.fl.run import run_fl
    cfg, ddir = _toy_cfg_and_data(tmp_path)
    cfg["dp"] = {"epsilons": [5], "delta": 1e-5, "max_grad_norm": 1.0}
    kw = {"data_dir": ddir, "hp": HP, "target_eps": 5.0}
    full = run_fl(cfg, 0, "dpfull", rounds=6, **kw)
    part = run_fl(cfg, 0, "dpcut", rounds=6, stop_after=4, **kw)          # interrupted after 4 of the 6 planned rounds
    assert part["summary"]["rounds_done"] == 4
    with pytest.raises(RuntimeError):                                      # a changed plan changes sigma: refuse to resume
        run_fl(cfg, 0, "dpcut", rounds=8, resume=True, **kw)
    cut = run_fl(cfg, 0, "dpcut", rounds=6, resume=True, **kw)
    art = tmp_path / "art"
    s_full = torch.load(art / full["run_id"] / "final_state.pt", weights_only=False)
    s_cut = torch.load(art / cut["run_id"] / "final_state.pt", weights_only=False)
    assert max(float((s_full[k] - s_cut[k]).abs().max()) for k in s_full) < 1e-5
    ef, ec = full["summary"]["dp"], cut["summary"]["dp"]
    assert ef["eps_max"] == pytest.approx(ec["eps_max"]) and ef["rows"] == ec["rows"]           # same steps -> same epsilon after resume
    assert ef["within_2pct"] and ef["eps_max"] <= 1.02 * 5.0
    assert [r["steps"] for r in ef["rows"]] == [6 * dp_utils.steps_per_epoch(n, HP["batch_size"]) for n in full["partition_sizes"]]
    sig = full["dp"]["sigmas"]
    assert len(sig) == 3 and all(s > 0 for s in sig)
    ck = core.load_checkpoint(art / cut["run_id"])
    assert ck["dp_steps"] == {i: r["steps"] for i, r in enumerate(ef["rows"])}                  # counters live in the checkpoint


def test_a_round_with_a_missing_client_aborts():
    from types import SimpleNamespace

    from ppfeddata.fl.app import require_all_replies
    ok, bad = SimpleNamespace(has_error=lambda: False), SimpleNamespace(has_error=lambda: True)
    require_all_replies([ok, ok, ok], 3, 1)
    with pytest.raises(RuntimeError, match="2 of 3"):
        require_all_replies([ok, ok, bad], 3, 7)
    with pytest.raises(RuntimeError, match="2 of 3"):
        require_all_replies([ok, ok], 3, 7)


# ----------------------------------------------------------------------------- DP-specific hyper-parameter search
def test_proxy_subsample_is_stratified_and_sorted():
    from ppfeddata.tune_dp import proxy_subsample
    y = np.repeat(np.arange(4), [1000, 400, 50, 5])
    idx = proxy_subsample(y, 4, 0.2, seed=0)
    assert np.array_equal(idx, np.sort(idx)) and len(np.unique(idx)) == len(idx)
    assert np.bincount(y[idx], minlength=4).tolist() == [200, 80, 10, 1]            # proportions kept, rare class keeps >= 1 row
    assert np.array_equal(idx, proxy_subsample(y, 4, 0.2, seed=0))


def test_hp_from_params_overrides_only_the_searched_keys():
    from ppfeddata.tune_dp import hp_from_params
    base = {"latent_dim": 16, "hidden": [256, 128], "beta": 0.4, "lr": 1e-3, "batch_size": 256, "beta_warmup_epochs": 10, "epochs": 30, "patience": 5}
    hp, clip = hp_from_params(base, {"latent_dim": 8, "width": 64, "batch_size": 512, "beta": 0.2, "lr": 4e-3, "max_grad_norm": 7.5})
    assert hp["hidden"] == [64, 32] and hp["batch_size"] == 512 and hp["latent_dim"] == 8 and clip == 7.5
    assert hp["beta_warmup_epochs"] == 10 and hp["epochs"] == 30 and base["hidden"] == [256, 128]      # base untouched


def test_dp_trial_runs_end_to_end_on_toy_data():
    from ppfeddata.tune_dp import dp_trial, proxy_subsample
    from tests.test_models import make_schema, synthetic_data
    schema = make_schema()
    schema["label_map"] = {"c0": 0, "c1": 1, "c2": 2}
    Xt, yt = synthetic_data(900, 0)
    Xv, yv = synthetic_data(300, 1)
    data = {"train": {"X": Xt, "y": yt}, "val": {"X": Xv, "y": yv}}
    cfg = {"fl": {"rounds": 3, "local_epochs": 1}, "dp": {"delta": 1e-5}, "tune": {"syn_per_class": 100, "rf_trees": 10}}
    hp = {**HP, "batch_size": 128}
    sub = proxy_subsample(yt, 3, 0.5)
    r = dp_trial(cfg, hp, 2.0, 5.0, data, schema, sub, seed=0)
    assert 0.0 <= r["macro_f1"] <= 1.0 and r["sigma"] > 0 and r["val_loss"] > 0
    assert r["eps_check"] <= 5.0 * 1.0001                                              # sigma was calibrated for the whole plan
    assert r["steps"] == 3 * dp_utils.steps_per_epoch(len(sub), 128)
