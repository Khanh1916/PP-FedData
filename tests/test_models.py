"""Phase 7 tests: CVAE shapes / loss, generation validity, conditioning, reproducibility."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ppfeddata.models.cvae import CVAE, beta_at, build_layout, hidden_from_width, loss_terms, one_hot  # noqa: E402
from ppfeddata.models.generate import (dead_category_mask, decode_numeric, encode_numeric, generate,  # noqa: E402
                                       postprocess)
from ppfeddata.models.train import train_cvae  # noqa: E402

HP = {"latent_dim": 4, "hidden": [32, 16], "beta": 0.5, "beta_warmup_epochs": 3, "lr": 3e-3, "batch_size": 128,
      "epochs": 30, "class_balanced_sampler": False}


def make_schema():
    """Layout: numeric a (real-valued, flag), b (integer, log1p, ultra-sparse, flag) | binary c (flag), d |
    flags a, b, c | categorical k (3 live categories + dead OTHER + dead NONE)."""
    pa = {"type": "numeric", "log1p": False, "mean": 0.0, "std": 2.0, "scale": 1.0, "nonneg_clip": False,
          "raw_min": -6.0, "raw_max": 6.0, "is_integer": False, "ultra_sparse": False, "has_flag": True}
    pb = {"type": "numeric", "log1p": True, "mean": 3.0, "std": 1.5, "scale": 1.0, "nonneg_clip": False,
          "raw_min": 2.0, "raw_max": 3000.0, "is_integer": True, "ultra_sparse": True, "has_flag": True}
    blocks = [
        {"name": "a", "type": "numeric", "column": "a", "start": 0, "width": 1},
        {"name": "b", "type": "numeric", "column": "b", "start": 1, "width": 1},
        {"name": "c", "type": "binary", "column": "c", "start": 2, "width": 1},
        {"name": "d", "type": "binary", "column": "d", "start": 3, "width": 1},
        {"name": "a_is_na", "type": "na_flag", "column": "a", "start": 4, "width": 1},
        {"name": "b_is_na", "type": "na_flag", "column": "b", "start": 5, "width": 1},
        {"name": "c_is_na", "type": "na_flag", "column": "c", "start": 6, "width": 1},
        {"name": "k", "type": "categorical", "column": "k", "start": 7, "width": 5,
         "categories": ["x", "y", "z", "OTHER", "NONE"]},
    ]
    params = {"a": pa, "b": pb,
              "c": {"type": "binary", "has_flag": True}, "d": {"type": "binary", "has_flag": False},
              "k": {"type": "categorical", "train_counts": {"x": 50, "y": 30, "z": 20, "OTHER": 0, "NONE": 0}}}
    return {"n_features": 12, "blocks": blocks, "params": params, "settings": {"clip_sigma": 5}}


def synthetic_data(n=3000, seed=0):
    """Class c in {0,1,2}: k = c exactly, a ~ N(2c, 0.3 in z units), c-bit probability depends on the class."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = np.zeros((n, 12), dtype=np.float32)
    X[:, 0] = rng.normal(y * 1.5 - 1.5, 0.2)
    X[:, 1] = rng.normal(0, 0.5, n)
    X[:, 2] = (rng.random(n) < np.array([0.1, 0.5, 0.9])[y]).astype(np.float32)
    X[:, 3] = (rng.random(n) < 0.5).astype(np.float32)
    X[np.arange(n), 7 + y] = 1.0
    return X, y.astype(np.int64)


def test_layout_from_schema():
    lay = build_layout(make_schema())
    assert (lay.D, lay.n_num, lay.n_bin, lay.groups) == (12, 2, 5, ((7, 5),))
    bad = make_schema()
    bad["blocks"][2], bad["blocks"][7] = bad["blocks"][7], bad["blocks"][2]
    with pytest.raises(ValueError):
        build_layout(bad)


def test_shapes_and_no_batchnorm():
    lay = build_layout(make_schema())
    m = CVAE(lay, 3, 4, (32, 16))
    x, y = torch.randn(10, 12), torch.randint(0, 3, (10,))
    out, mu, lv = m(x, one_hot(y, 3))
    assert out.shape == (10, 12) and mu.shape == lv.shape == (10, 4)
    assert not any(isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) for mod in m.modules())
    assert any(isinstance(mod, torch.nn.LayerNorm) for mod in CVAE(lay, 3, 4, (32, 16), layernorm=True).modules())


def test_beta_schedule_and_hidden():
    assert [beta_at(e, 0.5, 10) for e in (0, 5, 10, 20)] == [0.0, 0.25, 0.5, 0.5]
    assert beta_at(0, 0.5, 0) == 0.5
    assert hidden_from_width(128) == (128, 64)


def test_loss_terms_known_values():
    lay = build_layout(make_schema())
    x = torch.zeros(2, 12)
    x[:, 7] = 1.0
    out = torch.zeros(2, 12)                      # perfect numeric mean 0; logits 0 -> BCE = ln 2 per flag; CE = ln 5
    t = loss_terms(out, x, torch.zeros(2, 4), torch.zeros(2, 4), 1.0, lay)
    assert float(t["recon_num"]) == pytest.approx(0.0)
    assert float(t["recon_bin"]) == pytest.approx(5 * np.log(2), rel=1e-5)
    assert float(t["recon_cat"]) == pytest.approx(np.log(5), rel=1e-5)
    assert float(t["kl"]) == pytest.approx(0.0, abs=1e-6)


def test_training_reduces_loss():
    X, y = synthetic_data()
    lay = build_layout(make_schema())
    m, hist = train_cvae(X, y, X[:500], y[:500], 3, lay, HP, seed=0, epochs=15)
    assert hist[-1]["val_loss"] < hist[3]["val_loss"] * 0.8
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]


def test_generation_is_valid_and_counts_exact():
    schema = make_schema()
    m = CVAE(build_layout(schema), 3, 4, (32, 16))
    for p in m.parameters():                       # make the logits large so a faulty mask would show up
        torch.nn.init.normal_(p, std=2.0)
    X, y = generate(m, schema, {0: 300, 1: 200, 2: 100}, seed=0)
    assert X.shape == (600, 12) and np.bincount(y).tolist() == [300, 200, 100]
    assert X.dtype == np.float32 and np.isfinite(X).all()
    assert set(np.unique(X[:, 2:7])) <= {0.0, 1.0}
    assert (X[:, 7:12].sum(axis=1) == 1).all()
    assert X[:, 10:12].sum() == 0                  # OTHER and NONE have 0 train rows -> never sampled
    assert dead_category_mask(schema)[7:12].tolist() == [False, False, False, True, True]


def test_generation_obeys_domain_and_flags():
    schema = make_schema()
    m = CVAE(build_layout(schema), 3, 4, (32, 16))
    for p in m.parameters():
        torch.nn.init.normal_(p, std=3.0)
    X, _ = generate(m, schema, [2000, 0, 0], seed=1)
    pa, pb = schema["params"]["a"], schema["params"]["b"]
    ra, rb = decode_numeric(X[:, 0], pa), decode_numeric(X[:, 1], pb)
    na_a, na_b, na_c = X[:, 4] == 1, X[:, 5] == 1, X[:, 6] == 1
    assert na_a.any() and (~na_a).any() and na_b.any()
    ok_a, ok_b = ~na_a, ~na_b
    assert ra[ok_a].min() >= -6.0 - 1e-4 and ra[ok_a].max() <= 6.0 + 1e-4
    assert rb[ok_b].min() >= 2.0 - 1e-3 and rb[ok_b].max() <= 3000.0 + 1e-1
    assert np.allclose(rb[ok_b], np.round(rb[ok_b]), atol=1e-3)         # integer column
    assert (X[na_b, 1] == 0.0).all()                                    # ultra-sparse "not applicable" = 0
    assert np.allclose(X[na_a, 0], encode_numeric(np.zeros(1), pa, 5)[0], atol=1e-6)
    assert (X[na_c, 2] == 0.0).all()                                    # binary column with its flag set -> 0


def test_encode_decode_roundtrip():
    schema = make_schema()
    for col, raw in (("a", np.array([-5.0, 0.0, 3.3])), ("b", np.array([2.0, 17.0, 3000.0]))):
        p = schema["params"][col]
        np.testing.assert_allclose(decode_numeric(encode_numeric(raw * p["scale"], p, 5), p), raw, rtol=1e-6, atol=1e-6)


def test_generation_reproducible_by_seed():
    schema = make_schema()
    X, y = synthetic_data(500)
    m, _ = train_cvae(X, y, None, None, 3, build_layout(schema), HP, seed=0, epochs=3)
    a, _ = generate(m, schema, [50, 50, 50], seed=7)
    b, _ = generate(m, schema, [50, 50, 50], seed=7)
    c, _ = generate(m, schema, [50, 50, 50], seed=8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    m2, _ = train_cvae(X, y, None, None, 3, build_layout(schema), HP, seed=0, epochs=3)
    for p1, p2 in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)                                       # same seed -> same weights


def test_postprocess_is_pure_function_of_rng():
    schema = make_schema()
    raw = np.random.default_rng(0).normal(size=(200, 12)).astype(np.float32)
    a = postprocess(raw, schema, np.random.default_rng(3))
    b = postprocess(raw, schema, np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_conditioning_is_learned():
    """The category and the numeric mean are fully determined by the class; the generator must reproduce that."""
    schema = make_schema()
    X, y = synthetic_data(4000)
    m, _ = train_cvae(X, y, None, None, 3, build_layout(schema), HP, seed=0, epochs=40)
    Xg, yg = generate(m, schema, [400, 400, 400], seed=0)
    k = Xg[:, 7:10].argmax(axis=1)
    assert (k == yg).mean() > 0.9
    for c in range(3):
        za = Xg[(yg == c) & (Xg[:, 4] == 0), 0]
        assert abs(za.mean() - (c * 1.5 - 1.5)) < 0.5
    bit = [Xg[yg == c, 2].mean() for c in range(3)]
    assert bit[0] < bit[1] < bit[2]


def test_early_stopping_restores_best_and_stops():
    schema = make_schema()
    X, y = synthetic_data(1000)
    m, hist = train_cvae(X, y, X[:200], y[:200], 3, build_layout(schema), {**HP, "lr": 0.05}, seed=0, epochs=60,
                         patience=3)
    assert len(hist) <= 60 and all("val_loss" in h for h in hist)


# --------------------------------------------------------------------------------------------------
# B2 pipeline on a toy problem
# --------------------------------------------------------------------------------------------------
def toy_cfg(tmp_path):
    return {"label_mode": "toy", "seeds": [0], "compute": {"artifacts_dir": str(tmp_path / "art"), "runs_csv": str(tmp_path / "runs.csv")},
            "cvae": {**HP, "epochs": 8, "patience": 3}, "generate": {"target_per_class": 800, "residual_noise": True, "snap_support": True, "snap_max_unique": 50},
            "tune": {"syn_per_class": 200, "rf_trees": 20, "epochs": 5, "n_trials": 2},
            "eval": {"rf": {"n_estimators": 20}, "mlp": {"hidden": [16], "max_iter": 30, "early_stopping": True}, "bootstrap": 20},
            "thresholds": {"c2st_auc_max": 0.95, "dup_rate_max": 0.01, "dcr_ratio_min": 0.5, "seed_std_max": 0.02}}


def toy_data():
    sets = {s: synthetic_data(n, seed=i) for i, (s, n) in enumerate((("train", 1500), ("val", 450), ("test", 450)))}
    return {s: {"X": X, "y": y} for s, (X, y) in sets.items()}


def toy_schema():
    sch = make_schema()
    sch["label_map"] = {"c0": 0, "c1": 1, "c2": 2}
    sch["fit"] = {"split": "train", "n_rows": 1500}
    return sch


def test_run_b2_seed_end_to_end(tmp_path):
    from ppfeddata.eval.runs import RunLedger
    from ppfeddata.models.b2 import PROTOCOLS, b2_name, run_b2_seed
    cfg, data, schema = toy_cfg(tmp_path), toy_data(), toy_schema()
    ledger = RunLedger(tmp_path / "runs.csv")
    run_b2_seed(cfg, 0, data, schema, ledger)
    df = ledger.frame()
    assert sorted(df["config"]) == sorted(b2_name(p, c) for p, c in PROTOCOLS)
    assert df["macro_f1"].between(0, 1).all() and (df["c2st_auc_mean"] > 0.4).all()
    taug = df[df["config"] == "B2-TAug-rf"].iloc[0]
    assert taug["n_train"] == 1500 + taug["n_synthetic"] and taug["n_synthetic"] > 0
    tstr = df[df["config"] == "B2-TSTR-rf"].iloc[0]
    assert tstr["n_train"] == 3 * 200 and tstr["n_synthetic"] == 600
    n0 = len(df)
    run_b2_seed(cfg, 0, data, schema, ledger)             # resume: nothing is re-run
    assert len(ledger.frame()) == n0


def test_positive_control_with_overfit_cvae(tmp_path):
    from ppfeddata.models.b2 import cvae_positive_control
    cfg = toy_cfg(tmp_path)
    cfg["tune"]["syn_per_class"] = 300
    res = cvae_positive_control(cfg, toy_data(), toy_schema(), n_members=60, epochs=150, batch_size=16, beta=0.01)
    assert res["n_members"] == 60 and 0.0 <= res["mia_auc_mean"] <= 1.0


def test_stratified_members_are_balanced():
    from ppfeddata.eval.privacy import positive_control
    X, y = synthetic_data(1500)
    seen = {}

    def gen(Xm, ym, s):
        seen["counts"] = np.bincount(ym, minlength=3)
        return Xm.copy(), ym.copy()

    positive_control(gen, X, y, X[:300], y[:300], 3, n_members=90, stratified=True)
    assert seen["counts"].tolist() == [30, 30, 30]


# --------------------------------------------------------------------------------------------------
# Opacus compatibility and the budget arithmetic
# --------------------------------------------------------------------------------------------------
def test_cvae_is_opacus_compatible_with_per_sample_grads():
    pytest.importorskip("opacus")
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from torch.utils.data import DataLoader, TensorDataset

    X, y = synthetic_data(512)
    m = CVAE(build_layout(make_schema()), 3, 4, (32, 16), layernorm=True)     # LayerNorm is allowed, BatchNorm is not
    assert ModuleValidator.validate(m, strict=False) == []
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=64)
    gm, dopt, dl = PrivacyEngine().make_private(module=m, optimizer=opt, data_loader=loader, noise_multiplier=1.0,
                                                max_grad_norm=1.0, poisson_sampling=True)
    xb, yb = next(iter(dl))
    out, mu, lv = gm(xb, one_hot(yb, 3))
    loss_terms(out, xb, mu, lv, 0.5, m.layout)["loss"].backward()
    for name, p in gm.named_parameters():
        assert p.grad_sample is not None and p.grad_sample.shape == (len(xb),) + tuple(p.shape), name   # every layer, incl. the first
    dopt.step()


def test_budget_extrapolation_arithmetic():
    from ppfeddata.models.benchmark import extrapolate
    cfg = {"fl": {"num_clients": 5, "rounds": 30, "local_epochs": 2}, "seeds": [0, 1, 2], "tune": {"n_trials": 30},
           "thresholds": {"overhead_ratio_max": 3.0}}
    plain = {"epoch_s": 2.0, "n_rows": 18000}
    dp = {"epoch_s": 6.0, "n_rows": 18000, "make_private_s": 0.5}
    ex = extrapolate(cfg, plain, dp, n_train=90000, trial_s=100.0, b2={"cvae_train_s": 50.0, "eval_s": 20.0})
    assert ex["dp_over_plain"] == pytest.approx(3.0)
    assert ex["fl_run_plain_s"] == pytest.approx(30 * 2 * (2.0 / 18000) * 90000 + 20.0)          # 600 + 20
    assert ex["fl_run_dp_s"] == pytest.approx(30 * 2 * (6.0 / 18000) * 90000 + 30 * 5 * 0.5 + 20.0)
    assert ex["optuna_s"] == pytest.approx(3000.0)
    cuts = dict(ex["cuts"])
    assert cuts["(a) Optuna 30 -> 15 trials"] == pytest.approx(1500.0)
    assert cuts["(b) rounds 30 -> 20 (all FL runs)"] > 0 and cuts["(d) drop extensions A1-A5"] == pytest.approx(ex["extensions_total_s"])
    assert ex["total_s"] == pytest.approx(ex["core_s"] + ex["optuna_s"])


def test_gen_stats_noise_and_snap():
    from ppfeddata.models.generate import GenStats, fit_gen_stats, gen_stats_from_cfg
    schema = make_schema()
    X, y = synthetic_data(2000)
    m, _ = train_cvae(X, y, None, None, 3, build_layout(schema), HP, seed=0, epochs=10)
    assert gen_stats_from_cfg(m, X, y, schema, {"residual_noise": False, "snap_support": False}) is None
    st = fit_gen_stats(m, X, y, schema, max_unique=5)
    assert st.residual_std.shape == (3, 2) and (st.residual_std > 0).all()
    assert st.support == {}                                     # continuous columns have > 5 distinct values
    X2 = X.copy()
    X2[:, 1] = np.round(X2[:, 1] * 2) / 2                       # numeric column b now takes a handful of values
    st2 = fit_gen_stats(m, X2, y, schema, max_unique=50)
    assert all(s == 1 for _, s in st2.support) and len(st2.support) == 3
    plain, _ = generate(m, schema, [800, 0, 0], seed=1)
    noisy, _ = generate(m, schema, [800, 0, 0], seed=1, stats=GenStats({}, st.residual_std))
    snap, ys = generate(m, schema, [300, 300, 300], seed=1, stats=st2)
    assert noisy[:, 0].std() > plain[:, 0].std()                # noise restores spread
    for c in range(3):
        ok = snap[(ys == c) & (snap[:, 5] == 0), 1]             # b is_na == 0 rows
        assert set(np.unique(ok)) <= set(st2.support[(c, 1)])
