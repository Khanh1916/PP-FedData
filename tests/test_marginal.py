"""Optimisation O3: federated DP marginal generator (accounting, discretisation, Chow-Liu tree, sampling)."""
import math

import numpy as np
import pandas as pd
import pytest

from ppfeddata.models import marginal as mg
from tests.test_models import make_schema, synthetic_data


def _schema():
    return {**make_schema(), "label_map": {"A": 0, "B": 1, "C": 2}}


@pytest.mark.parametrize("eps", [0.5, 1.0, 5.0, 10.0])
def test_rho_conversion_round_trips(eps):
    r = mg.rho_from_eps(eps, 1e-5)
    assert mg.eps_from_rho(r, 1e-5) == pytest.approx(eps, rel=1e-9)
    assert mg.sigma_for(r, 4) == pytest.approx(math.sqrt(4 / (2 * r)))


def test_spanning_tree_follows_the_strongest_links():
    w = np.array([[0, 5, 1, 0], [5, 0, 1, 4], [1, 1, 0, 3], [0, 4, 3, 0]], dtype=float)
    e = mg.max_spanning_tree(w)
    assert len(e) == 3 and {frozenset(x) for x in e} == {frozenset((0, 1)), frozenset((1, 3)), frozenset((3, 2))}
    assert e[0][0] == 0                                          # rooted at attribute 0, parents first


def test_coarse_bins_follow_the_quantiles_and_never_cross():
    a = mg.Attr("x", "numeric", 0, 1, mg.FINE)
    h = np.zeros((2, mg.FINE))
    h[:, 10], h[:, 11], h[:, 40] = 100, 100, 200
    m = mg.coarse_maps([h], [a], bins=4)[0]
    assert np.all(np.diff(m) >= 0) and m.max() <= 3
    assert m[0] == m[10] < m[40] < m[63]                         # the two heavy regions fall in different coarse bins


def test_noise_free_fit_reproduces_the_class_structure():
    s = _schema()
    X, y = synthetic_data(3000)
    parts = [np.arange(0, 1000), np.arange(1000, 2000), np.arange(2000, 3000)]
    m = mg.fit(X, y, parts, s, eps=1e7, delta=1e-5, bins=8)
    Xs, ys = mg.sample(m, s, [1000, 1000, 1000], seed=0)
    assert Xs.shape == (3000, 12) and np.array_equal(np.bincount(ys), [1000, 1000, 1000])
    assert np.all(Xs[:, 7:12].sum(1) == 1) and not Xs[:, 10:12].any()           # one-hot, dead OTHER / NONE never drawn
    assert np.mean(Xs[np.arange(3000), 7 + ys] == 1) > 0.97                     # k = class kept
    means = [Xs[ys == c, 0].mean() for c in range(3)]
    assert means[0] < means[1] < means[2]
    assert m.info["eps"] == pytest.approx(1e7, rel=1e-6) and m.info["noise_share"] == pytest.approx(1 / 3)


def test_distributed_and_local_noise_accounting():
    s = _schema()
    X, y = synthetic_data(1500)
    parts = [np.arange(0, 500), np.arange(500, 1000), np.arange(1000, 1500)]
    d = mg.fit(X, y, parts, s, eps=5.0, delta=1e-5)
    loc = mg.fit(X, y, parts, s, eps=5.0, delta=1e-5, noise_share=1.0)
    assert d.info["eps"] == pytest.approx(5.0) and loc.info["eps"] == pytest.approx(5.0)
    assert d.info["eps_one_honest"] > 5.0 and loc.info["eps_one_honest"] == pytest.approx(5.0)
    assert d.info["sigmas"] == loc.info["sigmas"]               # same total per client; distributed splits it


def test_skellam_rdp_tends_to_the_gaussian_and_calibrates_the_target():
    a, m = 8, 37
    for mu in (1e3, 1e5):
        g = a * m / (2 * mu)
        assert g <= mg.skellam_rdp(a, m, mu) <= g * (1 + 10 / mu ** 0.5)
    sizes, split = [37, 666, 37], (0.3, 0.1, 0.6)
    for eps in (1.0, 5.0):
        mus = mg.skellam_mus(eps, 1e-5, sizes, split)
        assert mg.skellam_eps(list(zip(sizes, mus)), 1e-5) == pytest.approx(eps, rel=1e-4)
        assert mg.skellam_eps(list(zip(sizes, [m * 0.9 for m in mus])), 1e-5) > eps          # less noise -> above the budget


def test_skellam_shares_sum_to_the_full_variance_and_stay_integers():
    rng = np.random.default_rng(0)
    tot = sum(mg.client_noise((200000,), 50.0, 0.2, rng, "skellam") for _ in range(5))
    assert np.all(tot == np.round(tot)) and tot.var() == pytest.approx(50.0, rel=0.02) and abs(tot.mean()) < 0.1


def test_skellam_fit_releases_integers_and_reports_epsilon_by_honest_clients():
    s = _schema()
    X, y = synthetic_data(1500)
    parts = [np.arange(0, 500), np.arange(500, 1000), np.arange(1000, 1500)]
    m = mg.fit(X, y, parts, s, eps=5.0, delta=1e-5, mechanism="skellam")
    assert m.info["mechanism"] == "skellam" and m.info["eps"] == pytest.approx(5.0, rel=1e-4)
    assert all(np.array_equal(t, np.round(t)) for t in m.fine + m.edge_tables)
    curve = mg.eps_honest_curve(m.info["scales"], len(m.attrs), 1e-5, 3, "skellam")
    assert curve[3] == pytest.approx(5.0, rel=1e-4) and curve[1] > curve[2] > curve[3]
    Xs, ys = mg.sample(m, s, [100, 100, 100], 0)
    assert Xs.shape == (300, 12)


def test_mg_rows_in_the_scorecard():
    from ppfeddata import aggregate as ag
    from ppfeddata import scorecard as sc
    from tests.test_scorecard import RARE, _gen, _summary
    p = ag.parse_config("MGd-eps5-TSTR-rf")
    assert p["method"] == "MG" and p["eps"] == 5.0 and p["family"] == "o3 marginal distributed"
    s = pd.concat([_summary(), pd.DataFrame(_gen("MGd-eps5", 0.38, 0.38, eps=5.0) + _gen("MGl-eps5", 0.33, 0.33, eps=5.0))], ignore_index=True)
    by = {r["label"]: r for r in sc.build(s, rare=RARE)["rows"]}
    assert by["MGd-eps5"]["secagg"] and by["MGd-eps5"]["dp_mode"] == "distributed" and by["MGd-eps5"]["valid"]
    assert not by["MGl-eps5"]["secagg"] and by["MGl-eps5"]["dp_mode"] == "local"
    assert ag.parse_config("MGs-eps1-TAug-mlp")["family"] == "o3 marginal distributed skellam secagg+"
    s2 = pd.concat([s, pd.DataFrame(_gen("MGs-eps5", 0.40, 0.40, eps=5.0))], ignore_index=True)
    r = {x["label"]: x for x in sc.build(s2, rare=RARE)["rows"]}["MGs-eps5"]
    assert r["secagg"] and r["dp_mode"] == "distributed" and r["valid"]


def test_flower_encoding_keeps_integers_exact():
    from ppfeddata.fl import mg_app
    p = mg_app.secagg_params({"fl": {"num_clients": 5}, "secagg": {}})["params"]
    assert p["quantization_range"] == 2 * p["clipping_range"]          # quantisation step 1: integers are not rounded
    assert 5 * p["quantization_range"] < p["modulus_range"] == 2 ** 32
