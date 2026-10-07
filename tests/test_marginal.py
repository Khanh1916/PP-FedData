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
