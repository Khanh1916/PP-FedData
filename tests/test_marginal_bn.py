"""Optimisation O3: per-class Bayesian-network generalisation of FedDP-Marginal and its membership-inference score."""
import numpy as np
import pytest

from ppfeddata.models import marginal as mg
from ppfeddata.models import marginal_bn as bn
from tests.test_models import make_schema, synthetic_data


def _schema():
    return {**make_schema(), "label_map": {"A": 0, "B": 1, "C": 2}}


PARTS = [np.arange(0, 1000), np.arange(1000, 2000), np.arange(2000, 3000)]


def test_structure_degree_1_is_the_spanning_tree_and_degree_2_has_two_parents():
    w = np.array([[0, 5, 1, 0], [5, 0, 1, 4], [1, 1, 0, 3], [0, 4, 3, 0]], dtype=float)
    order, par = bn.structure(w, 1)
    assert {frozenset((p[0], j)) for j, p in par.items() if p} == {frozenset(e) for e in mg.max_spanning_tree(w)}
    order2, par2 = bn.structure(w, 2)
    assert par2[0] == () and all(len(par2[j]) == min(2, order2.index(j)) for j in order2)
    assert all(set(par2[j]) <= set(order2[:order2.index(j)]) for j in order2)        # parents are placed before


@pytest.mark.parametrize("kw", [{}, {"degree": 2, "bins": 4}, {"class_trees": True}, {"class_maps": True, "bins_rare": 2, "rare_limit": 1100}, {"fine": 32}])
def test_options_keep_the_accounting_and_learn_the_classes(kw):
    s = _schema()
    X, y = synthetic_data(3000)
    m = bn.fit(X, y, PARTS, s, eps=1e7, delta=1e-5, **{"bins": 8, **kw})
    ref = mg.fit(X, y, PARTS, s, eps=5.0, delta=1e-5, mechanism="skellam")
    m5 = bn.fit(X, y, PARTS, s, eps=5.0, delta=1e-5, **{"bins": 8, **kw})
    assert m5.info["eps"] == pytest.approx(ref.info["eps"]) and m5.info["scales"] == pytest.approx(ref.info["scales"])
    Xs, ys = bn.sample(m, s, [500, 500, 500], 0)
    assert Xs.shape == (1500, 12) and np.all(Xs[:, 7:12].sum(1) == 1)
    assert np.mean(Xs[np.arange(1500), 7 + ys] == 1) > 0.95
    means = [Xs[ys == c, 0].mean() for c in range(3)]
    assert means[0] < means[1] < means[2]


def test_loglik_is_a_normalised_log_probability_and_favours_members_without_noise():
    s = _schema()
    X, y = synthetic_data(400, seed=1)
    Xo, yo = synthetic_data(400, seed=2)
    m = bn.fit(X, y, [np.arange(0, 200), np.arange(200, 400)], s, eps=1e7, delta=1e-5, bins=32, fine=128)
    li, lo = bn.loglik(m, X, y), bn.loglik(m, Xo, yo)
    assert np.all(li <= 0) and li.mean() > lo.mean()                       # members are fitted better (overfitted, no noise)
    t = mg.fit(X, y, [np.arange(0, 200), np.arange(200, 400)], s, eps=1e7, delta=1e-5, bins=32, mechanism="skellam")
    assert bn.loglik_tree(t, X, y).mean() > bn.loglik_tree(t, Xo, yo).mean()
