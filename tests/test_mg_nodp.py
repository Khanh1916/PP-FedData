"""Round Q1: FedDP-Marginal without DP (exact sums), the plain Flower releases and the report of the protection matrix."""
import numpy as np
import pytest

from ppfeddata.models import marginal as mg
from tests.test_models import make_schema, synthetic_data


def _schema():
    return {**make_schema(), "label_map": {"A": 0, "B": 1, "C": 2}}


def _fit_nodp(n=1500, K=3, seed=0):
    s = _schema()
    X, y = synthetic_data(n)
    parts = np.array_split(np.arange(n), K)
    return s, X, y, parts, mg.fit(X, y, parts, s, None, 1e-5, bins=8, seed=seed)


def test_no_dp_means_no_noise_and_no_epsilon():
    sc = mg.noise_scales(None, 1e-5, 5, mg.SPLIT, "skellam")
    assert sc == {s: 0.0 for s in mg.STAGES}
    rng = np.random.default_rng(0)
    assert not mg.client_noise((3, 4), 0.0, 0.2, rng, "skellam").any() and not mg.client_noise((2,), 7.0, 0.2, rng, "none").any()
    assert mg.privacy_info(sc, 5, 1e-5, 0.2, "skellam") == {"eps": None, "eps_one_honest": None}
    assert mg.eps_honest_curve(sc, 5, 1e-5, 5, "skellam") == {}


def test_no_dp_fit_releases_the_exact_counts():
    s, X, y, parts, m = _fit_nodp()
    attrs = mg.attributes(s)
    F = mg.discretise_fine(X, attrs, float(s["settings"]["clip_sigma"]))
    exact = mg.client_tables("fine", F, np.asarray(y, dtype=np.int64), attrs, 3)
    assert all(np.array_equal(a, b) for a, b in zip(m.fine, exact))                  # the sum over the clients is the pooled count
    edges = mg.client_tables("edges", F, np.asarray(y, dtype=np.int64), attrs, 3, m.maps, m.edges)
    assert np.array_equal(m.root_table, edges[0]) and all(np.array_equal(a, b) for a, b in zip(m.edge_tables, edges[1:]))
    assert m.info["eps"] is None and m.info["eps_target"] is None and m.info["mechanism"] == "none"
    Xs, ys = mg.sample(m, s, [100, 100, 100], seed=0)
    Xs2, _ = mg.sample(m, s, [100, 100, 100], seed=0)
    assert Xs.shape == (300, 12) and np.array_equal(Xs, Xs2)


def test_no_dp_model_has_nothing_to_fuse():
    *_, m = _fit_nodp()
    with pytest.raises(ValueError, match="nothing to fuse"):
        mg.refine(m, fuse=True)


def test_membership_attack_runs_without_noise():
    from ppfeddata.mia_marginal import fit_score
    s, X, y, _, _ = _fit_nodp()
    sc = fit_score(X, np.asarray(y, dtype=np.int64), np.arange(0, 1500, 2), s, None, 1e-5, {"bins": 8, "split": list(mg.SPLIT), "options": None}, 3, 0)
    assert sc.shape == (1500,) and np.isfinite(sc).all()


def test_plain_strategy_adds_the_client_vectors_exactly():
    from flwr.common import Code, FitRes, Status, ndarrays_to_parameters

    from ppfeddata.fl.mg_app import _strategy_class
    st = _strategy_class()(3, secure=False)
    vecs = [np.array([1, 2, 2 ** 20 - 1], dtype=np.int32), np.array([0, 5, 1], dtype=np.int32), np.array([7, 0, 3], dtype=np.int32)]
    res = [(None, FitRes(Status(Code.OK, "OK"), ndarrays_to_parameters([v]), 1, {"partition-id": i})) for i, v in enumerate(vecs)]
    params, _ = st.aggregate_fit(1, res, [])
    assert params is None and np.array_equal(st.total, np.sum(vecs, axis=0).astype(np.float64)) and st.pids == [0, 1, 2]
    with pytest.raises(RuntimeError, match="2 of 3"):
        st.aggregate_fit(1, res[:2], [])


def test_grid_tap_names_a_plain_round_by_its_message_type():
    from types import SimpleNamespace

    from flwr.app import ConfigRecord, RecordDict

    from ppfeddata.fl.secagg import TapGrid

    class Grid:
        def send_and_receive(self, messages, *a, **k):
            return list(messages)

    msg = SimpleNamespace(content=RecordDict({"fitins.config": ConfigRecord({"payload": "{}"})}), metadata=SimpleNamespace(message_type="train"),
                          has_error=lambda: False)                  # a Message needs a running Flower task; the tap reads only these fields
    tap = TapGrid(Grid())
    tap.send_and_receive([msg])
    assert tap.stages[0]["stage"] == "train" and tap.stages[0]["bytes_down"] == tap.stages[0]["bytes_up"] > 0


def test_report_renders_the_matrix_and_the_costs():
    from ppfeddata.mg_nodp import render
    row = lambda c, f, mb, s, auc, eps=float("nan"): {"config": c, "tstr_f1": f, "tstr_f1_std": 0.01, "bin_f1": 0.9, "rare_recall": 0.3,   # noqa: E731
                                                       "eps": eps, "mb_per_round": mb, "seconds": s, "mia_auc": auc}
    out = {"bins": 16, "matrix": {"plain": row("MGp-nodp", 0.44, 0.10, 5.0, 0.58), "secagg": row("MGs-nodp", 0.44, 0.15, 7.0, 0.58),
                                  "dp_local": [{**row("MGl-eps5", 0.37, 0.28, 1.0, None, 5.0), "bytes_estimated": True, "flower": False}],
                                  "dp_secagg": [row("MGs-eps5", 0.41, 0.15, 7.0, 0.52, 5.0)]},
           "dp_cost": [{"eps": 5.0, "config": "MGs-eps5", "classifier": "rf", "delta": -0.03, "lo": -0.04, "hi": -0.02, "sigma": 0.01, "effect": "worse",
                        "stream_effect": "none", "rare_delta": -0.02}],
           "secagg_utility": [{"classifier": "rf", "delta": 0.0, "lo": 0.0, "hi": 0.0, "sigma": 0.01, "effect": "none", "stream_effect": "none", "rare_delta": 0.0}],
           "vs_b3": [], "secagg_cost": {"bytes_ratio": 1.5, "time_ratio": 1.4, "mb_per_round": [0.15, 0.10], "seconds": [7.0, 5.0]},
           "same_release": {0: {"tables": True, "synthetic": True}, 1: None},
           "mia": {"nodp": 0.56, "nodp_rare": 0.58, "control_nonoise_half": 0.583, "dp": {"MGs-eps5": 0.522}, "threshold": 0.55},
           "r6_aside": {"taug_nodp": 0.4447, "taug_mg_eps5": 0.4482, "delta": -0.0035, "lo": -0.004, "hi": -0.003, "sigma": 0.0007, "effect": "worse",
                        "synthetic_mia_nodp": 0.498}}
    md = render(out)
    assert "| no DP, SecAgg+ | MGs-nodp | 0.440 ± 0.010 | 0.900 | 0.300 | - | 0.150 | 7.0 | 0.580 |" in md
    assert "| DP (distributed), SecAgg+ | MGs-eps5 |" in md and "| 5.00 |" in md
    assert "| 0.280 (est.) | n/a (in-process) | n/a |" in md                    # the local variant: estimated bytes, no Flower time, no attack
    assert "| 5 | MGs-eps5 | rf | -0.030 | [-0.040, -0.020] | 0.010 | worse † | -0.020 |" in md          # † : the stream bootstrap disagrees
    assert "(1.50x); FL seconds 7.0 against 5.0 (1.40x)" in md and "| rf | +0.000 | [+0.000, +0.000] | 0.010 | no effect | +0.000 |" in md
    assert "| 0 | yes | yes |" in md and "| 1 | n/a | n/a |" in md
    assert "Without DP: calibrated AUC 0.560 (rare classes 0.580); threshold 0.55" in md and "MGs-eps5 0.522" in md
    assert "Δ -0.0035 [-0.0040, -0.0030], seed std 0.0007, worse" in md
