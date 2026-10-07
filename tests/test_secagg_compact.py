"""Optimisation O2 (bandwidth): uint16 / uint32 masked vectors in Flower's SecAgg+ give the same modular sum as int64, with fewer bytes."""
import numpy as np
import pytest

pytest.importorskip("flwr")

from flwr.common import bytes_to_ndarray, ndarray_to_bytes  # noqa: E402
from flwr.common.secure_aggregation.ndarrays_arithmetic import parameters_addition  # noqa: E402

from ppfeddata.fl import secagg  # noqa: E402


@pytest.mark.parametrize("bits", [16, 32])
def test_compact_sum_equals_the_int64_modular_sum(bits):
    m = 2 ** bits
    rng = np.random.default_rng(0)
    clients = [[rng.integers(-3 * m, 3 * m, size=(7, 5), dtype=np.int64), rng.integers(-3 * m, 3 * m, size=4, dtype=np.int64)] for _ in range(5)]
    secagg.set_compact(True)
    try:
        sent = [secagg._client_mod(c, m) for c in clients]
        assert all(a.dtype == secagg.COMPACT_DTYPES[m] for c in sent for a in c)
        wire = [[bytes_to_ndarray(ndarray_to_bytes(a)) for a in c] for c in sent]           # serialised as Flower does
        agg = wire[0]
        for c in wire[1:]:
            agg = parameters_addition(agg, c)                                                  # the server's sum, wrapping in uint
        agg = secagg._server_mod(agg, m)
    finally:
        secagg.set_compact(False)
    exact = [sum(c[i] for c in clients) % m for i in range(2)]
    assert all(np.array_equal(a.astype(np.int64), e) for a, e in zip(agg, exact))
    full = len(ndarray_to_bytes(np.zeros(35, np.int64)))
    assert len(ndarray_to_bytes(sent[0][0])) < full


def test_set_compact_installs_and_removes_the_patch():
    import importlib
    cm = importlib.import_module("flwr.client.mod.secure_aggregation.secaggplus_mod")
    sw = importlib.import_module("flwr.server.workflow.secure_aggregation.secaggplus_workflow")
    secagg.set_compact(True)
    assert cm.parameters_mod is secagg._client_mod and sw.parameters_mod is secagg._server_mod
    secagg.set_compact(False)
    assert cm.parameters_mod is secagg._ORIGINAL["client"] and sw.parameters_mod is secagg._ORIGINAL["server"]


def test_compact_levels_are_parsed_and_scored():
    import pandas as pd

    from ppfeddata import aggregate as ag
    from ppfeddata import scorecard as sc
    from tests.test_scorecard import RARE, _gen, _summary
    p = ag.parse_config("M3o-t27-eps5-c16-plain-TSTR-rf")
    assert p["method"] == "M3" and p["eps"] == 5.0 and p["variant"] == "plain" and p["family"].startswith("o2 bandwidth c16")
    assert ag.parse_config("M2-c32-TAug-mlp")["method"] == "M2"
    s = pd.concat([_summary(), pd.DataFrame(_gen("M2-c16", 0.40, 0.40, bytes_=1.6e6) + _gen("M3o-t27-eps5-c16-plain", 0.27, 0.27, eps=5.0, bytes_=0.6e6))],
                  ignore_index=True)
    by = {r["label"]: r for r in sc.build(s, rare=RARE)["rows"]}
    assert by["M2-c16"]["secagg"] and not by["M2-c16"]["dp"] and by["M2-c16"]["valid"]
    assert by["M3o-eps5-c16 (plain)"]["dp"] and by["M3o-eps5-c16 (plain)"]["secagg"]


def test_compact_spec_checks_that_the_sum_cannot_wrap():
    cfg = {"fl": {"num_clients": 5}, "secagg": {}}
    sp = secagg.secagg_spec(cfg, compact=True, modulus_range=2 ** 16, quantization_range=2 ** 13)
    assert sp["compact"] and sp["params"]["modulus_range"] == 2 ** 16
    with pytest.raises(ValueError, match="wrap"):
        secagg.secagg_spec(cfg, compact=True, modulus_range=2 ** 16, quantization_range=2 ** 14)
    with pytest.raises(ValueError, match="2\\^16 or 2\\^32"):
        secagg.secagg_spec(cfg, compact=True, modulus_range=2 ** 24, quantization_range=2 ** 13)
    assert not secagg.secagg_spec(cfg)["compact"]
