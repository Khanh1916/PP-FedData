"""Phase 10 tests: secure aggregation (Flower SecAgg+) - correctness T-SA1, hiding T-SA2, dropout T-SA3, clipping range.

Most tests run Flower's REAL workflow (`SecAggPlusWorkflow`) and REAL client mod (`secaggplus_mod`) in one process through a
minimal in-process grid, so they need no Ray and see every message. One slow test goes through the real Flower simulation
(Ray) for M2-vs-FedAvg and checkpoint/resume.
"""
import copy
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flwr")

from flwr.app import Context, Error, Message, RecordDict  # noqa: E402
from flwr.client.mod import secaggplus_mod  # noqa: E402
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters  # noqa: E402
from flwr.compat.common import recorddict_compat as compat  # noqa: E402

from ppfeddata.fl import core, secagg  # noqa: E402


@pytest.fixture(autouse=True)
def flower_task_identity():
    """Flower stamps every Message with the identity of the running task (set by its simulation runtime; set here by hand)."""
    from flwr.supercore.task_identity import TaskIdentity
    old = (TaskIdentity._task_id, TaskIdentity._run_id, TaskIdentity._node_id)
    TaskIdentity.task_id, TaskIdentity.run_id, TaskIdentity.node_id = 1, 1, 1
    yield
    TaskIdentity._task_id, TaskIdentity._run_id, TaskIdentity._node_id = old


PARAMS = {"num_shares": 5, "reconstruction_threshold": 3, "clipping_range": 8.0, "max_weight": 100000.0,
          "quantization_range": 2 ** 22, "modulus_range": 2 ** 32}
SIZES = [500, 3000, 20000, 40000, 25000]                     # 5 clients like the study: from the minimum client to a dominant one
TEMPLATE = OrderedDict(a=torch.zeros(40, 30), b=torch.zeros(30), c=torch.zeros(7, 5, 9))     # 1,545 floats in three tensors
BIG = OrderedDict(a=torch.zeros(300, 300), b=torch.zeros(100, 100), c=torch.zeros(500))        # 100,500 floats, like the real CVAE (105,564)


class InProcessGrid:
    """Delivers each message to `secaggplus_mod` (wrapping `train_fn`) living in this process. `dropped` = node ids that fail in the
    stage that carries the training request (the reply is an error message, as when a client app crashes)."""

    def __init__(self, n_clients, train_fn, dropped=()):
        self.node_ids = [100 + i for i in range(n_clients)]
        self.ctx = {nid: Context(run_id=1, node_id=nid, node_config={"partition-id": i}, state=RecordDict(), run_config={})
                    for i, nid in enumerate(self.node_ids)}
        self.run = SimpleNamespace(run_id=1)
        self.train_fn, self.dropped = train_fn, set(dropped)

    def get_node_ids(self):
        return self.node_ids

    def send_and_receive(self, messages, timeout=None):
        from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS, Key, Stage
        out = []
        for m in messages:
            m = copy.deepcopy(m)              # a real transport serialises each message; the mod pops keys, and the workflow shares one record between messages
            nid = m.metadata.dst_node_id
            stage = m.content.config_records[RECORD_KEY_CONFIGS][Key.STAGE]
            if nid in self.dropped and stage == Stage.COLLECT_MASKED_VECTORS:
                out.append(Message(error=Error(code=1, reason="simulated client failure"), reply_to=m))
                continue
            out.append(secaggplus_mod(m, self.ctx[nid], lambda msg, c: self.train_fn(msg, c)))
        return out


def make_updates(seed=0, scale=0.3, template=TEMPLATE):
    """Fixed per-client 'trained' weights (independent of the global model) so the exact weighted mean is known."""
    rng = np.random.default_rng(seed)
    return [OrderedDict((k, torch.tensor(rng.normal(0, scale, v.shape), dtype=torch.float32)) for k, v in template.items()) for _ in SIZES]


def make_train_fn(updates, sizes, diag):
    """Plays the role of `app.train_secagg`: returns the client's fixed weights in the legacy FitRes layout and, like the real
    client, writes its diagnostics. `diag` = (spec, run dir)."""
    def train(msg, ctx):
        pid = int(ctx.node_config["partition-id"])
        fitins = compat.recorddict_to_fitins(msg.content, keep_input=True)
        arrays = secagg.state_to_arrays(updates[pid])
        secagg.client_diagnostics(diag[0], diag[1], int(fitins.config["round"]), pid, arrays, sizes[pid])
        fitres = FitRes(status=Status(code=Code.OK, message="OK"), parameters=ndarrays_to_parameters(arrays),
                        num_examples=sizes[pid], metrics={"partition-id": pid, "steps": 3, "train_seconds": 0.0})
        return Message(content=compat.fitres_to_recorddict(fitres, keep_input=True), reply_to=msg)
    return train


def run_secure(updates, tmp, rounds=1, dropped=(), sizes=SIZES, params=PARAMS, capture=False, allow_dropout=False, template=TEMPLATE):
    """Drives `secagg.run_rounds` (the real round loop) through the in-process grid. Returns (final state, strategy, grid, evaluated states)."""
    tmp.mkdir(parents=True, exist_ok=True)
    sp = {"secagg": {"params": params, "diagnostics": capture, "debug_rounds": [1]}}
    grid = InProcessGrid(len(sizes), make_train_fn(updates, sizes, (sp, tmp)), dropped)
    strategy = secagg.make_strategy_class()(0, len(sizes), None, False, allow_dropout)
    seen = []
    ctx = Context(run_id=1, node_id=0, node_config={}, state=RecordDict(), run_config={})
    state = OrderedDict((k, v.clone()) for k, v in template.items())
    final = secagg.run_rounds(sp, grid, ctx, strategy, state, 0, rounds, lambda r, arr: seen.append(arr), tmp)
    return final, strategy, grid, seen


def exact_mean(updates, sizes=SIZES, keep=None):
    keep = list(range(len(sizes))) if keep is None else keep
    w = np.array([sizes[i] for i in keep], dtype=np.float64)
    return {k: np.average(np.stack([updates[i][k].numpy().astype(np.float64) for i in keep]), axis=0, weights=w) for k in updates[0]}


def max_err(final, want):
    return max(float(np.abs(final[k].numpy().astype(np.float64) - want[k]).max()) for k in want)


# ----------------------------------------------------------------------------- T-SA1: correctness
def test_tsa1_secure_average_equals_weighted_fedavg_within_the_quantisation_bound(tmp_path):
    bound = secagg.quantization_error_bound(5, PARAMS, sum(SIZES))
    assert bound < 1e-4                                                  # the gate of the spec is reachable with this max_weight
    for seed in range(3):
        ups = make_updates(seed)
        final, strat, _, _ = run_secure(ups, tmp_path / f"s{seed}")
        want = exact_mean(ups)
        err = max_err(final, want)
        assert err <= bound, (seed, err, bound)                          # hard bound of the stochastic quantiser
        assert err < 1e-4
        unweighted = {k: np.mean([u[k].numpy() for u in ups], axis=0) for k in want}
        assert max(float(np.abs(unweighted[k] - want[k]).max()) for k in want) > 1e-2      # the weights matter, a wrong weighting would be noticed
        assert strat.steps == {i: 3 for i in range(5)} and strat.n_by_pid == dict(enumerate(SIZES))


def test_tsa1_quantisation_is_unbiased_and_the_default_max_weight_would_be_wrong(tmp_path):
    ups = make_updates(5)
    final, *_ = run_secure(ups, tmp_path / "a")
    want = exact_mean(ups)
    e = np.concatenate([(final[k].numpy().astype(np.float64) - want[k]).ravel() for k in want])
    assert abs(e.mean()) < 0.1 * e.std()                                 # stochastic rounding: no systematic drift
    # Flower's default max_weight = 1000 is below the client sizes: the weighting is silently distorted, so the config must raise it
    bad, *_ = run_secure(ups, tmp_path / "b", params={**PARAMS, "max_weight": 1000.0})
    assert max_err(bad, want) > 1e-2


def test_secagg_spec_raises_max_weight_above_flower_default_and_keeps_other_defaults():
    sa = secagg.secagg_spec({"secagg": {"num_shares": 5, "reconstruction_threshold": 3, "clipping_range": 8.0}})
    p = sa["params"]
    assert p["max_weight"] > max(SIZES) and p["quantization_range"] == 2 ** 22 and p["modulus_range"] == 2 ** 32 and p["clipping_range"] == 8.0
    assert isinstance(p["clipping_range"], float) and isinstance(p["modulus_range"], int)      # the client mod type-checks these


def test_rounds_chain_and_step_counters_add_up(tmp_path):
    """Two rounds of the real loop (clients return fixed weights, so the global model must equal the exact mean after each)."""
    ups = make_updates(1)
    final, strat, _, seen = run_secure(ups, tmp_path, rounds=2)
    assert len(seen) == 3                                                 # evaluate(0), evaluate(1), evaluate(2)
    assert max_err(final, exact_mean(ups)) < 1e-4
    assert strat.steps == {i: 6 for i in range(5)}                       # step counters add up over rounds


# ----------------------------------------------------------------------------- T-SA2: hiding
def test_tsa2_masked_vectors_look_uniform_and_are_uncorrelated_with_the_update(tmp_path):
    ups = make_updates(2, template=BIG)
    run_secure(ups, tmp_path, capture=True, template=BIG)
    z = np.load(tmp_path / "secagg_debug" / "masked_r0001.npz")          # the masked vectors the server received, kept for the debug round
    assert sorted(z.files) == [f"c{i}" for i in range(5)]
    for pid in range(5):
        plain = secagg.flat(secagg.state_to_arrays(ups[pid]))
        st = secagg.masking_stats(z[f"c{pid}"], plain, PARAMS, SIZES[pid])
        assert abs(st["rho"]) < 0.05, st                                 # spec: |rho| < 0.05
        assert st["ks_stat"] < 0.05 and st["ks_p"] > 1e-4, st            # approximately uniform on [0, modulus)
        assert st["rho_unmasked"] > 0.99 and st["ks_stat_unmasked"] > 0.5      # positive control: without the mask both statistics do fire
        assert st["min"] >= 0 and st["max"] < st["mod"]


def test_tsa2_even_the_sum_of_all_masked_vectors_stays_noise_until_the_unmask_stage(tmp_path):
    ups = make_updates(3, template=BIG)
    run_secure(ups, tmp_path, capture=True, template=BIG)
    z = np.load(tmp_path / "secagg_debug" / "masked_r0001.npz")
    total = sum(z[f"c{i}"].astype(np.int64) for i in range(5)) % PARAMS["modulus_range"]
    want = exact_mean(ups)
    plain_mean = secagg.flat([want[k] for k in BIG])
    # the pairwise masks cancel in the sum but each client's private mask does not (it is removed only in the unmask stage,
    # when the clients reveal shares of their seeds), so the sum of the masked vectors is still noise to the server
    assert abs(np.corrcoef(total[1:].astype(np.float64), plain_mean)[0, 1]) < 0.1


# ----------------------------------------------------------------------------- T-SA3: dropout
def test_tsa3_one_client_drops_and_the_aggregate_of_the_others_is_still_correct(tmp_path):
    ups = make_updates(4)
    final, strat, _, _ = run_secure(ups, tmp_path, dropped=[102], allow_dropout=True)       # node 102 = client 2 (the 20k-row one)
    assert strat.n_by_pid == {0: SIZES[0], 1: SIZES[1], 3: SIZES[3], 4: SIZES[4]}
    want = exact_mean(ups, keep=[0, 1, 3, 4])
    bound = secagg.quantization_error_bound(4, PARAMS, sum(SIZES[i] for i in (0, 1, 3, 4)))
    err = max_err(final, want)
    assert err <= bound and err < 1e-4
    assert max(float(np.abs(exact_mean(ups)[k] - want[k]).max()) for k in want) > 1e-2      # the dropped client really is excluded


def test_tsa3_two_clients_may_drop_with_threshold_three_of_five_but_three_may_not(tmp_path):
    ups = make_updates(6)
    final, strat, *_ = run_secure(ups, tmp_path / "two", dropped=[100, 104], allow_dropout=True)    # 3 of 5 remain = threshold
    assert sorted(strat.n_by_pid) == [1, 2, 3]
    assert max_err(final, exact_mean(ups, keep=[1, 2, 3])) < 1e-4
    with pytest.raises(RuntimeError, match="halted"):                     # 2 left < threshold: the protocol stops, the run loop aborts
        run_secure(ups, tmp_path / "three", dropped=[100, 101, 104], allow_dropout=True)


def test_a_missing_client_aborts_the_run_unless_dropout_is_allowed(tmp_path):
    with pytest.raises(RuntimeError, match="4 of 5 clients"):
        run_secure(make_updates(8), tmp_path, dropped=[103])


# ----------------------------------------------------------------------------- clipping range
def test_values_beyond_clipping_range_are_silently_clipped_and_the_check_sees_it(tmp_path):
    ups = make_updates(9)
    big = OrderedDict((k, v.clone()) for k, v in ups[3].items())
    big["b"][0] = 12.0                                                    # one weight beyond +/- clipping_range (8)
    ups[3] = big
    # the quantiser clips the WEIGHTED value w * n_i / max_weight; with max_weight = n_3 client 3 has factor 1, so 12 is cut to 8
    final, *_ = run_secure(ups, tmp_path / "run", params={**PARAMS, "max_weight": 40000.0})
    want = exact_mean(ups)
    assert float(np.abs(final["b"].numpy() - want["b"]).max()) > 0.1       # distorted: no error is raised, the value is just cut
    sp = {"secagg": {"diagnostics": False, "params": {**PARAMS, "max_weight": 40000.0}}}
    secagg.client_diagnostics(sp, tmp_path, 1, 0, secagg.state_to_arrays(ups[0]), SIZES[0])
    secagg.client_diagnostics(sp, tmp_path, 1, 3, secagg.state_to_arrays(ups[3]), SIZES[3])
    s = secagg.clip_summary(tmp_path, 8.0)
    assert not s["ok"] and s["n_beyond_range"] == 1 and s["max_abs_w"] == pytest.approx(12.0)
    assert s["max_abs_weighted"] == pytest.approx(12.0)                  # n_3 / max_weight = 1, so weighted = raw for this client
    assert secagg.clip_summary(tmp_path, 20.0)["ok"]


def test_clip_files_are_trimmed_on_resume_and_last_write_wins(tmp_path):
    sp = {"secagg": {"diagnostics": False}}
    for rnd, v in ((1, 1.0), (2, 2.0), (3, 3.0)):
        secagg.client_diagnostics(sp, tmp_path, rnd, 0, [np.array([v])])
    secagg.reset_client_files(tmp_path, 2)
    assert secagg.clip_summary(tmp_path, 8.0)["max_abs_w"] == 2.0
    secagg.client_diagnostics(sp, tmp_path, 3, 0, [np.array([1.5])])
    s = secagg.clip_summary(tmp_path, 8.0)
    assert s["max_abs_w"] == 2.0 and s["n_updates"] == 3


# ----------------------------------------------------------------------------- bytes
def test_byte_counter_reports_masked_upload_larger_than_the_float32_model_and_four_stages(tmp_path):
    run_secure(make_updates(10), tmp_path, capture=True)
    rows = secagg.read_secagg_rounds(tmp_path)
    assert len(rows) == 1 and [s["stage"] for s in rows[0]["stages"]] == list(secagg.STAGES)
    n_par = sum(v.numel() for v in TEMPLATE.values())
    up = [s for s in rows[0]["stages"] if s["stage"] == "collect_masked_vectors"][0]["bytes_up"]
    assert up > 5 * n_par * 7                                             # Flower sends the masked vector as int64 (8 bytes), not float32 (4)
    assert rows[0]["bytes_down"] > 5 * n_par * 4 and rows[0]["agg_max_abs_err"] < 1e-3


# ----------------------------------------------------------------------------- Flower simulation (slow)
def test_flower_simulation_secagg_matches_fedavg_and_resumes_exactly(tmp_path):
    pytest.importorskip("ray")
    from tests.test_fl import _toy_cfg_and_data
    from tests.test_models import HP

    from ppfeddata.fl.run import run_fl
    cfg, ddir = _toy_cfg_and_data(tmp_path)
    cfg["secagg"] = {"num_shares": 3, "reconstruction_threshold": 2, "max_weight": 1000.0}      # toy clients have fewer than 1000 rows
    sa = secagg.secagg_spec(cfg, diagnostics=True, debug_rounds=(1,))
    kw = {"data_dir": ddir, "hp": HP, "save_states": True}
    plain = run_fl(cfg, 0, "plain", rounds=4, **kw)
    sec = run_fl(cfg, 0, "sec", rounds=4, secagg=sa, **kw)
    cut = run_fl(cfg, 0, "cut", rounds=4, secagg=sa, stop_after=2, **kw)             # interrupted after 2 of 4 rounds
    assert cut["summary"]["rounds_done"] == 2
    cut = run_fl(cfg, 0, "cut", rounds=4, secagg=sa, resume=True, **kw)
    art = tmp_path / "art"
    # (1) round-1 global model: identical inputs, so secure vs plain differ by quantisation only
    s1p = torch.load(art / plain["run_id"] / "states" / "round_0001.pt", weights_only=False)
    s1s = torch.load(art / sec["run_id"] / "states" / "round_0001.pt", weights_only=False)
    d1 = max(float((s1p[k] - s1s[k]).abs().max()) for k in s1p)
    bound = secagg.quantization_error_bound(3, sa["params"], sum(sec["partition_sizes"]))
    assert d1 <= bound * 1.01 + 1e-7, (d1, bound)
    # (2) the server-side comparison with the plain mean of the same updates agrees, for every round
    rows = secagg.read_secagg_rounds(art / sec["run_id"])
    assert len(rows) == 4 and all(r["agg_max_abs_err"] <= bound * 1.01 for r in rows)
    # (3) exact resume (stochastic rounding is seeded per client and round)
    sf = torch.load(art / sec["run_id"] / "final_state.pt", weights_only=False)
    sc = torch.load(art / cut["run_id"] / "final_state.pt", weights_only=False)
    assert max(float((sf[k] - sc[k]).abs().max()) for k in sf) < 1e-6
    assert [r["round"] for r in core.read_round_log(art / cut["run_id"])] == [0, 1, 2, 3, 4]
    assert sec["summary"]["client_steps"] == cut["summary"]["client_steps"] == plain["summary"]["client_steps"]
    # (4) bookkeeping: the clipping check saw every update, the summary has the stage times and bytes
    s = sec["summary"]["secagg"]
    assert s["clip"]["n_updates"] == 3 * 4 and s["clip"]["ok"] and set(s["stage_seconds"]) == set(secagg.STAGES) and s["bytes_per_round"] > 0
    assert (art / sec["run_id"] / "secagg_debug" / "masked_r0001.npz").exists()
    assert not list((art / sec["run_id"] / "secagg_debug").glob("plain_r0002*"))          # plain dumps of non-debug rounds are deleted
    # (5) the training metrics that travel in the clear contain no data-dependent number
    assert all(r["train_loss"] is None for r in core.read_round_log(art / sec["run_id"]) if r["round"] > 0)


# ----------------------------------------------------------------------------- control: aggregate noise
def test_aggregate_noise_control_is_deterministic_seeded_by_round_and_has_the_requested_size():
    st = OrderedDict(w=torch.zeros(300, 400), b=torch.ones(50))
    same = core.add_aggregate_noise(st, 0.0, 1, 3)
    assert all(torch.equal(same[k], st[k]) for k in st) and same["w"] is not st["w"]
    a = core.add_aggregate_noise(st, 1e-3, 1, 3)
    assert all(torch.equal(a[k], core.add_aggregate_noise(st, 1e-3, 1, 3)[k]) for k in st)                 # same (seed, round): same noise
    assert not torch.equal(a["w"], core.add_aggregate_noise(st, 1e-3, 1, 4)["w"])                           # round changes the draw
    assert not torch.equal(a["w"], core.add_aggregate_noise(st, 1e-3, 2, 3)["w"])                           # noise seed changes the draw
    assert float(a["w"].std()) == pytest.approx(1e-3, rel=0.02) and abs(float(a["w"].mean())) < 1e-5
    assert torch.equal(st["b"], torch.ones(50))                                                              # the input is not modified
