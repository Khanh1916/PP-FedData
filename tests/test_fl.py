"""Phase 8 tests: Dirichlet partition, FedAvg, client determinism, checkpoint/resume through a real Flower simulation."""
import json
from collections import OrderedDict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flwr")

from tests.test_models import HP, make_schema, synthetic_data  # noqa: E402

from ppfeddata.fl import core  # noqa: E402
from ppfeddata.models.cvae import build_layout  # noqa: E402
from ppfeddata.partition import dirichlet_partition, label_table, make_partition, partition_path  # noqa: E402


# ----------------------------------------------------------------------------- partition
def test_partition_is_a_disjoint_cover_with_min_size():
    y = np.random.default_rng(0).integers(0, 6, 5000)
    parts, tries = dirichlet_partition(y, 5, 0.5, seed=0, min_size=300, n_classes=6)
    cat = np.concatenate(parts)
    assert len(cat) == len(y) and len(np.unique(cat)) == len(y)          # every row exactly once
    assert min(len(p) for p in parts) >= 300 and tries >= 1
    again, _ = dirichlet_partition(y, 5, 0.5, seed=0, min_size=300, n_classes=6)
    assert all(np.array_equal(a, b) for a, b in zip(parts, again))      # same seed, same split
    other, _ = dirichlet_partition(y, 5, 0.5, seed=1, min_size=300, n_classes=6)
    assert not all(np.array_equal(a, b) for a, b in zip(parts, other))


def _skew(y, parts, k=6):
    """Mean total-variation distance between a client's label distribution and the global one."""
    g = np.bincount(y, minlength=k) / len(y)
    return float(np.mean([0.5 * np.abs(np.bincount(y[p], minlength=k) / len(p) - g).sum() for p in parts]))


def test_small_alpha_is_more_skewed_than_large_alpha():
    y = np.random.default_rng(1).integers(0, 6, 20000)
    lo, _ = dirichlet_partition(y, 5, 0.1, 0, 200, 6)
    hi, _ = dirichlet_partition(y, 5, 100.0, 0, 200, 6)
    assert _skew(y, lo) > 0.3 > 0.05 > _skew(y, hi)


def test_partition_redraws_until_clients_are_big_enough_and_fails_when_impossible():
    y = np.random.default_rng(2).integers(0, 3, 600)
    _, tries = dirichlet_partition(y, 4, 0.05, 0, min_size=100, n_classes=3)
    assert tries > 1                                                      # alpha 0.05 almost always leaves a client short
    with pytest.raises(ValueError):
        dirichlet_partition(y, 4, 0.5, 0, min_size=200)                  # 4 x 200 > 600 rows
    with pytest.raises(RuntimeError):
        dirichlet_partition(y, 4, 0.01, 0, min_size=149, n_classes=3, max_tries=5)


def test_make_partition_stores_and_reloads(tmp_path):
    cfg = {"label_mode": "toy", "paths": {"work_dir": str(tmp_path)}, "fl": {"num_clients": 3, "dirichlet_alpha": 0.5, "min_client_size": 50}}
    y = np.random.default_rng(3).integers(0, 3, 1000)
    p1, m1 = make_partition(cfg, y, ["a", "b", "c"], seed=2)
    f = partition_path(cfg, 0.5, 2)
    assert f.exists() and f.name == "alpha0.5_seed2_toy.json" and json.loads(f.read_text())["sizes"] == m1["sizes"]
    p2, _ = make_partition(cfg, y, ["a", "b", "c"], seed=2)
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))
    t = label_table(y, p1, ["a", "b", "c"])
    assert t["n"].sum() == 1000 and (t[["a", "b", "c"]].sum(axis=1) == t["n"]).all()


# ----------------------------------------------------------------------------- FedAvg
def test_flower_aggregation_equals_hand_computed_weighted_mean():
    from flwr.app import ArrayRecord, MetricRecord, RecordDict
    from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords
    rng = np.random.default_rng(0)
    states, ns = [], [int(n) for n in rng.integers(50, 5000, 5)]
    for _ in ns:
        states.append(OrderedDict(w=torch.tensor(rng.normal(size=(7, 3)), dtype=torch.float32), b=torch.tensor(rng.normal(size=4), dtype=torch.float32)))
    recs = [RecordDict({"arrays": ArrayRecord(s), "metrics": MetricRecord({"num-examples": n})}) for s, n in zip(states, ns)]
    got = aggregate_arrayrecords(recs, "num-examples").to_torch_state_dict()
    want = core.fedavg_numpy([{k: v.numpy() for k, v in s.items()} for s in states], ns)
    for k in want:
        assert np.abs(got[k].numpy() - want[k]).max() < 1e-6
    assert np.abs(want["w"] - np.mean([s["w"].numpy() for s in states], axis=0)).max() > 1e-3   # weights matter


def test_fedavg_numpy_known_values():
    out = core.fedavg_numpy([{"a": np.array([0.0, 0.0])}, {"a": np.array([4.0, 8.0])}], [1, 3])
    np.testing.assert_allclose(out["a"], [3.0, 6.0])


# ----------------------------------------------------------------------------- local training and checkpoints
def _toy_state_and_data():
    schema = make_schema()
    lay = build_layout(schema)
    X, y = synthetic_data(600)
    return lay, X, y, core.init_state(lay, 3, HP, seed=0)


def test_local_train_is_deterministic_and_depends_on_round_and_client():
    lay, X, y, st = _toy_state_and_data()
    a = core.local_train(st, X, y, lay, 3, HP, 1, round_idx=1, seed=0, client_id=0)
    b = core.local_train(st, X, y, lay, 3, HP, 1, round_idx=1, seed=0, client_id=0)
    c = core.local_train(st, X, y, lay, 3, HP, 1, round_idx=2, seed=0, client_id=0)
    d = core.local_train(st, X, y, lay, 3, HP, 1, round_idx=1, seed=0, client_id=1)
    key = "out.weight"
    assert torch.equal(a["state"][key], b["state"][key])
    assert not torch.equal(a["state"][key], c["state"][key]) and not torch.equal(a["state"][key], d["state"][key])
    assert a["n"] == 600 and a["steps"] == int(np.ceil(600 / HP["batch_size"]))
    assert not torch.equal(a["state"][key], st[key])
    assert torch.equal(st[key], core.init_state(lay, 3, HP, seed=0)[key])      # the global state passed in is not modified in place


def test_init_state_is_common_for_a_seed():
    lay, *_ = _toy_state_and_data()
    s1, s2, s3 = (core.init_state(lay, 3, HP, seed=s) for s in (0, 0, 1))
    assert all(torch.equal(s1[k], s2[k]) for k in s1) and not all(torch.equal(s1[k], s3[k]) for k in s1)


def test_checkpoint_roundtrip_keeps_two_and_log_truncates(tmp_path):
    lay, X, y, st = _toy_state_and_data()
    for r in (2, 4, 6):
        core.save_checkpoint(tmp_path, st, r, {0: 3 * r, 1: 2 * r})
        core.append_round_log(tmp_path, {"round": r, "x": 1})
    ck = core.load_checkpoint(tmp_path)
    assert ck["round"] == 6 and ck["client_steps"] == {0: 18, 1: 12} and ck["dp_steps"] == {}
    assert len(list((tmp_path / "ckpt").glob("round_*.pt"))) == 2
    core.truncate_round_log(tmp_path, 4)
    assert [r["round"] for r in core.read_round_log(tmp_path)] == [2, 4]
    assert core.load_checkpoint(tmp_path / "nowhere") is None


# ----------------------------------------------------------------------------- Flower simulation (slow)
def _toy_cfg_and_data(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    schema = make_schema()
    schema["label_map"] = {"c0": 0, "c1": 1, "c2": 2}
    (data_dir / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    for name, n, seed in (("train", 900, 0), ("val", 300, 1)):
        X, y = synthetic_data(n, seed)
        np.savez(data_dir / f"{name}.npz", X=X, y=y)
    cfg = {"label_mode": "toy", "paths": {"work_dir": str(tmp_path / "work")}, "seeds": [0],
           "fl": {"num_clients": 3, "rounds": 6, "local_epochs": 1, "dirichlet_alpha": 0.5, "min_client_size": 60},
           "compute": {"artifacts_dir": str(tmp_path / "art"), "checkpoint_every_rounds": 2}}
    return cfg, data_dir


def test_flower_run_checkpoint_and_resume_equals_uninterrupted(tmp_path):
    from ppfeddata.fl.run import run_fl
    cfg, ddir = _toy_cfg_and_data(tmp_path)
    kw = {"data_dir": ddir, "hp": HP}
    full = run_fl(cfg, 0, "full", rounds=6, **kw)
    part = run_fl(cfg, 0, "cut", rounds=4, **kw)                         # "stopped" after 4 rounds
    assert part["summary"]["rounds_done"] == 4
    cut = run_fl(cfg, 0, "cut", rounds=6, resume=True, **kw)             # resumed from the round-4 checkpoint to 6
    art = tmp_path / "art"
    s_full = torch.load(art / full["run_id"] / "final_state.pt", weights_only=False)
    s_cut = torch.load(art / cut["run_id"] / "final_state.pt", weights_only=False)
    assert max(float((s_full[k] - s_cut[k]).abs().max()) for k in s_full) < 1e-5
    assert full["summary"]["rounds_done"] == cut["summary"]["rounds_done"] == 6
    assert [r["round"] for r in core.read_round_log(art / cut["run_id"])] == [0, 1, 2, 3, 4, 5, 6]
    assert cut["summary"]["client_steps"] == full["summary"]["client_steps"]
    sizes = full["partition_sizes"]
    assert full["summary"]["client_steps"] == {str(i): 6 * int(np.ceil(n / HP["batch_size"])) for i, n in enumerate(sizes)}
    rows = core.read_round_log(art / full["run_id"])
    assert rows[0]["round"] == 0 and rows[-1]["val_loss"] < rows[0]["val_loss"]          # the global model improves
    assert all(r["bytes"] > 0 and r["round_seconds"] > 0 for r in rows[1:])
    done = run_fl(cfg, 0, "full", rounds=6, **kw)                        # already finished: nothing is re-run
    assert done["summary"]["rounds_done"] == 6
    again = run_fl(cfg, 0, "full", rounds=6, agg_noise=1.0, **kw)        # a finished run keeps the spec it was run with
    assert again["summary"]["rounds_done"] == 6
    assert json.loads((art / full["run_id"] / "spec.json").read_text())["agg_noise"] == 0.0
