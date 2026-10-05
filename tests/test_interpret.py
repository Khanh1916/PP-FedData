"""Phase 12 tests: the shared paired bootstrap, the independent epsilon, and the interpretation rules (R1-R6, red flags, recommendation) on a
synthetic world whose answers are known by construction."""
import json
import re
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ppfeddata import aggregate as ag
from ppfeddata import interpret as it
from ppfeddata.eval import compare as cp
from ppfeddata.eval import dp_check as dc
from ppfeddata.eval.runs import run_id
from ppfeddata.eval.stats import paired_bootstrap_diff
from ppfeddata.eval.utility import confusion, macro_f1_from_confusion
from ppfeddata.fl import dp_utils
from ppfeddata.interpret_report import render
from tests.test_aggregate import CLASSES, make_cfg, make_ledger

K = len(CLASSES)


@pytest.fixture(autouse=True)
def tuned_trial_21(monkeypatch):
    """The tuned DP family name comes from configs/best_cvae_dp.yaml; fix it here so the tests do not depend on that file."""
    import ppfeddata.fl.m1 as m1
    monkeypatch.setattr(m1, "tuned_setup", lambda cfg: ("M1d-t21", {}, 1.0))


# --------------------------------------------------------------------------------------------------
# eval/compare.py
# --------------------------------------------------------------------------------------------------
def noisy(y, acc, rng, k=K):
    """Predictions that are right with probability `acc`, otherwise a different random class."""
    p = y.copy()
    wrong = rng.random(len(y)) > acc
    p[wrong] = (y[wrong] + rng.integers(1, k, wrong.sum())) % k
    return p


def small_world(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(K), n // K * 0 + 50)                    # 50 rows per class
    return rng, y


def test_point_confusion_and_metrics_match_the_reference_implementation():
    rng, y = small_world()
    p = noisy(y, 0.7, rng)
    b = cp.PairedBootstrap(y, K, n_boot=20, seed=1)
    b.add({"a": p})
    assert np.array_equal(b.point["a"], confusion(y, p, K))
    assert float(cp.MACRO_F1(b.point["a"])) == pytest.approx(macro_f1_from_confusion(confusion(y, p, K)), abs=1e-12)
    rec = [(p[y == c] == c).mean() for c in range(K)]
    assert float(cp.recall_of(2, 3)(b.point["a"])) == pytest.approx(np.mean(rec[2:4]))
    with pytest.raises(ValueError):
        cp.recall_of()(b.point["a"])
    assert b.conf3("a").shape == (20, K, K) and (b.conf3("a").sum(axis=(1, 2)) == len(y)).all()          # stratified: every resample keeps the test size


def test_row_bootstrap_reproduces_the_phase6_paired_bootstrap():
    rng, y = small_world()
    pa, pb = noisy(y, 0.8, rng), noisy(y, 0.6, rng)
    old = paired_bootstrap_diff(y, pa, pb, K, n_boot=200, seed=3)
    b = cp.PairedBootstrap(y, K, n_boot=200, seed=3)
    b.add({"a": pa, "b": pb})
    new = cp.compare_runs(b, ["a"], ["b"])
    assert new["delta"] == pytest.approx(old["diff"], abs=1e-12)
    assert new["lo"] == pytest.approx(old["lo"], abs=1e-9) and new["hi"] == pytest.approx(old["hi"], abs=1e-9)


def test_resamples_are_shared_deterministic_and_independent_of_what_is_added():
    rng, y = small_world()
    pa, pb = noisy(y, 0.8, rng), noisy(y, 0.6, rng)
    one = cp.PairedBootstrap(y, K, n_boot=30, seed=5)
    one.add({"a": pa, "b": pb})
    two = cp.PairedBootstrap(y, K, n_boot=30, seed=5)
    two.add({"b": pb})
    two.add({"a": pa})                                                           # separate call: the same resamples, so still paired
    assert np.array_equal(one.conf["a"], two.conf["a"]) and np.array_equal(one.conf["b"], two.conf["b"])
    other = cp.PairedBootstrap(y, K, n_boot=30, seed=6)
    other.add({"a": pa})
    assert not np.array_equal(one.conf["a"], other.conf["a"])
    same = cp.PairedBootstrap(y, K, n_boot=30, seed=5)
    same.add({"a": pa, "b": pb})
    one.add({"a": pb})                                                           # already present keys are not recomputed
    assert np.array_equal(same.conf["a"], one.conf["a"])
    with pytest.raises(ValueError):
        one.add({"bad": pa[:-1]})
    with pytest.raises(ValueError):
        one.add({"bad2": np.full_like(pa, K)})


def test_effect_rule_needs_both_the_interval_and_the_seed_std():
    c = lambda **kw: {**{"delta": 0.05, "lo": 0.03, "hi": 0.07, "sigma": 0.01}, **kw}          # noqa: E731
    assert cp.effect_of(c()) == "better"
    assert cp.effect_of(c(delta=-0.05, lo=-0.07, hi=-0.03)) == "worse"
    assert cp.effect_of(c(lo=-0.01)) == "none"                                                   # interval includes 0
    assert cp.effect_of(c(sigma=0.08)) == "none"                                                 # interval excludes 0 but |delta| is inside the seed std
    assert cp.effect_of(c(delta=-0.02, lo=0.01, hi=0.03)) == "none"                              # inconsistent sign never counts


def test_identical_runs_give_a_zero_interval_and_a_clear_gain_is_found():
    rng, y = small_world(seed=2)
    seeds = {s: (noisy(y, 0.85, rng), noisy(y, 0.60, rng)) for s in (0, 1, 2)}
    b = cp.PairedBootstrap(y, K, n_boot=100, seed=0)
    b.add({("good", s): g for s, (g, _) in seeds.items()} | {("bad", s): w for s, (_, w) in seeds.items()})
    same = cp.compare_runs(b, [("good", s) for s in seeds], [("good", s) for s in seeds])
    assert same["delta"] == 0 and same["lo"] == 0 and same["hi"] == 0 and same["effect"] == "none"
    gain = cp.compare_runs(b, [("good", s) for s in seeds], [("bad", s) for s in seeds])
    assert gain["effect"] == "better" and gain["lo"] > 0 and gain["n_seeds"] == 3 and len(gain["per_seed"]) == 3
    assert cp.compare_runs(b, [("bad", s) for s in seeds], [("good", s) for s in seeds])["effect"] == "worse"
    with pytest.raises(ValueError):
        cp.compare_runs(b, [("good", 0)], [("good", 0), ("good", 1)])


def test_stream_bootstrap_is_wider_when_the_errors_come_in_whole_streams():
    rng = np.random.default_rng(4)
    sizes = [10, 20, 30, 40, 20, 30]                                              # 4 classes x 6 streams of unequal size (150 packets per class)
    y = np.repeat(np.arange(4), sum(sizes))
    streams = np.array([f"c{c}s{i}" for c in range(4) for i, n in enumerate(sizes) for _ in range(n)])
    wrong_a = {f"c{c}s{i}" for c in range(4) for i in rng.choice(6, 1, replace=False)}            # a: one bad stream per class
    wrong_b = {f"c{c}s{i}" for c in range(4) for i in rng.choice(6, 2, replace=False)}            # b: two bad streams per class
    pa, pb = y.copy(), y.copy()
    for pred, bad in ((pa, wrong_a), (pb, wrong_b)):
        m = np.isin(streams, list(bad))
        pred[m] = (y[m] + 1) % 4
    rows = cp.PairedBootstrap(y, 4, n_boot=300, seed=0)
    rows.add({"a": pa, "b": pb})
    st = cp.PairedBootstrap(y, 4, n_boot=300, seed=0, mode="streams", clusters=streams)
    st.add({"a": pa, "b": pb})
    wr, ws = (lambda c: c["hi"] - c["lo"])(cp.compare_runs(rows, ["a"], ["b"])), (lambda c: c["hi"] - c["lo"])(cp.compare_runs(st, ["a"], ["b"]))
    assert ws > 1.5 * wr, (wr, ws)
    assert np.array_equal(st.point["a"], rows.point["a"])                          # the full-data confusion is the same
    assert st.conf3("a").sum(axis=(1, 2)).std() > 0                                # the number of rows varies between stream resamples
    with pytest.raises(ValueError, match="several classes"):
        cp.PairedBootstrap(np.array([0, 1]), 2, mode="streams", clusters=np.array(["s", "s"]))
    with pytest.raises(ValueError, match="stream id"):
        cp.PairedBootstrap(y, 4, mode="streams")


# --------------------------------------------------------------------------------------------------
# eval/dp_check.py
# --------------------------------------------------------------------------------------------------
GRID = [(0.0123, 1.21, 3420), (0.0073, 1.085, 4740), (0.05, 2.0, 600), (0.5, 3.0, 300), (1.0, 3.0, 60), (0.02, 0.9, 2000)]


def test_independent_epsilon_equals_opacus_on_the_same_integer_orders():
    from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
    orders = list(range(2, 64))
    for q, s, t in GRID:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref, _ = get_privacy_spent(orders=orders, rdp=compute_rdp(q=q, noise_multiplier=s, steps=t, orders=orders), delta=1e-5)
        assert dc.epsilon_independent(s, q, t, 1e-5, orders=orders) == pytest.approx(float(ref), rel=1e-9)


def test_independent_epsilon_is_close_to_and_not_below_the_project_accountant():
    for q, s, t in GRID[:3]:
        mine, theirs = dc.epsilon_independent(s, q, t, 1e-5), dp_utils.epsilon(s, q, t, 1e-5)
        assert theirs * (1 - 1e-9) <= mine <= theirs * 1.02, (q, s, t, mine, theirs)           # Opacus also searches fractional orders: never larger


def test_rdp_closed_forms_and_monotonicity():
    for s in (0.8, 2.0):
        for a in (2, 5, 17):
            assert dc.rdp_int(1.0, s, a) == pytest.approx(a / (2 * s * s), rel=1e-12)             # q = 1 is the plain Gaussian mechanism
            assert dc.rdp_int(0.0, s, a) == pytest.approx(0.0, abs=1e-12)                         # nothing is sampled
    assert dc.epsilon_independent(1.0, 0.01, 0, 1e-5) == 0.0 and dc.epsilon_independent(0.0, 0.01, 10, 1e-5) == float("inf")
    base = dc.epsilon_independent(1.2, 0.01, 1000, 1e-5)
    assert dc.epsilon_independent(1.2, 0.01, 2000, 1e-5) > base > dc.epsilon_independent(2.0, 0.01, 1000, 1e-5)


def test_loader_length_rule_matches_a_real_opacus_loader():
    import torch
    from opacus.data_loader import DPDataLoader
    from torch.utils.data import DataLoader, TensorDataset
    quirk = [n for n in (50501, 47105) if dc.loader_steps_per_epoch(n, 512) != -(-n // 512)]
    assert quirk == [50501, 47105]                                                  # 99 and 93 batches per epoch: 1 / (1 / x) lands just below x
    for n in (50501, 47105, 40106, 7557, 600):
        dl = DPDataLoader.from_data_loader(DataLoader(TensorDataset(torch.zeros(n, 1)), batch_size=512))
        assert len(dl) == dc.loader_steps_per_epoch(n, 512), n


@pytest.mark.filterwarnings("ignore:Optimal order")
def test_recompute_run_agrees_with_the_project_epsilon_and_explains_the_step_gap():
    sizes, bs, rounds, epochs, delta = [3000, 50501, 1200], 512, 30, 2, 1e-5
    sig = dp_utils.calibrate_clients(sizes, 5.0, delta, bs, rounds, epochs)
    spec = {"partition_sizes": sizes, "hp": {"batch_size": bs}, "rounds": rounds, "local_epochs": epochs, "dp": {"sigmas": sig, "delta": delta, "target_eps": 5.0}}
    plan = {i: rounds * epochs * dp_utils.steps_per_epoch(n, bs) for i, n in enumerate(sizes)}
    loader = {i: rounds * epochs * dc.loader_steps_per_epoch(n, bs) for i, n in enumerate(sizes)}
    reported = dp_utils.epsilon_table(sizes, sig, loader, bs, delta, 5.0)["eps_max"]
    res = dc.recompute_run(spec, {str(k): v for k, v in loader.items()}, reported)
    assert abs(res["rel_diff"]) < 0.02 and res["steps_match_loader"] and not res["steps_match_plan"]      # client 1: 98 batches per epoch, not 99
    ok = dc.recompute_run(spec, plan, reported)
    assert ok["steps_match_plan"] and ok["eps_independent"] >= res["eps_independent"]
    bad = dc.recompute_run(spec, {0: 10, 1: 10, 2: 10}, reported)
    assert not bad["steps_match_plan"] and not bad["steps_match_loader"] and bad["rel_diff"] < -0.5


# --------------------------------------------------------------------------------------------------
# The synthetic world
# --------------------------------------------------------------------------------------------------
ACC = {"B0-rf": 0.60, "B0-mlp": 0.40, "B1a-rf": 0.68, "B1b-rf": 0.72, "B1b-mlp": 0.62}


def accuracy(name):
    """Accuracy of the synthetic run `name`: TAug-RF equals B0-RF (flips only), TAug-MLP is a bit above B0-MLP, TSTR-MLP is above B0-MLP but below SMOTE
    (the class-balance effect), DP generators are poor in TSTR."""
    if name in ACC:
        return ACC[name]
    dp = name.startswith(("M1d", "M3d", "M1-"))
    if "-TSTR-" in name:
        return 0.30 if dp else (0.52 if name.endswith("rf") else 0.55)
    return 0.45 if name.endswith("mlp") else 0.60


def make_world(tmp_path, n_per_class=120, per_stream=6, n_boot=40):
    """A ledger with the shape of the real one (from test_aggregate), the real-test split on disk, one prediction file per ledger row (accuracy by config)
    and the specs of the DP runs with the epsilon the project's accountant gives."""
    df = make_ledger(tmp_path, hashes=("h0", "h1", "h2"))
    cfg = make_cfg(tmp_path, df)
    y = np.repeat(np.arange(K), n_per_class)
    streams = np.array([f"{CLASSES[c]}:{i // per_stream}" for c in range(K) for i in range(n_per_class)])
    groups = np.array([f"g{c}" for c in range(K) for _ in range(n_per_class)])
    pdir = tmp_path / "work" / "processed" / "6class"
    pdir.mkdir(parents=True)
    np.savez(pdir / "test.npz", y=y, stream_id=streams, group_id=groups)
    (pdir / "feature_schema.json").write_text(json.dumps({"label_map": {c: i for i, c in enumerate(CLASSES)}}), encoding="utf-8")
    rng = np.random.default_rng(11)
    b0, made = {}, {}
    rows = []
    twin = lambda n: n.replace("M2-", "B3-", 1) if n.startswith("M2-") else n.replace("M3d-t21-eps5", "M1d-t21-eps5", 1) if n.startswith("M3d-t21-eps5") else None      # noqa: E731
    for _, r in df.sort_values(["seed", "config"]).iterrows():
        name, s = r["config"], int(r["seed"])
        if name in ACC:
            p = noisy(y, ACC[name], rng)
            if name == "B0-rf":
                b0[s] = p
        elif twin(name) and (twin(name), s) in made:
            p = made[(twin(name), s)].copy()                                       # secure aggregation does not change what the model learns here: identical predictions
        elif "-TAug-rf" in name:
            p = b0[s].copy() if s in b0 else noisy(y, 0.60, rng)                 # real data dominate: the same predictions up to a few flips
            flip = rng.random(len(y)) < 0.01
            p[flip] = (y[flip] + 1) % K
        else:
            p = noisy(y, accuracy(name), rng)
        made[(name, s)] = p
        d = tmp_path / "art" / run_id(name, s, "6class") / "preds"
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "test.npz", y_pred=p.astype(np.int16))
        conf = confusion(y, p, K)
        row = r.to_dict()
        row["macro_f1"] = macro_f1_from_confusion(conf)
        for c in range(K):
            row[f"recall_{CLASSES[c]}"] = conf[c, c] / conf[c].sum()
        rows.append(row)
    df = pd.DataFrame(rows)
    # DP runs: specs with calibrated noise, step counters of the loader, and the epsilon of the project's accountant in the ledger
    sizes, bs, rounds, epochs, delta = [3000, 2500, 2000, 1500, 1000], 512, 30, 2, 1e-5
    calib = {}
    for fl in sorted({ag.parse_config(n)["fl_run"] for n in df["config"] if ag.parse_config(n)["eps"] == ag.parse_config(n)["eps"]} - {None}):
        eps = ag.parse_config(next(n for n in df["config"] if ag.parse_config(n)["fl_run"] == fl))["eps"]
        if eps not in calib:                                                          # one calibration per target epsilon
            sig = dp_utils.calibrate_clients(sizes, eps, delta, bs, rounds, epochs)
            steps = {str(i): rounds * epochs * dc.loader_steps_per_epoch(n, bs) for i, n in enumerate(sizes)}
            calib[eps] = (sig, steps, dp_utils.epsilon_table(sizes, sig, steps, bs, delta, eps)["eps_max"])
        sig, steps, e_max = calib[eps]
        for s in (0, 1, 2):
            d = tmp_path / "art" / run_id(fl, s, "6class")
            if not d.exists():
                continue
            (d / "spec.json").write_text(json.dumps({"partition_sizes": sizes, "hp": {"batch_size": bs}, "rounds": rounds, "local_epochs": epochs,
                                                     "dp": {"sigmas": sig, "delta": delta, "target_eps": eps}}), encoding="utf-8")
            lines = (d / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
            last = json.loads(lines[-1])
            last["client_dp_steps"] = steps
            (d / "rounds.jsonl").write_text("\n".join(lines[:-1] + [json.dumps(last)]) + "\n", encoding="utf-8")
        mask = df["config"].map(lambda n, fl=fl: ag.parse_config(n)["fl_run"] == fl)
        df.loc[mask, "dp_eps_max"] = e_max
    df.to_csv(tmp_path / "art" / "runs.csv", index=False)
    cfg.update(paths={"work_dir": str(tmp_path / "work"), "shared_manifest_dir": str(tmp_path / "manifests")}, eval={"bootstrap": n_boot}, tune={"syn_per_class": 5000})
    cfg["thresholds"].update(seed_std_max=0.02, eps_max_recommend=5.0, seed_std_redflag=0.05, f1_near_one=0.95, eps_recompute_rtol=0.10)
    (tmp_path / "art" / "B2_positive_control_6class.json").write_text(json.dumps({"mia_auc_mean": 0.52, "copier_sensitivity": {"0.0": 0.97, "0.05": 0.66, "0.5": 0.50}}), encoding="utf-8")
    return cfg, df


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("world")
    cfg, df = make_world(tmp)
    ann = ag.annotate(ag.load_ledger(cfg), cfg)
    summ = ag.build_summary(ann, cfg)
    R = it.interpret(cfg, summ, ann, n_boot=40)
    return SimpleNamespace(cfg=cfg, df=ann, summ=summ, R=R, tmp=tmp)


def row_of(rows, label, **kw):
    return next(r for r in rows if r["label"] == label and all(r.get(k) == v for k, v in kw.items()))


# --------------------------------------------------------------------------------------------------
# interpret.py
# --------------------------------------------------------------------------------------------------
def test_predictions_reproduce_the_ledger_and_the_meta_is_complete(world):
    m = world.R["meta"]
    assert m["integrity_max_abs_diff_vs_ledger"] < 1e-12 and m["bootstrap_modes"] == ["rows", "streams"] and m["seeds"] == [0, 1, 2]
    assert m["n_test_rows"] == K * 120 and m["n_groups"] == K and m["n_streams"] == K * 20 and m["rare_classes"] == ["DELAYED", "SYN", "INVALID", "WILL"]


def test_r1_finds_the_balancing_references_and_no_gain_of_the_rf_generators(world):
    rows = world.R["R1"]["rows"]
    smote = row_of(rows, "B1b")
    assert smote["reference"] and smote["rf"]["macro_f1"]["effect"] == "better" and smote["mlp"]["macro_f1"]["effect"] == "better"
    assert smote["rf"]["rare_mean"]["delta"] > 0.05
    for lab in ("B2", "B3", "M2", "M1-eps5", "M3-eps5"):
        r = row_of(rows, lab)
        assert not r["reference"] and r["rf"]["macro_f1"]["effect"] == "none", lab               # same predictions as B0 up to 1 % flips
        assert set(r["rf"]["per_class"]) == {"DELAYED", "SYN", "INVALID", "WILL"}
    assert "mlp" not in row_of(rows, "B1a") and row_of(rows, "B3")["mlp"]["macro_f1"]["delta"] > 0       # B1a is RF only; the MLP gains a bit


def test_r2_cvae_is_worse_than_the_simple_methods(world):
    rows = world.R["R2"]["rows"]
    assert len(rows) == 6 and all(r["macro_f1"]["effect"] == "worse" for r in rows)
    assert {(r["generator"], r["reference"], r["classifier"]) for r in rows} == {(g, ref, c) for g in ("B2", "B3") for ref, c in (("B1a", "rf"), ("B1b", "rf"), ("B1b", "mlp"))}


def test_r3_reports_the_federation_loss_for_the_four_protocol_classifier_pairs(world):
    rows = world.R["R3"]["rows"]
    assert [(r["protocol"], r["classifier"]) for r in rows] == [("TSTR", "rf"), ("TSTR", "mlp"), ("TAug", "rf"), ("TAug", "mlp")]
    r = rows[0]
    assert r["relative_loss"] == pytest.approx(r["macro_f1"]["delta"] / r["macro_f1"]["mean_a"])


def test_r4_dp_cost_curves_privacy_rows_and_the_positive_control(world):
    r4 = world.R["R4"]
    tstr = [c for c in r4["cost"] if c["protocol"] == "TSTR"]
    assert len(r4["cost"]) == 12 and all(c["macro_f1"]["effect"] == "worse" for c in tstr)         # accuracy 0.30 against 0.52
    assert [p["eps"] for p in r4["privacy"]] == ["inf", "10", "5", "1"] and r4["privacy"][0]["eps_max"] is None and all(p["eps_max"] <= float(p["eps"]) for p in r4["privacy"][1:])
    assert set(r4["curves"]) >= {"TSTR-rf", "TSTR-mlp"} and r4["curves"]["TSTR-rf"]["monotone_within_noise"]
    assert r4["mia"]["near_chance"] and r4["mia"]["approaches_chance"] is False
    pc = r4["positive_control"]
    assert pc["available"] and pc["overfit_cvae_detected"] is False and pc["copier_detected_at_zero_noise"] is True


def test_curve_check_flags_a_step_that_hurts_by_more_than_the_seed_std():
    ok = it._curve_checks({1: (0.2, 0.02), 5: (0.3, 0.02), 10: (0.31, 0.02)})
    assert ok["monotone_within_noise"] and not ok["flat"]
    bad = it._curve_checks({1: (0.3, 0.01), 5: (0.2, 0.01), 10: (0.25, 0.01)})
    assert not bad["monotone_within_noise"] and bad["violations"][0]["from_eps"] == 1
    flat = it._curve_checks({1: (0.2, 0.05), 5: (0.22, 0.05), 10: (0.21, 0.05)})
    assert flat["flat"] and flat["monotone_within_noise"]
    low = it._curve_checks({1: (9.0, 0.1), 5: (7.0, 0.1), 10: (8.0, 0.1)}, higher_is_better=False)                  # ELBO: lower is better
    assert not low["monotone_within_noise"]


def test_r5_secagg_is_inside_the_noise_and_the_overhead_is_quantified(world):
    r5 = world.R["R5"]
    assert len(r5["pairs"]) == 8 and all(p["macro_f1"]["effect"] == "none" for p in r5["pairs"])
    o = {x["pair"]: x for x in r5["overhead"]}
    assert o["M2 / B3"]["time_ratio"] == pytest.approx((2.1 + 0.03) / (1.8 + 0.03), abs=0.02) and o["M2 / B3"]["bytes_per_param_ratio"] == pytest.approx((6.4e6 / 1e5) / (4e6 / 1e5))


def test_r6_filters_ties_and_the_protection_tie_break(world):
    r6 = world.R["R6"]
    by = {c["label"]: c for c in r6["candidates"]}
    assert not by["B2"]["eligible"] and not by["B2"]["federated"]                           # pooled data
    assert by["B3"]["eligible"] and by["M2"]["eligible"]
    assert all(not by[k]["ok_overhead"] and not by[k]["eligible"] for k in ("M1-eps1", "M1-eps5", "M1-eps10", "M3-eps5"))
    assert not by["M1-eps10"]["ok_eps"] and by["M1-eps5"]["ok_eps"] and all(c["ok_mia"] for c in by.values())
    assert set(r6["tied"]) == {"B3", "M2"} and r6["recommended"] == "M2"                    # equal utility: SecAgg is the stronger protection
    assert set(r6["tstr_ranking"]["tied"]) == {"B3", "M2"} and r6["tstr_ranking"]["recommended"] == "M2"          # ranked by the synthetic-only macro-F1 the answer is the same
    alt = r6["dp_alternative"]
    assert alt["recommended"] == "M1-eps1" and set(alt["tied"]) >= {"M1-eps1", "M1-eps5", "M3-eps5"} and "M1-eps10" not in alt["tied"]


def test_r6_without_any_eligible_configuration_gives_no_recommendation(world):
    ctx = it.build_context(world.cfg, world.summ, world.df, n_boot=20)
    cfg = json.loads(json.dumps(world.cfg))
    cfg["thresholds"]["overhead_ratio_max"] = 0.5
    ctx.cfg = cfg
    cost = {r["label"]: r for r in ag.cost_rows(world.summ, ctx.ents)}
    r6 = it.r6(ctx, cost)
    assert r6["recommended"] is None and r6["literal"] is None and r6["tied"] == [] and not any(c["eligible"] for c in r6["candidates"])


def test_red_flags_on_the_world(world):
    F = {f["id"]: f for f in world.R["flags"]}
    assert not F["F1"]["triggered"] and not F["F3"]["triggered"] and not F["F4"]["triggered"] and not F["F5"]["triggered"]
    f2 = F["F2"]
    assert f2["triggered"] and "investigated" in f2["status"] and f2["evidence"]["summary"]["hit_classifiers"] == ["mlp"]
    assert f2["evidence"]["summary"]["below_balanced_in_all_hits"] is True
    assert f2["evidence"]["summary"]["recall_signature_in_all_hits"] is False                # this world's errors are uniform over the classes: no class-balancing signature
    hit = [r for r in f2["evidence"]["rows"] if r["vs_b0"]["effect"] == "better"]
    assert {r["label"] for r in hit} == {"B2", "B3", "M2"} and all(r["vs_balanced_real"]["B1b"]["effect"] == "worse" for r in hit)
    ev5 = F["F5"]["evidence"]
    assert ev5["n_runs"] >= 9 and ev5["max_abs_rel_diff"] < 0.02 and ev5["n_counters_unexplained"] == 0
    assert F["F4"]["evidence"]["n_in_matrix"] > 0 and F["F4"]["evidence"]["soft_breakdown"]["by_method"] is not None


def test_f2_stays_open_when_tstr_beats_even_the_balanced_real_data(world):
    ctx = it.build_context(world.cfg, world.summ, world.df, n_boot=20)
    ctx.seeds_of = dict(ctx.seeds_of)
    # pretend the balanced real reference is the weak B0-mlp: TSTR-mlp then beats both
    orig = ctx.cmp
    ctx.cmp = lambda a, b, metric=cp.MACRO_F1: orig(a, "B0-mlp" if b == "B1b-mlp" else b, metric)
    f = it.flag_tstr_over_trtr(ctx)
    assert f["triggered"] and "OPEN" in f["status"] and not f["evidence"]["summary"]["below_balanced_in_all_hits"]


def test_f1_f3_f4_f5_trigger_when_the_numbers_say_so(world):
    summ, df, cfg = world.summ.copy(), world.df.copy(), world.cfg
    # F1: B0 near 1
    s1 = summ.copy()
    s1.loc[s1["config"] == "B0-rf", "macro_f1_mean"] = 0.99
    assert it.flag_near_one(SimpleNamespace(cfg=cfg, summ=s1))["triggered"]
    # F3: duplicates
    d3 = df.copy()
    d3.loc[d3["prefix"] == "B3", "dup_rate"] = 0.2
    f3 = it.flag_copying(SimpleNamespace(cfg=cfg, df=d3))
    assert f3["triggered"] and f3["evidence"]["max_dup_rate"][0]["value"] == 0.2
    d3b = df.copy()
    d3b.loc[d3b["prefix"] == "B2", "dcr_ratio_mean"] = 0.1
    f3b = it.flag_copying(SimpleNamespace(cfg=cfg, df=d3b))
    assert f3b["triggered"] and f3b["evidence"]["min_dcr_ratio"][0]["value"] == 0.1
    # F4: spread, once outside and once inside the matrix
    s4 = summ.copy()
    s4.loc[s4["config"] == "M1-eps1-plain-TSTR-rf", "macro_f1_std"] = 0.09
    f4 = it.flag_seed_spread(SimpleNamespace(cfg=cfg, summ=s4))
    assert f4["triggered"] and "outside the spec matrix" in f4["status"] and f4["evidence"]["n_above_redflag_in_matrix"] == 0
    s4.loc[s4["config"] == "B3-TSTR-rf", "macro_f1_std"] = 0.09
    assert "including rows of the spec matrix" in it.flag_seed_spread(SimpleNamespace(cfg=cfg, summ=s4))["status"]
    # F5: a reported epsilon 1.5 times too large, or counters that match neither the plan nor the loader
    d5 = df.copy()
    d5.loc[d5["config"].str.startswith("M1d-t21-eps5"), "dp_eps_max"] *= 1.5
    f5 = it.flag_epsilon(SimpleNamespace(cfg=cfg, df=d5))
    assert f5["triggered"] and f5["evidence"]["n_far"] > 0 and f5["evidence"]["max_abs_rel_diff"] > 0.3
    rj = world.tmp / "art" / "M1d-t21-eps5_0" / "rounds.jsonl"
    saved = rj.read_text(encoding="utf-8")                                                          # the world is shared by the module: put the log back afterwards
    rj.write_text(json.dumps({"round": 30, "client_dp_steps": {str(i): 7 for i in range(5)}}) + "\n", encoding="utf-8")
    try:
        f5b = it.flag_epsilon(SimpleNamespace(cfg=cfg, df=df))
        assert f5b["triggered"] and f5b["evidence"]["n_counters_unexplained"] >= 1                 # counters that match neither the plan nor the loader
    finally:
        rj.write_text(saved, encoding="utf-8")


def test_to_jsonable_and_write_json_round_trip(world, tmp_path):
    assert it.to_jsonable({"a": np.float64(1.5), "b": np.int64(3), "c": np.bool_(True), "d": float("nan"), "e": (np.float32(2.0),)}) == {"a": 1.5, "b": 3, "c": True, "d": None, "e": [2.0]}
    p = it.write_json(world.R, tmp_path / "x" / "interpretation.json")
    back = json.loads(p.read_text(encoding="utf-8"))
    assert back["R6"]["recommended"] == "M2" and back["meta"]["n_runs"] == world.R["meta"]["n_runs"] and len(back["flags"]) == 5


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def tables_are_well_formed(text):
    """Every block of table lines has the same number of cells in each line (a stray pipe in a cell breaks a table)."""
    block = []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            block.append(len(re.findall(r"(?<!\\)\|", line)))
        else:
            if block and len(set(block)) != 1:
                return False
            block = []
    return True


def test_rendered_section_has_every_part_and_well_formed_tables(world):
    text = "\n".join(render(world.R, world.cfg))
    for h in ("## 9. Interpretation (Phase 12)", "### 9.0 Answers at a glance", "### 9.1 R1", "### 9.2 R2", "### 9.3 R3", "### 9.4 R4", "### 9.5 R5", "### 9.6 R6", "### 9.7 Red flags",
              "### 9.8 What secure aggregation and differential privacy each protect", "### 9.9 Why a CVAE, given R1 and R2", "How a difference is judged"):
        assert h in text, h
    assert tables_are_well_formed(text)
    ans = text.split("### 9.1 R1")[0]
    for k in ("**R1", "**R2", "**R3", "**R4", "**R5", "**R6", "**Why a CVAE at all?**", "**Red flags**"):
        assert k in ans, k
    assert "premise of the spec" in ans and "Ranked by TSTR instead of TAug" in text and "What R6 does not say" in text and "section 9.9 explains why the study uses it" in text
    assert "the tie is broken in favour of the stronger protection: **M2**" in ans or "**M2**" in ans
    assert "does not detect an over-fitted CVAE" in text and "honest-but-curious" in text
    assert "R1 found 0 of 7 generators better than real data only for the RF" in text
    assert "Removed by the overhead filter: M1-eps1, M1-eps5, M1-eps10, M3-eps5" in text


def test_report_text_is_generated_from_the_verdicts_not_fixed(world):
    R = json.loads(json.dumps(it.to_jsonable(world.R)))
    for r in R["R2"]["rows"]:                                                            # pretend the CVAE beat the simple methods
        r["macro_f1"]["effect"] = "better"
    R["R6"]["recommended"], R["R6"]["literal"], R["R6"]["tied"] = None, None, []
    txt = "\n".join(render(R, world.cfg))
    r2 = next(line for line in txt.splitlines() if line.startswith("- **R2"))
    assert "Not in general" in r2 and "6 of 6 comparisons worse" not in r2 and "0 of 6" not in r2.split("TAug with the CVAE")[0]
    assert "No configuration satisfies all three conditions" in txt
    f2 = next(f for f in R["flags"] if f["id"] == "F2")["evidence"]["summary"]
    assert "does not show the signature of class balancing" in txt or "does not show the signature" in txt                # the world has no class-balancing pattern
    f2["recall_signature_in_all_hits"], f2["below_balanced_in_all_hits"] = True, False
    t2 = "\n".join(render(R, world.cfg))
    assert "the signature of class balancing" in t2 and "does not show the signature" not in t2 and "In at least one flagged pair TSTR is not below that reference" in t2


def test_why_block_has_both_settings_and_the_numbers_of_the_summary(world):
    W = world.R["why"]
    ids = [x["id"] for x in W["rows"]]
    assert ids[:5] == ["B1b:TRTR", "B1a:TRTR", "B0:TRTR", "B2:TAug", "B3:TAug"] and "B3:TSTR" in ids and "B2:TSTR" not in ids         # B2 pools the data: not in the 'clients' setting
    by = {x["id"]: x for x in W["rows"]}
    assert by["B0:TRTR"]["needs_pooled_real_data"] and not by["M2:TSTR"]["needs_pooled_real_data"] and "mlp" not in by["B1a:TRTR"]["macro_f1"]
    assert by["B3:TSTR"]["macro_f1"]["rf"] == pytest.approx(world.summ.set_index("config").loc["B3-TSTR-rf", "macro_f1_mean"])
    assert W["retention_primary"]["B3"] == pytest.approx(by["B3:TSTR"]["macro_f1"]["rf"] / W["b0"]["rf"] - 1)
    assert set(W["c2st"]) >= {"B2", "B3", "M2"} and W["train_seconds"]["B3"] == pytest.approx(80.0)
    dpl = W["dp_tstr_relative_loss"]
    assert dpl["n"] == 6 and dpl["min"] < dpl["max"] < 0                                                          # accuracy 0.30 against 0.52 / 0.55


def test_why_section_follows_the_verdicts_and_degrades_without_data(world):
    R = json.loads(json.dumps(it.to_jsonable(world.R)))
    txt = "\n".join(render(R, world.cfg))
    sec = txt.split("### 9.9")[1]
    assert "do not support the premise" in sec and "negative result for the premise" in sec and "the premise of the study, not the outcome of a comparison" in sec
    assert "not tested" in sec.lower() and "federated training of the IDS classifier itself" in sec and "WGAN-GP" in sec and "SPEC_DEVIATIONS 6.7" in sec
    assert "(1) the real data can be pooled" in sec and "**M2** by the rule of 9.6" in sec and "or M1-eps1 if a formal guarantee is required" in sec
    # the bullet names both classifiers with the SMOTE reference
    bullet = next(line for line in txt.splitlines() if line.startswith("- **Why a CVAE at all?**"))
    assert "RF 0 of 7" in bullet and "MLP" in bullet and "for SMOTE" in bullet and "without pooling the raw data" in bullet
    # pretend the CVAE did help the RF and beat SMOTE once: the text must change with the verdicts
    R2 = json.loads(json.dumps(R))
    next(r for r in R2["R1"]["rows"] if r["label"] == "B3")["rf"]["macro_f1"]["effect"] = "better"
    R2["R2"]["rows"][0]["macro_f1"]["effect"] = "better"
    t2 = "\n".join(render(R2, world.cfg))
    s2 = t2.split("### 9.9")[1]
    assert "support the premise only in part" in s2 and "negative result for the premise" not in s2 and "(1) the real data can be pooled" not in s2
    assert "support the CVAE only in part" in t2 and "section 9.9 explains why the study uses it" not in t2 and "See R1 and R2 for how a CVAE pipeline compares" in t2
    # without the block the section says so instead of failing
    R3 = json.loads(json.dumps(R))
    del R3["why"]
    t3 = "\n".join(render(R3, world.cfg))
    assert "### 9.9 Why a CVAE, given R1 and R2" in t3 and "_(not available)_" in t3 and "**Why a CVAE at all?**" not in t3
    # a changed ranking by TSTR is reported as such
    R4 = json.loads(json.dumps(R))
    R4["R6"]["tstr_ranking"]["recommended"] = "B3"
    assert "a different recommendation from the one above" in "\n".join(render(R4, world.cfg))


def test_aggregate_with_interpretation_writes_json_and_section_9(world, tmp_path):
    out = ag.aggregate(world.cfg, out_dir=tmp_path / "res", n_boot=30)
    text = (tmp_path / "res" / "reports" / "final_report.md").read_text(encoding="utf-8")
    assert out["interpretation"] and json.loads((tmp_path / "res" / "interpretation.json").read_text(encoding="utf-8"))["R6"]["recommended"] == "M2"
    assert "## 9. Interpretation (Phase 12)" in text and "not applied yet" not in text and "Section 9 applies the interpretation rules" in text
    assert "## 10. Limitations" in text and tables_are_well_formed(text)


def test_aggregate_without_interpretation_or_without_predictions_keeps_the_placeholder(world, tmp_path):
    ag.aggregate(world.cfg, out_dir=tmp_path / "a", with_interpretation=False)
    t = (tmp_path / "a" / "reports" / "final_report.md").read_text(encoding="utf-8")
    assert "not applied yet" in t and "Not generated" in t and not (tmp_path / "a" / "interpretation.json").exists()
    cfg = json.loads(json.dumps(world.cfg))
    cfg["paths"]["work_dir"] = str(tmp_path / "nowhere")                                  # no test split on disk
    out = ag.aggregate(cfg, out_dir=tmp_path / "b")
    assert out["interpretation"] is None and "not applied yet" in (tmp_path / "b" / "reports" / "final_report.md").read_text(encoding="utf-8")
