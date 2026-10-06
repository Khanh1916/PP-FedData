"""Follow-up tests: the federated classifier baseline and the sensitivity worlds (A4, A5), on tiny synthetic data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ppfeddata import fed_classifier as fc
from ppfeddata import sensitivity as se
from ppfeddata.eval.runs import run_id, save_predictions

CLASSES = ["NORMAL", "BCF", "DELAYED", "SYN"]
K = len(CLASSES)


def blobs(n_per_class, seed, d=6):
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(123).normal(0, 4, size=(K, d))        # the same class centres in every split
    y = np.repeat(np.arange(K), n_per_class)
    X = centers[y] + rng.normal(0, 1, size=(len(y), d))
    return X.astype(np.float32), y.astype(np.int64)


def write_world(root: Path, seed=0, groups=True):
    """processed/<mode>/{train,val,test}.npz + feature_schema.json in `root`/processed/6class."""
    d = root / "processed" / "6class"
    d.mkdir(parents=True)
    sizes = {"train": 400, "val": 60, "test": 80}
    for i, (split, n) in enumerate(sizes.items()):
        X, y = blobs(n // K, seed + i)
        g = np.array([f"{split}-{c}-{j % 3}" for j, c in enumerate(y)])
        s = np.array([f"{split}:{j // 2}" for j in range(len(y))])
        np.savez(d / f"{split}.npz", X=X, y=y, group_id=g, stream_id=s)
    (d / "feature_schema.json").write_text(json.dumps({"label_map": {c: i for i, c in enumerate(CLASSES)}, "fit": {"split": "train", "n_rows": sizes["train"]}}), encoding="utf-8")
    return d


@pytest.fixture(autouse=True)
def fast_learning(monkeypatch):
    monkeypatch.setattr(fc, "LR", 5e-2)                                  # tiny data, few steps: a large step size lets the toy problem be learnt in a few rounds


def make_cfg(tmp_path, rounds=3):
    return {"label_mode": "6class", "seeds": [0, 1], "quota": {"NORMAL": [60000, 1, 1], "BCF": [20000, 1, 1], "DELAYED": [3000, 1, 1], "SYN": [3000, 1, 1]},
            "fl": {"num_clients": 3, "rounds": rounds, "local_epochs": 3, "dirichlet_alpha": 5.0, "min_client_size": 20}, "eval": {"bootstrap": 20},
            "paths": {"work_dir": str(tmp_path), "shared_manifest_dir": str(tmp_path / "m")},
            "compute": {"artifacts_dir": str(tmp_path / "art"), "runs_csv": str(tmp_path / "res" / "runs.csv")}}


def test_class_weights_are_sklearns_balanced_and_zero_for_an_absent_class():
    w = fc.class_weights(np.array([60, 20, 10, 0]))
    assert w.tolist() == pytest.approx([90 / (4 * 60), 90 / (4 * 20), 90 / (4 * 10), 0.0])


def test_fedavg_round_is_the_weighted_average_in_client_order(monkeypatch):
    k, d = 3, 4
    base = fc.make_mlp(d, k).state_dict()
    shifts = iter([1.0, -2.0, 0.5])

    def fake_train(model, X, y, epochs, weight, seed):                     # every client moves every weight by its own constant
        c = next(shifts)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(c)
    monkeypatch.setattr(fc, "train_epochs", fake_train)
    sizes = [10, 30, 60]
    clients = [(torch.zeros(n, d), torch.zeros(n, dtype=torch.long)) for n in sizes]
    out = fc.fedavg_round({a: b.clone() for a, b in base.items()}, clients, d, k, 1, None, 0, 1)
    expected = (10 * 1.0 + 30 * -2.0 + 60 * 0.5) / 100
    for name, v in out.items():
        assert torch.allclose(v, base[name].float() + expected, atol=1e-6), name


def test_one_run_learns_separable_data_is_deterministic_and_saves_predictions(tmp_path):
    write_world(tmp_path)
    cfg = make_cfg(tmp_path, rounds=6)
    from ppfeddata.eval.baselines import load_data
    from ppfeddata.partition import make_partition
    data, schema = load_data(cfg)
    parts, _ = make_partition(cfg, data["train"]["y"], CLASSES, seed=0)
    a = fc.run_one("FedMLP", cfg, data, CLASSES, 0, parts)
    b = fc.run_one("FedMLP", cfg, data, CLASSES, 0, parts)
    assert a["test_macro_f1"] == b["test_macro_f1"] and a["best_round"] == b["best_round"] and a["val_curve"] == b["val_curve"]          # same seed, same run
    assert a["test_macro_f1"] > 0.9 and set(a["recall"]) == set(CLASSES) and 1 <= a["best_round"] <= 6
    saved = np.load(tmp_path / "art" / run_id("FedMLP-mlp", 0, "6class") / "preds" / "test.npz")["y_pred"]
    assert saved.shape == data["test"]["y"].shape
    cent = fc.run_one("CentMLPcw", cfg, data, CLASSES, 0, None)                                                                     # pooled control, class-weighted: needs no partition
    assert cent["test_macro_f1"] > 0.9
    long = fc.run_one("FedMLP@8", cfg, data, CLASSES, 0, parts)                                                                     # a longer budget: its own run id
    assert len(long["val_curve"]) == 8 and long["run_id"] == run_id("FedMLP-r8-mlp", 0, "6class")


def test_run_all_writes_the_json_with_references_and_paired_comparisons(tmp_path):
    write_world(tmp_path)
    cfg = make_cfg(tmp_path, rounds=3)
    y_test = np.load(tmp_path / "processed" / "6class" / "test.npz")["y"]
    rng = np.random.default_rng(0)
    for ref in fc.REFERENCES:                                                                                                       # saved predictions of the main study's runs
        for s in cfg["seeds"]:
            p = np.where(rng.random(len(y_test)) < 0.7, y_test, (y_test + 1) % K)
            save_predictions(cfg, run_id(ref, s, "6class"), "test", p, None)
    summ = pd.DataFrame([{"config": r, "macro_f1_mean": 0.5, "macro_f1_std": 0.01, **{f"recall_{c}_mean": 0.4 for c in CLASSES}} for r in fc.REFERENCES])
    (tmp_path / "res").mkdir()
    summ.to_csv(tmp_path / "res" / "summary.csv", index=False)
    r = fc.run_all(cfg, names=("FedMLP", "CentMLP"), longer=4)
    assert set(r["classifiers"]) == {"FedMLP", "CentMLP", "FedMLP@4", "CentMLP@4"} and r["longer_rounds"] == 4 and r["rare"] == ["DELAYED", "SYN"]
    assert all(len(v["per_seed"]) == 2 and v["macro_f1_mean"] > 0.5 for v in r["classifiers"].values())
    assert set(r["references"]) == set(fc.REFERENCES) and r["references"]["B0-mlp"]["rare_recall_mean"] == pytest.approx(0.4)
    c = r["comparisons"]["FedMLP"]["B0-mlp"]
    assert c["macro_f1"]["effect"] == "better" and c["macro_f1"]["n_seeds"] == 2 and "rare_recall" in c                               # the references are 70 % accurate, the MLP ~100 %
    on_disk = json.loads((tmp_path / "res" / "fed_classifier.json").read_text(encoding="utf-8"))
    assert on_disk["classifiers"]["FedMLP"]["macro_f1_mean"] == pytest.approx(r["classifiers"]["FedMLP"]["macro_f1_mean"], abs=1e-6)


# --------------------------------------------------------------------------------------------------
# sensitivity worlds
# --------------------------------------------------------------------------------------------------
def test_world_cfg_changes_one_setting_and_keeps_every_output_apart(tmp_path):
    cfg = {**make_cfg(tmp_path), "split": {"split_seed": 0, "test_groups_per_subclass": 1}, "train_sampling": {"max_rows_per_stream": None}}
    assert se.world_cfg(cfg, se.MAIN) is cfg
    a4 = se.world_cfg(cfg, "A4-s2")
    assert a4["split"] == {"split_seed": 2, "test_groups_per_subclass": 1} and cfg["split"]["split_seed"] == 0                    # the main config is not touched
    assert a4["paths"]["work_dir"].endswith("_sens/A4-s2") or a4["paths"]["work_dir"].endswith("_sens\\A4-s2")
    assert "_sens" in a4["compute"]["artifacts_dir"] and a4["compute"]["runs_csv"].startswith(a4["compute"]["artifacts_dir"])
    assert a4["paths"]["shared_manifest_dir"] != cfg["paths"]["shared_manifest_dir"]
    from ppfeddata.utils import config_hash
    assert config_hash(a4) != config_hash(cfg)                                                                                      # a different experiment, a different hash
    a5 = se.world_cfg(cfg, "A5-cap20")
    assert a5["train_sampling"]["max_rows_per_stream"] == 20 and "M1-eps5" in se.SCENARIOS["A5-cap20"]["runs"] and "M1-eps5" not in se.SCENARIOS["A4-s1"]["runs"]


def fake_world(cfg, tag, accs, seed0):
    """A world with its own test split and saved predictions: accs maps a ledger name to its accuracy."""
    w = se.world_cfg(cfg, tag)
    write_world(Path(w["paths"]["work_dir"]), seed=seed0)
    y = np.load(Path(w["paths"]["work_dir"]) / "processed" / "6class" / "test.npz")["y"]
    rng = np.random.default_rng(seed0)
    for name, acc in accs.items():
        for s in cfg["seeds"]:
            save_predictions(w, run_id(name, s, "6class"), "test", np.where(rng.random(len(y)) < acc, y, (y + 1) % K), None)
    Path(w["compute"]["runs_csv"]).parent.mkdir(parents=True, exist_ok=True)
    Path(w["compute"]["runs_csv"]).write_text("run_id\n", encoding="utf-8")
    return w


def test_report_compares_the_worlds_with_the_rule_of_the_spec(tmp_path):
    cfg = {**make_cfg(tmp_path), "split": {"split_seed": 0}, "train_sampling": {"max_rows_per_stream": None}}
    base = {"B0-rf": 0.55, "B0-mlp": 0.40, "B3-TAug-rf": 0.55, "B3-TAug-mlp": 0.75, "B3-TSTR-rf": 0.45, "B3-TSTR-mlp": 0.90}
    fake_world(cfg, se.MAIN, base, 1)
    fake_world(cfg, "A4-s1", base, 2)                                                                                                # same accuracies: same verdicts
    fake_world(cfg, "A4-s2", {**base, "B3-TAug-rf": 0.30}, 3)                                                                       # one verdict changes (TAug-RF is now worse than B0)
    res = se.report(cfg, ["A4-s1", "A4-s2"], n_boot=30)
    assert [w["tag"] for w in res["worlds"]] == ["main", "A4-s1", "A4-s2"]                                                          # A5 has no runs: left out
    w = res["worlds"][0]
    assert w["n_test_rows"] == 80 and w["macro_f1"]["B0-rf"]["per_seed"] and w["train_rows_per_stream"]["mean"] == pytest.approx(2.0)
    v = {x["label"]: x for x in se.verdicts(res)}
    assert v["R1-rf"]["verdicts"]["A4-s2"] == "worse" and not v["R1-rf"]["same_everywhere"] and v["R1-rf"]["verdicts"]["main"] == "none"
    assert v["R1-mlp"]["verdicts"]["main"] == "better" and v["TSTR-vs-B0-mlp"]["verdicts"]["A4-s1"] == "better"
    md = (tmp_path / "res" / "reports" / "sensitivity.md").read_text(encoding="utf-8")
    js = json.loads((tmp_path / "res" / "sensitivity.json").read_text(encoding="utf-8"))
    assert len(js["worlds"]) == 3 and "same verdict in every world that has it" in md
    assert "A4-s2" in md and "NO" in md and "comparisons have the same verdict in every world that has them (3 worlds in all)" in md
    for line in md.splitlines():
        assert not line.startswith("|") or line.count("|") >= 3


def test_report_without_any_world_says_so(tmp_path):
    cfg = {**make_cfg(tmp_path), "split": {"split_seed": 0}, "train_sampling": {"max_rows_per_stream": None}}
    res = se.report(cfg, [], n_boot=10)
    assert res["worlds"] == [] and "no world has runs yet" in (tmp_path / "res" / "reports" / "sensitivity.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------------
# the text that uses the two result files
# --------------------------------------------------------------------------------------------------
def _cmp(delta, effect):
    return {"delta": delta, "lo": delta - 0.01, "hi": delta + 0.01, "effect": effect, "n_seeds": 3}


def fake_fed(eff_cw="none", eff_plain="worse"):
    cl = {n: {"macro_f1_mean": m, "macro_f1_std": 0.01, "rare_recall_mean": r} for n, m, r in (("FedMLP", 0.32, 0.15), ("FedMLPcw", 0.43, 0.31), ("CentMLPcw", 0.51, 0.42), ("FedMLPcw@100", 0.44, 0.30))}
    ref = {"B3-TSTR-mlp": {"label": "CVAE + FL, synthetic data only", "macro_f1_mean": 0.42, "macro_f1_std": 0.01, "rare_recall_mean": 0.37},
           "B0-mlp": {"label": "real data only (pooled)", "macro_f1_mean": 0.33, "macro_f1_std": 0.01, "rare_recall_mean": 0.16}}
    cmp = {"FedMLP": {"B3-TSTR-mlp": {"macro_f1": _cmp(-0.10, eff_plain), "rare_recall": _cmp(-0.2, "worse")}},
           "FedMLPcw": {"B3-TSTR-mlp": {"macro_f1": _cmp(0.009, eff_cw), "rare_recall": _cmp(-0.05, "worse")}, "B0-mlp": {"macro_f1": _cmp(0.1, "better"), "rare_recall": _cmp(0.15, "better")}}}
    return {"classifiers": cl, "references": ref, "comparisons": cmp, "num_clients": 5, "alpha": 0.5, "rounds": 30, "local_epochs": 2, "longer_rounds": 100}


def test_fed_text_follows_the_verdicts_and_degrades_without_the_file():
    from ppfeddata import interpret_report as ir
    assert ir._fed_clause({}) is None and ir._fed_paragraph({}, {}) == [] and ir._fed_facts({"fed_classifier": {"classifiers": {"FedMLP": {}}}}) is None
    R = {"fed_classifier": fake_fed()}
    c = ir._fed_clause(R)
    assert "0.430 with class weights" in c and "0.320 without" in c and "0.420 for the CVAE route" in c and "not different from (inside the noise) it (delta +0.009)" in c and "below it (delta -0.100)" in c
    par = "\n".join(ir._fed_paragraph(R, {}))
    assert "not shown to be worse than the federated CVAE" in par and "| FedMLPcw@100 |" in par and "federation costs 0.080" in par and "B3-TSTR-mlp (CVAE + FL" in par
    worse = "\n".join(ir._fed_paragraph({"fed_classifier": fake_fed(eff_cw="worse")}, {}))
    assert "keeps part of the case" in worse and "not shown to be worse" not in worse


def test_sensitivity_bullet_and_the_a4_limitation_use_the_worlds():
    from ppfeddata import interpret_report as ir
    from ppfeddata import limitations as lim

    def world(tag, eff, shared=None):
        return {"tag": tag, "test_groups_shared_with_main": shared, "comparisons": {"R1-rf": {"what": "R1 (RF)", "macro_f1": {"effect": eff}}}}
    S = {"worlds": [world("main", "none"), world("A4-s1", "better", 1), world("A5-cap20", "none")]}
    b = ir._sensitivity_bullet({"sensitivity": S})
    assert "0 of 1 comparisons" in b and "R1 (RF) (main none, A4-s1 better, A5-cap20 none)" in b
    assert ir._sensitivity_bullet({}) is None and ir._sensitivity_bullet({"sensitivity": {"worlds": [world("main", "none")]}}) is None
    t = lim.a4_text({"sensitivity": S}, None)
    assert "A4 was run with 1 other choices" in t and "shared with the main study: 1" in t and "0 of 1 comparisons" in t                    # A5 is not an A4 world
    assert "was not run" in lim.a4_text(None, None)


def test_readme_check_reads_the_clean_install_record():
    from ppfeddata import acceptance as ac
    heads = lambda n: "\n".join(ac.README_NEED[n]) + "\n"                                                                              # noqa: E731
    texts = {n: heads(n) for n in ac.README_NEED}
    assert "not tried on a clean machine" in ac.check_readme(texts, set()).evidence
    r = ac.check_readme(texts, set(), "ran pytest: 346 passed")
    assert r.status == "PARTIAL" and "clean-install record" in r.evidence and "raw-data phases were not re-run" in r.evidence
