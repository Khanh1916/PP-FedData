"""Follow-up: the federated MLP with distributed DP (DP-FedSGD, the noise split over the clients, `fed_distributed.py`), on tiny synthetic data."""
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from ppfeddata import fed_distributed as fd
from ppfeddata.eval.runs import run_id, save_predictions
from ppfeddata.fl import dp_utils
from tests.test_fed_classifier import CLASSES, K, make_cfg, write_world


def test_clipped_sum_clips_every_row_on_its_own():
    torch.manual_seed(0)
    model = fd.make_mlp(5, 3, (8,))
    params = {k: v.detach() for k, v in model.named_parameters()}
    X, y, w = torch.randn(7, 5) * 5, torch.tensor([0, 1, 2, 0, 1, 2, 0]), torch.tensor([1.0, 2.0, 0.5])
    got, n = fd.clipped_sum(model, params, X, y, w, clip=0.3, chunk=3)
    want, clipped = {k: torch.zeros_like(v) for k, v in params.items()}, 0
    for i in range(len(X)):                                                  # the same thing one row at a time with autograd
        model.zero_grad()
        (torch.nn.functional.cross_entropy(model(X[i:i + 1]), y[i:i + 1]) * w[y[i]]).backward()
        g = {k: p.grad.clone() for k, p in model.named_parameters()}
        norm = math.sqrt(sum(float(v.pow(2).sum()) for v in g.values()))
        clipped += norm > 0.3
        for k in want:
            want[k] += g[k] * min(1.0, 0.3 / norm)
    assert n == clipped > 0 and all(torch.allclose(got[k], want[k], atol=1e-5) for k in want)


def test_the_shares_of_the_clients_add_up_to_the_noise_of_central_dp_sgd():
    sigma, clip, k_clients, denom = 1.3, 2.0, 5, 10.0
    total = sum(fd.noisy_share(torch.zeros(200_000), sigma, clip, k_clients, denom, torch.Generator().manual_seed(c)) for c in range(k_clients))
    one = fd.noisy_share(torch.zeros(200_000), sigma, clip, k_clients, denom, torch.Generator().manual_seed(9))
    assert float(total.std()) == pytest.approx(sigma * clip / denom, rel=0.01)                 # the sum: sigma * C, as one central DP-SGD step
    assert float(one.std()) == pytest.approx(sigma * clip / denom / math.sqrt(k_clients), rel=0.01)


def _data(tmp_path):
    from ppfeddata.eval.baselines import load_data
    from ppfeddata.partition import make_partition
    write_world(tmp_path)
    cfg = {**make_cfg(tmp_path), "dp": {"delta": 1e-5}, "secagg": {"clipping_range": 16.0}}
    data, _ = load_data(cfg)
    parts, _ = make_partition(cfg, data["train"]["y"], CLASSES, seed=0)
    return cfg, data, parts


def test_one_run_spends_the_target_epsilon_learns_and_is_deterministic(tmp_path):
    cfg, data, parts = _data(tmp_path)
    hp = {"lr": 5e-2, "hidden": (16,), "clip": 1.0, "batch": 64}
    a = fd.train(data, K, parts, hp, 10.0, 1e-5, 0, fd.secagg_params(cfg), epochs=6)
    b = fd.train(data, K, parts, hp, 10.0, 1e-5, 0, fd.secagg_params(cfg), epochs=6)
    n = len(data["train"]["y"])
    assert a["rounds"] == fd.rounds_for(n, 64, 6) and a["q"] == pytest.approx(64 / n)
    assert a["eps"] <= 10.0 * 1.001 and a["eps"] == pytest.approx(10.0, rel=0.01) and a["eps_one_honest"] > a["eps"]
    assert a["eps_one_honest"] == pytest.approx(dp_utils.epsilon(a["sigma"] / math.sqrt(len(parts)), a["q"], a["rounds"], 1e-5))
    assert a["curve"] == b["curve"] and a["val_macro_f1"] > 0.6 and a["quant_clipped_frac"] == 0.0     # same seed, same run; the toy problem is learnt
    c = fd.train(data, K, parts, {**hp, "sched": "cos"}, None, 1e-5, 0, fd.secagg_params(cfg), epochs=6)    # the control: no noise, epsilon not claimed
    assert c["sigma"] == 0.0 and c["eps"] is None and c["eps_one_honest"] is None and c["val_macro_f1"] > 0.6


def test_cosine_schedule_decays_the_learning_rate_to_zero():
    assert fd.lr_at(0.02, 1, 100) == fd.lr_at(0.02, 100, 100) == 0.02                            # constant
    assert fd.lr_at(0.02, 1, 100, "cos") == pytest.approx(0.02) and fd.lr_at(0.02, 51, 100, "cos") == pytest.approx(0.01)
    assert 0 < fd.lr_at(0.02, 100, 100, "cos") < 1e-4


def _cmp(delta, effect):
    return {"delta": delta, "lo": delta - 0.01, "hi": delta + 0.01, "effect": effect, "n_seeds": 3}


def fake_distributed():
    cl, cmpx, refs = {}, {}, {}
    for e, (d_fw, eff) in {1: (-0.05, "worse"), 5: (0.0, "none"), 10: (0.04, "better")}.items():
        cl[f"FedMLPddp{e}"] = {"macro_f1_mean": 0.3 + e / 100, "macro_f1_std": 0.01, "rare_recall_mean": 0.2,
                               "dp": {"target_eps": float(e), "eps_one_honest": 3.0 * e, "rounds": 2593, "eps_max_over_seeds": float(e)}}
        names = {"framework": f"MGs-eps{e}-TSTR-mlp", "MGd": f"MGd-eps{e}-TSTR-mlp", "local DP": f"FedMLPcwT-dp{e}-mlp"}
        cmpx[f"FedMLPddp{e}"] = {role: {"against": n, "macro_f1": _cmp(d_fw if role == "framework" else 0.1, eff if role == "framework" else "better"),
                                        "rare_recall": _cmp(0.0, "none")} for role, n in names.items()}
        refs.update({n: {"macro_f1_mean": 0.35, "macro_f1_std": 0.01, "rare_recall_mean": 0.2, "bytes_total": 5e5 if "MG" in n else None,
                         "eps_one_honest": 2.0 * e if "MG" in n else None} for n in names.values()})
    cl["FedMLPddp-nonoise"] = {"macro_f1_mean": 0.40, "macro_f1_std": 0.01, "rare_recall_mean": 0.3, "dp": {"target_eps": None, "eps_one_honest": None, "rounds": 2593}}
    search = {"lr": [0.02, 0.05], "clip": [2.0, 8.0, 32.0], "sched": ["const", "cos"], "batch": [1024]}
    trials = [{"hp": {"lr": lr, "clip": c, "batch": 1024, "sched": s}} for lr in search["lr"] for c in search["clip"] for s in search["sched"]]
    per = {f"{e}": {"chosen": {"lr": 0.02, "clip": 8.0, "batch": 1024, "sched": "cos"}, "trials": trials} for e in (1, 5, 10)}
    compact = {"classifiers": {f"FedMLPddp{e}": {"macro_f1_mean": m} for e, m in ((1, 0.346), (5, 0.354), (10, 0.346))}}
    refs["MGs-nodp-TSTR-mlp"] = {"macro_f1_mean": 0.45, "macro_f1_std": 0.01, "rare_recall_mean": 0.3, "bytes_total": None, "eps_one_honest": None}
    nonoise_cmp = {"against": "MGs-nodp-TSTR-mlp", "macro_f1": _cmp(-0.05, "worse"), "rare_recall": _cmp(0.0, "none")}
    return {"classifiers": cl, "comparisons": cmpx, "references": refs, "delta": 1e-5, "bytes_total_est": 7.7e9, "compact": compact, "nonoise_comparison": nonoise_cmp,
            "tuning": {"per_eps": per, "search": search, "seed": 0, "epochs": 30}}


def test_report_and_limitation_follow_the_distributed_results(tmp_path):
    from ppfeddata import interpret_report as ir
    from ppfeddata import limitations as lim
    R = {"fed_classifier": {"distributed": fake_distributed(), "protected": {"classifiers": {"FedMLPcwT": {"macro_f1_mean": 0.425}}}}}
    text = "\n".join(ir._fed_distributed(R))
    assert "above in 1, not different in 1 and below in 1 of 3 epsilons (delta -0.050 to +0.040)" in text and "7.7 GB" in text and "0.50-0.50 MB" in text
    assert "schedule {const, cos}, expected batch 1024; 12 trials per epsilon, 30 passes" in text and "| 2,593 |" in text and "| 0.02, 8, cos |" in text
    assert "+0.100 to +0.100 against DP at each client. Communication:" in text and "tuned at epsilon 5 only" in text
    assert "grows to 3.0 / 15.0 / 30.0 (one release per round), against 2.0 / 10.0 / 20.0 for FedDP-Marginal" in text and "edge of the grid" not in text   # middle values: no edge
    assert "the same recipe reaches 0.400 ± 0.010, against 0.450 for FedDP-Marginal without DP (TSTR-MLP; delta -0.050 [-0.060, -0.040] worse)" in text
    assert "DP costs the direct classifier 0.090 / 0.050 / 0.000 and FedDP-Marginal 0.100 / 0.100 / 0.100 at epsilon 1 / 5 / 10" in text
    assert "the recipe itself (one clipped gradient step per round, no client drift) is 0.025 below FedAvg of the same MLP without DP (0.425)." in text
    assert "compact search at epsilon 5 only, applied to every epsilon, gave 0.346 / 0.354 / 0.346" in text
    D = fake_distributed()
    D["tuning"]["per_eps"]["10"]["chosen"]["clip"] = 32.0                                          # top of a 3-value grid; the lr grid has 2 values: never flagged
    assert "(the clipping bound at epsilon 10 chosen at the edge of the grid" in "\n".join(ir._fed_distributed({"fed_classifier": {"distributed": D}}))
    assert "above FedDP-Marginal (TSTR-MLP, same epsilon) in 1 of 3 epsilons" in ir._fed_dist_clause(R)
    assert ir._fed_distributed({}) == [] and ir._fed_dist_clause({"fed_classifier": {}}) is None
    res = tmp_path / "res"
    res.mkdir()
    cfg = {"label_mode": "6class", "compute": {"runs_csv": str(res / "runs.csv")}}
    (res / "fed_classifier.json").write_text(json.dumps({"distributed": fake_distributed()}), encoding="utf-8")
    t = {i["topic"]: i for i in lim.limitations(cfg)}["What was not compared"]
    assert "with distributed DP (DP-FedSGD" in t["text"] and t["text"].endswith("a discrete (Skellam) mechanism for the classifier trained with distributed DP.") and "R.7" in t["source"]
    (res / "fed_classifier.json").write_text("{}", encoding="utf-8")
    assert {i["topic"]: i for i in lim.limitations(cfg)}["What was not compared"]["text"].endswith("a classifier trained by FL with distributed DP.")


def test_run_all_adds_the_distributed_block_and_compares_at_the_same_epsilon(tmp_path, monkeypatch):
    from ppfeddata.tune_marginal import framework_name
    cfg, data, _ = _data(tmp_path)
    monkeypatch.setattr(fd, "SEARCH", {"lr": (5e-2,), "clip": (1.0, 4.0), "sched": ("const",), "batch": (64,)})
    monkeypatch.setattr(fd, "EPOCHS", 4)
    y_test, rng = data["test"]["y"], np.random.default_rng(0)
    eps_list = (1, 10)
    names = [n for e in eps_list for n in fd.against(e).values()]
    for name in names:                                                      # saved predictions of the runs compared with (70 % accurate)
        for s in cfg["seeds"]:
            save_predictions(cfg, run_id(name, s, "6class"), "test", np.where(rng.random(len(y_test)) < 0.7, y_test, (y_test + 1) % K), None)
    summ = [{"config": n, "macro_f1_mean": 0.4, "macro_f1_std": 0.01, "bytes_per_round_mean": 1000.0, "rounds_mean": 3.0, **{f"recall_{c}_mean": 0.3 for c in CLASSES}}
            for n in names if not n.startswith("FedMLPcwT")]
    summ.append({"config": "M2-TSTR-rf", "macro_f1_mean": 0.4, "macro_f1_std": 0.0, "sa_bytes_per_round_mean": 600.0, "cvae_params_mean": 10.0})
    (tmp_path / "res").mkdir()
    pd.DataFrame(summ).to_csv(tmp_path / "res" / "summary.csv", index=False)
    compact = {"tuning": {"chosen": {"lr": 0.02}}, "classifiers": {"FedMLPddp1": {"macro_f1_mean": 0.346}}, "comparisons": {}}   # a block of the compact search
    old = {"classifiers": {"FedMLP": {"macro_f1_mean": 0.3}}, "tuning": {"non_dp": {"chosen": {"hidden": [16, 8]}}},
           "protected": {"classifiers": {"FedMLPcwT-dp1": {"macro_f1_mean": 0.2, "macro_f1_std": 0.0, "rare_recall_mean": 0.1}}}, "distributed": compact}
    (tmp_path / "res" / "fed_classifier.json").write_text(json.dumps(old), encoding="utf-8")
    r = fd.run_all(cfg, eps_list=eps_list, n_boot=20)
    D = r["distributed"]
    assert r["classifiers"] == old["classifiers"] and r["protected"] == old["protected"]                 # the rest of the file is kept
    assert D["compact"]["classifiers"]["FedMLPddp1"]["macro_f1_mean"] == 0.346 and D["compact"]["tuning"] == compact["tuning"]
    per = D["tuning"]["per_eps"]
    assert set(per) == {"1", "10"} and all(len(v["trials"]) == 2 and v["chosen"]["hidden"] == [16, 8] for v in per.values()) and D["tuning"]["epochs"] == 4
    assert set(D["classifiers"]) == {"FedMLPddp1", "FedMLPddp10", "FedMLPddp-nonoise"} and D["classifiers"]["FedMLPddp1"]["dp"]["eps_max_over_seeds"] <= 1.001
    nn = D["classifiers"]["FedMLPddp-nonoise"]
    assert nn["dp"]["eps_max_over_seeds"] is None and nn["dp"]["sigma"] == 0.0 and nn["hp"] == per["1"]["chosen"] and "FedMLPddp-nonoise" not in D["comparisons"]
    assert D["nonoise_comparison"] is None                                                     # no prediction of FedDP-Marginal without DP in this world
    c = D["comparisons"]["FedMLPddp1"]
    assert set(c) == {"framework", "MGd", "local DP"} and c["framework"]["against"] == f"{framework_name(1.0)}-TSTR-mlp" and c["MGd"]["macro_f1"]["n_seeds"] == 2
    assert D["references"]["FedMLPcwT-dp1-mlp"]["macro_f1_mean"] == 0.2 and D["references"]["MGd-eps10-TSTR-mlp"]["bytes_total"] == 3000.0
    assert D["bytes_total_est"] == pytest.approx(60.0 * D["n_params"] * D["classifiers"]["FedMLPddp1"]["dp"]["rounds"])
    again = fd.run_all(cfg, eps_list=eps_list, n_boot=20)                                                   # the cache: nothing is trained again
    assert again["distributed"]["classifiers"]["FedMLPddp10"]["per_seed"] == D["classifiers"]["FedMLPddp10"]["per_seed"]
    assert again["distributed"]["compact"] == D["compact"]                                                  # the compact entry survives a rewrite
    assert fd.compact_entry({"tuning": {"per_eps": {}}}) is None and fd.compact_entry(None) is None
