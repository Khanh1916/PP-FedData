"""Optimisation round O0: scorecard, dominance with the seed rule, Pareto front, frozen baseline."""
import json
import math

import pandas as pd
import pytest

from ppfeddata import scorecard as sc

CLASSES = ["NORMAL", "BCF", "DELAYED", "SYN", "INVALID", "WILL"]
RARE = CLASSES[2:]


def _gen(prefix, f1, val, eps=None, std=0.01, bytes_=4e6, t=100.0, rare=0.3):
    rows = []
    for proto in ("TSTR", "TAug"):
        for clf, dv in (("rf", 0.0), ("mlp", 0.01)):
            r = {"config": f"{prefix}-{proto}-{clf}", "n_seeds": 3, "val_macro_f1_mean": val + dv, "macro_f1_mean": f1 + (0.45 if proto == "TAug" else 0) + dv,
                 "macro_f1_std": std, "bin_f1_mean": 0.9, "bin_f1_std": 0.01, "bytes_per_round_mean": bytes_, "fl_total_s_mean": t,
                 "dp_eps_max_mean": eps if eps else math.nan}
            for c in CLASSES:
                r[f"recall_{c}_mean"], r[f"recall_{c}_std"] = (rare if c in RARE else 0.8), 0.01
            rows.append(r)
    return rows


def _summary(m1_f1=(0.25, 0.26, 0.27)):
    rows = [{"config": f"B0-{c}", "n_seeds": 3, "macro_f1_mean": 0.45, "macro_f1_std": 0.001, "val_macro_f1_mean": 0.45,
             **{f"recall_{k}_mean": 0.5 for k in CLASSES}} for c in ("rf", "mlp")]
    rows += _gen("B3", 0.42, 0.40) + _gen("M2", 0.42, 0.40, bytes_=6e6)
    for e, f in zip((1, 5, 10), m1_f1):
        rows += _gen(f"M1d-t21-eps{e}-plain", f, f, eps=float(e), bytes_=1.5e6, t=280, rare=0.15)
        rows += _gen(f"M1d-t21-eps{e}", f + 0.03, f + 0.03, eps=float(e), bytes_=1.5e6, t=280, rare=0.15)
    rows += _gen("M3d-t21-eps5-plain", 0.26, 0.26, eps=5.0, bytes_=2.2e6, t=280, rare=0.15)
    rows += _gen("M3d-t21-eps5", 0.29, 0.29, eps=5.0, bytes_=2.2e6, t=280, rare=0.15)
    return pd.DataFrame(rows)


def _by_label(out):
    return {r["label"]: r for r in out["rows"]}


def test_classifier_is_chosen_on_validation_not_test():
    s = _summary()
    s.loc[s["config"] == "B3-TSTR-rf", "macro_f1_mean"] = 0.99          # best on TEST, worse on validation: must not be picked
    r = _by_label(sc.build(s, rare=RARE))["B3"]
    assert r["tstr_classifier"] == "mlp" and r["tstr_f1"] == pytest.approx(0.43)


def test_residual_noise_twins_are_listed_but_not_valid_nor_on_the_front():
    out = sc.build(_summary(), rare=RARE)
    by = _by_label(out)
    twin = by["M1-eps5 (residual noise)"]
    assert not twin["valid"] and not twin["pareto"] and "without DP" in twin["invalid_reason"]
    assert by["M1-eps5"]["valid"] and by["M1-eps5"]["variant"] == "plain"
    assert all(by[n]["valid"] for n in ("B3", "M2", "M3-eps5"))


def test_taug_gain_is_against_b0_with_the_same_classifier():
    r = _by_label(sc.build(_summary(), rare=RARE))["B3"]
    assert r["taug_classifier"] == "mlp" and r["taug_gain"] == pytest.approx(0.42 + 0.45 + 0.01 - 0.45)


def test_utility_differences_inside_the_seed_std_are_ties():
    a = {"tstr_f1": 0.30, "tstr_f1_std": 0.02}
    assert sc._cmp_util(a, {"tstr_f1": 0.31, "tstr_f1_std": 0.005}, "tstr_f1") == 0
    assert sc._cmp_util(a, {"tstr_f1": 0.35, "tstr_f1_std": 0.005}, "tstr_f1") == -1
    assert sc._cmp_util(a, {"tstr_f1": math.nan}, "tstr_f1") == 0


def test_epsilon_and_cost_lower_is_better_infinite_is_worst():
    assert sc._cmp_low(1.0, 5.0, sc.EPS_TOL) == 1 and sc._cmp_low(5.0, 1.0, sc.EPS_TOL) == -1
    assert sc._cmp_low(4.99, 5.0, sc.EPS_TOL) == 0
    assert sc._cmp_low(5.0, math.inf, sc.EPS_TOL) == 1 and sc._cmp_low(math.inf, math.inf, sc.EPS_TOL) == 0


def test_a_worse_epsilon_with_no_better_utility_is_dominated():
    out = sc.build(_summary(m1_f1=(0.25, 0.25, 0.25)), rare=RARE)
    by = _by_label(out)
    assert not by["M1-eps10"]["pareto"] and set(by["M1-eps10"]["dominated_by"]) >= {"M1-eps1"}
    assert by["M1-eps1"]["pareto"]


def test_secagg_and_no_dp_rows_are_on_the_front_when_they_lead_on_utility():
    out = sc.build(_summary(), rare=RARE)
    assert {"B3", "M2", "M1-eps1", "M3-eps5"} <= set(out["front"])


def test_direct_route_is_a_reference_and_never_on_the_front():
    fed = {"protected": {"classifiers": {"FedMLPcwT-dp5": {"macro_f1_mean": 0.9, "macro_f1_std": 0.01, "rare_recall_mean": 0.9, "seeds": [0, 1, 2],
                                                         "dp": {"eps_max_over_seeds": 4.999}}}}}
    out = sc.build(_summary(), fed=fed, rare=RARE)
    d = _by_label(out)["FedMLPcwT-dp5"]
    assert d["reference"] and not d["pareto"] and "FedMLPcwT-dp5" not in out["front"]
    assert all(not any(x == "FedMLPcwT-dp5" for x in r["dominated_by"]) for r in out["rows"])


def test_deltas_against_the_baseline_use_the_seed_rule():
    base = sc.build(_summary(), rare=RARE)
    cur = sc.build(_summary(m1_f1=(0.25, 0.40, 0.27)), rare=RARE, base=base)
    d = {x["label"]: x for x in cur["vs_baseline"]}
    assert d["M1-eps5"]["tstr_f1_effect"] == "better" and d["M1-eps5"]["tstr_f1"] == pytest.approx(0.14)
    assert d["M1-eps1"]["tstr_f1_effect"] == "no change"


def test_run_writes_files_and_freezes_the_baseline_once(tmp_path):
    _summary().to_csv(tmp_path / "summary.csv", index=False)
    cfg = {"compute": {"runs_csv": str(tmp_path / "runs.csv"), "artifacts_dir": str(tmp_path)}, "label_mode": "6class", "quota": {}}
    sc.run(cfg, freeze_baseline=True)
    assert (tmp_path / "scorecard_baseline.json").exists() and (tmp_path / "reports" / "scorecard.md").exists()
    j = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))
    assert next(r for r in j["rows"] if r["label"] == "B3")["eps"] == "inf"
    with pytest.raises(FileExistsError):
        sc.run(cfg, freeze_baseline=True)
    again = sc.run(cfg)
    assert again["vs_baseline"] and all(d["tstr_f1_effect"] == "no change" for d in again["vs_baseline"])
    assert "Change against the baseline" in (tmp_path / "reports" / "scorecard.md").read_text(encoding="utf-8")


def test_optimised_families_are_parsed_and_both_variants_are_valid():
    from ppfeddata import aggregate as ag
    p = ag.parse_config("M3o-t7-eps5-plain-TSTR-rf")
    assert p["method"] == "M3" and p["family"] == "o1 t7" and p["variant"] == "plain" and p["eps"] == 5.0
    s = pd.concat([_summary(), pd.DataFrame(_gen("M1o-t7-eps5-plain", 0.30, 0.30, eps=5.0, bytes_=1.5e6, t=280) + _gen("M1o-t7-eps5", 0.33, 0.33, eps=5.0, bytes_=1.5e6, t=280)
                                            + _gen("M3o-t7-eps5", 0.33, 0.33, eps=5.0, bytes_=2.2e6, t=280))], ignore_index=True)
    by = _by_label(sc.build(s, rare=RARE))
    assert by["M1o-eps5"]["valid"] and by["M1o-eps5"]["variant"] == "standard" and by["M1o-eps5 (plain)"]["variant"] == "plain"
    assert by["M3o-eps5"]["secagg"] and by["M3o-eps5"]["dp"] and not by["M1o-eps5"]["secagg"]
    assert "M1-eps5" in by["M1o-eps5"].get("dominated_by", []) or by["M1o-eps5"]["pareto"]
    assert "M1o-eps5" in by["M1-eps5"]["dominated_by"]                 # better utility at the same epsilon and cost
