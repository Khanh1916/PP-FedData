"""Phase 11 tests: parsing of ledger names, summary statistics, the six figures and the auto-generated final report, on a synthetic ledger."""
import json

import numpy as np
import pandas as pd
import pytest

from ppfeddata import aggregate as ag
from ppfeddata.eval.runs import run_id

CLASSES = ["NORMAL", "BCF", "DELAYED", "SYN", "INVALID", "WILL"]
QUOTA = {"NORMAL": [60000, 2000, 5000], "BCF": [20000, 2000, 5000], "DELAYED": [3000, 2000, 5000], "SYN": [3000, 2000, 5000],
         "INVALID": [1500, 2000, 5000], "WILL": [1000, 2000, 5000]}
GEN = [(p, c) for p in ("TSTR", "TAug") for c in ("rf", "mlp")]


@pytest.fixture(autouse=True)
def tuned_trial_21(monkeypatch):
    import ppfeddata.fl.m1 as m1
    monkeypatch.setattr(m1, "tuned_setup", lambda cfg: ("M1d-t21", {}, 1.0))


def make_ledger(tmp_path, with_dp=True, with_secagg=True, hashes=("h1", "h1", "h1"), seeds=(0, 1, 2), with_a1=False):
    """A ledger with the shape of the real one; every metric is a deterministic function of (configuration, seed)."""
    rng = np.random.default_rng(0)
    rows, fl_runs = [], set()

    def add(config, proto, clf, fl_run=None, eps=np.nan, plain=False, bytes_=4_000_000, params=100_000, hash_="h1", base=0.4, alpha=0.5):
        for s in seeds:
            r = {"run_id": run_id(config, s, "6class"), "config": config, "classifier": clf, "protocol": proto, "seed": s, "label_mode": "6class", "config_hash": hash_,
                 "git_commit": "c" * 40, "macro_f1": base + 0.01 * s + rng.normal(0, 0.002), "balanced_acc": base, "pr_auc_macro": base, "bin_f1": 0.9, "val_macro_f1": base,
                 "wasserstein_mean": 0.12 if not plain else 0.2, "js_mean": 0.1, "corr_dist_mean": 0.3, "c2st_auc_mean": 0.999, "dup_rate": 1e-4, "dcr_ratio_mean": 0.95,
                 "mia_auc_mean": 0.49, "fit_time_s": 2.0, "alpha": alpha, "bytes_per_round": bytes_, "cvae_params": params, "fl_total_s": 80.0, "cvae_train_s": 80.0}
            for i, c in enumerate(CLASSES):
                r[f"recall_{c}"] = min(1.0, 0.2 + 0.1 * i + base / 10 + 0.01 * s)
                r[f"f1_{c}"] = 0.5
            if not np.isnan(eps):
                r.update({"dp_target_eps": eps, "dp_eps_max": eps * 0.999, "dp_eps_median": eps * 0.9, "dp_sigma_min": 0.7, "dp_sigma_max": 3.0, "dp_ratio_to_target": 0.999})
            rows.append(r)
        if fl_run:
            fl_runs.add(fl_run)

    for n, c, b in (("B0-rf", "rf", 0.45), ("B0-mlp", "mlp", 0.33), ("B1a-rf", "rf", 0.49), ("B1b-rf", "rf", 0.50), ("B1b-mlp", "mlp", 0.47)):
        add(n, "TRTR", c, hash_=hashes[0], base=b)
    for p, c in GEN:
        add(f"B2-{p}-{c}", p, c, hash_=hashes[1], base=0.41 if p == "TSTR" else 0.447)
        add(f"B3-{p}-{c}", p, c, "B3", hash_=hashes[1], base=0.39 if p == "TSTR" else 0.447)
        add(f"B3-plain-{p}-{c}", p, c, "B3", hash_=hashes[1], base=0.38 if p == "TSTR" else 0.446)
    if with_dp:
        for e in (1, 5, 10):
            for p, c in GEN:
                for suf in ("-plain", ""):
                    add(f"M1d-t21-eps{e}{suf}-{p}-{c}", p, c, f"M1d-t21-eps{e}", eps=e, plain=bool(suf), bytes_=1_400_000, params=36_000, hash_=hashes[1],
                        base=(0.2 + 0.01 * e) if p == "TSTR" else 0.445)
                add(f"M1-eps{e}-plain-{p}-{c}", p, c, f"M1-eps{e}", eps=e, plain=True, hash_=hashes[1], base=0.19)
        # candidate trials of the DP search: one seed, must not be taken as the headline family
        add("M1d-t17-eps5-plain-TSTR-rf", "TSTR", "rf", "M1d-t17-eps5", eps=5, plain=True, hash_=hashes[1], base=0.2)
    if with_secagg:
        for p, c in GEN:
            add(f"M2-{p}-{c}", p, c, "M2", bytes_=6_400_000, hash_=hashes[2], base=0.39 if p == "TSTR" else 0.447)
        if with_dp:
            for p, c in GEN:
                for suf in ("-plain", ""):
                    add(f"M3d-t21-eps5{suf}-{p}-{c}", p, c, "M3d-t21-eps5", eps=5, plain=bool(suf), bytes_=2_250_000, params=36_000, hash_=hashes[2], base=0.25 if p == "TSTR" else 0.447)
    if with_a1:
        for a, b in ((0.1, 0.33), (10.0, 0.40)):
            for p, c in GEN:
                add(f"A1-a{a:g}-{p}-{c}", p, c, f"A1-a{a:g}", hash_=hashes[1], base=b if p == "TSTR" else 0.447, alpha=a)
    df = pd.DataFrame(rows)
    # the FL round logs the timing figures are computed from, and the spec of every FL run (client sizes)
    sec = {"B3": 1.8, "M2": 2.1, "M1d-t21-eps1": 8.0, "M1d-t21-eps5": 8.1, "M1d-t21-eps10": 8.1, "M3d-t21-eps5": 8.3, "M1-eps1": 20.0, "M1-eps5": 20.0, "M1-eps10": 20.0, "M1d-t17-eps5": 8.0,
           "A1-a0.1": 1.8, "A1-a10": 1.8}
    sizes = {"A1-a0.1": [41000, 20000, 7000, 2100, 917], "A1-a10": [14000] * 5}
    for fl in fl_runs:
        for s in seeds:
            d = tmp_path / "art" / run_id(fl, s, "6class")
            d.mkdir(parents=True, exist_ok=True)
            (d / "rounds.jsonl").write_text("".join(json.dumps({"round": r, "round_seconds": 25.0 if r == 1 else sec[fl] + 0.01 * r}) + "\n" for r in range(0, 6)), encoding="utf-8")
            (d / "spec.json").write_text(json.dumps({"partition_sizes": sizes.get(fl, [30000, 20000, 12000, 6000, 2000]), "hp": {"batch_size": 1000}, "local_epochs": 2}), encoding="utf-8")
    return df


def make_cfg(tmp_path, df):
    (tmp_path / "art").mkdir(exist_ok=True)
    csv = tmp_path / "art" / "runs.csv"
    df.to_csv(csv, index=False)
    return {"label_mode": "6class", "seeds": [0, 1, 2], "quota": QUOTA, "dp": {"delta": 1e-5}, "fl": {"num_clients": 5, "rounds": 30, "local_epochs": 2, "dirichlet_alpha": 0.5},
            "thresholds": {"dup_rate_max": 0.01, "dcr_ratio_min": 0.5, "mia_auc_max": 0.55, "c2st_auc_max": 0.95, "overhead_ratio_max": 3.0},
            "compute": {"artifacts_dir": str(tmp_path / "art"), "runs_csv": str(csv)},
            "paths": {"work_dir": str(tmp_path / "data"), "shared_manifest_dir": str(tmp_path / "manifests")}}


# ----------------------------------------------------------------------------- names
def test_parse_config_names():
    p = ag.parse_config
    assert p("B0-mlp")["method"] == "B0" and p("B0-mlp")["classifier"] == "mlp" and p("B1b-rf")["method"] == "B1b"
    b = p("B2-TAug-rf")
    assert (b["method"], b["protocol"], b["classifier"], b["variant"]) == ("B2", "TAug", "rf", "standard") and b["fl_run"] is None
    assert p("B3-plain-TSTR-mlp")["variant"] == "plain" and p("B3-plain-TSTR-mlp")["fl_run"] == "B3" and p("B3-plain-TSTR-mlp")["family"].startswith("epsilon-infinity")
    m = p("M1d-t21-eps5-plain-TAug-rf")
    assert (m["method"], m["variant"], m["eps"], m["fl_run"]) == ("M1", "plain", 5.0, "M1d-t21-eps5") and m["family"] == "dp-tuned t21"
    assert p("M1-eps10-TSTR-rf")["family"] == "phase7 hyper-parameters" and p("M1-eps10-TSTR-rf")["eps"] == 10.0 and p("M1-eps10-TSTR-rf")["variant"] == "standard"
    m3 = p("M3d-t21-eps5-TSTR-rf")
    assert (m3["method"], m3["family"], m3["fl_run"]) == ("M3", "dp-tuned t21", "M3d-t21-eps5")
    assert p("M2-TAug-mlp")["fl_run"] == "M2" and p("A1-a0.1-TSTR-rf")["method"] == "A1" and p("A1-a0.1-TSTR-rf")["alpha"] == 0.1
    assert p("something-else")["method"] is None


def test_dp_families_picks_the_tuned_trial_that_has_every_epsilon(tmp_path):
    df = make_ledger(tmp_path)
    assert ag.dp_families(set(df["config"])) == {"M1": "M1d-t21", "M3": "M3d-t21"}       # t17 is a one-epsilon candidate, M1-eps* is the Phase 7 family
    assert ag.dp_families({"B0-rf", "M2-TSTR-rf"}) == {}


# ----------------------------------------------------------------------------- summary
def test_summary_mean_std_over_seeds_and_matrix_flags(tmp_path):
    df = make_ledger(tmp_path)
    cfg = make_cfg(tmp_path, df)
    d = ag.annotate(ag.load_ledger(cfg), cfg)
    summ = ag.build_summary(d, cfg)
    assert summ["config"].is_unique and len(summ) == df["config"].nunique()
    x = df[df["config"] == "B2-TAug-rf"]["macro_f1"].to_numpy()
    r = summ[summ["config"] == "B2-TAug-rf"].iloc[0]
    assert r["macro_f1_mean"] == pytest.approx(x.mean()) and r["macro_f1_std"] == pytest.approx(x.std(ddof=0)) and r["n_seeds"] == 3 and r["seeds"] == "0,1,2"
    assert ag.ms(summ, "B2-TAug-rf", "macro_f1") == f"{x.mean():.4f} ± {x.std():.4f}" and ag.ms(summ, "nope", "macro_f1") == "n/a"
    flags = summ.set_index("config")["in_matrix"]
    assert flags["M1d-t21-eps5-plain-TAug-rf"] and flags["M1d-t21-eps5-TAug-rf"] and flags["M3d-t21-eps5-plain-TSTR-rf"] and flags["B0-rf"]      # the headline family and its noise twin
    assert not flags["M1-eps5-plain-TAug-rf"] and not flags["M1d-t17-eps5-plain-TSTR-rf"]                                                         # reference family and candidate trial
    assert r["round_s_median_mean"] if "round_s_median_mean" in r else True
    assert summ[summ["config"] == "B3-TSTR-rf"].iloc[0]["round_s_median_mean"] == pytest.approx(1.8 + 0.01 * 3, abs=0.02)      # median of rounds 2..5, round 1 (start-up) excluded
    assert np.isnan(summ[summ["config"] == "B2-TAug-rf"].iloc[0]["round_s_median_mean"])                                       # not federated: no round log


def test_single_seed_configuration_shows_no_std(tmp_path):
    df = make_ledger(tmp_path, seeds=(0,))
    cfg = make_cfg(tmp_path, df)
    summ = ag.build_summary(ag.annotate(ag.load_ledger(cfg), cfg), cfg)
    assert ag.ms(summ, "B0-rf", "macro_f1").endswith("(1 seed)")


# ----------------------------------------------------------------------------- aggregate end to end
def test_aggregate_writes_summary_six_figures_and_the_report(tmp_path):
    df = make_ledger(tmp_path, hashes=("h0", "h1", "h2"))
    cfg = make_cfg(tmp_path, df)
    out = ag.aggregate(cfg, out_dir=tmp_path / "res", with_interpretation=False)
    res = tmp_path / "res"
    assert (res / "summary.csv").exists() and len(pd.read_csv(res / "summary.csv")) == df["config"].nunique()
    names = {"f1_by_config.png", "recall_rare_classes.png", "utility_privacy.png", "fidelity_vs_eps.png", "overhead.png", "pareto.png"}
    assert names == {p.name for p in (res / "figures").glob("*.png")} and all((res / "figures" / n).stat().st_size > 5000 for n in names)
    text = (res / "reports" / "final_report.md").read_text(encoding="utf-8")
    for h in ("## 1. Provenance and completeness", "## 2a. Utility, RF", "## 2b. Utility, MLP", "## 3. Recall of the rare classes (DELAYED, SYN, INVALID, WILL", "## 4. Privacy",
              "## 5. Fidelity", "## 6. Cost", "## 8. Reference rows", "## 9. Interpretation (Phase 12)", "## 10. Limitations"):
        assert h in text, h
    x = df[df["config"] == "M2-TAug-rf"]["macro_f1"].to_numpy()
    assert f"{x.mean():.4f} ± {x.std():.4f}" in text                                           # a number of the table is the one of the ledger
    for hh in ("h0", "h1", "h2"):
        assert hh in text                                                                       # all three config hashes are listed in the provenance
    assert "M1-eps5" in text and "M3-eps5" in text and "not applied yet" in text and out["rows"] == df["config"].nunique()
    assert "Every configuration was produced with one config hash" in text and "B1a" in text.split("## 2a.")[1].split("## 2b.")[0] and "B1a" not in text.split("## 2b.")[1].split("## 3.")[0]
    assert "![Pareto](../figures/pareto.png)" in text and "8b." not in text                  # no A1 rows in the ledger: no A1 section
    cost =ag.cost_rows(ag.build_summary(ag.annotate(ag.load_ledger(cfg), cfg), cfg), ag.entries(ag.build_summary(ag.annotate(ag.load_ledger(cfg), cfg), cfg)))
    by = {r["label"]: r for r in cost}
    assert by["B3"]["ratio"] == pytest.approx(1.0) and by["M2"]["ratio"] > 1.1 and by["M1-eps5"]["ratio"] > 4 and by["M2"]["bytes_per_param"] == pytest.approx(6_400_000 / 100_000)


def test_aggregate_degrades_gracefully_without_dp_and_secagg(tmp_path):
    df = make_ledger(tmp_path, with_dp=False, with_secagg=False)
    cfg = make_cfg(tmp_path, df)
    out = ag.aggregate(cfg, out_dir=tmp_path / "res", with_interpretation=False)
    text = (tmp_path / "res" / "reports" / "final_report.md").read_text(encoding="utf-8")
    assert len(out["figures"]) == 6 and "M1-eps5" not in text.split("## 2a.")[1].split("## 3.")[0] and "B3" in text


def test_report_has_the_a1_section_with_utility_and_partition_columns(tmp_path):
    df = make_ledger(tmp_path, with_a1=True)
    cfg = make_cfg(tmp_path, df)
    out = ag.aggregate(cfg, out_dir=tmp_path / "res", with_interpretation=False)
    text = (tmp_path / "res" / "reports" / "final_report.md").read_text(encoding="utf-8")
    sec = text.split("## 8b.")[1].split("## 9.")[0]
    assert sec.index("| 0.1 |") < sec.index("| 0.5 (B3) |") < sec.index("| 10 |")                      # alphas in increasing order, B3 is the alpha of the config
    x = df[df["config"] == "A1-a0.1-TSTR-rf"]["macro_f1"].to_numpy()
    assert f"{x.mean():.4f} ± {x.std():.4f}" in sec
    assert "917 - 41,000" in sec and "14,000 - 14,000" in sec and "2,000 - 30,000" in sec                  # client rows, min - max over the seeds (from spec.json)
    assert "| 60 |" in sec and "| 28 |" in sec and "| 42 |" in sec                                          # effective optimiser steps / round: sum_i n_i/N * ceil(n_i/1000) * 2
    assert "TSTR-rf recall, mean over DELAYED/SYN/INVALID/WILL" in sec and "FL final val loss" in sec and "SPEC_DEVIATIONS 8.6" in sec
    assert out["rows"] == df["config"].nunique()


def test_aggregate_rejects_an_empty_ledger(tmp_path):
    cfg = make_cfg(tmp_path, make_ledger(tmp_path))
    pd.DataFrame(columns=["run_id", "label_mode"]).to_csv(cfg["compute"]["runs_csv"], index=False)
    with pytest.raises(FileNotFoundError):
        ag.aggregate(cfg, out_dir=tmp_path / "res", with_interpretation=False)
