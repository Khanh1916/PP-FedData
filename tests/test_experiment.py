"""Phase 11 tests: the experiment matrix (configs/exp), planning against a run ledger, execution with resume, failure isolation, the
isolated trial workspace, gate G4 and the trial-vs-existing comparison. The runners are replaced by fakes: no model is trained."""
import json

import numpy as np
import pandas as pd
import pytest

from ppfeddata import run_experiment as rx
from ppfeddata.eval.runs import RunLedger, run_id

CFG = {"label_mode": "6class", "seeds": [0, 1], "compute": {"artifacts_dir": None, "runs_csv": None}}


@pytest.fixture(autouse=True)
def tuned_trial_21(monkeypatch):
    """The tuned DP family name comes from configs/best_cvae_dp.yaml; fix it here so the tests do not depend on that file."""
    import ppfeddata.fl.m1 as m1
    monkeypatch.setattr(m1, "tuned_setup", lambda cfg: ("M1d-t21", {}, 1.0))


@pytest.fixture
def cfg(tmp_path):
    c = json.loads(json.dumps(CFG))
    c["compute"]["artifacts_dir"] = str(tmp_path / "art")
    c["compute"]["runs_csv"] = str(tmp_path / "art" / "runs.csv")
    return c


def fake_write(cfg, exp, seed, resume=True, metric=0.5):
    """What a real runner leaves behind: one ledger row and one prediction file per expected ledger name."""
    led = RunLedger(rx.runs_csv_path(cfg))
    for n in rx.ledger_names(cfg, exp):
        rid = run_id(n, seed, cfg["label_mode"])
        led.append({"run_id": rid, "config": n, "seed": seed, "label_mode": cfg["label_mode"], "macro_f1": metric, "balanced_acc": metric,
                    "fit_time_s": 6.0, "cvae_train_s": 60.0 if exp.kind not in ("baseline",) else np.nan, "cvae_gen_s": 3.0, "fidpriv_s": 12.0,
                    "config_hash": "abc", "recall_BCF": 0.9})
        d = rx.artifacts_dir(cfg) / rid / "preds"
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "test.npz", y_pred=np.zeros(3, dtype=np.int16))


FAKES = {k: fake_write for k in rx.RUNNERS}


# ----------------------------------------------------------------------------- the matrix files
def test_matrix_files_are_valid_and_cover_the_spec_matrix():
    exps = rx.load_matrix()
    ids = {e.id for e in exps}
    assert {"B0", "B1a", "B1b", "B2", "B3", "M1-eps1", "M1-eps5", "M1-eps10", "M2", "M3"} <= ids               # the matrix of the spec
    assert {e.id for e in exps if e.group == "matrix"} == {"B0", "B1a", "B1b", "B2", "B3", "M1-eps1", "M1-eps5", "M1-eps10", "M2", "M3"}
    assert {"A1-a0.1", "A1-a10", "A2", "A3", "A4", "A5"} <= ids and all(e.group == "extension" for e in exps if e.id.startswith("A"))
    assert [e.kind for e in exps if e.group == "matrix"] == sorted((e.kind for e in exps if e.group == "matrix"), key=rx.KINDS.index)      # cheap first
    m1 = [e for e in exps if e.kind == "m1" and e.group == "matrix"]
    assert [e.params["eps"] for e in m1] == [1, 5, 10]                                                       # numeric order, not 1, 10, 5
    assert not [e for e in exps if e.id in ("A2", "A3", "A4", "A5") and e.runnable]                          # no runner yet, and the plan says so


def test_matrix_loader_rejects_bad_files(tmp_path):
    (tmp_path / "X.yaml").write_text("id: X\ntitle: t\ngroup: matrix\nkind: nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown kind"):
        rx.load_matrix(tmp_path)
    (tmp_path / "X.yaml").write_text("id: Y\ntitle: t\ngroup: matrix\nkind: b2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file name"):
        rx.load_matrix(tmp_path)
    (tmp_path / "X.yaml").write_text("id: X\ntitle: t\ngroup: matrix\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        rx.load_matrix(tmp_path)


def test_ledger_names_per_kind(cfg):
    by_id = {e.id: e for e in rx.load_matrix()}
    n = {k: rx.ledger_names(cfg, by_id[k]) for k in by_id}
    assert n["B0"] == ["B0-rf", "B0-mlp"] and n["B1a"] == ["B1a-rf"] and n["B1b"] == ["B1b-rf", "B1b-mlp"]
    assert n["B2"] == ["B2-TSTR-rf", "B2-TSTR-mlp", "B2-TAug-rf", "B2-TAug-mlp"] and n["B3"][0] == "B3-TSTR-rf" and n["M2"][-1] == "M2-TAug-mlp"
    assert n["M1-eps5"][0] == "M1d-t21-eps5-plain-TSTR-rf" and n["M1-eps5"][4] == "M1d-t21-eps5-TSTR-rf" and len(n["M1-eps5"]) == 8      # plain first, then with noise
    assert n["M1p7-eps10"][0] == "M1-eps10-plain-TSTR-rf"                                                    # the Phase 7 family keeps its old names
    assert n["M3"][0] == "M3d-t21-eps5-plain-TSTR-rf" and len(n["M3"]) == 8
    assert n["A1-a0.1"][0] == "A1-a0.1-TSTR-rf" and n["A1-a10"][0] == "A1-a10-TSTR-rf"
    assert rx.fl_run_name(cfg, by_id["M3"]) == "M3d-t21-eps5" and rx.fl_run_name(cfg, by_id["B3-plain"]) == "B3" and rx.fl_run_name(cfg, by_id["B2"]) is None


# ----------------------------------------------------------------------------- plan
def test_plan_distinguishes_done_partial_and_todo(cfg):
    exps = {e.id: e for e in rx.load_matrix()}
    fake_write(cfg, exps["B2"], 0)                                                       # complete
    led = RunLedger(rx.runs_csv_path(cfg))
    led.append({"run_id": run_id("M2-TSTR-rf", 0, "6class"), "config": "M2-TSTR-rf", "seed": 0, "label_mode": "6class"})      # one of four rows
    (rx.artifacts_dir(cfg) / "B3_0" / "ckpt").mkdir(parents=True)                       # an interrupted FL run: checkpoint only
    (rx.artifacts_dir(cfg) / "B3_0" / "ckpt" / "latest.json").write_text(json.dumps({"file": "round_0010.pt", "round": 10}))
    fake_write(cfg, exps["B0"], 0)
    for p in (rx.artifacts_dir(cfg) / run_id("B0-mlp", 0, "6class") / "preds").glob("*.npz"):    # a ledger row whose predictions are gone
        p.unlink()
    rows = {(r["id"], r["seed"]): r for r in rx.plan(cfg, list(exps.values()), [0, 1], groups=("matrix",))}
    assert rows[("B2", 0)]["status"] == "done" and rows[("B2", 1)]["status"] == "todo"
    assert rows[("M2", 0)]["status"] == "partial" and rows[("M2", 0)]["rows_done"] == 1 and rows[("M2", 0)]["rows_expected"] == 4
    assert rows[("B3", 0)]["status"] == "partial" and rows[("B3", 0)]["checkpoint_round"] == 10
    assert rows[("B0", 0)]["status"] == "partial" and rows[("B0", 0)]["rows_without_predictions"] == 1
    assert rows[("M3", 1)]["status"] == "todo"
    ext = rx.plan(cfg, list(exps.values()), [0], groups=("extension",))
    assert {r["id"]: r["status"] for r in ext}["A4"] == "not-runnable" and {r["id"]: r["status"] for r in ext}["A1-a10"] == "todo"
    assert [r["id"] for r in rx.plan(cfg, list(exps.values()), [0], only=["M3", "B2"])] == ["B2", "M3"]


def test_measured_minutes_uses_the_ledger_and_falls_back_to_the_file_estimate(cfg):
    exps = {e.id: e for e in rx.load_matrix()}
    assert rx.measured_minutes(pd.DataFrame(), cfg, exps["B2"]) is None
    fake_write(cfg, exps["B2"], 0)
    led = RunLedger(rx.runs_csv_path(cfg)).frame()
    # 4 fits x 6 s + training 60 s + one variant: 3 s generation + 12 s fidelity/privacy = 99 s
    assert rx.measured_minutes(led, cfg, exps["B2"]) == pytest.approx(99 / 60)
    fake_write(cfg, exps["M1-eps5"], 0)
    led = RunLedger(rx.runs_csv_path(cfg)).frame()
    assert rx.measured_minutes(led, cfg, exps["M1-eps5"]) == pytest.approx((8 * 6 + 60 + 2 * 15) / 60)         # two variants (plain, with noise), FL time once
    row = [r for r in rx.plan(cfg, list(exps.values()), [0], groups=("matrix",)) if r["id"] == "M3"][0]
    assert row["est_minutes"] == exps["M3"].est_minutes


# ----------------------------------------------------------------------------- execute, G4, isolation
def test_execute_runs_only_what_is_missing_in_cheap_first_order_and_isolates_failures(cfg):
    exps = rx.load_matrix()
    by = {e.id: e for e in exps}
    fake_write(cfg, by["B0"], 0)
    calls = []

    def runner(kind):
        def run(c, exp, seed, resume):
            calls.append((exp.id, seed))
            if exp.id == "M2":
                raise RuntimeError("boom")
            fake_write(c, exp, seed)
        return run

    s = rx.execute(cfg, exps, [0], stage="trial", workspace="main", runners={k: runner(k) for k in rx.RUNNERS})
    assert ("B0", 0) not in calls and calls[0] == ("B1a", 0)                                  # B0 was already done; the rest in the order of KINDS
    assert [c[0] for c in calls] == ["B1a", "B1b", "B2", "B3", "M2", "M1-eps1", "M1-eps5", "M1-eps10", "M3"]
    st = {r["id"]: r for r in s["items"]}
    assert st["M2"]["status"] == "failed" and "boom" in st["M2"]["error"] and st["M3"]["status"] == "done" and st["B0"]["status"] == "done"
    saved = json.loads(rx.status_path(cfg).read_text())
    assert saved["stage"] == "trial" and {i["id"]: i["status"] for i in saved["items"]}["M2"] == "failed"
    calls.clear()                                                                               # a second session: only the failure is retried
    s2 = rx.execute(cfg, exps, [0], stage="trial", workspace="main", runners={k: runner(k) for k in rx.RUNNERS})
    assert calls == [("M2", 0)] and {r["id"]: r["status"] for r in s2["items"]}["M2"] == "failed"


def test_a_runner_that_leaves_rows_missing_is_reported_incomplete(cfg):
    exps = [e for e in rx.load_matrix() if e.id == "B2"]
    s = rx.execute(cfg, exps, [0], stage="trial", workspace="main", runners={k: (lambda c, e, sd, r: None) for k in rx.RUNNERS})
    assert s["items"][0]["status"] == "incomplete"


def test_full_stage_needs_the_g4_approval_but_a_dry_run_does_not(cfg):
    exps = rx.load_matrix()
    with pytest.raises(PermissionError, match="Gate G4"):
        rx.execute(cfg, exps, [0, 1], stage="full", runners={k: fake_write for k in rx.RUNNERS})
    assert not RunLedger(rx.runs_csv_path(cfg)).frame().shape[0]
    s = rx.execute(cfg, exps, [0, 1], stage="full", dry_run=True)
    assert s["dry_run"] and all(r["status"] == "todo" for r in s["items"]) and len(s["items"]) == 20           # 10 configurations x 2 seeds
    s = rx.execute(cfg, exps, [0], stage="full", approved=True, only=["B2"], runners={k: fake_write for k in rx.RUNNERS})
    assert s["approved_g4"] and s["workspace"] == "main" and s["items"][0]["status"] == "done"


def test_trial_stage_runs_one_seed_in_an_isolated_workspace(cfg):
    exps = rx.load_matrix()
    by = {e.id: e for e in exps}
    fake_write(cfg, by["B2"], 0)                                                                # an existing result in the main workspace
    before = RunLedger(rx.runs_csv_path(cfg)).frame().copy()
    s = rx.execute(cfg, exps, [1, 0], stage="trial", only=["B2", "M2"], runners={k: fake_write for k in rx.RUNNERS})
    assert s["workspace"] == "trial" and s["seeds"] == [1]                                       # one seed only, the first one given
    tcfg = rx.workspace_cfg(cfg, "trial")
    assert tcfg["compute"]["artifacts_dir"].endswith("_trial") and tcfg["compute"]["runs_csv"].replace("\\", "/").endswith("_trial/runs.csv")
    assert cfg["compute"]["artifacts_dir"] != tcfg["compute"]["artifacts_dir"]                   # the caller's config is not modified
    trial = RunLedger(rx.runs_csv_path(tcfg)).frame()
    assert set(trial["seed"]) == {1} and {"B2-TSTR-rf", "M2-TAug-mlp"} <= set(trial["config"])
    pd.testing.assert_frame_equal(RunLedger(rx.runs_csv_path(cfg)).frame(), before)               # the main ledger is untouched
    assert rx.status_path(tcfg).exists() and not (rx.artifacts_dir(cfg) / "experiment_status_6class.json").exists()


def test_compare_workspaces_reports_identical_and_different_rows(cfg):
    exps = {e.id: e for e in rx.load_matrix()}
    tcfg = rx.workspace_cfg(cfg, "trial")
    fake_write(cfg, exps["B2"], 0, metric=0.5)
    fake_write(tcfg, exps["B2"], 0, metric=0.5)
    cmp_ = rx.compare_workspaces(cfg, tcfg, [0])
    assert len(cmp_) == 4 and cmp_["identical"].all() and (cmp_["max_abs_diff"] == 0).all()
    fake_write(tcfg, exps["B2"], 0, metric=0.5 + 1e-3)
    cmp_ = rx.compare_workspaces(cfg, tcfg, [0])
    assert not cmp_["identical"].any() and cmp_["max_abs_diff"].max() == pytest.approx(1e-3)
    # a few bytes of difference in a byte count (SecAgg messages depend on random keys) is tolerated, a different result is not
    fake_write(tcfg, exps["B2"], 0, metric=0.5)
    led = RunLedger(rx.runs_csv_path(cfg)).frame()
    led["bytes_per_round"] = 6_400_000.0
    led.to_csv(rx.runs_csv_path(cfg), index=False)
    led = RunLedger(rx.runs_csv_path(tcfg)).frame()
    led["bytes_per_round"] = 6_400_033.0
    led.to_csv(rx.runs_csv_path(tcfg), index=False)
    cmp_ = rx.compare_workspaces(cfg, tcfg, [0])
    assert cmp_["identical"].all() and (cmp_["bytes_abs_diff"] == 33).all() and (cmp_["bytes_rel_diff"] < 1e-5).all()
    led["bytes_per_round"] = 7_000_000.0
    led.to_csv(rx.runs_csv_path(tcfg), index=False)
    assert not rx.compare_workspaces(cfg, tcfg, [0])["identical"].any()
    fake_write(tcfg, exps["M2"], 0)                                                              # not in the main ledger
    cmp_ = rx.compare_workspaces(cfg, tcfg, [0])
    assert (~cmp_[cmp_["config"].str.startswith("M2")]["in_main_ledger"]).all()
    assert rx.compare_workspaces(cfg, rx.workspace_cfg({**cfg, "label_mode": "6class", "compute": {**cfg["compute"], "artifacts_dir": cfg["compute"]["artifacts_dir"] + "x",
                                                                                                    "runs_csv": cfg["compute"]["artifacts_dir"] + "x/none.csv"}}, "main"), [0]).empty


def test_trial_report_is_written_from_the_session(cfg, tmp_path):
    exps = rx.load_matrix()
    fake_write(cfg, {e.id: e for e in exps}["B2"], 0)
    s = rx.execute(cfg, exps, [0], stage="trial", only=["B2"], runners={k: fake_write for k in rx.RUNNERS})
    out = tmp_path / "g4.md"
    res = rx.write_trial_report(cfg, s, exps, out)
    text = out.read_text(encoding="utf-8")
    assert "Gate G4" in text and "B2" in text and "identical in every compared metric" in text and res["n_identical"] == 4 and res["n_failed"] == 0


def test_a1_runner_passes_the_alpha_names_the_run_and_skips_a_finished_seed(cfg, monkeypatch):
    """The real A1 runner with the heavy parts replaced: the Dirichlet alpha reaches run_fl, the run is called A1-a<alpha>, the evaluation gets
    the same name, and a seed whose rows are all in the ledger does not train again."""
    import ppfeddata.eval.baselines as bl
    import ppfeddata.fl.b3 as b3
    import ppfeddata.fl.run as flrun
    import ppfeddata.models.b2 as b2
    exp = {e.id: e for e in rx.load_matrix()}["A1-a0.1"]
    seen = {"fl": [], "eval": []}
    summary = {"fl_total_s": 50.0, "mean_round_s": 1.7, "bytes_per_round": 100, "server_peak_rss_gb": 0.5, "final_val_loss": 8.0, "client_seconds_mean": 1.0}

    def fake_fl(cfg_, seed, name, alpha=None, resume=True, **kw):
        seen["fl"].append((name, seed, alpha))
        return {"alpha": alpha, "num_clients": 5, "rounds": 30, "local_epochs": 2, "summary": summary}

    def fake_eval(cfg_, model, name, seed, data, schema, ledger, extra):
        seen["eval"].append((name, seed, extra["alpha"]))
        fake_write(cfg_, exp, seed)                                          # the rows the plan expects for this configuration

    monkeypatch.setattr(bl, "load_data", lambda c: ({}, {}))
    monkeypatch.setattr(flrun, "run_fl", fake_fl)
    monkeypatch.setattr(b3, "load_fl_model", lambda c, r, s: object())
    monkeypatch.setattr(b2, "evaluate_generator", fake_eval)

    rx._r_a1(cfg, exp, 1, True)
    assert seen["fl"] == [("A1-a0.1", 1, 0.1)] and seen["eval"] == [("A1-a0.1", 1, 0.1)]
    assert rx.run_status(cfg, exp, 1, RunLedger(rx.runs_csv_path(cfg)).frame())["status"] == "done"
    rx._r_a1(cfg, exp, 1, True)                                              # everything is in the ledger: nothing trains again
    assert len(seen["fl"]) == 1
