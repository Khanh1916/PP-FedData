"""Phase 11: experiment orchestration (the `run` command).

One file per configuration in `configs/exp/` (B0 ... M3, the reference families and the extensions) says what a configuration is and
which runner produces it. `plan` compares the matrix with the run ledger and the artifacts, `execute` runs what is missing (every
runner is resume-aware: a finished run is skipped, an interrupted FL run restarts from its checkpoint), `compare_workspaces`
checks a trial against the results that already exist.

Gate G4 is part of the tool. Stage `trial` runs ONE seed (the first of `seeds`) in an isolated workspace (`artifacts/_trial`, with its
own run ledger) so that nothing that exists is touched; stage `full` runs every seed in the main workspace and refuses to start
without `approved=True` (the `--approve-g4` flag, to be given only after the user has looked at the trial).

The headline DP family of the matrix is the one with the DP-specific hyper-parameters (`M1d-t<trial>`, `M3d-t<trial>`, SPEC_DEVIATIONS 9.11-9.14);
the Phase 7 family (`M1`) is kept as the `reference` group.
"""
from __future__ import annotations

import copy
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from ppfeddata.eval.runs import RunLedger, artifacts_dir, run_id, runs_csv_path
from ppfeddata.models.b2 import PROTOCOLS, method_name

logger = logging.getLogger("ppfeddata.run_experiment")

EXP_DIR = Path(__file__).resolve().parents[2] / "configs" / "exp"
KINDS = ("baseline", "b2", "b3", "b3plain", "m2", "m1", "m3", "a1", "needs_data")        # also the execution order: cheap first
GROUPS = ("matrix", "reference", "extension")
TRIAL_DIRNAME = "_trial"


# --------------------------------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------------------------------
@dataclass
class Exp:
    id: str
    title: str
    group: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    est_minutes: float | None = None
    note: str = ""

    @property
    def runnable(self) -> bool:
        return self.kind != "needs_data"

    def sort_key(self) -> tuple:
        return (KINDS.index(self.kind), float(self.params.get("eps", self.params.get("alpha", 0))), self.id)


def load_matrix(exp_dir: str | Path = EXP_DIR) -> list[Exp]:
    exps = []
    for f in sorted(Path(exp_dir).glob("*.yaml")):
        d = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        miss = {"id", "title", "group", "kind"} - set(d)
        if miss:
            raise ValueError(f"{f.name}: missing keys {sorted(miss)}")
        if d["kind"] not in KINDS:
            raise ValueError(f"{f.name}: unknown kind {d['kind']!r}; expected one of {KINDS}")
        if d["group"] not in GROUPS:
            raise ValueError(f"{f.name}: unknown group {d['group']!r}; expected one of {GROUPS}")
        if d["id"] != f.stem:
            raise ValueError(f"{f.name}: id {d['id']!r} must equal the file name")
        exps.append(Exp(d["id"], d["title"], d["group"], d["kind"], dict(d.get("params") or {}), d.get("est_minutes"), d.get("note", "")))
    return sorted(exps, key=Exp.sort_key)


def family_prefix(cfg: dict[str, Any], exp: Exp) -> str:
    """Ledger prefix of a DP family: M1 / M1d-t<trial> (kind m1) and M3 / M3d-t<trial> (kind m3)."""
    base = "M1" if exp.kind == "m1" else "M3"
    if exp.params.get("family", "tuned") == "tuned":
        from ppfeddata.fl.m1 import tuned_setup
        return tuned_setup(cfg)[0].replace("M1", base, 1)
    return base


def dp_name(cfg: dict[str, Any], exp: Exp) -> str:
    return f"{family_prefix(cfg, exp)}-eps{float(exp.params['eps']):g}"


def ledger_names(cfg: dict[str, Any], exp: Exp) -> list[str]:
    """Every ledger `config` name one seed of this configuration writes."""
    prot = lambda prefix: [method_name(prefix, a, b) for a, b in PROTOCOLS]      # noqa: E731
    k, p = exp.kind, exp.params
    if k == "baseline":
        return list(p["names"])
    if k in ("b2", "b3", "m2"):
        return prot(k.upper())
    if k == "b3plain":
        return prot("B3-plain")
    if k in ("m1", "m3"):
        name = dp_name(cfg, exp)
        return prot(name + "-plain") + prot(name)               # plain decoder (covered by epsilon) and with residual noise
    if k == "a1":
        return prot(f"A1-a{float(p['alpha']):g}")
    return []


def fl_run_name(cfg: dict[str, Any], exp: Exp) -> str | None:
    k = exp.kind
    if k in ("b3", "b3plain"):
        return "B3"
    if k == "m2":
        return "M2"
    if k in ("m1", "m3"):
        return dp_name(cfg, exp)
    if k == "a1":
        return f"A1-a{float(exp.params['alpha']):g}"
    return None


# --------------------------------------------------------------------------------------------------
# Workspaces
# --------------------------------------------------------------------------------------------------
def workspace_cfg(cfg: dict[str, Any], workspace: str) -> dict[str, Any]:
    """`main` = the config as it is; `trial` = same config, artifacts and run ledger under artifacts/_trial (nothing existing is touched;
    the processed data, the client partitions and the hyper-parameter files are shared read-only inputs)."""
    if workspace == "main":
        return cfg
    if workspace != "trial":
        raise ValueError(f"unknown workspace {workspace!r}")
    c = copy.deepcopy(cfg)
    root = Path(cfg["compute"]["artifacts_dir"]) / TRIAL_DIRNAME
    c["compute"]["artifacts_dir"] = str(root)
    c["compute"]["runs_csv"] = str(root / "runs.csv")
    return c


# --------------------------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------------------------
def _checkpoint_round(rdir: Path) -> int | None:
    p = rdir / "ckpt" / "latest.json"
    return int(json.loads(p.read_text(encoding="utf-8"))["round"]) if p.exists() else None


def run_status(cfg: dict[str, Any], exp: Exp, seed: int, ledger: pd.DataFrame) -> dict[str, Any]:
    names, mode = ledger_names(cfg, exp), cfg["label_mode"]
    ids = [run_id(n, seed, mode) for n in names]
    have = set(ledger["run_id"]) if len(ledger) else set()
    n_done = sum(i in have for i in ids)
    no_preds = [i for i in ids if i in have and not (artifacts_dir(cfg) / i / "preds" / "test.npz").exists()]
    fl = fl_run_name(cfg, exp)
    ck = _checkpoint_round(artifacts_dir(cfg) / run_id(fl, seed, mode)) if fl else None
    if names and n_done == len(names) and not no_preds:
        status = "done"
    elif n_done or ck is not None or no_preds:
        status = "partial"
    else:
        status = "todo"
    return {"rows_done": n_done, "rows_expected": len(names), "rows_without_predictions": len(no_preds), "checkpoint_round": ck, "status": status}


def measured_minutes(ledger: pd.DataFrame, cfg: dict[str, Any], exp: Exp) -> float | None:
    """Wall-clock estimate of one seed from what the ledger recorded for it (classifier fits, FL/CVAE training once per seed, generation and
    fidelity/privacy once per variant). None when the configuration has no rows yet."""
    names = ledger_names(cfg, exp)
    sub = ledger[ledger["config"].isin(names)] if len(ledger) else ledger
    if not len(sub):
        return None
    z = lambda v: float(np.nan_to_num(v))                      # noqa: E731
    per_seed = []
    for _, g in sub.groupby("seed"):
        t = z(g["fit_time_s"].sum()) + (z(g["prep_time_s"].sum()) if "prep_time_s" in g else 0.0)
        t += z(g["cvae_train_s"].iloc[0]) if "cvae_train_s" in g else 0.0
        variant = g["config"].str.rsplit("-", n=2).str[0]
        for _, gv in g.groupby(variant):
            t += (z(gv["cvae_gen_s"].iloc[0]) if "cvae_gen_s" in gv else 0.0) + (z(gv["fidpriv_s"].iloc[0]) if "fidpriv_s" in gv else 0.0)
        per_seed.append(t)
    return float(np.median(per_seed)) / 60.0


def plan(cfg: dict[str, Any], exps: list[Exp], seeds: list[int], groups: tuple[str, ...] = ("matrix",), only: list[str] | None = None) -> list[dict[str, Any]]:
    ledger = RunLedger(runs_csv_path(cfg)).frame()
    if len(ledger):
        ledger = ledger[ledger["label_mode"] == cfg["label_mode"]]
    rows = []
    for e in exps:
        if e.group not in groups or (only and e.id not in only):
            continue
        for s in seeds:
            if not e.runnable:
                rows.append({"id": e.id, "group": e.group, "kind": e.kind, "seed": s, "status": "not-runnable", "rows_done": 0, "rows_expected": 0,
                             "checkpoint_round": None, "rows_without_predictions": 0, "est_minutes": None, "note": e.note})
                continue
            st = run_status(cfg, e, s, ledger)
            est = measured_minutes(ledger, cfg, e) if st["status"] != "done" else None
            rows.append({"id": e.id, "group": e.group, "kind": e.kind, "seed": s, **st, "est_minutes": est if est is not None else e.est_minutes, "note": e.note})
    return rows


# --------------------------------------------------------------------------------------------------
# Runners (one call = one configuration, one seed; each is resume-aware)
# --------------------------------------------------------------------------------------------------
def _r_baseline(cfg, exp, seed, resume):
    from ppfeddata.eval.baselines import run_baselines
    run_baselines(cfg, list(exp.params["names"]), [seed], resume=resume)


def _r_b2(cfg, exp, seed, resume):
    from ppfeddata.eval.baselines import load_data
    from ppfeddata.models.b2 import run_b2_seed
    data, schema = load_data(cfg)            # the MIA positive control of `run_b2` is a one-off per label mode, not part of a seed
    run_b2_seed(cfg, seed, data, schema, RunLedger(runs_csv_path(cfg)), resume)


def _r_b3(cfg, exp, seed, resume):
    from ppfeddata.fl.b3 import run_b3
    run_b3(cfg, [seed], resume=resume)


def _r_b3plain(cfg, exp, seed, resume):
    from ppfeddata.fl.m1 import run_m1
    run_m1(cfg, [], [seed], resume=resume, tuned=True)          # no epsilon: only the B3-plain evaluation of the seed


def _r_m1(cfg, exp, seed, resume):
    from ppfeddata.fl.m1 import run_m1
    run_m1(cfg, [float(exp.params["eps"])], [seed], resume=resume, tuned=exp.params.get("family", "tuned") == "tuned")


def _r_m2(cfg, exp, seed, resume):
    from ppfeddata.fl.m2 import run_m2
    run_m2(cfg, [seed], resume=resume)


def _r_m3(cfg, exp, seed, resume):
    from ppfeddata.fl.m2 import run_m3
    run_m3(cfg, float(exp.params["eps"]), [seed], resume=resume, tuned=exp.params.get("family", "tuned") == "tuned")


def _r_a1(cfg, exp, seed, resume):
    """Extension A1: B3 with another Dirichlet alpha (same everything else)."""
    from ppfeddata.eval.baselines import load_data
    from ppfeddata.fl.b3 import fl_extra, load_fl_model
    from ppfeddata.fl.m1 import _all_done
    from ppfeddata.fl.run import run_fl
    from ppfeddata.models.b2 import evaluate_generator
    alpha, name = float(exp.params["alpha"]), f"A1-a{float(exp.params['alpha']):g}"
    data, schema = load_data(cfg)
    ledger = RunLedger(runs_csv_path(cfg))
    if resume and _all_done(ledger, [name], seed, cfg["label_mode"]):
        return
    run = run_fl(cfg, seed, name, alpha=alpha, resume=resume)
    evaluate_generator(cfg, load_fl_model(cfg, run, schema), name, seed, data, schema, ledger, fl_extra(run))        # fl_extra already carries `alpha`


RUNNERS: dict[str, Callable[..., None]] = {"baseline": _r_baseline, "b2": _r_b2, "b3": _r_b3, "b3plain": _r_b3plain, "m1": _r_m1, "m2": _r_m2, "m3": _r_m3, "a1": _r_a1}


# --------------------------------------------------------------------------------------------------
# Execute
# --------------------------------------------------------------------------------------------------
def status_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["compute"]["artifacts_dir"]) / f"experiment_status_{cfg['label_mode']}.json"


G4_MESSAGE = ("Gate G4: stage `full` needs the user's approval after the one-seed trial. Run `run --stage trial`, let the user read "
              "results/reports/g4_trial.md, and only then pass --approve-g4.")


def execute(cfg: dict[str, Any], exps: list[Exp], seeds: list[int], *, stage: str = "trial", workspace: str | None = None, approved: bool = False,
            groups: tuple[str, ...] = ("matrix",), only: list[str] | None = None, resume: bool = True, dry_run: bool = False,
            runners: dict[str, Callable[..., None]] | None = None) -> dict[str, Any]:
    """Run what is missing. Returns the session record (also written to artifacts/experiment_status_<mode>.json of the workspace)."""
    if stage not in ("trial", "full"):
        raise ValueError(f"unknown stage {stage!r}")
    if stage == "full" and not approved and not dry_run:
        raise PermissionError(G4_MESSAGE)
    workspace = workspace or ("trial" if stage == "trial" else "main")
    seeds = list(seeds)
    if stage == "trial" and len(seeds) > 1:
        logger.warning("trial stage runs one seed: using seed %d only", seeds[0])
        seeds = seeds[:1]
    wcfg = workspace_cfg(cfg, workspace)
    runners = runners or RUNNERS
    rows = plan(wcfg, exps, seeds, groups, only)
    session: dict[str, Any] = {"stage": stage, "workspace": workspace, "approved_g4": bool(approved), "label_mode": cfg["label_mode"], "seeds": seeds,
                               "groups": list(groups), "started": datetime.now().isoformat(timespec="seconds"), "dry_run": dry_run, "items": rows}
    todo = [r for r in rows if r["status"] in ("todo", "partial")]
    logger.info("plan (%s, workspace %s, seeds %s): %d items, %d done, %d to run, ~%.0f min", stage, workspace, seeds, len(rows),
                sum(r["status"] == "done" for r in rows), len(todo), sum(r["est_minutes"] or 0 for r in todo))
    for r in rows:
        logger.info("  %-12s seed %d  %-12s rows %d/%d%s", r["id"], r["seed"], r["status"], r["rows_done"], r["rows_expected"],
                    f"  checkpoint round {r['checkpoint_round']}" if r["checkpoint_round"] else "")
    if dry_run:
        session["finished"] = datetime.now().isoformat(timespec="seconds")
        return session
    exp_by_id = {e.id: e for e in exps}
    path = status_path(wcfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    for r in todo:
        exp = exp_by_id[r["id"]]
        t0 = time.perf_counter()
        logger.info(">>> %s seed %d (%s)", exp.id, r["seed"], exp.title)
        try:
            runners[exp.kind](wcfg, exp, r["seed"], resume)
            r["error"] = None
        except Exception as e:                                  # one failing configuration must not stop the others
            r["error"] = f"{type(e).__name__}: {e}"[:600]
            logger.error("%s seed %d FAILED: %s\n%s", exp.id, r["seed"], r["error"], traceback.format_exc())
        r["seconds"] = time.perf_counter() - t0
        led = RunLedger(runs_csv_path(wcfg)).frame()
        led = led[led["label_mode"] == wcfg["label_mode"]] if len(led) else led
        r.update(run_status(wcfg, exp, r["seed"], led))
        r["status"] = "done" if r["status"] == "done" else ("failed" if r["error"] else "incomplete")
        logger.info("<<< %s seed %d: %s in %.1f min", exp.id, r["seed"], r["status"], r["seconds"] / 60)
        session["finished"] = datetime.now().isoformat(timespec="seconds")
        path.write_text(json.dumps(session, indent=2, default=str), encoding="utf-8")
    session["finished"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(session, indent=2, default=str), encoding="utf-8")
    bad = [r for r in rows if r["status"] in ("failed", "incomplete", "todo", "partial")]
    logger.info("session finished: %d of %d items done, %d not done", sum(r["status"] == "done" for r in rows), len(rows), len(bad))
    return session


# --------------------------------------------------------------------------------------------------
# Trial vs existing results
# --------------------------------------------------------------------------------------------------
COMPARE = ["macro_f1", "balanced_acc", "pr_auc_macro", "val_macro_f1", "c2st_auc_mean", "wasserstein_mean", "js_mean", "corr_dist_mean",
           "dcr_ratio_mean", "dup_rate", "mia_auc_mean", "dp_eps_max", "dp_sigma_max"]
COUNTED = ["bytes_per_round", "sa_bytes_per_round"]      # bytes counted on the wire: the size of a SecAgg message depends on random keys and node ids
BYTES_RTOL = 1e-4


def compare_workspaces(main: dict[str, Any], trial: dict[str, Any], seeds: list[int], exps: list[Exp] | None = None) -> pd.DataFrame:
    """One row per ledger row of the trial that also exists in the main ledger: largest absolute difference over the result metrics (a
    reproducible pipeline gives 0) and the relative difference of the byte counts, which may move by a few bytes in SecAgg runs."""
    a = RunLedger(runs_csv_path(main)).frame()
    b = RunLedger(runs_csv_path(trial)).frame()
    if not len(b):
        return pd.DataFrame()
    a = a.set_index("run_id")
    out = []
    for _, r in b[b["seed"].isin(seeds)].iterrows():
        if r["run_id"] not in a.index:
            out.append({"run_id": r["run_id"], "config": r["config"], "seed": r["seed"], "in_main_ledger": False})
            continue
        m = a.loc[r["run_id"]]
        diffs = {c: abs(float(r[c]) - float(m[c])) for c in COMPARE + [c for c in b.columns if c.startswith("recall_")]
                 if c in b.columns and c in a.columns and pd.notna(r[c]) and pd.notna(m[c])}
        worst = max(diffs, key=diffs.get) if diffs else None
        byte_diffs = {c: abs(float(r[c]) - float(m[c])) for c in COUNTED if c in b.columns and c in a.columns and pd.notna(r[c]) and pd.notna(m[c])}
        byte_rel = max((d / max(abs(float(m[c])), 1.0) for c, d in byte_diffs.items()), default=0.0)
        out.append({"run_id": r["run_id"], "config": r["config"], "seed": int(r["seed"]), "in_main_ledger": True, "n_metrics": len(diffs),
                    "max_abs_diff": diffs[worst] if worst else float("nan"), "worst_metric": worst, "bytes_abs_diff": max(byte_diffs.values(), default=0.0), "bytes_rel_diff": byte_rel,
                    "identical": bool(diffs and max(diffs.values()) == 0.0 and byte_rel < BYTES_RTOL),
                    "config_hash_trial": r.get("config_hash"), "config_hash_main": m.get("config_hash")})
    return pd.DataFrame(out)


def compare_final_states(main: dict[str, Any], trial: dict[str, Any], exps: list[Exp], seeds: list[int]) -> list[dict[str, Any]]:
    """Largest absolute difference between the final global weights of the FL runs of the two workspaces."""
    import torch
    out, seen = [], set()
    for e in exps:
        name = fl_run_name(main, e) if e.runnable else None
        for s in seeds:
            if not name or (name, s) in seen:
                continue
            seen.add((name, s))
            fa = artifacts_dir(main) / run_id(name, s, main["label_mode"]) / "final_state.pt"
            fb = artifacts_dir(trial) / run_id(name, s, trial["label_mode"]) / "final_state.pt"
            if fa.exists() and fb.exists():
                sa, sb = torch.load(fa, weights_only=False), torch.load(fb, weights_only=False)
                out.append({"run": run_id(name, s, main["label_mode"]), "max_abs_diff": float(max((sa[k] - sb[k]).abs().max() for k in sa))})
    return out


def median_round_seconds(cfg: dict[str, Any], name: str, seed: int) -> float | None:
    from ppfeddata.fl import core
    rows = [r["round_seconds"] for r in core.read_round_log(artifacts_dir(cfg) / run_id(name, seed, cfg["label_mode"])) if r["round"] > 1]
    return float(np.median(rows)) if rows else None


def write_trial_report(cfg: dict[str, Any], session: dict[str, Any], exps: list[Exp], out: str | Path = "./results/reports/g4_trial.md") -> dict[str, Any]:
    """The document the user reads at Gate G4: what ran, how long, whether it reproduces the existing results, what is left."""
    from ppfeddata.eval.baselines import _table

    trial, seeds = workspace_cfg(cfg, "trial"), session["seeds"]
    items = session["items"]
    cmp_ = compare_workspaces(cfg, trial, seeds, exps)
    fs = compare_final_states(cfg, trial, [e for e in exps if e.group == "matrix"], seeds)
    rows = [{"configuration": r["id"], "seed": r["seed"], "status": r["status"], "ledger rows": f"{r['rows_done']}/{r['rows_expected']}",
             "minutes": f"{r['seconds'] / 60:.1f}" if r.get("seconds") else "-", "error": (r.get("error") or "")[:80]} for r in items]
    tot = sum(r.get("seconds") or 0 for r in items) / 60
    L = ["# Gate G4 - one-seed trial of the whole experiment matrix", "",
         f"Auto-generated by `ppfeddata run --stage trial`. Seed {seeds}, label mode {cfg['label_mode']}, started {session['started']}, finished {session.get('finished')}. "
         "The trial ran in an **isolated workspace** (`artifacts/_trial`, its own run ledger); the processed data, the client partitions and the hyper-parameter files "
         "are the shared read-only inputs, so no existing result was touched or reused. It is a from-scratch run of the matrix with one command.", "",
         "## What ran", "", _table(rows), "", f"Total {tot:.1f} min (the sum of the configurations; the evaluation of every generator is included).", ""]
    if len(cmp_):
        ok = cmp_[cmp_["in_main_ledger"].fillna(False)]
        ident = int(ok["identical"].sum()) if len(ok) else 0
        worst = ok.sort_values("max_abs_diff", ascending=False).head(8)
        L += ["## Does it reproduce the results that already exist?", "",
              f"{len(ok)} of {len(cmp_)} ledger rows of the trial also exist in the main ledger (same run_id). **{ident} of {len(ok)} are identical in every compared metric**; "
              f"the largest difference of a result metric over all rows is {float(ok['max_abs_diff'].max()):.2e}. Result metrics compared: " + ", ".join(COMPARE) + ", the per-class recalls. "
              f"Byte counts ({', '.join(COUNTED)}) are compared with a relative tolerance of {BYTES_RTOL:g}: the largest difference is {float(ok['bytes_abs_diff'].max()):.0f} bytes "
              f"({float(ok['bytes_rel_diff'].max()):.1e} relative), in SecAgg runs, because the size of a SecAgg message depends on random keys and node ids.", ""]
        if ident < len(ok):
            L += [_table([{"run": r["run_id"], "max abs diff": f"{r['max_abs_diff']:.2e}", "worst metric": r["worst_metric"]} for _, r in worst.iterrows()]), ""]
        hs = sorted(set(cmp_["config_hash_trial"].dropna()) | set(ok["config_hash_main"].dropna()))
        L += [f"Config hashes: the trial used {sorted(set(cmp_['config_hash_trial'].dropna()))}; the existing rows carry {sorted(set(ok['config_hash_main'].dropna()))}. "
              "A different hash with identical metrics means the config changed only in keys the method does not use (e.g. the SecAgg section after the baselines had run).", ""]
    if fs:
        L += ["### FL final weights (trial vs existing)", "", _table([{"run": f["run"], "max abs diff of final weights": f"{f['max_abs_diff']:.2e}"} for f in fs]), ""]
    med = []
    for e in exps:
        name = fl_run_name(cfg, e) if e.runnable and e.group == "matrix" else None
        for s in seeds:
            if name:
                a, b = median_round_seconds(cfg, name, s), median_round_seconds(trial, name, s)
                if a and b:
                    med.append({"FL run": run_id(name, s, cfg["label_mode"]), "existing s/round": f"{a:.2f}", "trial s/round": f"{b:.2f}"})
    if med:
        L += ["### Seconds per round (median, rounds 2+): measured when the run was made vs in the trial", "",
              "Both columns come from this machine. The trial ran the configurations back to back in one session, but small scripts and unit tests ran in parallel at times, "
              "so treat the timings as indicative, not as a controlled benchmark.", "", _table(med), ""]
    L += ["## Not done", ""]
    bad = [r for r in items if r["status"] != "done"]
    L += ([f"- {r['id']} seed {r['seed']}: {r['status']} {r.get('error') or ''}" for r in bad] or ["- Nothing failed: every configuration of the matrix completed."]) + [""]
    sec = {r["id"]: (r.get("seconds") or 0) / 60 for r in items}
    n_seeds = len(cfg["seeds"])
    rest = (n_seeds - 1) * tot
    a1 = sum(e.est_minutes or 0 for e in exps if e.group == "extension" and e.kind == "a1") * n_seeds
    if sec.get("B3"):
        a1_meas = 2 * n_seeds * (sec["B3"])
        a1 = a1_meas
    L += ["## Decisions asked at Gate G4", "",
          f"1. **The main matrix.** All {n_seeds} seeds of every configuration already exist in `results/runs.csv` (made in Phases 6 to 10, each phase with its own gate), so a full run would skip them all. "
          "Either (a) keep those results as the final matrix, or (b) repeat seeds " + ", ".join(str(x) for x in cfg["seeds"][1:]) + " in the isolated workspace so that the whole matrix comes from one commit "
          f"(`run --stage full --workspace trial --approve-g4`, about {rest:.0f} min from the timings above).",
          f"2. **Extensions** (optional, cut first when the budget is short): A1 (Dirichlet alpha 0.1 and 10 on B3) has a runner, about {a1:.0f} min for {n_seeds} seeds "
          "(`run --stage full --groups extension --only A1-a0.1 A1-a10 --approve-g4`). A2 to A5 need the data pipeline (Phases 3 and 4) re-run for other quotas, split seeds or `max_rows_per_stream` "
          "and have no runner yet; their cost has not been measured.",
          "3. **Compute budget** (`compute.budget_hours` is not set): the orchestrator runs extensions last and prints the estimate before starting.", ""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(L), encoding="utf-8")
    return {"n_rows_compared": int(len(cmp_)), "n_identical": int(cmp_["identical"].sum()) if len(cmp_) and "identical" in cmp_ else 0, "n_failed": len(bad)}
