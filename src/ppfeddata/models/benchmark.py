"""Phase 7.4: compute benchmark on ONE client-sized dataset, then extrapolation to the whole experiment matrix.

Measured (each in a fresh process so the peak RAM is clean): seconds per epoch of the plain CVAE and of DP-SGD
(Opacus, Poisson sampling) at n = 18,000 rows (and 9,000 to check that the cost is linear in n), the one-off cost of
`make_private`, and the peak RAM. Everything else in `results/reports/compute_budget.md` is arithmetic on those numbers
and on the measured B2 / Optuna / baseline times in `results/runs.csv` and the Optuna study; assumptions are printed.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("ppfeddata.models.benchmark")

N_CLIENT = 18000
REPORT = Path("./results/reports/compute_budget.md")


# --------------------------------------------------------------------------------------------------
# Child process: one measurement
# --------------------------------------------------------------------------------------------------
def measure(kind: str, cfg: dict[str, Any], n_rows: int = N_CLIENT, epochs: int = 6, seed: int = 0,
            noise_multiplier: float = 1.0, max_grad_norm: float = 1.0) -> dict[str, Any]:
    import torch

    from ppfeddata.eval.baselines import load_data
    from ppfeddata.eval.overhead import peak_rss_gb
    from ppfeddata.models.cvae import CVAE, beta_at, build_layout, loss_terms, numpy_to_tensor, one_hot
    from ppfeddata.models.train import fit_epoch, seed_torch
    from ppfeddata.tune import load_best_cvae

    data, schema = load_data(cfg)
    hp = load_best_cvae(cfg)
    k = len(schema["label_map"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data["train"]["y"]), size=n_rows, replace=False)
    X, y = numpy_to_tensor(data["train"]["X"][idx]), torch.as_tensor(data["train"]["y"][idx], dtype=torch.long)
    seed_torch(seed)
    model = CVAE(build_layout(schema), k, int(hp["latent_dim"]), tuple(hp["hidden"]))
    opt = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]))
    bs, beta = int(hp["batch_size"]), float(hp["beta"])
    res: dict[str, Any] = {"kind": kind, "n_rows": n_rows, "batch_size": bs, "epochs_timed": epochs,
                           "n_params": int(sum(p.numel() for p in model.parameters()))}
    times: list[float] = []
    if kind == "plain":
        for ep in range(epochs):
            t0 = time.perf_counter()
            fit_epoch(model, opt, X, y, bs, beta_at(ep, beta, 5), rng)
            times.append(time.perf_counter() - t0)
        res["steps_per_epoch"] = int(np.ceil(n_rows / bs))
    elif kind == "dp":
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        from torch.utils.data import DataLoader, TensorDataset

        errors = ModuleValidator.validate(model, strict=False)
        res["opacus_validation_errors"] = [str(e) for e in errors]
        if errors:
            raise RuntimeError(f"model is not Opacus-compatible: {errors}")
        loader = DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=True)
        t0 = time.perf_counter()
        pe = PrivacyEngine()
        gs_model, dp_opt, dp_loader = pe.make_private(module=model, optimizer=opt, data_loader=loader,
                                                      noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
                                                      poisson_sampling=True)
        res["make_private_s"] = time.perf_counter() - t0
        for ep in range(epochs):
            gs_model.train()
            t0, steps = time.perf_counter(), 0
            for xb, yb in dp_loader:
                if len(xb) == 0:
                    continue
                out, mu, logvar = gs_model(xb, one_hot(yb, k))
                loss = loss_terms(out, xb, mu, logvar, beta_at(ep, beta, 5), gs_model._module.layout)["loss"]
                dp_opt.zero_grad(set_to_none=True)
                loss.backward()
                dp_opt.step()
                steps += 1
            times.append(time.perf_counter() - t0)
        res["steps_per_epoch"] = steps
        res["noise_multiplier"] = noise_multiplier
    else:
        raise ValueError(kind)
    res["epoch_seconds"] = times
    res["first_epoch_s"] = times[0]
    res["epoch_s"] = float(np.mean(times[1:])) if len(times) > 1 else times[0]   # warm epochs
    res["peak_rss_gb"] = peak_rss_gb()
    return res


# --------------------------------------------------------------------------------------------------
# Parent: orchestration + extrapolation
# --------------------------------------------------------------------------------------------------
def _child(kind: str, n_rows: int, label_mode: str, config_path: str) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "ppfeddata.models.benchmark", "--measure", kind, "--n-rows", str(n_rows),
           "--label-mode", label_mode, "--config", config_path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return json.loads(out[-1])


def _optuna_trial_seconds(cfg: dict[str, Any]) -> tuple[float, int]:
    from ppfeddata.tune import study_path
    p = study_path(cfg)
    if not p.exists():
        return float("nan"), 0
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=f"cvae_{cfg['label_mode']}", storage=f"sqlite:///{p.resolve().as_posix()}")
    d = [t.duration.total_seconds() for t in study.trials if t.state.name == "COMPLETE" and t.duration]
    return (float(np.mean(d)) if d else float("nan")), len(d)


def _b2_eval_seconds(cfg: dict[str, Any]) -> dict[str, float]:
    from ppfeddata.eval.runs import RunLedger, runs_csv_path
    df = RunLedger(runs_csv_path(cfg)).frame()
    df = df[(df.get("label_mode") == cfg["label_mode"]) & df["config"].astype(str).str.startswith("B2-")] if len(df) else df
    if not len(df) or "fidpriv_s" not in df:
        return {}
    per_seed = df.groupby("seed").agg(train=("cvae_train_s", "first"), gen=("cvae_gen_s", "first"), fp=("fidpriv_s", "first"),
                                      fit=("fit_time_s", "sum"))
    return {"cvae_train_s": float(per_seed["train"].mean()), "gen_s": float(per_seed["gen"].mean()),
            "eval_s": float((per_seed["gen"] + per_seed["fp"] + per_seed["fit"]).mean()), "seeds": int(len(per_seed))}


def extrapolate(cfg: dict[str, Any], plain: dict[str, Any], dp: dict[str, Any], n_train: int, trial_s: float,
                b2: dict[str, float]) -> dict[str, Any]:
    fl, nseed = cfg["fl"], len(cfg["seeds"])
    R, E, C = int(fl["rounds"]), int(fl["local_epochs"]), int(fl["num_clients"])
    per_row_plain, per_row_dp = plain["epoch_s"] / plain["n_rows"], dp["epoch_s"] / dp["n_rows"]
    eval_s = b2.get("eval_s", float("nan"))

    def fl_run(per_row: float, rounds: int, setup: float = 0.0) -> float:
        # clients run one after the other: sum_i t_epoch_i = per_row * sum_i n_i = per_row * n_train
        return rounds * E * per_row * n_train + rounds * C * setup

    runs = {"plain": fl_run(per_row_plain, R) + eval_s, "dp": fl_run(per_row_dp, R, dp.get("make_private_s", 0.0)) + eval_s}
    ratio = cfg["thresholds"]["overhead_ratio_max"]
    rows = [("B2 (centralised CVAE)", nseed, b2.get("cvae_train_s", float("nan")) + eval_s),
            ("B3 (FL)", nseed, runs["plain"]),
            ("M1-eps1 (FL + DP)", nseed, runs["dp"]), ("M1-eps5 (FL + DP)", nseed, runs["dp"]),
            ("M1-eps10 (FL + DP)", nseed, runs["dp"]),
            ("M2 (FL + SecAgg), no crypto cost assumed", nseed, runs["plain"]),
            ("M3 (FL + DP + SecAgg), no crypto cost assumed", nseed, runs["dp"])]
    n_trials = int(cfg["tune"]["n_trials"])
    optuna_s = n_trials * trial_s
    core = sum(n * t for _, n, t in rows)
    total = core + optuna_s
    ext = [("A1 non-IID alpha {0.1, 10} on B3 (alpha 0.5 = B3)", 2 * nseed * runs["plain"]),
           ("A2 imbalance {10:1, 60:1, 200:1} on B3 (FL cost only; re-sampling not included)", 3 * nseed * runs["plain"]),
           ("A3 11-class mode on B3", nseed * runs["plain"]),
           ("A4 two other split seeds, B0 + B3 (B0 cost negligible)", 2 * nseed * runs["plain"]),
           ("A5 max_rows_per_stream on B3 and M1-eps5", nseed * (runs["plain"] + runs["dp"]))]
    ext_total = sum(t for _, t in ext)
    # cuts, in the order the spec prescribes; each saving is computed against the full plan, not cumulatively
    cuts = [("(a) Optuna 30 -> 15 trials", (n_trials - 15) * trial_s),
            ("(b) rounds 30 -> 20 (all FL runs)", sum(n * (t - eval_s) for _, n, t in rows[1:]) * (1 - 20 / R)),
            ("(c) seeds of M1-eps1 and M1-eps10: 3 -> 2", 2 * runs["dp"]),
            ("(d) drop extensions A1-A5", ext_total)]
    secagg_extra = (ratio - 1) * nseed * (runs["plain"] + runs["dp"])
    return {"per_row_plain_s": per_row_plain, "per_row_dp_s": per_row_dp, "dp_over_plain": per_row_dp / per_row_plain,
            "fl_run_plain_s": runs["plain"], "fl_run_dp_s": runs["dp"], "rows": rows, "optuna_s": optuna_s,
            "core_s": core, "total_s": total, "secagg_worst_factor": ratio, "secagg_worst_extra_s": secagg_extra,
            "cuts": cuts, "extensions": ext, "extensions_total_s": ext_total, "eval_s": eval_s, "n_train": n_train}


def run_benchmark(cfg: dict[str, Any], config_path: str = "configs/default.yaml") -> dict[str, Any]:
    import torch

    from ppfeddata.eval.baselines import load_data
    from ppfeddata.eval.runs import RunLedger, runs_csv_path
    from ppfeddata.utils import config_hash, git_commit

    mode = cfg["label_mode"]
    res = {}
    for kind in ("plain", "dp"):
        for n in (N_CLIENT // 2, N_CLIENT):
            logger.info("measuring %s at n=%d", kind, n)
            res[f"{kind}_{n}"] = _child(kind, n, mode, config_path)
    plain, dp = res[f"plain_{N_CLIENT}"], res[f"dp_{N_CLIENT}"]
    n_train = len(load_data(cfg)[0]["train"]["y"])
    trial_s, n_done = _optuna_trial_seconds(cfg)
    b2 = _b2_eval_seconds(cfg)
    ex = extrapolate(cfg, plain, dp, n_train, trial_s, b2)
    df = RunLedger(runs_csv_path(cfg)).frame()
    base_s = float(df[df["config"].astype(str).str.startswith(("B0", "B1"))]["fit_time_s"].sum()) if len(df) else float("nan")
    env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": bool(torch.cuda.is_available()),
           "threads": torch.get_num_threads(), "cpu": platform.processor(), "machine": platform.machine(),
           "git_commit": git_commit(), "config_hash": config_hash(cfg)}
    try:
        import opacus
        env["opacus"] = opacus.__version__
    except ImportError:
        pass
    b3_check = _b3_check(cfg, ex)
    raw = {"env": env, "b3_check": b3_check, "measurements": res, "extrapolation": {k: v for k, v in ex.items()}, "optuna_trial_s": trial_s,
           "optuna_trials_done": n_done, "b2": b2, "baselines_fit_s_total": base_s}
    out = Path(cfg["compute"]["artifacts_dir"]) / f"compute_benchmark_{mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
    write_report(cfg, raw)
    return raw


def _b3_check(cfg: dict[str, Any], ex: dict[str, Any]) -> dict[str, Any] | None:
    from ppfeddata.eval.runs import run_id
    from ppfeddata.fl.core import read_round_log

    rows, seeds = [], []
    for sd in cfg["seeds"]:
        rl = [r for r in read_round_log(Path(cfg["compute"]["artifacts_dir"]) / run_id("B3", sd, cfg["label_mode"])) if r["round"] > 0]
        if len(rl) > 1:
            rows.append(rl)
            seeds.append(sd)
    if not rows:
        return None
    return {"seeds": seeds, "first_round_s": float(np.mean([r[0]["round_seconds"] for r in rows])),
            "median_round_s": float(np.median([x["round_seconds"] for r in rows for x in r[1:]])),
            "sum_client_s": float(np.median([sum(x["client_seconds"]) for r in rows for x in r[1:]])),
            "pred_round_s": float(cfg["fl"]["local_epochs"] * ex["per_row_plain_s"] * ex["n_train"])}


def _h(s: float) -> str:
    return "n/a" if s != s else f"{s / 3600:.2f} h" if s >= 600 else f"{s:.0f} s"


def write_report(cfg: dict[str, Any], raw: dict[str, Any], out: str | Path = REPORT) -> None:
    from ppfeddata.eval.baselines import _table
    m, ex, env = raw["measurements"], raw["extrapolation"], raw["env"]
    fl = cfg["fl"]
    budget = cfg["compute"].get("budget_hours")
    L = ["# Compute budget (Phase 7.4)", "",
         f"Auto-generated by `ppfeddata benchmark`. Hardware/software: Python {env['python']}, torch {env['torch']}"
         f"{', Opacus ' + env['opacus'] if 'opacus' in env else ''}, CUDA available: {env['cuda']} (all timings are CPU, "
         f"{env['threads']} threads), {env['cpu']}. Commit `{(env['git_commit'] or 'n/a')[:10]}`, config hash `{env['config_hash'][:10]}`. "
         "Timings depend on this machine; a Colab session will differ, so re-run `ppfeddata benchmark` there before using the totals.", "",
         "## Measured on one client-sized dataset", ""]
    rows = []
    for key, r in m.items():
        rows.append({"run": key, "rows": r["n_rows"], "batch": r["batch_size"], "steps/epoch": r["steps_per_epoch"],
                     "first epoch s": f"{r['first_epoch_s']:.2f}", "warm epoch s": f"{r['epoch_s']:.2f}",
                     "make_private s": f"{r.get('make_private_s', 0):.2f}" if "make_private_s" in r else "-",
                     "peak RAM GB": f"{r['peak_rss_gb']:.2f}", "params": r["n_params"]})
    L += [_table(rows), "",
          f"Warm epoch = mean of the epochs after the first (6 epochs timed). DP-SGD = Opacus {env.get('opacus', '?')}, Poisson sampling, "
          f"flat clipping; the noise multiplier does not change the cost. Linearity check (time per row at n=9,000 vs 18,000): "
          f"plain {m['plain_9000']['epoch_s'] / 9000 * 1e6:.1f} vs {m['plain_18000']['epoch_s'] / 18000 * 1e6:.1f} us/row, "
          f"DP {m['dp_9000']['epoch_s'] / 9000 * 1e6:.1f} vs {m['dp_18000']['epoch_s'] / 18000 * 1e6:.1f} us/row. "
          "Epochs of a fraction of a second are noisy and per-step overhead makes small clients slightly more expensive per row, "
          "so the n = 18,000 figure is used for the extrapolation.", "",
          f"**DP-SGD costs {ex['dp_over_plain']:.2f}x the plain epoch** (this is the cost ratio that matters for the matrix).", "",
          "## Extrapolation", "",
          f"Formula (spec 7.4): FL run = rounds x local_epochs x sum_i t_epoch_i, clients sequential, t_epoch_i linear in n_i, so "
          f"sum_i t_epoch_i = (s/row) x n_train ({ex['per_row_plain_s'] * 1e6:.1f} us/row plain, {ex['per_row_dp_s'] * 1e6:.1f} us/row DP; "
          f"n_train = {ex['n_train']}); rounds {fl['rounds']}, local epochs {fl['local_epochs']}, "
          f"{fl['num_clients']} clients. DP runs add `make_private` once per client per round. Each run also pays the evaluation "
          f"(generate + fidelity + privacy + 2 classifiers x TSTR/TAug) measured in B2: {_h(ex['eval_s'])}.", ""]
    rows = [{"config": n, "runs": c, "time per run": _h(t), "total": _h(c * t)} for n, c, t in ex["rows"]]
    rows.append({"config": f"Optuna ({cfg['tune']['n_trials']} trials)", "runs": 1,
                 "time per run": _h(raw["optuna_trial_s"]) + " per trial", "total": _h(ex["optuna_s"])})
    rows.append({"config": "**TOTAL core matrix + Optuna**", "runs": "", "time per run": "", "total": _h(ex["total_s"])})
    L += [_table(rows), "",
          f"Already spent: baselines B0/B1 fit time {_h(raw['baselines_fit_s_total'])} (all seeds), Optuna {raw['optuna_trials_done']} trials.", "",
          "Assumptions to check later: (1) SecAgg (M2/M3) is counted with no crypto cost because Phase 10 has not been written; "
          f"the worst case allowed by `thresholds.overhead_ratio_max` = {ex['secagg_worst_factor']} x the per-run time would add "
          f"up to {_h(ex['secagg_worst_extra_s'])}; "
          "(2) time per epoch is taken as linear in the number of rows (see the linearity check above for how close this is); (3) the single-machine simulation measures compute, "
          "not network latency.", ""]
    v = raw.get("b3_check")
    if v:
        L += ["## Check against the measured B3 runs (Phase 8)", "",
              f"B3 (no DP) on the real data, seeds {v['seeds']}: median over rounds 2+ of the summed client training time per round = "
              f"{v['sum_client_s']:.2f} s (predicted sequential cost per round = local_epochs x s/row x n_train = {v['pred_round_s']:.2f} s); "
              f"median wall time per round = {v['median_round_s']:.2f} s because the simulation runs the {fl['num_clients']} clients in parallel "
              f"(Ray actors, 2 torch threads each); round 1 = {v['first_round_s']:.1f} s because it includes starting Ray. "
              f"The measured sequential client time is {v['sum_client_s'] / v['pred_round_s'] * 100 - 100:+.0f} % against the formula (so the formula is about right for a "
              "client-by-client run, with per-round overheads it does not model), while the parallel wall time per round is lower; "
              "each FL run also pays a fixed start-up of about half a minute (round 1).", ""]
    L += ["## Budget decision", ""]
    if budget is None:
        L += ["`compute.budget_hours` is not set (null), so no cut is applied. The spec says to cut in this order if the total "
              "exceeds the budget or does not fit the Colab sessions; the saving of each cut (against the totals above):", ""]
    else:
        L += [f"`compute.budget_hours` = {budget} h; total {_h(ex['total_s'])}: **{'WITHIN' if ex['total_s'] <= budget * 3600 else 'OVER'}** budget.", ""]
    rows = [{"cut (spec order)": n, "saves": _h(s)} for n, s in ex["cuts"]]
    L += [_table(rows), "",
          "Extensions A1-A5 (cost if they were run; approximate):", "",
          _table([{"extension": n, "approx. cost": _h(t)} for n, t in ex["extensions"]]),
          "", f"All extensions together: {_h(ex['extensions_total_s'])}. Never cut: seeds of B0-B3, M1-eps5, M2, M3.", ""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(L), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", choices=["plain", "dp"], required=True)
    ap.add_argument("--n-rows", type=int, default=N_CLIENT)
    ap.add_argument("--label-mode", default="6class")
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args(argv)
    from ppfeddata.utils import load_config
    cfg = load_config(a.config)
    cfg["label_mode"] = a.label_mode
    print(json.dumps(measure(a.measure, cfg, a.n_rows)))


if __name__ == "__main__":
    main()

