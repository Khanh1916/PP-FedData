"""Optimisation O2(a): search and final runs of M3-distributed (DP-FedSGD with the noise split over SecAgg, `fl/dpfedsgd.py`).

Same protocol as the O1 search (`tune_dp_full.py`): one Optuna study per epsilon, every trial a full run on the real partition (seed 0),
fitness = validation macro-F1 of a RF on `tune.syn_per_class` rows per class, the better of the plain decoder and the DP residual noise
(here released with distributed noise, `noise_share` = 1/K); the test split and the ledger are not touched. The space differs: one
gradient step per round, so the number of rounds and the expected total batch per round replace rounds x local epochs x client batch.
The final runs (`--final`) train the best trial of each epsilon with every seed as `M3f-t<n>-eps<e>` (and `-plain`) into the run ledger.
Limitation (as in O1): tuned on non-private validation data, one seed per trial.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import yaml

from ppfeddata.eval.baselines import load_data
from ppfeddata.fl import dpfedsgd
from ppfeddata.models.cvae import hidden_from_width
from ppfeddata.tune import load_best_cvae
from ppfeddata.tune_dp_full import n_complete, score_run, write_best

logger = logging.getLogger("ppfeddata.tune_fedsgd")

BEST_PATH = Path("./configs/best_cvae_fedsgd.yaml")
SPACE = {"latent_dim": [8, 16, 32], "width": [64, 128, 256], "batch_size": [512, 1024, 2048, 4096], "rounds": [500, 1000, 2000, 4000],
         "beta": [0.1, 2.0], "lr": [5e-4, 2e-2], "max_grad_norm": [0.1, 50.0], "cw_power": [0.0, 0.5, 1.0], "stat_frac": [0.02, 0.05, 0.1],
         "res_clip": [0.5, 1.0, 2.0, 4.0]}
# first trial of every study: the shape of the O1 winner at epsilon 5 (trial 27) with a central DP-SGD schedule
ANCHOR = {"latent_dim": 8, "width": 64, "batch_size": 2048, "rounds": 2000, "beta": 0.136, "lr": 0.005, "max_grad_norm": 1.0, "cw_power": 0.5,
          "stat_frac": 0.02, "res_clip": 1.0}


def trial_name(eps: float, number: int) -> str:
    return f"O2f-eps{eps:g}-t{number}"


def final_name(trial: int, eps: float) -> str:
    return f"M3f-t{int(trial)}-eps{eps:g}"


def hp_from_params(base: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    return {**base, "latent_dim": int(p["latent_dim"]), "hidden": list(hidden_from_width(int(p["width"]))), "beta": float(p["beta"]),
            "lr": float(p["lr"]), "cw_power": float(p["cw_power"])}


def run_kwargs(base: dict[str, Any], p: dict[str, Any], eps: float) -> dict[str, Any]:
    return {"hp": hp_from_params(base, p), "target_eps": float(eps), "rounds": int(p["rounds"]), "batch_size": int(p["batch_size"]),
            "clip": float(p["max_grad_norm"]), "stat_frac": float(p["stat_frac"]), "res_clip": float(p["res_clip"])}


def suggest(trial) -> dict[str, Any]:
    c, f = trial.suggest_categorical, trial.suggest_float
    return {"latent_dim": c("latent_dim", SPACE["latent_dim"]), "width": c("width", SPACE["width"]), "batch_size": c("batch_size", SPACE["batch_size"]),
            "rounds": c("rounds", SPACE["rounds"]), "beta": f("beta", *SPACE["beta"], log=True), "lr": f("lr", *SPACE["lr"], log=True),
            "max_grad_norm": f("max_grad_norm", *SPACE["max_grad_norm"], log=True), "cw_power": c("cw_power", SPACE["cw_power"]),
            "stat_frac": c("stat_frac", SPACE["stat_frac"]), "res_clip": c("res_clip", SPACE["res_clip"])}


def study_path(cfg: dict[str, Any], eps: float) -> Path:
    return Path(cfg["compute"]["artifacts_dir"]) / f"optuna_fedsgd_eps{eps:g}_{cfg['label_mode']}.db"


def run_search(cfg: dict[str, Any], eps: float, n_trials: int = 12, seed: int = 0) -> dict[str, Any]:
    import optuna

    from ppfeddata.fl import secagg

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data, schema = load_data(cfg)
    base, sa, K = load_best_cvae(cfg), secagg.secagg_spec(cfg), int(cfg["fl"]["num_clients"])
    path = study_path(cfg, eps)
    path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=f"cvae_fedsgd_eps{eps:g}_{cfg['label_mode']}", direction="maximize",
                                storage=f"sqlite:///{path.resolve().as_posix()}", load_if_exists=True)
    if not study.trials:
        study.enqueue_trial(ANCHOR)
    study.sampler = optuna.samplers.TPESampler(seed=seed + len(study.trials))       # no repeated startup points on resume (O1.5)

    def objective(trial) -> float:
        p = suggest(trial)
        t0 = time.perf_counter()
        run = dpfedsgd.run(cfg, seed, trial_name(eps, trial.number), secagg=sa, **run_kwargs(base, p, eps))
        r = score_run(cfg, run, schema, data, seed, noise_share=1.0 / K)
        d = run["summary"]["dp"]
        for key, v in (("variant", r["variant"]), ("eps_max", d["eps_all_honest"]), ("eps_one_honest", d["eps_one_honest"]),
                       ("fl_total_s", run["summary"]["fl_total_s"]), ("clipped_frac", run["summary"]["clipped_frac"])):
            trial.set_user_attr(key, v)
        for v in ("plain", "dps"):
            for m, x in (r[v] or {}).items():
                trial.set_user_attr(f"{v}_{m}", x)
        logger.info("fedsgd eps %g trial %d %s -> val macro-F1 %.4f (%s; plain %.4f), eps %.3f (one honest %.1f), %.0fs", eps, trial.number,
                    {a: (round(b, 4) if isinstance(b, float) else b) for a, b in p.items()}, r["value"], r["variant"], r["plain"]["macro_f1"],
                    d["eps_all_honest"], d["eps_one_honest"], time.perf_counter() - t0)
        return r["value"]

    logger.info("O2 DP-FedSGD search, eps %g: %d trials done", eps, n_complete(study))
    while n_complete(study) < n_trials and len(study.trials) < 3 * n_trials:
        study.optimize(objective, n_trials=1, catch=(RuntimeError, ValueError))
    y = write_best(cfg, eps, study, BEST_PATH)
    s = y.get("searches", {}).get(f"eps{eps:g}")
    if s:
        s["fitness"] = "val macro-F1 of RF on syn_per_class rows per class, better of plain decoder and DP residual noise (noise split over the clients); DP-FedSGD run, seed 0"
        BEST_PATH.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    return y


def run_final(cfg: dict[str, Any], eps_list: list[float] | None = None, seeds: list[int] | None = None, resume: bool = True,
              best: str | Path = BEST_PATH) -> list[str]:
    from ppfeddata.eval.runs import RunLedger, runs_csv_path
    from ppfeddata.fl import dp_stats, secagg
    from ppfeddata.fl.b3 import load_fl_model
    from ppfeddata.fl.m1 import _all_done
    from ppfeddata.models.b2 import evaluate_generator

    y = yaml.safe_load(Path(best).read_text(encoding="utf-8"))
    eps_list = [float(e) for e in (eps_list or cfg["dp"]["epsilons"])]
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    data, schema = load_data(cfg)
    ledger, mode, base = RunLedger(runs_csv_path(cfg)), cfg["label_mode"], load_best_cvae(cfg)
    sa, K = secagg.secagg_spec(cfg), int(cfg["fl"]["num_clients"])
    names = []
    for eps in eps_list:
        s = y["searches"][f"eps{eps:g}"]
        name = final_name(s["best_trial"], eps)
        names.append(name)
        for seed in seeds:
            if resume and _all_done(ledger, [name, name + "-plain"], seed, mode):
                logger.info("skip %s seed %d", name, seed)
                continue
            run = dpfedsgd.run(cfg, seed, name, secagg=sa, resume=resume, **run_kwargs(base, s["params"], eps))
            sm, d = run["summary"], run["summary"]["dp"]
            model = load_fl_model(cfg, run, schema)
            extra = {"alpha": run["alpha"], "num_clients": K, "rounds": int(s["params"]["rounds"]), "local_epochs": 1, "fl_total_s": sm["fl_total_s"],
                     "mean_round_s": sm["mean_round_s"], "bytes_per_round": sm["bytes_per_round_est"], "sa_bytes_per_round": sm["bytes_per_round_est"],
                     "bytes_estimated": True, "fl_final_val_loss": sm["final_val_loss"], "cvae_train_s": sm["fl_total_s"], "cvae_val_loss": sm["final_val_loss"],
                     "dp_mode": "distributed", "dp_target_eps": float(eps), "dp_eps_max": d["eps_all_honest"], "dp_eps_median": d["eps_all_honest"],
                     "dp_eps_one_honest": d["eps_one_honest"], "dp_delta": d["delta"], "dp_sigma_min": d["sigma"], "dp_sigma_max": d["sigma"],
                     "dp_clip": float(s["params"]["max_grad_norm"]), "dp_ratio_to_target": d["eps_all_honest"] / float(eps), "dp_steps_total": int(d["rounds"]),
                     "dp_q": d["q"], "dp_trial": int(s["best_trial"]), "dp_stat_frac": float(s["params"]["stat_frac"]), "dp_sigma_stat": d["sigma_stat"],
                     "dp_res_clip": float(s["params"]["res_clip"]), "cw_power": float(s["params"]["cw_power"]), "fl_rounds": int(s["params"]["rounds"]),
                     "dp_clipped_frac": sm["clipped_frac"], "sa_quant_clipped_frac": sm["quant_clipped_frac"]}
            st = dp_stats.stats_for_run(model, run, data["train"]["X"], data["train"]["y"], noise_share=1.0 / K)
            logger.info("%s seed %d: epsilon %.3f (one honest client %.1f), %.0fs", name, seed, d["eps_all_honest"], d["eps_one_honest"], sm["fl_total_s"])
            evaluate_generator(cfg, model, name + "-plain", seed, data, schema, ledger, extra, use_stats=False)
            evaluate_generator(cfg, model, name, seed, data, schema, ledger, {**extra, "residual_noise_dp": True}, stats=st)
    return names

