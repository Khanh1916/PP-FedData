"""Optimisation O1(b): DP hyper-parameter search at FULL scale, one study per epsilon.

The Phase 9 search (`tune_dp.py`) used a one-client proxy (20 % of train, no FedAvg) at epsilon 5 only, and its winner was used for every
epsilon; the scorecard of O0 then found M1-eps10 dominated by M1-eps1 and M1-eps5. Here every trial is a real federated run (the same
Flower code path as M1: `fl.run.run_fl`, 5 non-IID clients, seed 0) at the target epsilon of its study, and the search also covers the
FL schedule (rounds, local epochs), the class weights of the loss (`cw_power`, O1(c)) and the DP residual statistics (`stat_frac`,
`res_clip`, O1(a)).

Fitness, on the REAL VALIDATION split only (the ledger and the test split are not touched): macro-F1 of a RF (`tune.rf_trees` trees)
trained on `tune.syn_per_class` rows per class from the trained model, for the plain decoder and for the decoder with the DP residual
noise; the trial's value is the better of the two and the variant is recorded. Validation binary F1 and rare-class recall are recorded
too. Limitation (as in Phase 7 and 9): the choice uses non-private validation data and one seed per trial; it is not covered by epsilon.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ppfeddata.eval.baselines import load_data
from ppfeddata.eval.utility import compute_metrics, full_proba, make_classifier
from ppfeddata.fl import dp_stats
from ppfeddata.fl.b3 import load_fl_model
from ppfeddata.fl.run import run_fl
from ppfeddata.models.cvae import hidden_from_width
from ppfeddata.models.generate import generate
from ppfeddata.tune import load_best_cvae

logger = logging.getLogger("ppfeddata.tune_dp_full")

BEST_PATH = Path("./configs/best_cvae_dp_full.yaml")
SPACE = {"latent_dim": [8, 16, 32], "width": [64, 128, 256], "batch_size": [256, 512, 1024, 2048], "beta": [0.1, 2.0], "lr": [5e-4, 2e-2],
         "max_grad_norm": [0.5, 50.0], "rounds": [15, 30, 50], "local_epochs": [1, 2, 4], "cw_power": [0.0, 0.5, 1.0],
         "stat_frac": [0.02, 0.05, 0.1], "res_clip": [0.5, 1.0, 2.0, 4.0]}
# the M1d-t21 setting used so far (Phase 9), with the residual statistics added: the first trial of every study
ANCHOR = {"latent_dim": 16, "width": 128, "batch_size": 512, "beta": 0.2848, "lr": 0.0045, "max_grad_norm": 5.873, "rounds": 30,
          "local_epochs": 2, "cw_power": 0.0, "stat_frac": 0.05, "res_clip": 1.0}


def skip_reason(p: dict[str, Any]) -> str | None:
    """Settings that cannot run on this machine (O1.4): with width 256 the per-sample gradients of Opacus in 5 parallel Ray actors run out
    of memory and the actors die after 3 attempts (~8 min lost per trial). The choices of `SPACE` stay unchanged, because an existing
    study rejects a categorical distribution that differs from its earlier trials; such trials are pruned instead."""
    if int(p["width"]) == 256:
        return "width 256: out of memory with 5 Ray actors (O1.4)"
    if int(p["width"]) == 128 and int(p["batch_size"]) == 2048:
        return "width 128 with batch 2048: out of memory with 5 Ray actors (O1.8)"
    return None


def trial_name(eps: float, number: int) -> str:
    return f"O1-eps{eps:g}-t{number}"


def study_path(cfg: dict[str, Any], eps: float) -> Path:
    return Path(cfg["compute"]["artifacts_dir"]) / f"optuna_dpfull_eps{eps:g}_{cfg['label_mode']}.db"


def hp_from_params(base: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    return {**base, "latent_dim": int(p["latent_dim"]), "hidden": list(hidden_from_width(int(p["width"]))), "batch_size": int(p["batch_size"]),
            "beta": float(p["beta"]), "lr": float(p["lr"]), "cw_power": float(p["cw_power"])}


def run_kwargs(base: dict[str, Any], p: dict[str, Any], eps: float) -> dict[str, Any]:
    """Arguments of `run_fl` for a parameter set (shared by the search and the final runs)."""
    return {"target_eps": float(eps), "hp": hp_from_params(base, p), "max_grad_norm": float(p["max_grad_norm"]), "rounds": int(p["rounds"]),
            "local_epochs": int(p["local_epochs"]), "stat_frac": float(p["stat_frac"]), "res_clip": float(p["res_clip"])}


def val_scores(cfg: dict[str, Any], model, schema, data, stats, seed: int, rare: list[int]) -> dict[str, float]:
    Xs, ys = generate(model, schema, [int(cfg["tune"]["syn_per_class"])] * len(schema["label_map"]), seed, stats=stats)
    return val_scores_xy(cfg, Xs, ys, schema, data, seed, rare)


def val_scores_xy(cfg: dict[str, Any], Xs, ys, schema, data, seed: int, rare: list[int], n_jobs: int = -1) -> dict[str, float]:
    """Validation macro-F1, binary F1 and rare-class recall of a RF trained on the synthetic rows (Xs, ys)."""
    k, t = len(schema["label_map"]), cfg["tune"]
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    rf = make_classifier("rf", {"eval": {"rf": {"n_estimators": int(t["rf_trees"])}}}, seed).set_params(n_jobs=n_jobs).fit(Xs, ys)
    Xv, yv = data["val"]["X"], data["val"]["y"]
    m = compute_metrics(yv, rf.predict(Xv), full_proba(rf, Xv, k), classes)
    rec = [m["per_class"][classes[c]]["recall"] for c in rare]
    return {"macro_f1": float(m["macro_f1"]), "bin_f1": float(m["binary"]["f1"]), "rare_recall": float(np.mean(rec))}


def rare_labels(cfg: dict[str, Any], schema: dict[str, Any], limit: int = 5000) -> list[int]:
    lm, q = schema["label_map"], cfg.get("quota", {})
    r = [int(v) for c, v in lm.items() if c in q and q[c][0] <= limit]
    return sorted(r) or sorted(lm.values())[-4:]


def score_run(cfg: dict[str, Any], run: dict[str, Any], schema, data, seed: int = 0, noise_share: float = 1.0) -> dict[str, Any]:
    """`noise_share` < 1: the statistics release with its noise split over the clients (distributed DP, O2)."""
    model = load_fl_model(cfg, run, schema)
    rare = rare_labels(cfg, schema)
    plain = val_scores(cfg, model, schema, data, None, seed, rare)
    st = dp_stats.stats_for_run(model, run, data["train"]["X"], data["train"]["y"], noise_share)
    dps = val_scores(cfg, model, schema, data, st, seed, rare) if st is not None else None
    best = "dps" if dps is not None and dps["macro_f1"] > plain["macro_f1"] else "plain"
    return {"plain": plain, "dps": dps, "variant": best, "value": (dps if best == "dps" else plain)["macro_f1"],
            "eps_max": run["summary"]["dp"]["eps_max"], "fl_total_s": run["summary"]["fl_total_s"]}


def run_search(cfg: dict[str, Any], eps: float, n_trials: int = 30, seed: int = 0) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data, schema = load_data(cfg)
    base = load_best_cvae(cfg)
    path = study_path(cfg, eps)
    path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=f"cvae_dpfull_eps{eps:g}_{cfg['label_mode']}", direction="maximize",
                                storage=f"sqlite:///{path.resolve().as_posix()}", load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=seed))
    if not study.trials:
        study.enqueue_trial(ANCHOR)

    def objective(trial) -> float:
        p = {"latent_dim": trial.suggest_categorical("latent_dim", SPACE["latent_dim"]), "width": trial.suggest_categorical("width", SPACE["width"]),
             "batch_size": trial.suggest_categorical("batch_size", SPACE["batch_size"]), "beta": trial.suggest_float("beta", *SPACE["beta"], log=True),
             "lr": trial.suggest_float("lr", *SPACE["lr"], log=True), "max_grad_norm": trial.suggest_float("max_grad_norm", *SPACE["max_grad_norm"], log=True),
             "rounds": trial.suggest_categorical("rounds", SPACE["rounds"]), "local_epochs": trial.suggest_categorical("local_epochs", SPACE["local_epochs"]),
             "cw_power": trial.suggest_categorical("cw_power", SPACE["cw_power"]), "stat_frac": trial.suggest_categorical("stat_frac", SPACE["stat_frac"]),
             "res_clip": trial.suggest_categorical("res_clip", SPACE["res_clip"])}
        if (why := skip_reason(p)) is not None:
            logger.info("eps %g trial %d pruned: %s", eps, trial.number, why)
            raise optuna.TrialPruned(why)
        t0 = time.perf_counter()
        run = run_fl(cfg, seed, trial_name(eps, trial.number), resume=True, **run_kwargs(base, p, eps))
        r = score_run(cfg, run, schema, data, seed)
        for key in ("variant", "eps_max", "fl_total_s"):
            trial.set_user_attr(key, r[key])
        for v in ("plain", "dps"):
            if r[v]:
                for m, x in r[v].items():
                    trial.set_user_attr(f"{v}_{m}", x)
        logger.info("eps %g trial %d %s -> val macro-F1 %.4f (%s; plain %.4f), eps %.3f, %.0fs", eps, trial.number,
                    {a: (round(b, 4) if isinstance(b, float) else b) for a, b in p.items()}, r["value"], r["variant"], r["plain"]["macro_f1"],
                    r["eps_max"], time.perf_counter() - t0)
        return r["value"]

    # a sampler seeded with `seed` alone would repeat the same startup points on every resume (and in every study): offset by the trials so far
    study.sampler = optuna.samplers.TPESampler(seed=seed + len(study.trials))
    done = n_complete(study)
    logger.info("O1 search, eps %g: %d trials done, %d to run", eps, done, max(0, n_trials - done))
    # pruned and failed trials count in `study.optimize(n_trials=...)`; run until n_trials are COMPLETE, at most 3 x n_trials attempts in all
    while n_complete(study) < n_trials and len(study.trials) < 3 * n_trials:
        study.optimize(objective, n_trials=1, catch=(RuntimeError,))
    return write_best(cfg, eps, study)


def n_complete(study) -> int:
    return sum(t.state.name == "COMPLETE" for t in study.trials)


def write_best(cfg: dict[str, Any], eps: float, study, out: str | Path = BEST_PATH) -> dict[str, Any]:
    """Best trial per epsilon (on validation) into configs/best_cvae_dp_full.yaml; the other epsilons already there are kept."""
    out = Path(out)
    y = yaml.safe_load(out.read_text(encoding="utf-8")) if out.exists() else {}
    done = sorted([t for t in study.trials if t.state.name == "COMPLETE"], key=lambda t: t.value, reverse=True)
    if not done:
        return y
    b = done[0]
    y.setdefault("searches", {})[f"eps{eps:g}"] = {
        "epsilon": float(eps), "n_trials": len(done), "best_trial": int(b.number), "val_macro_f1": float(b.value), "variant": b.user_attrs.get("variant"),
        "anchor_val_macro_f1": float(next((t.value for t in done if t.number == 0), float("nan"))),
        "params": {k: (float(v) if isinstance(v, float) else v) for k, v in b.params.items()},
        "val": {k: float(v) for k, v in b.user_attrs.items() if isinstance(v, float)},
        "fitness": "val macro-F1 of RF on syn_per_class rows per class, better of plain decoder and DP residual noise; full FL run, seed 0",
        "limitation": "tuned on real non-private validation data, one seed per trial"}
    out.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    return y


# --------------------------------------------------------------------------------------------------
# Final runs of the chosen settings (3 seeds, into the run ledger)
# --------------------------------------------------------------------------------------------------
def final_name(method: str, trial: int, eps: float) -> str:
    """`M1o-t<trial>-eps<e>` (local DP) and `M3o-...` (local DP + SecAgg); the ledger adds `-plain` for the plain decoder."""
    return f"{method}o-t{int(trial)}-eps{eps:g}"


def run_final(cfg: dict[str, Any], eps_list: list[float] | None = None, seeds: list[int] | None = None, methods: tuple[str, ...] = ("M1", "M3"),
              resume: bool = True, best: str | Path = BEST_PATH) -> list[str]:
    """For every epsilon, the best trial of its search, trained with every seed, evaluated twice (plain decoder and DP residual noise,
    both inside epsilon). M3 = the same setting with Flower's SecAgg+ (each client still adds its full DP noise: M3-local)."""
    from ppfeddata.eval.runs import RunLedger, runs_csv_path
    from ppfeddata.fl import secagg
    from ppfeddata.fl.b3 import fl_extra
    from ppfeddata.fl.m1 import _all_done, dp_extra
    from ppfeddata.fl.m2 import sa_extra
    from ppfeddata.models.b2 import evaluate_generator

    y = yaml.safe_load(Path(best).read_text(encoding="utf-8"))
    eps_list = [float(e) for e in (eps_list or cfg["dp"]["epsilons"])]
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    data, schema = load_data(cfg)
    ledger, mode, base = RunLedger(runs_csv_path(cfg)), cfg["label_mode"], load_best_cvae(cfg)
    names = []
    for eps in eps_list:
        s = y["searches"][f"eps{eps:g}"]
        for method in methods:
            name = final_name(method, s["best_trial"], eps)
            names.append(name)
            kw = run_kwargs(base, s["params"], eps)
            if method == "M3":
                kw["secagg"] = secagg.secagg_spec(cfg)
            for seed in seeds:
                if resume and _all_done(ledger, [name, name + "-plain"], seed, mode):
                    logger.info("skip %s seed %d", name, seed)
                    continue
                run = run_fl(cfg, seed, name, resume=resume, **kw)
                model = load_fl_model(cfg, run, schema)
                extra = {**fl_extra(run), **dp_extra(run), "dp_trial": int(s["best_trial"]), "dp_stat_frac": float(s["params"]["stat_frac"]),
                         "dp_sigma_stat": float(run["dp"]["sigma_stat"]), "dp_res_clip": float(s["params"]["res_clip"]),
                         "cw_power": float(s["params"]["cw_power"]), "fl_rounds": int(s["params"]["rounds"]), "fl_local_epochs": int(s["params"]["local_epochs"])}
                if method == "M3":
                    extra.update(sa_extra(run))
                st = dp_stats.stats_for_run(model, run, data["train"]["X"], data["train"]["y"])
                logger.info("%s seed %d: epsilon max %.3f (target %g, statistics included), %.0fs", name, seed, run["summary"]["dp"]["eps_max"], eps,
                            run["summary"]["fl_total_s"])
                evaluate_generator(cfg, model, name + "-plain", seed, data, schema, ledger, extra, use_stats=False)
                evaluate_generator(cfg, model, name, seed, data, schema, ledger, {**extra, "residual_noise_dp": True}, stats=st)
    return names
