"""Phase 9 follow-up: DP-specific hyper-parameter search (Optuna TPE) for the CVAE trained with client-side DP-SGD.

The hyper-parameters of Phase 7 were tuned without DP. Under DP-SGD the useful settings differ (clipping bound C, batch
size, learning rate, model width), so they are searched again here, ONCE at one target epsilon (default 5) and then used for
every epsilon.

Proxy for speed: ONE client holding a class-stratified 20 % subsample of the train split (about the size of an average client),
trained with exactly the FL code path (`core.local_train_dp`: fresh Opacus engine and Adam every round, `fl.rounds` rounds of
`fl.local_epochs` epochs, sigma calibrated for the whole plan). No FedAvg, no non-IID: the proxy ranks configurations, and the
best ones are then checked in real federated runs (`fl.m1.verify_dp_candidates`).

Fitness: macro-F1 on the REAL VALIDATION split of a RF (100 trees) trained on `tune.syn_per_class` rows per class sampled from the
plain decoder (no residual noise, so the number depends only on the DP-trained weights). The test split is never used.
Limitation to report: the search uses real, non-private validation data and one seed per trial, so it is not itself covered by epsilon.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ppfeddata.eval.baselines import load_data
from ppfeddata.eval.utility import compute_metrics, full_proba, make_classifier
from ppfeddata.fl import core, dp_utils
from ppfeddata.models.cvae import build_layout, hidden_from_width
from ppfeddata.models.generate import generate
from ppfeddata.tune import load_best_cvae

logger = logging.getLogger("ppfeddata.tune_dp")

BEST_DP_PATH = Path("./configs/best_cvae_dp.yaml")
PROXY_FRACTION = 0.2
SPACE = {"latent_dim": [8, 16, 32], "width": [32, 64, 128, 256], "batch_size": [128, 256, 512, 1024],
         "beta": [0.1, 2.0], "lr": [2e-4, 5e-3], "max_grad_norm": [0.1, 20.0]}


def study_path(cfg: dict[str, Any], eps: float) -> Path:
    return Path(cfg["compute"]["artifacts_dir"]) / f"optuna_dp_eps{eps:g}_{cfg['label_mode']}.db"


def proxy_subsample(y: np.ndarray, n_classes: int, frac: float = PROXY_FRACTION, seed: int = 0) -> np.ndarray:
    """Class-stratified subsample (keeps the class proportions of the train split, at least 1 row per class)."""
    rng = np.random.default_rng(seed)
    idx = [rng.choice(np.flatnonzero(y == c), max(1, int(round(frac * (y == c).sum()))), replace=False) for c in range(n_classes) if (y == c).any()]
    return np.sort(np.concatenate(idx))


def dp_trial(cfg: dict[str, Any], hp: dict[str, Any], clip: float, eps: float, data, schema, sub: np.ndarray, seed: int = 0) -> dict[str, Any]:
    """Train one proxy client with DP-SGD under the FL schedule and score the plain decoder on VAL."""
    fl, k = cfg["fl"], len(schema["label_map"])
    lay = build_layout(schema)
    X, y = data["train"]["X"][sub], data["train"]["y"][sub]
    bs, rounds, le = int(hp["batch_size"]), int(fl["rounds"]), int(fl["local_epochs"])
    delta = float(cfg["dp"]["delta"])
    dp_utils.check_delta([len(X)], delta)
    sigma = dp_utils.calibrate_sigma(eps, delta, len(X), bs, rounds, le)
    state = core.init_state(lay, k, hp, seed)
    t0 = time.perf_counter()
    steps = 0
    for r in range(1, rounds + 1):
        out = core.local_train_dp(state, X, y, lay, k, hp, le, r, seed, 0, sigma, clip)
        state, steps = out["state"], steps + out["steps"]
    train_s = time.perf_counter() - t0
    model = core.make_model(lay, k, hp)
    model.load_state_dict(state)
    model.eval()
    t = cfg["tune"]
    Xs, ys = generate(model, schema, [int(t["syn_per_class"])] * k, seed, stats=None)
    rf = make_classifier("rf", {"eval": {"rf": {"n_estimators": int(t["rf_trees"])}}}, seed).fit(Xs, ys)
    Xv, yv = data["val"]["X"], data["val"]["y"]
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    m = compute_metrics(yv, rf.predict(Xv), full_proba(rf, Xv, k), classes)
    val = core.val_loss(state, Xv, yv, lay, k, hp)
    return {"macro_f1": float(m["macro_f1"]), "val_loss": float(val["loss"]), "sigma": float(sigma), "steps": int(steps),
            "eps_check": dp_utils.epsilon(sigma, dp_utils.sample_rate(len(X), bs), steps, delta), "train_seconds": train_s,
            "recall": {c: d["recall"] for c, d in m["per_class"].items()}}


def hp_from_params(base: dict[str, Any], p: dict[str, Any]) -> tuple[dict[str, Any], float]:
    hp = {**base, "latent_dim": int(p["latent_dim"]), "hidden": list(hidden_from_width(int(p["width"]))), "batch_size": int(p["batch_size"]),
          "beta": float(p["beta"]), "lr": float(p["lr"])}
    return hp, float(p["max_grad_norm"])


def run_tune_dp(cfg: dict[str, Any], eps: float = 5.0, n_trials: int = 24, seed: int = 0) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data, schema = load_data(cfg)
    k = len(schema["label_map"])
    sub = proxy_subsample(data["train"]["y"], k)
    base = load_best_cvae(cfg)
    path = study_path(cfg, eps)
    path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=f"cvae_dp_eps{eps:g}_{cfg['label_mode']}", direction="maximize", storage=f"sqlite:///{path.resolve().as_posix()}",
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=seed))
    if not study.trials:              # anchor points: the settings used so far, then two hand-picked "DP-friendly" starts
        study.enqueue_trial({"latent_dim": int(base["latent_dim"]), "width": int(base["hidden"][0]), "batch_size": int(base["batch_size"]),
                             "beta": float(base["beta"]), "lr": float(base["lr"]), "max_grad_norm": float(cfg["dp"]["max_grad_norm"])})
        study.enqueue_trial({"latent_dim": 16, "width": 64, "batch_size": 1024, "beta": 0.4, "lr": 1e-3, "max_grad_norm": 1.0})
        study.enqueue_trial({"latent_dim": 16, "width": 128, "batch_size": 512, "beta": 0.4, "lr": 1e-3, "max_grad_norm": 5.0})
    done = sum(t.state.name == "COMPLETE" for t in study.trials)

    def objective(trial) -> float:
        p = {"latent_dim": trial.suggest_categorical("latent_dim", SPACE["latent_dim"]), "width": trial.suggest_categorical("width", SPACE["width"]),
             "batch_size": trial.suggest_categorical("batch_size", SPACE["batch_size"]), "beta": trial.suggest_float("beta", *SPACE["beta"], log=True),
             "lr": trial.suggest_float("lr", *SPACE["lr"], log=True), "max_grad_norm": trial.suggest_float("max_grad_norm", *SPACE["max_grad_norm"], log=True)}
        hp, clip = hp_from_params(base, p)
        r = dp_trial(cfg, hp, clip, eps, data, schema, sub, seed)
        for key in ("val_loss", "sigma", "eps_check", "train_seconds", "recall"):
            trial.set_user_attr(key, r[key])
        logger.info("trial %d %s -> val TSTR macro-F1 %.4f, val ELBO %.2f, sigma %.2f (%.0fs)", trial.number,
                    {a: (round(b, 5) if isinstance(b, float) else b) for a, b in p.items()}, r["macro_f1"], r["val_loss"], r["sigma"], r["train_seconds"])
        return r["macro_f1"]

    todo = max(0, n_trials - done)
    logger.info("Optuna (DP, eps %g): %d trials done, %d to run", eps, done, todo)
    if todo:
        study.optimize(objective, n_trials=todo)
    return write_best_dp(cfg, study, eps)


def top_trials(study, n: int = 3):
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    return sorted(done, key=lambda t: t.value, reverse=True)[:n]


def write_best_dp(cfg: dict[str, Any], study, eps: float, out: str | Path = BEST_DP_PATH) -> dict[str, Any]:
    base = load_best_cvae(cfg)
    cands = []
    for t in top_trials(study, 3):
        hp, clip = hp_from_params(base, t.params)
        cands.append({"trial": int(t.number), "proxy_val_tstr_macro_f1": float(t.value), "proxy_val_elbo": float(t.user_attrs.get("val_loss", float("nan"))),
                      "max_grad_norm": clip, "cvae": {k: hp[k] for k in ("latent_dim", "hidden", "beta", "lr", "batch_size")}})
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    res = {"tuning": {"label_mode": cfg["label_mode"], "tuned_at_epsilon": float(eps), "n_trials": len(done),
                      "baseline_proxy_val_tstr_macro_f1": float(done[0].value) if done else None,
                      "fitness": "val macro-F1 of RF(100) trained on plain-decoder samples; proxy = one client, 20% stratified subsample, FL schedule",
                      "limitation": "tuned on real non-private validation data, one seed per trial, single-client proxy, one epsilon"},
           "candidates": cands}
    Path(out).write_text(yaml.safe_dump(res, sort_keys=False), encoding="utf-8")
    return res
