"""Phase 7.3: Optuna (TPE) search of the CVAE hyper-parameters on CENTRALISED, non-private data.

Fitness = macro-F1 of a RF (100 trees) trained on `syn_per_class` synthetic rows per class (TSTR) and scored on the REAL
VALIDATION split. The test split is never touched. The study lives in SQLite (artifacts/) so it resumes between sessions.
Limitation to report: tuning used real, centralised data (not private), and one seed per trial.
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
from ppfeddata.models.cvae import build_layout, hidden_from_width
from ppfeddata.models.generate import gen_stats_from_cfg, generate
from ppfeddata.models.train import train_cvae

logger = logging.getLogger("ppfeddata.tune")

BEST_PATH = Path("./configs/best_cvae.yaml")
SEARCH_SPACE = {"latent_dim": [8, 16, 32], "width": [64, 128, 256], "beta": [0.1, 2.0], "lr": [1e-4, 3e-3]}


def study_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["compute"]["artifacts_dir"]) / f"optuna_cvae_{cfg['label_mode']}.db"


def tstr_fitness(cfg: dict[str, Any], hp: dict[str, Any], data, schema, seed: int = 0, epochs: int | None = None,
                 hidden: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Train a CVAE with `hp`, draw `syn_per_class` rows per class, fit a RF, score macro-F1 on VAL."""
    t = cfg["tune"]
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    k = len(classes)
    t0 = time.perf_counter()
    model, hist = train_cvae(data["train"]["X"], data["train"]["y"], data["val"]["X"], data["val"]["y"], k,
                             build_layout(schema), hp, seed, epochs=epochs or int(t["epochs"]), hidden=hidden)
    t_train = time.perf_counter() - t0
    stats = gen_stats_from_cfg(model, data["train"]["X"], data["train"]["y"], schema, cfg["generate"])
    Xs, ys = generate(model, schema, [int(t["syn_per_class"])] * k, seed, stats=stats)
    rf = make_classifier("rf", {"eval": {"rf": {"n_estimators": int(t["rf_trees"])}}}, seed)
    rf.fit(Xs, ys)
    Xv, yv = data["val"]["X"], data["val"]["y"]
    m = compute_metrics(yv, rf.predict(Xv), full_proba(rf, Xv, k), classes)
    return {"macro_f1": float(m["macro_f1"]), "val_loss": float(hist[-1]["val_loss"]), "train_seconds": t_train,
            "recall": {c: d["recall"] for c, d in m["per_class"].items()}}


def suggest(trial) -> dict[str, Any]:
    s = SEARCH_SPACE
    return {"latent_dim": trial.suggest_categorical("latent_dim", s["latent_dim"]),
            "width": trial.suggest_categorical("width", s["width"]),
            "beta": trial.suggest_float("beta", *s["beta"], log=True),
            "lr": trial.suggest_float("lr", *s["lr"], log=True)}


def run_tune(cfg: dict[str, Any], n_trials: int | None = None, seed: int = 0) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data, schema = load_data(cfg)
    n_trials = int(n_trials if n_trials is not None else cfg["tune"]["n_trials"])
    path = study_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=f"cvae_{cfg['label_mode']}", direction="maximize",
                                storage=f"sqlite:///{path.resolve().as_posix()}", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    done = sum(t.state.name == "COMPLETE" for t in study.trials)
    if not study.trials:      # trial 0 = the default config, so the search is anchored to what we would use anyway
        d = cfg["cvae"]
        study.enqueue_trial({"latent_dim": int(d["latent_dim"]), "width": int(d["hidden"][0]), "beta": float(d["beta"]),
                             "lr": float(d["lr"])})

    def objective(trial) -> float:
        p = suggest(trial)
        hp = {**cfg["cvae"], "latent_dim": p["latent_dim"], "beta": p["beta"], "lr": p["lr"]}
        r = tstr_fitness(cfg, hp, data, schema, seed=seed, hidden=hidden_from_width(p["width"]))
        trial.set_user_attr("val_loss", r["val_loss"])
        trial.set_user_attr("train_seconds", r["train_seconds"])
        trial.set_user_attr("recall", r["recall"])
        logger.info("trial %d %s -> val TSTR macro-F1 %.4f (%.0fs)", trial.number, p, r["macro_f1"], r["train_seconds"])
        return r["macro_f1"]

    todo = max(0, n_trials - done)
    logger.info("Optuna: %d trials done, %d to run (target %d)", done, todo, n_trials)
    if todo:
        study.optimize(objective, n_trials=todo)
    return write_best(cfg, study)


def write_best(cfg: dict[str, Any], study, out: str | Path = BEST_PATH) -> dict[str, Any]:
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    best = study.best_trial
    p = best.params
    d = cfg["cvae"]
    res = {"cvae": {"latent_dim": int(p["latent_dim"]), "hidden": list(hidden_from_width(p["width"])),
                    "beta": float(p["beta"]), "lr": float(p["lr"]), "epochs": int(cfg["tune"]["epochs"]),
                    "beta_warmup_epochs": int(d["beta_warmup_epochs"]), "batch_size": int(d["batch_size"]),
                    "class_balanced_sampler": bool(d["class_balanced_sampler"]), "patience": d.get("patience")},
           "tuning": {"label_mode": cfg["label_mode"], "n_trials": len(done), "best_trial": int(best.number),
                      "best_val_tstr_macro_f1": float(best.value),
                      "default_val_tstr_macro_f1": float(done[0].value) if done else None,
                      "fitness": "macro-F1 of RF(100 trees) trained on syn_per_class synthetic rows per class, scored on REAL val",
                      "limitation": "tuned on centralised, non-private data with one seed per trial"}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(yaml.safe_dump(res, sort_keys=False), encoding="utf-8")
    return res


def load_best_cvae(cfg: dict[str, Any], path: str | Path = BEST_PATH) -> dict[str, Any]:
    """`cfg['cvae']` overridden by configs/best_cvae.yaml when it exists (and was tuned for this label mode)."""
    p = Path(path)
    hp = dict(cfg["cvae"])
    if p.exists():
        best = yaml.safe_load(p.read_text(encoding="utf-8"))
        if best.get("tuning", {}).get("label_mode") == cfg["label_mode"]:
            hp.update(best["cvae"])
    return hp
