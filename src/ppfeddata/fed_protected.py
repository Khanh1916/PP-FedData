"""Follow-up to `fed_classifier.py`: the directly federated detector with the same protections as the CVAE route, and with a little tuning.

`fed_classifier.py` found that an MLP trained by FedAvg with class weights matches the federated CVAE route when nothing is protected, but the MLP was untuned
and had no DP or SecAgg while the CVAE had both. This module gives the direct route the same chances:

- `FedMLPcwT`            class-weighted FedAvg MLP with the learning rate and the width chosen on VALIDATION data (a small grid, seed 0, 30 rounds);
- `FedMLPcwT-sa`         the same with the quantisation of SecAgg+ applied to the aggregation (see `quantised_average`);
- `FedMLPcwT-dp{1,5,10}` DP-SGD at the client (Opacus, Poisson sampling, per-sample clipping, the RDP accountant of `fl/dp_utils.py`, delta 1e-5, one sigma per client for
                         the whole run, epsilon = max over clients), learning rate and clipping bound chosen on validation at epsilon 5;
- `FedMLPcwT-dp5-sa`     both.

What the SecAgg variant is: the quantiser of Flower's SecAgg+ (weights scaled by n_i / max_weight, clipped to +-clipping_range, stochastic rounding to 2^22 levels,
integer sum, de-quantisation) applied to the clients' updates, in process. The masks cancel exactly in the real protocol and are not emulated, so this measures the
numerical effect of SecAgg on utility, not its protection or its cost; for the CVAE the real protocol was run (R5).
The class weights are computed from the clients' summed class counts, which the clients would have to share (labels are not protected, as in the rest of the study).
The model chosen at the end is the round with the best validation macro-F1, so the epsilon reported is that of the whole plan (an upper bound for the chosen round);
the selection itself uses non-private validation data and is not covered by epsilon, exactly as the early stopping of the CVAE.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ppfeddata.eval.compare import PairedBootstrap, compare_runs, recall_of
from ppfeddata.eval.runs import load_predictions, run_id, save_predictions
from ppfeddata.eval.utility import compute_metrics
from ppfeddata.fed_classifier import class_weights, file_label, macro_f1, make_mlp, predict, summarise
from ppfeddata.fl import dp_utils
from ppfeddata.partition import make_partition
from ppfeddata.utils import config_hash, git_commit, set_seed

logger = logging.getLogger("ppfeddata.fed_protected")

BATCH = 256
LR_GRID, WIDTH_GRID = (1.0e-3, 3.0e-3, 1.0e-2), (128, 256)
DP_LR_GRID, DP_CLIP_GRID = (1.0e-3, 5.0e-3, 2.0e-2), (0.5, 2.0, 8.0)
EPS_LIST = (1, 5, 10)
TUNED = "FedMLPcwT"
# label -> the run of the CVAE route it is compared with (the synthetic data alone train the same MLP: TSTR-MLP)
PAIRS = {TUNED: "B3-TSTR-mlp", f"{TUNED}-sa": "M2-TSTR-mlp", f"{TUNED}-dp1": "M1d-t21-eps1-plain-TSTR-mlp", f"{TUNED}-dp5": "M1d-t21-eps5-plain-TSTR-mlp",
         f"{TUNED}-dp10": "M1d-t21-eps10-plain-TSTR-mlp", f"{TUNED}-dp5-sa": "M3d-t21-eps5-plain-TSTR-mlp"}
PROTECTION = {TUNED: "none", f"{TUNED}-sa": "SecAgg", f"{TUNED}-dp1": "DP eps 1", f"{TUNED}-dp5": "DP eps 5", f"{TUNED}-dp10": "DP eps 10", f"{TUNED}-dp5-sa": "DP eps 5 + SecAgg"}


# --------------------------------------------------------------------------------------------------
# SecAgg+ numerics
# --------------------------------------------------------------------------------------------------
def quantised_average(states: list[dict[str, torch.Tensor]], sizes: list[int], clip: float = 16.0, max_weight: float = 100000.0, levels: int = 2 ** 22, seed: int = 0) -> dict[str, torch.Tensor]:
    """The weighted average as SecAgg+ computes it: each client sends round_stochastic((clip(w_i * theta_i) + clip) / (2 clip) * (levels - 1)), the server adds the
    integers (no wrap-around: n_clients * levels < 2^32) and returns sum / (sum_i w_i) after undoing the shift. Error per coordinate < n_clients * 2 clip / levels / sum w."""
    rng = np.random.default_rng(seed)
    w = np.array(sizes, dtype=np.float64) / float(max_weight)
    out = {}
    for k in states[0]:
        total = None
        for wi, s in zip(w, states):
            v = np.clip(wi * s[k].double().numpy(), -clip, clip)
            x = (v + clip) / (2 * clip) * (levels - 1)
            q = np.floor(x + rng.random(x.shape))
            total = q if total is None else total + q
        dec = total / (levels - 1) * (2 * clip) - len(states) * clip
        out[k] = torch.tensor(dec / w.sum(), dtype=s[k].dtype)
    return out


# --------------------------------------------------------------------------------------------------
# Local training and rounds
# --------------------------------------------------------------------------------------------------
def local_update(state: dict[str, torch.Tensor], X: torch.Tensor, y: torch.Tensor, d_in: int, k: int, hidden: tuple[int, ...], epochs: int, weight: torch.Tensor,
                 lr: float, seed: int, sigma: float | None = None, clip: float | None = None) -> tuple[dict[str, torch.Tensor], int]:
    """`epochs` of Adam from the global weights; with `sigma` DP-SGD through Opacus. The loss of a record is w[y] * cross-entropy (so that the per-sample gradient is the
    gradient of that record's own term, which Opacus clips). Returns (new state, optimiser steps counted, empty Poisson batches included)."""
    torch.manual_seed(seed)
    model = make_mlp(d_in, k, hidden)
    model.load_state_dict(state)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True, generator=g)
    steps = 0
    if sigma is None:
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                steps += 1
                opt.zero_grad()
                (F.cross_entropy(model(xb), yb, reduction="none") * weight[yb]).mean().backward()
                opt.step()
        return {a: b.detach().clone() for a, b in model.state_dict().items()}, steps
    from opacus import PrivacyEngine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs, dopt, dl = PrivacyEngine().make_private(module=model, optimizer=opt, data_loader=loader, noise_multiplier=float(sigma), max_grad_norm=float(clip), poisson_sampling=True,
                                                    noise_generator=torch.Generator().manual_seed(seed + 1))
        for _ in range(epochs):
            gs.train()
            for xb, yb in dl:
                steps += 1
                if len(xb) == 0:
                    continue
                dopt.zero_grad(set_to_none=True)
                (F.cross_entropy(gs(xb), yb, reduction="none") * weight[yb]).mean().backward()
                dopt.step()
    return {a: b.detach().clone() for a, b in gs._module.state_dict().items()}, steps


def run_protected(label: str, cfg: dict[str, Any], data: dict[str, dict[str, np.ndarray]], classes: list[str], seed: int, parts: list[np.ndarray], hp: dict[str, Any],
                  eps: float | None = None, secagg: bool = False, rounds: int | None = None, save: bool = True) -> dict[str, Any]:
    """One federated run of the tuned class-weighted MLP; hp = {lr, hidden, clip?}. Keeps the round with the best validation macro-F1."""
    k, fl = len(classes), cfg["fl"]
    rounds, local = int(rounds or fl["rounds"]), int(fl["local_epochs"])
    Xtr, ytr = torch.tensor(data["train"]["X"]), torch.tensor(data["train"]["y"], dtype=torch.long)
    Xva, yva = torch.tensor(data["val"]["X"]), data["val"]["y"].astype(np.int64)
    Xte, yte = torch.tensor(data["test"]["X"]), data["test"]["y"].astype(np.int64)
    weight = class_weights(np.bincount(data["train"]["y"], minlength=k))
    hidden, lr = tuple(hp["hidden"]), float(hp["lr"])
    sizes = [len(p) for p in parts]
    clients = [(Xtr[torch.as_tensor(p)], ytr[torch.as_tensor(p)]) for p in parts]
    sigmas = dp_utils.calibrate_clients(sizes, eps, float(cfg["dp"]["delta"]), BATCH, rounds, local) if eps is not None else None
    set_seed(seed)
    model = make_mlp(Xtr.shape[1], k, hidden)
    state = {a: b.clone() for a, b in model.state_dict().items()}
    steps = {i: 0 for i in range(len(parts))}
    best, best_f1, best_round, curve = None, -1.0, 0, []
    sa = cfg.get("secagg", {})
    for rnd in range(1, rounds + 1):
        ups = []
        for cid, (X, y) in enumerate(clients):
            s, n = local_update(state, X, y, Xtr.shape[1], k, hidden, local, weight, lr, (seed * 1_000_003 + rnd * 1009 + cid) % (2 ** 31 - 1),
                                None if sigmas is None else sigmas[cid], hp.get("clip"))
            ups.append(s)
            steps[cid] += n
        if secagg:
            state = quantised_average(ups, sizes, float(sa.get("clipping_range", 16.0)), float(sa.get("max_weight", 100000)), seed=seed * 1009 + rnd)
        else:
            tot = float(sum(sizes))
            state = {a: sum(u[a].float() * (n / tot) for u, n in zip(ups, sizes)) for a in state}
        model.load_state_dict(state)
        f1 = macro_f1(yva, predict(model, Xva)[0], k)
        curve.append(f1)
        if f1 > best_f1:
            best_f1, best_round, best = f1, rnd, {a: b.clone() for a, b in model.state_dict().items()}
    model.load_state_dict(best)
    pred, proba = predict(model, Xte)
    res = compute_metrics(yte, pred, proba, classes)
    out = {"seed": seed, "run_id": run_id(file_label(label) + "-mlp", seed, cfg["label_mode"]), "best_round": best_round, "val_macro_f1": best_f1, "test_macro_f1": res["macro_f1"],
           "balanced_acc": res["balanced_acc"], "recall": {c: d["recall"] for c, d in res["per_class"].items()}, "val_curve": curve}
    if sigmas is not None:
        t = dp_utils.epsilon_table(sizes, sigmas, steps, BATCH, float(cfg["dp"]["delta"]), eps)
        out["dp"] = {"target_eps": float(eps), "eps_max": t["eps_max"], "eps_median": t["eps_median"], "ratio_to_target": t["max_ratio_to_target"], "sigma_min": float(min(sigmas)),
                     "sigma_max": float(max(sigmas)), "steps_total": int(sum(steps.values()))}
    if save:
        save_predictions(cfg, out["run_id"], "test", pred, proba, meta={"run_id": out["run_id"], "config": label, "seed": seed, "hp": hp, "best_round": best_round, "eps": eps, "secagg": secagg,
                                                                      "config_hash": config_hash(cfg), "git_commit": git_commit()})
    logger.info("%s seed %d: test macro-F1 %.4f (val %.4f at round %d)%s", label, seed, res["macro_f1"], best_f1, best_round,
                f", eps {out['dp']['eps_max']:.3f}" if "dp" in out else "")
    return out


# --------------------------------------------------------------------------------------------------
# Tuning on validation (seed 0)
# --------------------------------------------------------------------------------------------------
def tune(cfg, data, classes, parts, dp: bool = False, base: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Small grids, validation macro-F1 only: lr x width without DP, lr x clipping bound with DP at epsilon 5 (width of the non-DP choice)."""
    trials = []
    grid = ([{"lr": lr, "hidden": base["hidden"], "clip": c} for lr in DP_LR_GRID for c in DP_CLIP_GRID] if dp
            else [{"lr": lr, "hidden": (w, w // 2)} for lr in LR_GRID for w in WIDTH_GRID])
    for hp in grid:
        r = run_protected("tune", cfg, data, classes, 0, parts, hp, eps=5.0 if dp else None, save=False)
        trials.append({"hp": {**hp, "hidden": list(hp["hidden"])}, "val_macro_f1": r["val_macro_f1"], "best_round": r["best_round"], "dp": dp})
    best = max(trials, key=lambda t: t["val_macro_f1"])
    return {**best["hp"], "hidden": tuple(best["hp"]["hidden"])}, trials


# --------------------------------------------------------------------------------------------------
# Comparison with the CVAE route
# --------------------------------------------------------------------------------------------------
def compare_pairs(cfg: dict[str, Any], y_test: np.ndarray, classes: list[str], rare: list[str], seeds: list[int], n_boot: int, labels: list[str]) -> dict[str, Any]:
    """Each protected direct classifier against the CVAE-route run with the same protection (paired bootstrap, rule of R1 / R2)."""
    k, mode = len(classes), cfg["label_mode"]
    boot = PairedBootstrap(y_test, k, n_boot, 0, "rows")
    preds = {}
    for lab in labels:
        for name, key in ((file_label(lab) + "-mlp", lab), (PAIRS[lab], PAIRS[lab])):
            try:
                for s in seeds:
                    preds[(key, s)] = load_predictions(cfg, run_id(name, s, mode))["y_pred"]
            except FileNotFoundError:
                preds = {a: b for a, b in preds.items() if a[0] != key}
    boot.add(preds)
    rare_metric = recall_of(*[classes.index(c) for c in rare])
    out = {}
    for lab in labels:
        ka, kb = [(lab, s) for s in seeds], [(PAIRS[lab], s) for s in seeds]
        if all(x in preds for x in ka + kb):
            out[lab] = {"against": PAIRS[lab], "macro_f1": compare_runs(boot, ka, kb), "rare_recall": compare_runs(boot, ka, kb, rare_metric)}
    return out


class Cache:
    """Finished tuning and runs of this command in `artifacts/fed_protected_cache.json`, so that a crash (or an interrupted session) resumes instead of starting again.
    The key includes the config hash, so a changed config never reuses old numbers."""

    def __init__(self, cfg: dict[str, Any]):
        self.path = Path(cfg["compute"]["artifacts_dir"]) / "fed_protected_cache.json"
        self.key = config_hash(cfg)
        d = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.d = d if d.get("config_hash") == self.key else {"config_hash": self.key}

    def get(self, name: str):
        return self.d.get(name)

    def put(self, name: str, value: Any) -> Any:
        from ppfeddata.interpret import to_jsonable
        self.d[name] = to_jsonable(value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.d), encoding="utf-8")
        return value


def run_all(cfg: dict[str, Any], seeds: list[int] | None = None, n_boot: int | None = None, out: str | Path | None = None, eps_list=EPS_LIST) -> dict[str, Any]:
    """Tune, run the six protected variants for every seed and add the result to results/fed_classifier.json (keys `tuning` and `protected`)."""
    from ppfeddata.eval.baselines import load_data
    from ppfeddata.interpret import to_jsonable

    seeds = [int(s) for s in (seeds if seeds is not None else cfg["seeds"])]
    data, schema = load_data(cfg)
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    q = cfg.get("quota", {})
    rare = [c for c in classes if c in q and q[c][0] <= 5000] or classes[-4:]
    p = Path(out) if out else Path(cfg["compute"]["runs_csv"]).parent / "fed_classifier.json"
    res = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    parts0, _ = make_partition(cfg, data["train"]["y"], classes, seed=0)
    cache = Cache(cfg)
    c1 = cache.get("tune_plain") or cache.put("tune_plain", dict(zip(("hp", "trials"), tune(cfg, data, classes, parts0))))
    hp, t1 = {**c1["hp"], "hidden": tuple(c1["hp"]["hidden"])}, c1["trials"]
    c2 = cache.get("tune_dp") or cache.put("tune_dp", dict(zip(("hp", "trials"), tune(cfg, data, classes, parts0, dp=True, base=hp))))
    hp_dp, t2 = {**c2["hp"], "hidden": tuple(c2["hp"]["hidden"])}, c2["trials"]
    logger.info("tuned: %s ; DP: %s", hp, hp_dp)
    variants = [(TUNED, None, False), (f"{TUNED}-sa", None, True)] + [(f"{TUNED}-dp{e}", float(e), False) for e in eps_list] + [(f"{TUNED}-dp5-sa", 5.0, True)]
    runs: dict[str, list[dict[str, Any]]] = {v[0]: [] for v in variants}
    for seed in seeds:
        parts, _ = make_partition(cfg, data["train"]["y"], classes, seed=seed)
        for lab, eps, sa in variants:
            key = f"run:{lab}:{seed}"
            runs[lab].append(cache.get(key) or cache.put(key, run_protected(lab, cfg, data, classes, seed, parts, hp_dp if eps is not None else hp, eps, sa)))
    prot = {}
    for lab, rs in runs.items():
        prot[lab] = summarise(rs, classes, rare)
        if rs and "dp" in rs[0]:
            prot[lab]["dp"] = {"eps_max_over_seeds": float(max(r["dp"]["eps_max"] for r in rs)), "target_eps": rs[0]["dp"]["target_eps"],
                               "ratio_to_target_max": float(max(r["dp"]["ratio_to_target"] for r in rs))}
    try:
        import pandas as pd
        summ = pd.read_csv(Path(cfg["compute"]["runs_csv"]).parent / "summary.csv").set_index("config")
        refs = {r: {"macro_f1_mean": float(summ.loc[r, "macro_f1_mean"]), "macro_f1_std": float(summ.loc[r, "macro_f1_std"]),
                    "rare_recall_mean": float(np.mean([summ.loc[r, f"recall_{c}_mean"] for c in rare]))} for r in PAIRS.values() if r in summ.index}
    except FileNotFoundError:
        refs = {}
    res["tuning"] = {"non_dp": {"chosen": {**hp, "hidden": list(hp["hidden"])}, "trials": t1, "seed": 0, "rounds": int(cfg["fl"]["rounds"])},
                     "dp": {"chosen": {**hp_dp, "hidden": list(hp_dp["hidden"])}, "trials": t2, "seed": 0, "at_epsilon": 5.0}, "selected_on": "validation macro-F1"}
    res["protected"] = {"classifiers": prot, "references": refs, "pairs": PAIRS, "protection": PROTECTION, "delta": float(cfg["dp"]["delta"]), "seeds": seeds,
                        "comparisons": compare_pairs(cfg, data["test"]["y"].astype(np.int64), classes, rare, seeds, int(n_boot or cfg["eval"]["bootstrap"]), list(PAIRS))}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    logger.info("wrote %s", p)
    return res
