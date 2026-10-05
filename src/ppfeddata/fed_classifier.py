"""Phase 13 follow-up: the IDS classifier trained directly by federated learning, the alternative that section 9.9 of the report listed as not tested.

When the raw data cannot be pooled, the study's route is a federated CVAE that produces synthetic data (B3, M1-M3). The obvious competitor trains the
detector itself by FedAvg on the same non-IID clients and needs no generator and no synthetic data. This module runs it on the SAME partitions as B3
(Dirichlet alpha, same seeds) and evaluates on the SAME real test split, so the numbers sit next to TSTR-MLP of the CVAE pipelines.

Four classifiers (all the same MLP, 128-64, trained with Adam, no hyper-parameter search; the round / epoch is chosen on the validation split only):
- `FedMLP`    FedAvg, plain cross-entropy;
- `FedMLPcw`  FedAvg, cross-entropy weighted by class (global weights from the clients' summed class counts, which the clients must therefore share: a leak the
              plain variant does not have);
- `CentMLP`, `CentMLPcw`  the same network and the same number of passes over the pooled data: the control that separates "federated" from "this MLP".

Written in plain PyTorch, in process: it measures utility, not Flower's overhead, and it is not protected by DP or SecAgg (the comparison is with the
unprotected CVAE pipelines B3 and M2, on utility only). Federated SMOTE and a DP / SecAgg version of this classifier are not done.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ppfeddata.eval.compare import MACRO_F1, PairedBootstrap, compare_runs, recall_of
from ppfeddata.eval.runs import load_predictions, run_id, save_predictions
from ppfeddata.eval.utility import compute_metrics
from ppfeddata.partition import make_partition
from ppfeddata.utils import config_hash, git_commit, set_seed

logger = logging.getLogger("ppfeddata.fed_classifier")

NAMES = ("FedMLP", "FedMLPcw", "CentMLP", "CentMLPcw")
REFERENCES = {"B0-mlp": "real data only (pooled)", "B1b-mlp": "SMOTE (pooled)", "B3-TSTR-mlp": "CVAE + FL, synthetic data only", "M2-TSTR-mlp": "CVAE + FL + SecAgg, synthetic data only"}
OUT = Path("results") / "fed_classifier.json"
LONGER = 100                             # the second budget (rounds), against the 30 of the federated CVAE
BATCH, LR = 256, 1.0e-3                  # the CVAE's own batch size and learning rate; nothing was tuned here


def make_mlp(d_in: int, k: int, hidden=(128, 64)) -> nn.Module:
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, k))


def class_weights(counts: np.ndarray) -> torch.Tensor:
    """sklearn's "balanced": n / (k * count); a class with no rows gets weight 0."""
    counts = np.asarray(counts, dtype=np.float64)
    w = np.divide(counts.sum(), len(counts) * counts, out=np.zeros_like(counts), where=counts > 0)
    return torch.tensor(w, dtype=torch.float32)


def train_epochs(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int, weight: torch.Tensor | None, seed: int) -> None:
    """`epochs` passes over (X, y) with a fresh Adam (what a federated client does each round)."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), BATCH):
            ix = order[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(X[ix]), y[ix]).backward()
            opt.step()


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    p = torch.softmax(model(X), dim=1)
    return p.argmax(1).numpy(), p.numpy()


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    return float(MACRO_F1(np.bincount(y * k + pred, minlength=k * k).reshape(k, k)))


def fedavg_round(global_state: dict[str, torch.Tensor], clients: list[tuple[torch.Tensor, torch.Tensor]], d_in: int, k: int, local_epochs: int,
                 weight: torch.Tensor | None, seed: int, rnd: int) -> dict[str, torch.Tensor]:
    """One FedAvg round: every client trains `local_epochs` from the global weights; the results are averaged with weights n_i, in client order."""
    total = float(sum(len(X) for X, _ in clients))
    acc: dict[str, torch.Tensor] = {k_: torch.zeros_like(v, dtype=torch.float32) for k_, v in global_state.items()}
    for cid, (X, y) in enumerate(clients):
        m = make_mlp(d_in, k)
        m.load_state_dict(global_state)
        train_epochs(m, X, y, local_epochs, weight, seed * 1_000_003 + rnd * 1009 + cid)
        for k_, v in m.state_dict().items():
            acc[k_] += v.float() * (len(X) / total)
    return acc


def file_label(label: str) -> str:
    """`FedMLP@100` (a longer budget) is stored as `FedMLP-r100`."""
    return label.replace("@", "-r")


def run_one(label: str, cfg: dict[str, Any], data: dict[str, dict[str, np.ndarray]], classes: list[str], seed: int, parts: list[np.ndarray] | None) -> dict[str, Any]:
    """Train `label` (a name of NAMES, optionally `name@rounds`) for one seed; the round with the best validation macro-F1 is kept.
    Returns the test result and saves the predictions."""
    name, _, extra = label.partition("@")
    k, fl = len(classes), cfg["fl"]
    rounds, local = int(extra or fl["rounds"]), int(fl["local_epochs"])
    Xtr, ytr = torch.tensor(data["train"]["X"]), torch.tensor(data["train"]["y"], dtype=torch.long)
    Xva, yva = torch.tensor(data["val"]["X"]), data["val"]["y"].astype(np.int64)
    Xte, yte = torch.tensor(data["test"]["X"]), data["test"]["y"].astype(np.int64)
    weighted = name.endswith("cw")
    weight = class_weights(np.bincount(data["train"]["y"], minlength=k)) if weighted else None
    set_seed(seed)
    model = make_mlp(Xtr.shape[1], k)
    state = {a: b.clone() for a, b in model.state_dict().items()}
    best, best_f1, best_step, history = None, -1.0, 0, []
    federated = name.startswith("Fed")
    clients = [(Xtr[torch.as_tensor(p)], ytr[torch.as_tensor(p)]) for p in parts] if federated else None
    for rnd in range(1, rounds + 1):
        if federated:
            state = fedavg_round(state, clients, Xtr.shape[1], k, local, weight, seed, rnd)
            model.load_state_dict(state)
        else:                                                         # the control: the same recipe (fresh Adam every `local` epochs) on the pooled data, so only the data layout differs
            train_epochs(model, Xtr, ytr, local, weight, seed * 1_000_003 + rnd)
        f1 = macro_f1(yva, predict(model, Xva)[0], k)
        history.append(f1)
        if f1 > best_f1:
            best_f1, best_step, best = f1, rnd, {a: b.clone() for a, b in model.state_dict().items()}
    model.load_state_dict(best)
    pred, proba = predict(model, Xte)
    res = compute_metrics(yte, pred, proba, classes)
    rid = run_id(file_label(label) + "-mlp", seed, cfg["label_mode"])
    save_predictions(cfg, rid, "test", pred, proba, meta={"run_id": rid, "config": label, "rounds": rounds, "seed": seed, "best_round": best_step, "val_macro_f1": best_f1,
                                                         "config_hash": config_hash(cfg), "git_commit": git_commit()})
    logger.info("%s seed %d: test macro-F1 %.4f (val %.4f at round %d)", label, seed, res["macro_f1"], best_f1, best_step)
    return {"seed": seed, "run_id": rid, "best_round": best_step, "val_macro_f1": best_f1, "test_macro_f1": res["macro_f1"], "balanced_acc": res["balanced_acc"],
            "recall": {c: d["recall"] for c, d in res["per_class"].items()}, "val_curve": history}


def summarise(rows: list[dict[str, Any]], classes: list[str], rare: list[str]) -> dict[str, Any]:
    f1 = np.array([r["test_macro_f1"] for r in rows])
    rec = {c: float(np.mean([r["recall"][c] for r in rows])) for c in classes}
    return {"macro_f1_mean": float(f1.mean()), "macro_f1_std": float(f1.std()), "per_seed": [float(x) for x in f1], "recall": rec,
            "rare_recall_mean": float(np.mean([rec[c] for c in rare])), "best_round": [r["best_round"] for r in rows], "seeds": [r["seed"] for r in rows]}


def compare_with_references(cfg: dict[str, Any], y_test: np.ndarray, classes: list[str], rare: list[str], seeds: list[int], n_boot: int, names=NAMES) -> dict[str, Any]:
    """Paired bootstrap (rows) of each classifier against each reference run of the main experiment, with the rule of R1 / R2."""
    k, mode = len(classes), cfg["label_mode"]
    boot = PairedBootstrap(y_test, k, n_boot, 0, "rows")
    preds: dict[tuple[str, int], np.ndarray] = {}
    refs_ok = []
    for n in list(names) + list(REFERENCES):
        base = file_label(n) + "-mlp" if n in names else n
        got = []
        for s in seeds:
            try:
                preds[(n, s)] = load_predictions(cfg, run_id(base, s, mode))["y_pred"]
                got.append(s)
            except FileNotFoundError:
                break
        if len(got) == len(seeds) and n in REFERENCES:
            refs_ok.append(n)
    boot.add(preds)
    rare_metric = recall_of(*[classes.index(c) for c in rare])
    out: dict[str, Any] = {}
    for a in names:
        if (a, seeds[0]) not in preds:
            continue
        out[a] = {}
        for b in refs_ok:
            ka, kb = [(a, s) for s in seeds], [(b, s) for s in seeds]
            out[a][b] = {"macro_f1": compare_runs(boot, ka, kb), "rare_recall": compare_runs(boot, ka, kb, rare_metric)}
    return out


def run_all(cfg: dict[str, Any], seeds: list[int] | None = None, names: tuple[str, ...] = NAMES, n_boot: int | None = None, out: str | Path | None = None,
            longer: int | None = LONGER) -> dict[str, Any]:
    """Every classifier at the federated budget of the config (`rounds` x `local_epochs`) and, if `longer`, again with `longer` rounds: a classifier may
    need more passes than a generator, and the best round is picked on validation either way."""
    from ppfeddata.eval.baselines import load_data

    names = tuple(names) + tuple(f"{n}@{longer}" for n in names if longer and longer != int(cfg["fl"]["rounds"]))
    seeds = [int(s) for s in (seeds if seeds is not None else cfg["seeds"])]
    data, schema = load_data(cfg)
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    q = cfg.get("quota", {})
    rare = [c for c in classes if c in q and q[c][0] <= 5000] or classes[-4:]
    fl = cfg["fl"]
    result: dict[str, Any] = {"classes": classes, "rare": rare, "seeds": seeds, "label_mode": cfg["label_mode"], "rounds": int(fl["rounds"]), "longer_rounds": longer, "local_epochs": int(fl["local_epochs"]),
                              "alpha": float(fl["dirichlet_alpha"]), "num_clients": int(fl["num_clients"]), "batch_size": BATCH, "lr": LR, "config_hash": config_hash(cfg), "git_commit": git_commit(),
                              "classifiers": {}}
    runs: dict[str, list[dict[str, Any]]] = {n: [] for n in names}
    for seed in seeds:
        parts, _ = make_partition(cfg, data["train"]["y"], classes, seed=seed)       # the partition B3 used for this seed
        for n in names:
            runs[n].append(run_one(n, cfg, data, classes, seed, parts))
    for n in names:
        result["classifiers"][n] = summarise(runs[n], classes, rare)
    # the references of the main experiment, from the summary
    try:
        import pandas as pd
        summ = pd.read_csv(Path(cfg["compute"]["runs_csv"]).parent / "summary.csv").set_index("config")
        result["references"] = {r: {"label": lab, "macro_f1_mean": float(summ.loc[r, "macro_f1_mean"]), "macro_f1_std": float(summ.loc[r, "macro_f1_std"]),
                                    "rare_recall_mean": float(np.mean([summ.loc[r, f"recall_{c}_mean"] for c in rare]))} for r, lab in REFERENCES.items() if r in summ.index}
    except FileNotFoundError:
        result["references"] = {}
    result["comparisons"] = compare_with_references(cfg, data["test"]["y"].astype(np.int64), classes, rare, seeds, int(n_boot or cfg["eval"]["bootstrap"]), names)
    p = Path(out) if out else Path(cfg["compute"]["runs_csv"]).parent / "fed_classifier.json"
    from ppfeddata.interpret import to_jsonable
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(result), indent=1), encoding="utf-8")
    logger.info("wrote %s", p)
    return result
