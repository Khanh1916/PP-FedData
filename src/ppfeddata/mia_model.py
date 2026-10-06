"""A membership-inference attack with access to the released model, with positive controls (spec D6c, for the paper).

The attack of Phase 6 sees only the synthetic data and failed its positive control: it detects a verbatim copier but not an over-fitted CVAE (SPEC_DEVIATIONS 7.7), so an AUC
near 0.5 proved nothing. What DP-SGD bounds is what the released WEIGHTS reveal, so here the adversary holds the global model of the last round and scores a record by how well
the model fits it:

- score `plain`       = - negative ELBO of the record under the model (beta = 1, 8 posterior samples, class-conditioned): members are fitted better;
- score `calibrated`  = ELBO_B(x) - ELBO_A(x), with a reference model B trained with the same recipe on rows disjoint from A's (the adversary's auxiliary data, e.g. public captures):
                        removes how hard a record is in itself (the difference-of-losses attack of Carlini et al. / the offline likelihood-ratio idea without the Gaussian fit).

Design: the train pool is split at random into halves H1 and H2 (per seed). Model A is trained on H1 only and model B on H2 only (the same federated recipe on the same Dirichlet kind
of non-IID clients; for DP the same epsilon). For a record in H1 the truth is "member of A", for a record in H2 "non-member of A" (member of B); the attack is also run the other way
round (target B, reference A) and the two directions are averaged. Members and non-members come from the same capture groups, so no distribution shift is mistaken for leakage.
AUC is computed inside each class (class-balanced by construction, as in Phase 6) and averaged; the rare classes (the thin ones, where one record weighs most) are reported on their own,
with TPR at 1 % FPR for the calibrated score.

Positive controls, on centralised CVAEs with the same scores: a CVAE over-fitted on 500 rows (the one of 7.7), one trained on 5,000 rows for 100 epochs and one on a full half for 30 epochs. A
useful attack must detect the first and the number must fall as the training regime gets milder; if it does not, it cannot support a "no leakage" claim for DP either.

Limits: the adversary here knows the training recipe, holds the weights (the DP threat model of DP-SGD, stronger than seeing synthetic data only) and has a reference trained on disjoint
rows; the DP guarantee covers one record, so a record is protected even if it is a rare one, but this attack does not test group privacy (all rows of a capture). The normalisation
statistics (mean / std) come from the full train pool, so they include the non-members (a tiny shared-population leak in the attacker's disfavour). SecAgg changes only what the server sees
during training, not the released weights, so it is not attacked separately (M2 equals B3 on this attack by construction up to the 1e-5 quantisation noise).
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from ppfeddata.fl import core
from ppfeddata.models.cvae import CVAE, build_layout, one_hot, recon_terms
from ppfeddata.models.train import train_cvae
from ppfeddata.utils import config_hash, git_commit

logger = logging.getLogger("ppfeddata.mia_model")

CONFIGS = ("B3", "M1-eps1", "M1-eps5", "M1-eps10")
EPS = {"M1-eps1": 1.0, "M1-eps5": 5.0, "M1-eps10": 10.0}
SAMPLES = 8
DETECT = 0.55                               # the config's mia_auc_max: an attack "detects" a generator when its AUC reaches this


# --------------------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def neg_elbo(model: CVAE, X: np.ndarray, y: np.ndarray, samples: int = SAMPLES, beta: float = 1.0, seed: int = 0, batch: int = 8192) -> np.ndarray:
    """Per-record negative ELBO (reconstruction + beta * KL) under the class-conditioned model, averaged over `samples` draws of z (fixed seed)."""
    model.eval()
    torch.manual_seed(seed)
    k, out = model.n_classes, np.zeros(len(X))
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(np.ascontiguousarray(X[i:i + batch]), dtype=torch.float32)
        yo = one_hot(torch.as_tensor(y[i:i + batch], dtype=torch.long), k)
        mu, logvar = model.encode(xb, yo)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
        acc = torch.zeros(len(xb))
        for _ in range(samples):
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            r = recon_terms(model.decode(z, yo), xb, model.layout)
            acc += r["num"] + r["bin"] + r["cat"] + beta * kl
        out[i:i + batch] = (acc / samples).numpy()
    return out


def class_auc(score: np.ndarray, member: np.ndarray, y: np.ndarray, classes: list[str]) -> dict[str, float]:
    """AUC (member = positive) inside each class that has both kinds of record."""
    res = {}
    for i, c in enumerate(classes):
        m = y == i
        if m.any() and 0 < member[m].sum() < m.sum():
            res[c] = float(roc_auc_score(member[m], score[m]))
    return res


def tpr_at(score: np.ndarray, member: np.ndarray, fpr: float) -> float:
    f, t, _ = roc_curve(member, score)
    return float(np.interp(fpr, f, t))


def attack(L_a: np.ndarray, L_b: np.ndarray, in_a: np.ndarray, y: np.ndarray, classes: list[str], rare: list[str]) -> dict[str, Any]:
    """Both scores against target A (members = `in_a`), reference B (members = the complement). Mean over the two directions."""
    out: dict[str, Any] = {}
    for name, sa, sb in (("plain", -L_a, -L_b), ("calibrated", L_b - L_a, L_a - L_b)):
        d1 = class_auc(sa, in_a, y, classes)                 # target A
        d2 = class_auc(sb, ~in_a, y, classes)                # target B (its members are the complement)
        per = {c: 0.5 * (d1[c] + d2[c]) for c in d1 if c in d2}
        out[name] = {"auc_by_class": per, "auc_mean": float(np.mean(list(per.values()))), "auc_rare_mean": float(np.mean([per[c] for c in rare if c in per])) if any(c in per for c in rare) else None}
        if name == "calibrated":
            out[name]["tpr_at_1pct_fpr"] = 0.5 * (tpr_at(sa, in_a, 0.01) + tpr_at(sb, ~in_a, 0.01))
            out[name]["tpr_at_0_1pct_fpr"] = 0.5 * (tpr_at(sa, in_a, 0.001) + tpr_at(sb, ~in_a, 0.001))
    return out


# --------------------------------------------------------------------------------------------------
# Worlds: halves of the train pool
# --------------------------------------------------------------------------------------------------
def halves(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(10_000 + seed).permutation(n)
    return np.sort(perm[: n // 2]), np.sort(perm[n // 2:])


def world_cfg(cfg: dict[str, Any], seed: int, role: str) -> dict[str, Any]:
    import copy
    c = copy.deepcopy(cfg)
    work = Path(cfg["paths"]["work_dir"]) / "_mia" / f"s{seed}{role}"
    root = Path(cfg["compute"]["artifacts_dir"]) / "_mia" / f"s{seed}{role}"
    c["paths"] = {**cfg["paths"], "work_dir": str(work), "shared_manifest_dir": str(work / "manifests")}
    c["compute"] = {**cfg["compute"], "artifacts_dir": str(root), "runs_csv": str(root / "runs.csv")}
    c["fl"] = {**cfg["fl"], "min_client_size": min(int(cfg["fl"]["min_client_size"]), 250)}
    return c


def prepare_world(cfg: dict[str, Any], seed: int, role: str) -> dict[str, Any]:
    """processed/<mode> of a world: train.npz holds only the rows of its half; val, test and the schema are the study's (the schema's fit record names the half's size)."""
    from ppfeddata.data.preprocess import processed_dir
    src, w = processed_dir(cfg), world_cfg(cfg, seed, role)
    dst = processed_dir(w)
    if (dst / "train.npz").exists():
        return w
    dst.mkdir(parents=True, exist_ok=True)
    z = dict(np.load(src / "train.npz", allow_pickle=True))
    h = halves(len(z["y"]), seed)[0 if role == "A" else 1]
    np.savez(dst / "train.npz", **{k: v[h] for k, v in z.items()})
    for s in ("val", "test"):
        shutil.copy2(src / f"{s}.npz", dst / f"{s}.npz")
    sch = json.loads((src / "feature_schema.json").read_text(encoding="utf-8"))
    sch["fit"] = {**sch.get("fit", {}), "split": "train", "n_rows": int(len(h)), "note": "MIA world: the normalisation statistics come from the full train pool"}
    (dst / "feature_schema.json").write_text(json.dumps(sch), encoding="utf-8")
    return w


def train_world(cfg: dict[str, Any], seed: int, role: str, config: str):
    """Federated run of `config` on one half; returns the final global model."""
    from ppfeddata.data.preprocess import processed_dir
    from ppfeddata.fl.m1 import tuned_setup
    from ppfeddata.fl.run import run_fl

    w = prepare_world(cfg, seed, role)
    schema = json.loads((processed_dir(w) / "feature_schema.json").read_text(encoding="utf-8"))
    if config == "B3":
        run = run_fl(w, seed, "B3")
    else:
        fam, hp, clip = tuned_setup(w)
        run = run_fl(w, seed, f"{fam}-eps{EPS[config]:g}", target_eps=EPS[config], hp=hp, max_grad_norm=clip)
    state = torch.load(Path(run["artifacts_dir"]) / run["run_id"] / "final_state.pt", weights_only=False)
    m = core.make_model(build_layout(schema), len(schema["label_map"]), run["hp"])
    m.load_state_dict(state)
    m.eval()
    dp = run["summary"].get("dp")
    return m, {"eps_max": dp["eps_max"] if dp else None, "val_loss": run["summary"]["final_val_loss"], "n_train": int(sum(run["partition_sizes"]))}


# --------------------------------------------------------------------------------------------------
# Positive controls
# --------------------------------------------------------------------------------------------------
def control(cfg: dict[str, Any], X, y, schema, classes, rare, seed: int, n: int, epochs: int, batch: int, beta: float | None) -> dict[str, Any]:
    """Two centralised CVAEs on disjoint random subsets of n rows; the same attack."""
    from ppfeddata.tune import load_best_cvae
    h1, h2 = halves(len(y), seed)
    a, b = h1[:n], h2[:n]
    hp = {**load_best_cvae(cfg), "batch_size": batch, "beta_warmup_epochs": 0, "class_balanced_sampler": False}
    if beta is not None:
        hp["beta"] = beta
    lay, k = build_layout(schema), len(classes)
    ma, _ = train_cvae(X[a], y[a], None, None, k, lay, hp, seed, epochs=epochs, patience=None)
    mb, _ = train_cvae(X[b], y[b], None, None, k, lay, hp, seed + 1, epochs=epochs, patience=None)
    idx = np.concatenate([a, b])
    in_a = np.concatenate([np.ones(len(a), bool), np.zeros(len(b), bool)])
    La, Lb = neg_elbo(ma, X[idx], y[idx], seed=seed), neg_elbo(mb, X[idx], y[idx], seed=seed)
    r = attack(La, Lb, in_a, y[idx], classes, rare)
    r.update({"n_per_model": int(n), "epochs": int(epochs), "batch_size": int(batch), "beta": float(hp["beta"])})
    return r


CONTROLS = {"overfit_500": dict(n=500, epochs=1500, batch=32, beta=0.01), "mild_5000": dict(n=5000, epochs=100, batch=256, beta=None), "half_30ep": dict(n=44250, epochs=30, batch=256, beta=None)}


# --------------------------------------------------------------------------------------------------
def run_all(cfg: dict[str, Any], seeds: list[int] | None = None, configs: tuple[str, ...] = CONFIGS, controls: tuple[str, ...] = tuple(CONTROLS), out: str | Path | None = None) -> dict[str, Any]:
    from ppfeddata.data.preprocess import processed_dir
    from ppfeddata.interpret import to_jsonable

    seeds = [int(s) for s in (seeds if seeds is not None else cfg["seeds"])]
    d = processed_dir(cfg)
    schema = json.loads((d / "feature_schema.json").read_text(encoding="utf-8"))
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    q = cfg.get("quota", {})
    rare = [c for c in classes if c in q and q[c][0] <= 5000] or classes[-4:]
    tr = np.load(d / "train.npz", allow_pickle=True)
    X, y = tr["X"], tr["y"].astype(np.int64)
    p = Path(out) if out else Path(cfg["compute"]["runs_csv"]).parent / "mia_model.json"
    res: dict[str, Any] = {"classes": classes, "rare": rare, "seeds": seeds, "label_mode": cfg["label_mode"], "samples": SAMPLES, "detect_threshold": DETECT, "n_train_pool": int(len(y)),
                           "config_hash": config_hash(cfg), "git_commit": git_commit(), "configs": {}, "controls": {}}
    for cname in controls:
        rows = []
        for s in seeds:
            logger.info("positive control %s seed %d", cname, s)
            rows.append(control(cfg, X, y, schema, classes, rare, s, **CONTROLS[cname]))
        res["controls"][cname] = _mean_rows(rows)
    for config in configs:
        rows = []
        for s in seeds:
            ma, ia = train_world(cfg, s, "A", config)
            mb, ib = train_world(cfg, s, "B", config)
            h1, h2 = halves(len(y), s)
            in_a = np.zeros(len(y), bool)
            in_a[h1] = True
            La, Lb = neg_elbo(ma, X, y, seed=s), neg_elbo(mb, X, y, seed=s)
            r = attack(La, Lb, in_a, y, classes, rare)
            r.update({"seed": s, "eps_max": max(v for v in (ia["eps_max"], ib["eps_max"]) if v is not None) if ia["eps_max"] is not None else None,
                      "train_elbo_member": float(La[h1].mean()), "train_elbo_nonmember": float(La[h2].mean())})
            rows.append(r)
            logger.info("%s seed %d: calibrated AUC %.3f (rare %.3f), plain %.3f", config, s, r["calibrated"]["auc_mean"], r["calibrated"]["auc_rare_mean"] or float("nan"), r["plain"]["auc_mean"])
        res["configs"][config] = _mean_rows(rows)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    logger.info("wrote %s", p)
    return res


def _mean_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and std over seeds of the attack numbers; the per-seed rows are kept."""
    out: dict[str, Any] = {"per_seed": rows, "n_seeds": len(rows)}
    for name in ("plain", "calibrated"):
        v = {k: [r[name][k] for r in rows if r[name].get(k) is not None] for k in ("auc_mean", "auc_rare_mean", "tpr_at_1pct_fpr", "tpr_at_0_1pct_fpr") if k in rows[0][name]}
        out[name] = {k: {"mean": float(np.mean(x)), "std": float(np.std(x))} for k, x in v.items() if x}
        cls = rows[0][name]["auc_by_class"]
        out[name]["auc_by_class"] = {c: float(np.mean([r[name]["auc_by_class"][c] for r in rows if c in r[name]["auc_by_class"]])) for c in cls}
    if rows[0].get("eps_max") is not None:
        out["eps_max"] = float(max(r["eps_max"] for r in rows))
    for k in ("n_per_model", "epochs", "batch_size", "beta"):
        if k in rows[0]:
            out[k] = rows[0][k]
    return out
