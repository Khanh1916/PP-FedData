"""Phase 7: B2 = centralised CVAE (no FL, no DP). Train on the train split, generate, evaluate on the REAL test split.

Per seed: one CVAE, `generate.target_per_class` synthetic rows per class; then
  TSTR (train on `tune.syn_per_class` synthetic rows per class) and TAug (real + synthetic up to the target per class),
  each with RF and MLP; fidelity and privacy of the synthetic set; training cost.
Also the MIA positive control (deliberately over-fitted CVAE), the open item of SPEC_DEVIATIONS 6.3.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ppfeddata.eval.baselines import _flat_metrics, _table, load_data
from ppfeddata.eval.fidelity import fidelity_report
from ppfeddata.eval.overhead import Timer, peak_rss_gb
from ppfeddata.eval.privacy import positive_control, privacy_report
from ppfeddata.eval.runs import RunLedger, artifacts_dir, load_predictions, run_id, runs_csv_path, save_predictions
from ppfeddata.eval.stats import paired_bootstrap_diff
from ppfeddata.eval.utility import build_train_set, compute_metrics, full_proba, make_classifier
from ppfeddata.models.cvae import build_layout
from ppfeddata.models.generate import gen_stats_from_cfg, generate
from ppfeddata.models.train import train_cvae
from ppfeddata.tune import load_best_cvae
from ppfeddata.utils import config_hash, git_commit

logger = logging.getLogger("ppfeddata.models.b2")

PROTOCOLS = [("TSTR", "rf"), ("TSTR", "mlp"), ("TAug", "rf"), ("TAug", "mlp")]


def b2_name(protocol: str, clf: str) -> str:
    return f"B2-{protocol}-{clf}"


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _classes(schema) -> list[str]:
    return [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]


def head(X: np.ndarray, y: np.ndarray, k: int, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """First `k` rows of every class (synthetic rows are i.i.d., so the first k are a fair subset)."""
    idx = np.concatenate([np.flatnonzero(y == c)[:k] for c in range(n_classes)])
    return X[idx], y[idx]


def _summ(fid: dict[str, Any], pri: dict[str, Any]) -> dict[str, float]:
    m = fid["mean"]
    return {"wasserstein_mean": m["wasserstein"], "js_mean": m["js"], "corr_dist_mean": m["corr_dist"],
            "c2st_auc_mean": m.get("c2st_auc", float("nan")), "dup_rate": pri["duplicate_rate"],
            "dcr_ratio_mean": pri["dcr_ratio_mean"], "mia_auc_mean": pri["mia_auc_mean"]}


def method_name(prefix: str, protocol: str, clf: str) -> str:
    return f"{prefix}-{protocol}-{clf}"


def evaluate_generator(cfg: dict[str, Any], model, prefix: str, seed: int, data, schema, ledger: RunLedger,
                       extra: dict[str, Any] | None = None, use_stats: bool = True, stats=None, sample_fn=None) -> dict[str, Any]:
    """Generate from `model`, then run the Phase 6 evaluation: fidelity + privacy of the synthetic set and TSTR / TAug with
    RF and MLP on the real test split. One ledger row per (protocol, classifier). Shared by B2 (centralised), B3 (FL) and M1.
    `use_stats=False` samples the plain decoder (no residual noise): every number then depends only on the trained weights.
    `stats` (optimisation O1): use these generation statistics (the DP residual std of `fl/dp_stats.py`) instead of computing them from the
    pooled train split. `sample_fn(class_counts, seed) -> (X, y)` (optimisation O3): another generator instead of `model` (then None)."""
    mode, k = cfg["label_mode"], len(schema["label_map"])
    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    classes = _classes(schema)
    target, spc = int(cfg["generate"]["target_per_class"]), int(cfg["tune"]["syn_per_class"])
    with Timer() as t_gen:
        if sample_fn is not None:
            Xs, ys = sample_fn([target] * k, seed)
        else:
            if stats is None:
                stats = gen_stats_from_cfg(model, Xtr, ytr, schema, cfg["generate"]) if use_stats else None
            Xs, ys = generate(model, schema, [target] * k, seed, stats=stats)
    base = artifacts_dir(cfg) / run_id(prefix, seed, mode)
    base.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(base / "synthetic.npz", X=Xs, y=ys.astype(np.int16))

    Xf, yf = head(Xs, ys, spc, k)
    with Timer() as t_fp:
        fid = fidelity_report(Xtr, ytr, Xf, yf, schema, classes, seed=seed)
        pri = privacy_report(Xf, yf, Xtr, ytr, data["val"]["X"], data["val"]["y"], classes, seed=seed)
    (base / "fidelity_privacy.json").write_text(json.dumps({"fidelity": fid, "privacy": pri}, indent=2), encoding="utf-8")
    common = {**_summ(fid, pri), "residual_noise": bool(stats is not None and stats.residual_std is not None), "cvae_gen_s": t_gen.seconds, "fidpriv_s": t_fp.seconds, "cvae_params": n_params(model) if model is not None else 0,
              **(extra or {})}
    out = {}
    for protocol, clf_name in PROTOCOLS:
        name = method_name(prefix, protocol, clf_name)
        Xa, ya, info = build_train_set(protocol, Xtr, ytr, Xs, ys, k, spc, target, seed)
        clf = make_classifier(clf_name, cfg, seed)
        with warnings.catch_warnings(), Timer() as t_fit:
            warnings.simplefilter("ignore")
            clf.fit(Xa, ya)
        rid = run_id(name, seed, mode)
        res = {}
        for split in ("test", "val"):
            X, y = data[split]["X"], data[split]["y"]
            pred, proba = clf.predict(X), full_proba(clf, X, k)
            res[split] = compute_metrics(y, pred, proba, classes)
            save_predictions(cfg, rid, split, pred, proba if split == "test" else None,
                             meta={"run_id": rid, "config": name, "seed": seed, "n_train": int(len(ya))} if split == "test" else None)
        row = {"run_id": rid, "config": name, "classifier": clf_name, "protocol": protocol, "seed": seed,
               "label_mode": mode, "config_hash": config_hash(cfg), "git_commit": git_commit(), "n_train": int(len(ya)),
               "n_synthetic": int(sum(info["n_synthetic"].values())), "fit_time_s": t_fit.seconds,
               "n_iter": int(getattr(clf, "n_iter_", 0)) if clf_name == "mlp" else 0,
               **common, **_flat_metrics(res["test"]), **_flat_metrics(res["val"], "val_")}
        ledger.append(row)
        out[name] = row
        logger.info("%s: test macro-F1 %.4f (val %.4f)", rid, row["macro_f1"], row["val_macro_f1"])
    return out


def run_b2_seed(cfg: dict[str, Any], seed: int, data, schema, ledger: RunLedger, resume: bool = True) -> None:
    mode, k = cfg["label_mode"], len(schema["label_map"])
    names = [b2_name(p, c) for p, c in PROTOCOLS]
    if resume and all(ledger.done(run_id(n, seed, mode)) for n in names):
        logger.info("skip B2 seed %d (all runs in the ledger)", seed)
        return
    hp = load_best_cvae(cfg)
    with Timer() as t_train:
        model, hist = train_cvae(data["train"]["X"], data["train"]["y"], data["val"]["X"], data["val"]["y"], k,
                                 build_layout(schema), hp, seed, patience=hp.get("patience"))
    base = artifacts_dir(cfg) / run_id("B2", seed, mode)
    base.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config(), "hp": hp, "history": hist}, base / "model.pt")
    extra = {"cvae_train_s": t_train.seconds, "cvae_epochs": len(hist), "cvae_peak_rss_gb": peak_rss_gb(),
             "cvae_val_loss": hist[-1].get("val_loss", float("nan"))}
    evaluate_generator(cfg, model, "B2", seed, data, schema, ledger, extra)


def copier_sensitivity(cfg: dict[str, Any], data, schema, noises=(0.0, 0.05, 0.2, 0.5), seed: int = 0,
                       n_members: int = 500) -> dict[str, float]:
    """MIA AUC against a generator that copies its members and adds Gaussian noise to the numeric columns (noise in
    units of the standardised feature). Shows how close to a copy a synthetic row must be for the attack to notice."""
    k, spc = len(schema["label_map"]), int(cfg["tune"]["syn_per_class"])
    n_num = build_layout(schema).n_num
    out = {}
    for noise in noises:
        def gen(Xm, ym, s, noise=noise):
            rng = np.random.default_rng(s)
            idx = rng.integers(0, len(Xm), spc * k)
            X = Xm[idx].copy()
            X[:, :n_num] += rng.normal(0, noise, (len(idx), n_num))
            return X, ym[idx]
        out[str(noise)] = positive_control(gen, data["train"]["X"], data["train"]["y"], data["val"]["X"], data["val"]["y"], k,
                                           n_members=n_members, seed=seed, stratified=True)["mia_auc_mean"]
    return out


def cvae_positive_control(cfg: dict[str, Any], data, schema, seed: int = 0, n_members: int = 500, epochs: int = 1500,
                          batch_size: int = 32, beta: float | None = 0.01) -> dict[str, Any]:
    """MIA against a CVAE deliberately over-fitted on `n_members` rows (no validation, no early stopping, no DP)."""
    k, hp = len(schema["label_map"]), load_best_cvae(cfg)
    hp = {**hp, "batch_size": batch_size, "beta": float(hp["beta"] if beta is None else beta),
          "beta_warmup_epochs": 0, "class_balanced_sampler": False}
    lay = build_layout(schema)
    spc = int(cfg["tune"]["syn_per_class"])

    def gen(Xm, ym, s):
        model, _ = train_cvae(Xm, ym, None, None, k, lay, hp, s, epochs=epochs, patience=None)
        return generate(model, schema, [spc] * k, s, stats=gen_stats_from_cfg(model, Xm, ym, schema, cfg["generate"]))

    res = positive_control(gen, data["train"]["X"], data["train"]["y"], data["val"]["X"], data["val"]["y"], k,
                           n_members=n_members, seed=seed, stratified=True)
    res.update({"epochs": epochs, "beta": hp["beta"], "batch_size": batch_size, "classes": _classes(schema)})
    return res


def run_b2(cfg: dict[str, Any], seeds: list[int] | None = None, resume: bool = True) -> None:
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    data, schema = load_data(cfg)
    ledger = RunLedger(runs_csv_path(cfg))
    for s in seeds:
        run_b2_seed(cfg, s, data, schema, ledger, resume)
    pc = artifacts_dir(cfg) / f"B2_positive_control_{cfg['label_mode']}.json"
    if not (resume and pc.exists()):
        res = cvae_positive_control(cfg, data, schema)
        res["copier_sensitivity"] = copier_sensitivity(cfg, data, schema)
        pc.write_text(json.dumps(res, indent=2), encoding="utf-8")
        logger.info("MIA positive control: AUC %.3f (members %d)", res["mia_auc_mean"], res["n_members"])


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def _ms(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.4f} ± {x.std():.4f}"


def write_b2_report(cfg: dict[str, Any], out: str | Path = "./results/reports/b2_cvae.md") -> dict[str, Any]:
    mode, th = cfg["label_mode"], cfg["thresholds"]
    df = RunLedger(runs_csv_path(cfg)).frame()
    df = df[df["label_mode"] == mode]
    data, schema = load_data(cfg)
    classes, k, nb = _classes(schema), len(schema["label_map"]), int(cfg["eval"]["bootstrap"])
    y = data["test"]["y"]
    hp = load_best_cvae(cfg)
    ref = [n for n in ("B0-rf", "B0-mlp", "B1a-rf", "B1b-rf", "B1b-mlp") if (df["config"] == n).any()]
    b2 = [b2_name(p, c) for p, c in PROTOCOLS if (df["config"] == b2_name(p, c)).any()]
    seeds = sorted(int(s) for s in df[df["config"].isin(b2)]["seed"].unique())
    L = [f"# Phase 7 - B2 centralised CVAE ({mode})", "",
         f"Auto-generated by `ppfeddata b2`. Real test split {len(y)} rows; seeds {seeds}. CVAE: latent {hp['latent_dim']}, "
         f"hidden {hp['hidden']}, beta {hp['beta']:.3f}, lr {hp['lr']:.2e}, up to {hp['epochs']} epochs "
         f"(early stopping patience {hp.get('patience')}). TSTR uses {cfg['tune']['syn_per_class']} synthetic rows per class; "
         f"TAug adds synthetic rows up to {cfg['generate']['target_per_class']} per class. Hyper-parameters were tuned on "
         "centralised, non-private data (`configs/best_cvae.yaml`).", ""]

    rows = []
    for n in ref + b2:
        s = df[df["config"] == n].sort_values("seed")
        rows.append({"config": n, "runs": len(s), "macro-F1 (test)": _ms(s["macro_f1"]), "balanced acc": _ms(s["balanced_acc"]),
                     "PR-AUC macro": _ms(s["pr_auc_macro"]), "macro-F1 (val)": _ms(s["val_macro_f1"])})
    L += ["## Utility (mean ± std over seeds)", "", _table(rows), ""]
    rows = [{"config": n, **{c: f"{df[df['config'] == n][f'recall_{c}'].mean():.3f}" for c in classes}} for n in ref + b2]
    L += ["## Recall per class (test)", "", _table(rows), ""]

    L += ["## Paired difference (same classifier, test macro-F1, stratified bootstrap)", "",
          "Criterion R1/R2 (spec): paired CI excludes 0 and |diff| > std of macro-F1 over seeds.", ""]
    rows = []
    for n in b2:
        clf = n.rsplit("-", 1)[1]
        for base in (f"B0-{clf}", "B1a-rf" if clf == "rf" else None, f"B1b-{clf}"):
            if base is None or base not in ref:
                continue
            for sd in seeds:
                ra, rb = run_id(n, sd, mode), run_id(base, sd, mode)
                if not ((df["run_id"] == ra).any() and (df["run_id"] == rb).any()):
                    continue
                d = paired_bootstrap_diff(y, load_predictions(cfg, ra)["y_pred"], load_predictions(cfg, rb)["y_pred"], k, nb, seed=sd)
                sd_std = float(df[df["config"] == n]["macro_f1"].std(ddof=0))
                rows.append({"comparison": f"{n} - {base}", "seed": sd, "diff": f"{d['diff']:+.4f}",
                             "95% CI": f"[{d['lo']:+.4f}, {d['hi']:+.4f}]", "excludes 0": d["excludes_zero"],
                             "|diff| > seed std": abs(d["diff"]) > sd_std})
    L += [_table(rows), ""]

    gate = {"c2st_ok": {}, "dup_ok": {}, "dcr_ok": {}}
    rows = []
    one = df[df["config"] == b2[0]].sort_values("seed") if b2 else df.iloc[0:0]
    for _, r in one.iterrows():
        sd = int(r["seed"])
        gate["c2st_ok"][sd] = bool(r["c2st_auc_mean"] < th["c2st_auc_max"])
        gate["dup_ok"][sd] = bool(r["dup_rate"] < th["dup_rate_max"])
        gate["dcr_ok"][sd] = bool(r["dcr_ratio_mean"] > th["dcr_ratio_min"])
        rows.append({"seed": sd, "C2ST AUC": f"{r['c2st_auc_mean']:.3f}", "Wasserstein": f"{r['wasserstein_mean']:.3f}",
                     "JS": f"{r['js_mean']:.3f}", "corr dist": f"{r['corr_dist_mean']:.3f}", "dup rate": f"{r['dup_rate']:.4f}",
                     "DCR ratio": f"{r['dcr_ratio_mean']:.3f}", "MIA AUC": f"{r['mia_auc_mean']:.3f}",
                     "train s": f"{r['cvae_train_s']:.0f}", "epochs": int(r["cvae_epochs"]), "params": int(r["cvae_params"])})
    L += ["## Fidelity, privacy and cost of the synthetic set (mean over classes)", "", _table(rows), ""]
    p = artifacts_dir(cfg) / run_id("B2", seeds[0], mode) / "fidelity_privacy.json" if seeds else None
    if p and p.exists():
        j = json.loads(p.read_text(encoding="utf-8"))
        rows = []
        for c in classes:
            f, pr = j["fidelity"].get(c), j["privacy"]["per_class"].get(c)
            if f and pr:
                rows.append({"class": c, "C2ST": f"{f['c2st_auc']:.3f}", "W": f"{f['wasserstein']:.3f}", "JS": f"{f['js']:.3f}",
                             "corr": f"{f['corr_dist']:.3f}", "dup": f"{pr['duplicate_rate']:.4f}",
                             "DCR ratio": f"{pr['ratio']:.3f}", "MIA AUC": f"{j['privacy']['mia_auc_per_class'].get(c, float('nan')):.3f}"})
        L += [f"Per class (seed {seeds[0]}):", "", _table(rows), ""]
    pcp = artifacts_dir(cfg) / f"B2_positive_control_{mode}.json"
    pc_ok = None
    if pcp.exists():
        pc = json.loads(pcp.read_text(encoding="utf-8"))
        pc_ok = bool(pc["mia_auc_mean"] > 0.6)
        cs = pc.get("copier_sensitivity", {})
        L += ["## MIA positive control", "",
              f"CVAE over-fitted on {pc['n_members']} members ({pc['epochs']} epochs, batch {pc['batch_size']}, beta {pc['beta']}, no DP): "
              f"MIA AUC {pc['mia_auc_mean']:.3f} (per class: " + ", ".join(f"{classes[int(c)]} {v:.2f}" for c, v in pc["per_class"].items()) + "). "
              + ("The attack detects memorisation, so an AUC near 0.5 on the real B2 is meaningful." if pc_ok else
                 "**The attack did not detect the over-fitted CVAE (AUC not clearly above 0.5), so it is too weak to show that "
                 "the generator does not memorise.** An AUC near 0.5 on B2 only says that no synthetic row is a near-copy of a training row."), ""]
        if cs:
            L += ["Sensitivity of the same attack to a pure copier (members resampled, Gaussian noise on the standardised numeric columns): "
                  + ", ".join(f"noise {n} -> AUC {v:.3f}" for n, v in cs.items())
                  + ". The attack detects exact copies and near-copies, and loses sensitivity once rows are perturbed by about 0.2 standard deviations.", ""]
    L += ["## Gate Phase 7 checks (thresholds from config)", "",
          f"- C2ST AUC < {th['c2st_auc_max']}: " + ", ".join(f"seed {s} {'OK' if v else 'FAIL'}" for s, v in gate["c2st_ok"].items()),
          f"- duplicate rate < {th['dup_rate_max']}: " + ", ".join(f"seed {s} {'OK' if v else 'FAIL'}" for s, v in gate["dup_ok"].items()),
          f"- DCR ratio > {th['dcr_ratio_min']}: " + ", ".join(f"seed {s} {'OK' if v else 'FAIL'}" for s, v in gate["dcr_ok"].items()),
          f"- all seeds present ({cfg['seeds']}): " + ("OK" if seeds == sorted(cfg["seeds"]) else "FAIL"), ""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(L), encoding="utf-8")
    gate["positive_control_ok"] = pc_ok
    return gate
