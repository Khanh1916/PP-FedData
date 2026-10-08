"""Optimisation O4: robustness of FedDP-Marginal to the federation (non-IID degree, number of clients) and to the 11-class labels.

Why one seed is enough for the distributed variant: the generator releases only SUMS over the clients of count tables, so with the noise
split over the clients (MGs) the released tables do not depend on how the rows are partitioned; only the noise draw changes. What does
depend on the federation is shown with more care: the local-DP variant (MGl, every client adds the full noise, so the sum carries K times
the variance) with 3 seeds at K = 10 and 20, and the epsilon with a single honest client (each client adds 1/K of the noise).

Runs (`ppfeddata robustness`), all at the settings chosen for 6 classes:
- MGs eps 5 through Flower SecAgg+, seed 0: Dirichlet alpha 0.1 and 10 (5 clients), 10 and 20 clients (alpha 0.5) -> `MGs-a0.1-eps5`, ...
- MGl eps 5, 3 seeds: 10 and 20 clients -> `MGl-k10-eps5`, `MGl-k20-eps5`;
- 11 classes (label_mode 11class, rare classes = train quota <= 5,000 rows): B0 (RF, MLP), MGs eps 1/5/10 and MGr where O3.3 chose a
  post-processing, 3 seeds.
The report (`results/robustness.json`, `results/reports/robustness.md`) reads the run ledger: TSTR macro-F1 with the classifier chosen on
validation, binary F1, rare-class recall, epsilon (and with one honest client), MB per run.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("ppfeddata.robustness")

FED = {"a0.1": (0.1, 5), "a10": (10.0, 5), "k10": (0.5, 10), "k20": (0.5, 20)}


def _cfg(cfg: dict[str, Any], alpha: float | None = None, k: int | None = None, mode: str | None = None) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    if alpha is not None:
        c["fl"]["dirichlet_alpha"] = float(alpha)
    if k is not None:
        c["fl"]["num_clients"] = int(k)
    if mode is not None:
        c["label_mode"] = mode
    return c


def run(cfg: dict[str, Any], resume: bool = True) -> dict[str, Any]:
    from ppfeddata.eval.baselines import run_baselines
    from ppfeddata.tune_marginal import run_final

    for tag, (alpha, k) in FED.items():
        run_final(_cfg(cfg, alpha, k), [5.0], seeds=[0], variants=("s",), resume=resume, tag=tag)
    for tag in ("k10", "k20"):
        alpha, k = FED[tag]
        run_final(_cfg(cfg, alpha, k), [5.0], variants=("l",), resume=resume, tag=tag)
    c11 = _cfg(cfg, mode="11class")
    run_baselines(c11, ["B0-rf", "B0-mlp"], resume=resume)
    run_final(c11, None, variants=("s", "r"), resume=resume)
    return report(cfg)


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def _rows(df: pd.DataFrame, prefix: str, rare: list[str]) -> dict[str, Any] | None:
    """The TSTR numbers of a generator prefix, classifier chosen by validation macro-F1 (mean over seeds)."""
    best = None
    for clf in ("rf", "mlp"):
        g = df[df["config"] == f"{prefix}-TSTR-{clf}"].drop_duplicates("seed")
        if len(g) and (best is None or g["val_macro_f1"].mean() > best[1]["val_macro_f1"].mean()):
            best = (clf, g)
    if best is None:
        return None
    clf, g = best
    rr = g[[f"recall_{c}" for c in rare if f"recall_{c}" in g]].astype(float).mean(axis=1)
    num = lambda c: float(pd.to_numeric(g[c], errors="coerce").mean()) if c in g else float("nan")      # noqa: E731
    return {"config": prefix, "classifier": clf, "n_seeds": int(g["seed"].nunique()), "tstr_f1": float(g["macro_f1"].mean()),
            "tstr_f1_std": float(g["macro_f1"].std(ddof=0)), "bin_f1": float(g["bin_f1"].mean()), "rare_recall": float(rr.mean()),
            "eps": num("dp_eps_max"), "eps_one_honest": num("dp_eps_one_honest"), "num_clients": num("num_clients"), "alpha": num("alpha"),
            "mb_per_run": num("bytes_per_round") * 3 / 1e6}


def _baseline(df: pd.DataFrame, name: str, rare: list[str]) -> dict[str, Any] | None:
    g = df[df["config"] == name].drop_duplicates("seed")
    if not len(g):
        return None
    rr = g[[f"recall_{c}" for c in rare if f"recall_{c}" in g]].astype(float).mean(axis=1)
    return {"config": name, "n_seeds": int(g["seed"].nunique()), "tstr_f1": float(g["macro_f1"].mean()), "tstr_f1_std": float(g["macro_f1"].std(ddof=0)),
            "bin_f1": float(g["bin_f1"].mean()), "rare_recall": float(rr.mean())}


def report(cfg: dict[str, Any]) -> dict[str, Any]:
    from ppfeddata.eval.runs import runs_csv_path
    from ppfeddata.interpret import to_jsonable

    df = pd.read_csv(runs_csv_path(cfg), low_memory=False)
    q6 = cfg.get("quota", {})
    q11 = cfg.get("quota_11class", {})
    rare6 = [c for c, v in q6.items() if v[0] <= 5000]
    rare11 = [c for c, v in q11.items() if v[0] <= 5000]
    d6, d11 = df[df["label_mode"] == "6class"], df[df["label_mode"] == "11class"]
    fed = [r for r in [_rows(d6, "MGs-eps5", rare6)] + [_rows(d6, f"MGs-{t}-eps5", rare6) for t in FED] if r]
    local = [r for r in [_rows(d6, "MGl-eps5", rare6)] + [_rows(d6, f"MGl-{t}-eps5", rare6) for t in ("k10", "k20")] if r]
    c11 = [r for r in [_baseline(d11, "B0-rf", rare11), _baseline(d11, "B0-mlp", rare11)] if r]
    c11 += [r for r in (_rows(d11, f"MG{v}-eps{e:g}", rare11) for v in ("s", "r") for e in (1.0, 5.0, 10.0)) if r]
    ref6 = [r for r in [_baseline(d6, "B0-rf", rare6), _baseline(d6, "B0-mlp", rare6)] + [_rows(d6, f"MGs-eps{e:g}", rare6) for e in (1.0, 5.0, 10.0)] if r]
    out = {"federation_distributed": fed, "federation_local": local, "eleven_classes": c11, "six_classes_reference": ref6,
           "rare_6class": rare6, "rare_11class": rare11}
    res = Path(cfg["compute"]["runs_csv"]).parent
    (res / "robustness.json").write_text(json.dumps(to_jsonable(out), indent=1), encoding="utf-8")
    (res / "reports").mkdir(parents=True, exist_ok=True)
    (res / "reports" / "robustness.md").write_text(render(out), encoding="utf-8")
    return out


def _f(x: Any, d: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if np.isnan(v) else f"{v:.{d}f}"


def render(out: dict[str, Any]) -> str:
    hdr = "| configuration | classifier | seeds | TSTR macro-F1 | binary F1 | rare recall | ε | ε (one honest client) | clients | Dirichlet α | MB per run |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    row = lambda r: (f"| {r['config']} | {r.get('classifier', '-')} | {r['n_seeds']} | {_f(r['tstr_f1'])} ± {_f(r['tstr_f1_std'])} | {_f(r['bin_f1'])} | "      # noqa: E731
                     f"{_f(r['rare_recall'])} | {_f(r.get('eps'), 2)} | {_f(r.get('eps_one_honest'), 1)} | {_f(r.get('num_clients'), 0)} | "
                     f"{_f(r.get('alpha'), 1)} | {_f(r.get('mb_per_run'), 2)} |")
    L = ["# Robustness of FedDP-Marginal (optimisation round O4)", "",
         "Generated by `ppfeddata robustness` from `results/runs.csv`; do not edit. TSTR on the real test split, classifier chosen on validation, "
         "mean ± std over seeds. The distributed variant releases only sums over the clients, so its tables do not depend on the partition: one "
         "seed per federation setting checks it; the local variant and the 11-class mode have 3 seeds.", "",
         "## Federation (ε 5): Dirichlet α and number of clients", "", "Distributed noise through Flower SecAgg+ (MGs):", "", hdr, sep]
    L += [row(r) for r in out["federation_distributed"]]
    L += ["", "Local DP (MGl: every client adds the full noise):", "", hdr, sep] + [row(r) for r in out["federation_local"]]
    L += ["", "## 11 classes (DoS and DDoS separated)", "", hdr, sep] + [row(r) for r in out["eleven_classes"]]
    L += ["", "Reference, 6 classes:", "", hdr, sep] + [row(r) for r in out["six_classes_reference"]]
    return "\n".join(L) + "\n"
