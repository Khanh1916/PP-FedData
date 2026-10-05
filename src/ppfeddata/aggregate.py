"""Phase 11: turn the run ledger into the final tables and figures.

`aggregate(cfg)` writes
  results/summary.csv                  one row per ledger configuration: mean and std (ddof = 0, as in the phase reports) over seeds
  results/figures/*.png                (1) macro-F1 by configuration, (2) recall of the rare classes, (3) utility-privacy curve, (4) fidelity vs epsilon,
                                       (5) overhead, (6) Pareto: macro-F1 against MIA AUC, point size = overhead
  results/reports/final_report.md      every number comes from the ledger, the round logs or the config; nothing is typed by hand

Conventions (the same as the phase reports):
- DP configurations are shown with the PLAIN decoder (`-plain`), the variant that depends only on the DP-trained weights and is therefore
  covered by epsilon; the residual-noise variant is listed next to it. B3 shown at epsilon = infinity is `B3-plain` for the same reason.
- The headline DP family is the one with the DP-specific hyper-parameters (`M1d-t<trial>`, `M3d-t<trial>`); the Phase 7 family (`M1`) is a reference table.
- `std` is over seeds; a configuration with one seed shows no std.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppfeddata.eval.runs import RunLedger, artifacts_dir, run_id, runs_csv_path

logger = logging.getLogger("ppfeddata.aggregate")

BASELINES = {"B0-rf": ("B0", "rf"), "B0-mlp": ("B0", "mlp"), "B1a-rf": ("B1a", "rf"), "B1b-rf": ("B1b", "rf"), "B1b-mlp": ("B1b", "mlp")}
_GEN = re.compile(r"^(?P<prefix>.+)-(?P<proto>TSTR|TAug)-(?P<clf>rf|mlp)$")
_EPS = re.compile(r"-eps(?P<eps>[0-9.]+)$")
_ALPHA = re.compile(r"^A1-a(?P<alpha>[0-9.]+)$")
COLORS = {"baseline": "#8c8c8c", "gen": "#1f77b4", "dp": "#ff7f0e", "secagg": "#2ca02c", "dpsa": "#9467bd", "ref": "#bcbd22"}
SUMMARY_METRICS = ["macro_f1", "balanced_acc", "pr_auc_macro", "bin_f1", "val_macro_f1", "wasserstein_mean", "js_mean", "corr_dist_mean", "c2st_auc_mean",
                   "dup_rate", "dcr_ratio_mean", "mia_auc_mean", "dp_eps_max", "dp_eps_median", "dp_sigma_min", "dp_sigma_max", "dp_ratio_to_target", "bytes_per_round",
                   "fl_total_s", "cvae_train_s", "fit_time_s", "peak_rss_gb", "fl_final_val_loss", "cvae_val_loss", "cvae_params", "sa_bytes_per_round", "sa_max_abs_w",
                   "round_s_median"]


# --------------------------------------------------------------------------------------------------
# Names -> meaning
# --------------------------------------------------------------------------------------------------
def parse_config(name: str) -> dict[str, Any]:
    """`M1d-t21-eps5-plain-TAug-rf` -> method M1, family dp-tuned, variant plain, protocol TAug, classifier rf, eps 5."""
    out: dict[str, Any] = {"config": name, "method": None, "family": "", "variant": "", "protocol": "", "classifier": "", "eps": np.nan, "alpha": np.nan,
                           "prefix": "", "fl_run": None}
    if name in BASELINES:
        out.update(method=BASELINES[name][0], classifier=BASELINES[name][1], protocol="TRTR")
        return out
    g = _GEN.match(name)
    if not g:
        return out
    prefix, plain = g["prefix"], g["prefix"].endswith("-plain")
    base = prefix[:-6] if plain else prefix
    out.update(protocol=g["proto"], classifier=g["clf"], prefix=prefix, variant="plain" if plain else "standard")
    m = _EPS.search(base)
    if m:
        out["eps"] = float(m["eps"])
    if base in ("B2", "B3", "M2"):
        out.update(method=base, fl_run=None if base == "B2" else base)
        if base == "B3" and plain:
            out["family"] = "epsilon-infinity reference"
    elif base.startswith(("M1d-t", "M3d-t")):
        trial = int(re.match(r"M[13]d-t(\d+)", base)[1])
        out.update(method=base[:2], family=f"dp-tuned t{trial}", fl_run=base)
    elif base.startswith(("M1-", "M3-")):
        out.update(method=base[:2], family="phase7 hyper-parameters", fl_run=base)
    elif _ALPHA.match(base):
        out.update(method="A1", family="extension", alpha=float(_ALPHA.match(base)["alpha"]), fl_run=base)
    if base == "B3":
        out["fl_run"] = "B3"
    return out


def dp_families(names) -> dict[str, str]:
    """Trial number of the tuned DP family found in the ledger names, per method: {'M1': 'M1d-t21', 'M3': 'M3d-t21'} (the highest-numbered
    trial that has all of eps 1, 5, 10 for M1)."""
    found: dict[str, dict[str, set[float]]] = {"M1": {}, "M3": {}}
    for n in names:
        p = parse_config(n)
        if p["family"].startswith("dp-tuned") and p["variant"] == "plain":
            found[p["method"]].setdefault(re.match(r"(M[13]d-t\d+)", p["prefix"])[1], set()).add(p["eps"])
    out = {}
    for m, d in found.items():
        if d:
            out[m] = max(d, key=lambda t: (len(d[t]), int(t.split("-t")[1])))
    return out


# --------------------------------------------------------------------------------------------------
# Loading and summarising
# --------------------------------------------------------------------------------------------------
def load_ledger(cfg: dict[str, Any]) -> pd.DataFrame:
    df = RunLedger(runs_csv_path(cfg)).frame()
    if not len(df):
        raise FileNotFoundError(f"empty or missing run ledger {runs_csv_path(cfg)}")
    return df[df["label_mode"] == cfg["label_mode"]].copy()


def median_round_seconds(cfg: dict[str, Any], fl_run: str, seed: int) -> float | None:
    """Median wall time of rounds 2+ of an FL run (round 1 includes starting Ray), from its round log."""
    p = artifacts_dir(cfg) / run_id(fl_run, seed, cfg["label_mode"]) / "rounds.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    xs = [r["round_seconds"] for r in rows if r.get("round", 0) > 1 and r.get("round_seconds") is not None]
    return float(np.median(xs)) if xs else None


def annotate(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add the parsed name parts and the per-run median round time."""
    meta = pd.DataFrame([parse_config(n) for n in df["config"].unique()]).set_index("config")
    out = df.join(meta[["method", "family", "variant", "eps", "prefix", "fl_run"]], on="config")
    out["eps_target"] = out["eps"]
    out["round_s_median"] = [median_round_seconds(cfg, r["fl_run"], int(r["seed"])) if isinstance(r["fl_run"], str) else np.nan for _, r in out.iterrows()]
    return out


def build_summary(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """One row per ledger configuration with mean and std over seeds. `df` must have been through `annotate`."""
    classes = [c[len("recall_"):] for c in df.columns if c.startswith("recall_")]          # label order, as the ledger columns
    metrics = [m for m in SUMMARY_METRICS + [f"recall_{c}" for c in classes] + [f"f1_{c}" for c in classes] if m in df.columns]
    rows = []
    for name, g in df.groupby("config", sort=False):
        r = {"config": name, **{k: g[k].iloc[0] for k in ("method", "family", "variant", "protocol", "classifier", "eps", "alpha")},
             "n_seeds": int(g["seed"].nunique()), "seeds": ",".join(str(int(s)) for s in sorted(g["seed"].unique())),
             "config_hashes": ",".join(sorted(set(g["config_hash"].dropna().astype(str)))), "git_commits": ",".join(sorted(set(c[:7] for c in g["git_commit"].dropna().astype(str))))}
        for m in metrics:
            x = pd.to_numeric(g[m], errors="coerce")
            r[f"{m}_mean"] = float(x.mean()) if x.notna().any() else np.nan
            r[f"{m}_std"] = float(x.std(ddof=0)) if x.notna().any() else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    shown = {n for e in entries(out) for n in e["cfgs"].values()}
    shown |= {n.replace("-plain-", "-") for n in shown if "-plain-" in n and not n.startswith("B3-plain")}          # the residual-noise twin of a plain DP row
    out.insert(out.columns.get_loc("n_seeds"), "in_matrix", out["config"].isin(shown))
    return out


def ms(summ: pd.DataFrame, config: str, metric: str, digits: int = 4, scale: float = 1.0) -> str:
    """'mean ± std' of a metric of a configuration, 'n/a' when there is none."""
    r = summ[summ["config"] == config]
    if not len(r) or pd.isna(r.iloc[0].get(f"{metric}_mean", np.nan)):
        return "n/a"
    r = r.iloc[0]
    m, s = r[f"{metric}_mean"] * scale, r[f"{metric}_std"] * scale
    return f"{m:.{digits}f} ± {s:.{digits}f}" if r["n_seeds"] > 1 else f"{m:.{digits}f} (1 seed)"


def mean_std(summ: pd.DataFrame, config: str, metric: str) -> tuple[float, float] | None:
    r = summ[summ["config"] == config]
    if not len(r) or pd.isna(r.iloc[0].get(f"{metric}_mean", np.nan)):
        return None
    return float(r.iloc[0][f"{metric}_mean"]), float(r.iloc[0][f"{metric}_std"])


# --------------------------------------------------------------------------------------------------
# The rows of the tables and figures
# --------------------------------------------------------------------------------------------------
def entries(summ: pd.DataFrame, fam: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Display entries in the order of the spec matrix. `cfgs` maps (protocol, classifier) to a ledger configuration name."""
    names = set(summ["config"])
    fam = dp_families(names) if fam is None else fam
    gen = lambda prefix: {(p, c): f"{prefix}-{p}-{c}" for p in ("TSTR", "TAug") for c in ("rf", "mlp")}      # noqa: E731
    base = lambda **kw: {("TRTR", c): n for c, n in kw.items()}                                               # noqa: E731
    E = [dict(label="B0", color="baseline", cfgs={("TRTR", "rf"): "B0-rf", ("TRTR", "mlp"): "B0-mlp"}),
         dict(label="B1a", color="baseline", cfgs={("TRTR", "rf"): "B1a-rf"}),
         dict(label="B1b", color="baseline", cfgs={("TRTR", "rf"): "B1b-rf", ("TRTR", "mlp"): "B1b-mlp"}),
         dict(label="B2", color="gen", cfgs=gen("B2")), dict(label="B3", color="gen", cfgs=gen("B3"))]
    if "M1" in fam:
        for e in (1, 5, 10):
            E.append(dict(label=f"M1-eps{e}", color="dp", eps=e, cfgs=gen(f"{fam['M1']}-eps{e}-plain")))
    E.append(dict(label="M2", color="secagg", cfgs=gen("M2")))
    if "M3" in fam:
        E.append(dict(label="M3-eps5", color="dpsa", eps=5, cfgs=gen(f"{fam['M3']}-eps5-plain")))
    return [e for e in E if any(n in names for n in e["cfgs"].values())]


def reference_entries(summ: pd.DataFrame) -> list[dict[str, Any]]:
    names = set(summ["config"])
    gen = lambda prefix: {(p, c): f"{prefix}-{p}-{c}" for p in ("TSTR", "TAug") for c in ("rf", "mlp")}      # noqa: E731
    E = [dict(label="B3-plain (eps = inf)", color="ref", cfgs=gen("B3-plain"))]
    E += [dict(label=f"M1 Phase 7 hp, eps {e}", color="ref", eps=e, cfgs=gen(f"M1-eps{e}-plain")) for e in (1, 5, 10)]
    return [e for e in E if any(n in names for n in e["cfgs"].values())]


# --------------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    _plt().close(fig)
    return path


def fig_f1_by_config(summ: pd.DataFrame, ents: list[dict[str, Any]], out: Path) -> Path:
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True)
    for i, proto in enumerate(("TAug", "TSTR")):
        for j, clf in enumerate(("rf", "mlp")):
            ax = axes[i, j]
            xs, labels = [], []
            for e in ents:
                key = ("TRTR", clf) if e["color"] == "baseline" else (proto, clf)
                if proto == "TSTR" and e["color"] == "baseline":
                    continue
                v = mean_std(summ, e["cfgs"].get(key, ""), "macro_f1")
                if v is None:
                    continue
                x = len(xs)
                ax.bar(x, v[0], yerr=v[1], capsize=3, color=COLORS[e["color"]])
                xs.append(x)
                labels.append(e["label"].replace("eps", "ε"))
            if proto == "TSTR":
                b0 = mean_std(summ, f"B0-{clf}", "macro_f1")
                if b0:
                    ax.axhline(b0[0], color="k", ls="--", lw=0.8, label=f"B0-{clf} (real data only)")
                    ax.legend(fontsize=7, loc="upper right")
            ax.set_xticks(xs, labels, rotation=40, ha="right", fontsize=8)
            ax.set_title(f"{clf.upper()}, " + ("real + synthetic (TAug); B0/B1 = real data without/with SMOTE" if proto == "TAug" else "synthetic only (TSTR)"), fontsize=9)
            ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("test macro-F1")
    axes[1, 0].set_ylabel("test macro-F1")
    fig.suptitle("Test macro-F1 by configuration (mean ± std over seeds; DP configurations: plain decoder)", fontsize=10)
    return _save(fig, out)


def rare_classes(cfg: dict[str, Any], summ: pd.DataFrame, limit: int = 5000) -> list[str]:
    """Classes with at most `limit` train rows (from the quotas), in the label order of the ledger columns."""
    have = [c[len("recall_"):-len("_mean")] for c in summ.columns if c.startswith("recall_") and c.endswith("_mean")]
    q = cfg.get("quota", {})
    rare = [c for c in have if c in q and q[c][0] <= limit]
    return rare or have[-4:]


def fl_partition_stats(cfg: dict[str, Any], fl_run: str, seeds: list[int]) -> dict[str, Any] | None:
    """Client sizes and effective optimiser steps per round of the FL run `fl_run` over `seeds`, read from their spec.json (None when absent)."""
    from ppfeddata.fl.b3 import effective_steps
    sizes, steps = [], []
    for s in seeds:
        p = artifacts_dir(cfg) / run_id(fl_run, s, cfg["label_mode"]) / "spec.json"
        if not p.exists():
            continue
        sp = json.loads(p.read_text(encoding="utf-8"))
        sizes.append([int(x) for x in sp["partition_sizes"]])
        steps.append(effective_steps(sizes[-1], int(sp["hp"]["batch_size"]), int(sp["local_epochs"])))
    if not sizes:
        return None
    return {"min": min(min(s) for s in sizes), "max": max(max(s) for s in sizes), "steps": float(np.mean(steps)), "n": len(sizes)}


def fig_recall_rare(cfg: dict[str, Any], summ: pd.DataFrame, ents: list[dict[str, Any]], out: Path, clf: str = "rf") -> Path:
    plt = _plt()
    classes = rare_classes(cfg, summ)
    items = []
    for e in ents:
        key = ("TRTR", clf) if e["color"] == "baseline" else ("TAug", clf)
        n = e["cfgs"].get(key)
        if n and (summ["config"] == n).any() and (e["color"] != "dp" or e.get("eps") == 5):
            items.append((e["label"], n, e["color"]))
    fig, ax = plt.subplots(figsize=(11, 4))
    w = 0.8 / max(len(items), 1)
    for i, (label, n, color) in enumerate(items):
        vals = [mean_std(summ, n, f"recall_{c}") or (np.nan, 0) for c in classes]
        ax.bar(np.arange(len(classes)) + i * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals], capsize=2, color=COLORS[color], alpha=0.35 + 0.65 * (i + 1) / len(items),
               edgecolor="k", linewidth=0.4, label=label.replace("eps", "ε"))
    ax.set_xticks(np.arange(len(classes)) + (len(items) - 1) * w / 2, classes)
    ax.set_ylabel(f"test recall ({clf.upper()})")
    ax.set_title("Recall of the rare classes (train rows <= 5000): real-only baselines and real + synthetic (TAug)", fontsize=10)
    ax.legend(fontsize=7, ncol=len(items), loc="upper center", bbox_to_anchor=(0.5, -0.07))
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def _eps_axis(summ: pd.DataFrame, fam: dict[str, str]) -> tuple[list[float], list[str]]:
    eps = sorted({float(e["eps"]) for e in entries(summ, fam) if e.get("eps") and e["label"].startswith("M1")})
    return eps, [f"{e:g}" for e in eps] + ["∞"]


def fig_utility_privacy(summ: pd.DataFrame, fam: dict[str, str], out: Path) -> Path:
    plt = _plt()
    eps, labels = _eps_axis(summ, fam)
    xs = list(range(len(eps) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, metric, title in ((axes[0], "macro_f1", "test macro-F1 (RF)"), (axes[1], "mia_auc_mean", "MIA AUC (0.5 = no signal; weak attack)"),
                              (axes[2], "dcr_ratio_mean", "DCR ratio (1 = no copying)")):
        for proto, style in (("TSTR", "o-"), ("TAug", "^-")):
            if metric != "macro_f1" and proto != "TSTR":
                continue
            m = [mean_std(summ, f"{fam['M1']}-eps{e:g}-plain-{proto}-rf", metric) for e in eps] if "M1" in fam else [None] * len(eps)
            m.append(mean_std(summ, f"B3-plain-{proto}-rf", metric))
            pts = [(x, v) for x, v in zip(xs, m) if v]
            if pts:
                ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts], yerr=[p[1][1] for p in pts], fmt=style, capsize=3, color=COLORS["dp"],
                            label=("DP (M1)" if metric != "macro_f1" else f"M1, {proto}-rf"))
            if "M3" in fam and 5.0 in eps:
                v = mean_std(summ, f"{fam['M3']}-eps5-plain-{proto}-rf", metric)
                if v:
                    ax.errorbar([eps.index(5.0)], [v[0]], yerr=[v[1]], fmt="s" if proto == "TSTR" else "D", capsize=3, color=COLORS["dpsa"], markersize=8,
                                label="M3 (DP + SecAgg)" if metric != "macro_f1" else f"M3, {proto}-rf")
        if metric == "macro_f1":
            b0 = mean_std(summ, "B0-rf", "macro_f1")
            if b0:
                ax.axhline(b0[0], color="k", ls="--", lw=0.8, label="B0-rf")
        if metric == "mia_auc_mean":
            ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        if metric == "dcr_ratio_mean":
            ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax.set_xticks(xs, labels)
        ax.set_xlabel("ε (worst client, record level; ∞ = no DP)")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    return _save(fig, out)


def fig_fidelity(summ: pd.DataFrame, fam: dict[str, str], out: Path) -> Path:
    plt = _plt()
    eps, labels = _eps_axis(summ, fam)
    xs = list(range(len(eps) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, metric, title in ((axes[0], "wasserstein_mean", "Wasserstein distance (lower is better)"), (axes[1], "c2st_auc_mean", "C2ST AUC (0.5 = indistinguishable)")):
        for suffix, style, name in (("-plain", "o-", "plain decoder"), ("", "s--", "with residual noise")):
            m = [mean_std(summ, f"{fam['M1']}-eps{e:g}{suffix}-TSTR-rf", metric) for e in eps] if "M1" in fam else [None] * len(eps)
            m.append(mean_std(summ, f"B3{'-plain' if suffix else ''}-TSTR-rf", metric))
            pts = [(x, v) for x, v in zip(xs, m) if v]
            if pts:
                ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts], yerr=[p[1][1] for p in pts], fmt=style, capsize=3, label=name)
        if metric == "c2st_auc_mean":
            ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_xticks(xs, labels)
        ax.set_xlabel("ε (∞ = B3, no DP)")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    return _save(fig, out)


def cost_rows(summ: pd.DataFrame, ents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per FL configuration: median seconds per round, ratio to B3, bytes per round, bytes per parameter (the DP-tuned model is smaller)."""
    base = mean_std(summ, "B3-TSTR-rf", "round_s_median")
    rows = []
    for e in ents:
        n = e["cfgs"].get(("TSTR", "rf"))
        if e["label"] in ("B0", "B1a", "B1b", "B2") or not n:
            continue
        t = mean_std(summ, n, "round_s_median")
        b = mean_std(summ, n, "bytes_per_round")
        p = mean_std(summ, n, "cvae_params")
        if t is None:
            continue
        rows.append({"label": e["label"], "color": e["color"], "s_round": t[0], "s_round_std": t[1], "ratio": t[0] / base[0] if base else np.nan,
                     "bytes": b[0] if b else np.nan, "params": p[0] if p else np.nan, "bytes_per_param": (b[0] / p[0]) if b and p else np.nan})
    return rows


def fig_overhead(summ: pd.DataFrame, ents: list[dict[str, Any]], out: Path) -> Path:
    plt = _plt()
    rows = cost_rows(summ, ents)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    xs = np.arange(len(rows))
    axes[0].bar(xs, [r["s_round"] for r in rows], yerr=[r["s_round_std"] for r in rows], capsize=3, color=[COLORS[r["color"]] for r in rows])
    for x, r in zip(xs, rows):
        axes[0].text(x, r["s_round"] + r["s_round_std"], f"×{r['ratio']:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(xs, [r["label"].replace("eps", "ε") for r in rows], rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("median seconds per round (rounds 2+)")
    axes[0].set_title("Time per round (label = ratio to B3)", fontsize=9)
    axes[1].bar(xs, [r["bytes_per_param"] for r in rows], color=[COLORS[r["color"]] for r in rows])
    for x, r in zip(xs, rows):
        axes[1].text(x, r["bytes_per_param"], f"{r['bytes'] / 1e6:.2f} MB", ha="center", va="bottom", fontsize=7)
    axes[1].set_xticks(xs, [r["label"].replace("eps", "ε") for r in rows], rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("bytes per model parameter per round")
    axes[1].set_title("Communication per round (label = total; the DP model is smaller)", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def fig_pareto(summ: pd.DataFrame, ents: list[dict[str, Any]], out: Path) -> Path:
    plt = _plt()
    cost = {r["label"]: r for r in cost_rows(summ, ents)}
    fig, ax = plt.subplots(figsize=(7, 5))
    for e in ents:
        if e["label"] in ("B0", "B1a", "B1b"):
            continue
        y = mean_std(summ, e["cfgs"].get(("TAug", "rf"), ""), "macro_f1")
        x = mean_std(summ, e["cfgs"].get(("TSTR", "rf"), ""), "mia_auc_mean")
        if y is None or x is None:
            continue
        c = cost.get(e["label"])
        size = 40 + 120 * (c["ratio"] if c and not np.isnan(c["ratio"]) else 1.0)
        ax.errorbar(x[0], y[0], xerr=x[1], yerr=y[1], fmt="none", ecolor="gray", alpha=0.5)
        if c:
            ax.scatter([x[0]], [y[0]], s=size, color=COLORS[e["color"]], edgecolor="k", alpha=0.8)
        else:                                                  # not federated: hollow marker of fixed size
            ax.scatter([x[0]], [y[0]], s=size, facecolors="none", edgecolors=COLORS[e["color"]], linewidths=1.8)
        ax.annotate(e["label"].replace("eps", "ε"), (x[0], y[0]), textcoords="offset points", xytext=(6, 5), fontsize=8)
    b0 = mean_std(summ, "B0-rf", "macro_f1")
    if b0:
        ax.axhline(b0[0], color="k", ls="--", lw=0.8)
        ax.text(ax.get_xlim()[0], b0[0], " B0-rf (real data only)", fontsize=7, va="bottom", ha="left")
    lo, hi = ax.get_ylim()
    ax.text(0.99, 0.01, f"y-axis span {hi - lo:.4f} macro-F1: differences are inside the seed std", transform=ax.transAxes, fontsize=7, ha="right", va="bottom", color="gray")
    ax.set_xlabel("MIA AUC of the synthetic set (0.5 = no signal; weak attack)")
    ax.set_ylabel("test macro-F1, real + synthetic (TAug, RF)")
    ax.set_title("Utility against empirical privacy risk; point size = time per round relative to B3\n(B2 is not federated: hollow, fixed size)", fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, out)


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def _table(rows: list[dict[str, Any]]) -> str:
    from ppfeddata.eval.baselines import _table as t
    return t(rows) if rows else "_(no data)_"


def versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for p in ("torch", "flwr", "opacus", "ray", "scikit-learn", "numpy", "pandas", "optuna", "imbalanced-learn"):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            pass
    return out


def provenance_rows(summ: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (h, c), g in summ.assign(_h=summ["config_hashes"], _c=summ["git_commits"]).groupby(["_h", "_c"]):
        rows.append({"config hash": h, "git commit": c, "ledger configurations": len(g), "methods": ", ".join(sorted({str(m) for m in g["method"].dropna()}))})
    return rows


def write_final_report(cfg: dict[str, Any], summ: pd.DataFrame, df: pd.DataFrame, figs: dict[str, Path], out: Path, fig_rel: str = "../figures",
                       interp: dict[str, Any] | None = None) -> Path:
    from ppfeddata.run_experiment import compare_workspaces, load_matrix, plan, workspace_cfg

    mode = cfg["label_mode"]
    fam = dp_families(set(summ["config"]))
    ents, refs = entries(summ, fam), reference_entries(summ)
    classes = [c[len("recall_"):-len("_mean")] for c in summ.columns if c.startswith("recall_") and c.endswith("_mean")]
    seeds = sorted(int(s) for s in df["seed"].unique())
    th = cfg["thresholds"]
    L = [f"# PP-FedData - final report ({mode})", "",
         "Auto-generated by `ppfeddata aggregate` from the run ledger (`results/runs.csv`), the FL round logs and the config; no number is typed by hand. "
         f"Seeds {seeds}; {len(df)} ledger rows ({df['config'].nunique()} configurations). Utility is measured on the real test split. "
         "DP configurations are shown with the **plain decoder**, the variant that depends only on the DP-trained weights and is covered by epsilon "
         "(epsilon = worst client, record level, delta " + f"{cfg['dp']['delta']:g}" + "); B3 at epsilon = infinity is B3-plain. The headline DP family "
         f"is {fam.get('M1', 'n/a')} / {fam.get('M3', 'n/a')} (hyper-parameters searched under DP-SGD, SPEC_DEVIATIONS 9.11-9.14). "
         + ("Section 9 applies the interpretation rules of Phase 12 (answers R1-R6, red flags, recommendation)." if interp is not None else
            "This version contains tables and figures only; the interpretation rules of Phase 12 (answers R1-R6, red flags, recommendation) are **not applied yet**."), ""]

    # ---- provenance and completeness
    exps = load_matrix()
    pl = plan(cfg, exps, seeds, groups=("matrix",))
    comp = [{"configuration": e.id, "title": e.title, **{f"seed {s}": next((r["status"] for r in pl if r["id"] == e.id and r["seed"] == s), "-") for s in seeds}}
            for e in exps if e.group == "matrix"]
    repro = ""
    tcfg = workspace_cfg(cfg, "trial")
    if runs_csv_path(tcfg).exists() and runs_csv_path(tcfg) != runs_csv_path(cfg):
        cmp_ = compare_workspaces(cfg, tcfg, seeds)
        ok = cmp_[cmp_["in_main_ledger"].fillna(False).astype(bool)] if len(cmp_) else cmp_
        if len(ok):
            repro = (f" **Reproduction check:** the isolated one-seed trial (seed {sorted(set(int(x) for x in ok['seed']))}) re-ran the matrix from scratch and reproduced "
                     f"{int(ok['identical'].sum())} of {len(ok)} ledger rows exactly (largest difference of a result metric {float(ok['max_abs_diff'].max()):.1e}; byte counts of SecAgg runs "
                     f"differ by up to {float(ok['bytes_abs_diff'].max()):.0f} bytes because they depend on random keys).")
    mixed = summ[summ["config_hashes"].str.contains(",") | summ["git_commits"].str.contains(",")]["config"].tolist()
    same = ("Every configuration was produced with one config hash and one git commit across its seeds. " if not mixed else
            f"{len(mixed)} configurations mix several hashes or commits across their seeds: {', '.join(mixed[:6])}{' ...' if len(mixed) > 6 else ''}. ")
    L += ["## 1. Provenance and completeness", "", "Status of the spec matrix in this ledger (done = every expected ledger row exists with its predictions):", "", _table(comp), "",
          same + "The hashes below differ between phases because the config file grew (new keys for later phases, none of them read by the earlier methods); "
          "`ppfeddata run --stage trial` re-runs the whole matrix from scratch under one commit and compares it with these rows (`results/reports/g4_trial.md`)." + repro, "",
          _table(provenance_rows(summ)), "",
          "Libraries installed when this report was generated: " + ", ".join(f"{k} {v}" for k, v in versions().items()) + ".", ""]
    q = cfg.get("quota", {})
    if q:
        L += ["Rows per class [train, val, test] (quotas; the splits are by capture group, see leakage_report.md):", "",
              _table([{"class": c, "train": v[0], "val": v[1], "test": v[2]} for c, v in q.items()]), ""]

    # ---- utility
    for clf in ("rf", "mlp"):
        rows = []
        for e in ents:
            if e["color"] == "baseline":
                n = e["cfgs"].get(("TRTR", clf))
                if not n:
                    continue                                                # e.g. B1a exists for the random forest only
                rows.append({"configuration": e["label"], "real-only / SMOTE": ms(summ, n, "macro_f1"), "TSTR (synthetic only)": "-", "TAug (real + synthetic)": "-",
                             "balanced acc (TAug / baseline)": ms(summ, n, "balanced_acc")})
            else:
                a, b = e["cfgs"][("TSTR", clf)], e["cfgs"][("TAug", clf)]
                rows.append({"configuration": e["label"], "real-only / SMOTE": "-", "TSTR (synthetic only)": ms(summ, a, "macro_f1"), "TAug (real + synthetic)": ms(summ, b, "macro_f1"),
                             "balanced acc (TAug / baseline)": ms(summ, b, "balanced_acc")})
        L += [f"## 2{'a' if clf == 'rf' else 'b'}. Utility, {clf.upper()} (test macro-F1, mean ± std over seeds)", "", _table(rows), ""]
    L += [f"![macro-F1 by configuration]({fig_rel}/{figs['f1'].name})", ""]

    # ---- rare classes
    rc = rare_classes(cfg, summ)
    rows = []
    for e in ents:
        key = ("TRTR", "rf") if e["color"] == "baseline" else ("TAug", "rf")
        n = e["cfgs"].get(key)
        if n:
            rows.append({"configuration": e["label"], **{c: ms(summ, n, f"recall_{c}", 3) for c in rc}})
    L += [f"## 3. Recall of the rare classes ({', '.join(rc)}; RF, real + synthetic or baseline)", "", _table(rows), "", f"![recall of the rare classes]({fig_rel}/{figs['recall'].name})", ""]
    rows = []
    for e in ents:
        n = e["cfgs"].get(("TRTR", "rf") if e["color"] == "baseline" else ("TAug", "rf"))
        if n:
            rows.append({"configuration": e["label"], **{c: ms(summ, n, f"recall_{c}", 3) for c in classes}})
    L += ["All classes:", "", _table(rows), ""]

    # ---- privacy
    prow = []
    for e in ents:
        n = e["cfgs"].get(("TSTR", "rf"))
        if e["color"] == "baseline" or not n or not (summ["config"] == n).any():
            continue
        r = summ[summ["config"] == n].iloc[0]
        prow.append({"configuration": e["label"], "target eps": f"{e['eps']:g}" if e.get("eps") else "-",
                     "eps achieved (max / median over clients)": f"{ms(summ, n, 'dp_eps_max', 3)} / {ms(summ, n, 'dp_eps_median', 3)}" if pd.notna(r.get("dp_eps_max_mean")) else "-",
                     "duplicate rate": ms(summ, n, "dup_rate"), "DCR ratio": ms(summ, n, "dcr_ratio_mean"), "MIA AUC": ms(summ, n, "mia_auc_mean")})
    L += ["## 4. Privacy: epsilon achieved and empirical checks", "",
          "The formal guarantee is epsilon (record level, packets of one TCP stream are correlated, labels are not protected; SecAgg adds no accounting). "
          "The duplicate rate, the DCR ratio and the membership-inference AUC are empirical checks of the synthetic set; the MIA only detects near-copies "
          "(positive control in b2_cvae.md), so 0.5 means 'no copying', not 'no leakage'.", "", _table(prow), "",
          f"Thresholds from the config (heuristics to be confirmed by the user): duplicate rate <= {th['dup_rate_max']}, DCR ratio >= {th['dcr_ratio_min']}, MIA AUC <= {th['mia_auc_max']}.", ""]

    # ---- fidelity
    frow = []
    for e in ents:
        n = e["cfgs"].get(("TSTR", "rf"))
        if e["color"] != "baseline" and n:
            frow.append({"configuration": e["label"], "Wasserstein": ms(summ, n, "wasserstein_mean"), "JS": ms(summ, n, "js_mean"), "correlation distance": ms(summ, n, "corr_dist_mean"),
                         "C2ST AUC": ms(summ, n, "c2st_auc_mean")})
    L += ["## 5. Fidelity of the synthetic set (mean over classes)", "",
          f"C2ST AUC close to 1 means a classifier tells synthetic from real rows almost perfectly (threshold in the config: <= {th['c2st_auc_max']}); the mean-squared-error decoder reproduces "
          "discrete columns as continuous values (SPEC_DEVIATIONS 7.1, 7.6), so this holds for every generator, DP or not.", "", _table(frow), "",
          f"![utility and privacy against epsilon]({fig_rel}/{figs['up'].name})", "", f"![fidelity against epsilon]({fig_rel}/{figs['fid'].name})", ""]

    # ---- cost
    crow = []
    for r in cost_rows(summ, ents):
        crow.append({"configuration": r["label"], "median s/round (rounds 2+)": f"{r['s_round']:.2f}", "ratio to B3": f"{r['ratio']:.2f}", "bytes/round": f"{r['bytes']:,.0f}",
                     "bytes/param/round": f"{r['bytes_per_param']:.1f}"})
    L += ["## 6. Cost", "",
          "Time per round is the median of rounds 2+ of the FL round logs (round 1 includes starting Ray), averaged over seeds. Bytes per round: plain FL = float32 model x clients x 2; "
          "SecAgg = counted on the grid (all four protocol stages). The DP-tuned model (hidden 128-64) is smaller than the B3 / M2 model (hidden 256-128), so compare bytes per parameter, "
          f"and compare M2 with B3 and M3 with M1. Overhead threshold in the config: {th['overhead_ratio_max']}x the plain FL round.", "", _table(crow), "",
          f"![overhead]({fig_rel}/{figs['cost'].name})", "", f"![Pareto]({fig_rel}/{figs['pareto'].name})", ""]

    # ---- DP detail: the residual-noise variant and the reference family
    rows = []
    for e in ents:
        if e["color"] in ("dp", "dpsa"):
            pfx = e["cfgs"][("TSTR", "rf")].rsplit("-TSTR-rf", 1)[0]
            std = pfx[:-6]
            rows.append({"configuration": e["label"], "TSTR-rf plain": ms(summ, f"{pfx}-TSTR-rf", "macro_f1"), "TSTR-rf with residual noise (not covered by eps)": ms(summ, f"{std}-TSTR-rf", "macro_f1"),
                         "TAug-rf plain": ms(summ, f"{pfx}-TAug-rf", "macro_f1"), "TAug-rf with residual noise": ms(summ, f"{std}-TAug-rf", "macro_f1")})
    L += ["## 7. DP configurations: plain decoder against the variant with residual noise", "",
          "The residual-noise scales are computed from the pooled train data and are not covered by epsilon (SPEC_DEVIATIONS 8.8, 9.4); the gap shows how much utility depends on that statistic.", "", _table(rows), ""]
    rrows = []
    for e in refs:
        rrows.append({"configuration": e["label"], "TSTR-rf": ms(summ, e["cfgs"][("TSTR", "rf")], "macro_f1"), "TSTR-mlp": ms(summ, e["cfgs"][("TSTR", "mlp")], "macro_f1"),
                      "TAug-rf": ms(summ, e["cfgs"][("TAug", "rf")], "macro_f1"), "TAug-mlp": ms(summ, e["cfgs"][("TAug", "mlp")], "macro_f1"),
                      "eps achieved (max)": ms(summ, e["cfgs"][("TSTR", "rf")], "dp_eps_max", 3) if e.get("eps") else "-"})
    L += ["## 8. Reference rows (not in the spec matrix)", "",
          "B3-plain is the epsilon = infinity point; the Phase 7 family used hyper-parameters tuned without DP and a clipping bound of 1 (SPEC_DEVIATIONS 9.5); it is kept to show what the DP-specific search changed.", "", _table(rrows), ""]
    a1 = sorted({float(a) for a in summ[summ["method"] == "A1"]["alpha"].dropna()})
    if a1:
        rare = [c for c in rare_classes(cfg, summ) if f"recall_{c}" in df.columns]
        rows, rows2 = [], []
        for a in sorted(set(a1) | {float(cfg["fl"]["dirichlet_alpha"])}):
            pre = "B3" if a == float(cfg["fl"]["dirichlet_alpha"]) else f"A1-a{a:g}"
            lab = f"{a:g}" + (" (B3)" if pre == "B3" else "")
            rows.append({"Dirichlet alpha": lab, **{f"{p}-{c}": ms(summ, f"{pre}-{p}-{c}", "macro_f1") for p in ("TSTR", "TAug") for c in ("rf", "mlp")}})
            ref = f"{pre}-TSTR-rf"
            sub = df[df["config"] == ref]
            seeds_ = sorted(int(s) for s in sub["seed"].unique())
            st = fl_partition_stats(cfg, pre, seeds_)
            rr = sub[[f"recall_{c}" for c in rare]].mean(axis=1) if len(sub) and rare else pd.Series(dtype=float)
            rows2.append({"Dirichlet alpha": lab, "seeds": len(seeds_),
                          "client rows, min - max over seeds": f"{st['min']:,} - {st['max']:,}" if st else "n/a",
                          "effective optimiser steps / round": f"{st['steps']:.0f}" if st else "n/a",
                          "FL final val loss (ELBO)": ms(summ, ref, "fl_final_val_loss"), "C2ST AUC": ms(summ, ref, "c2st_auc_mean"),
                          "Wasserstein": ms(summ, ref, "wasserstein_mean"),
                          "TSTR-rf recall, mean over " + "/".join(rare): f"{rr.mean():.3f} ± {rr.std(ddof=0):.3f}" if len(rr) else "n/a"})
        L += ["## 8b. Extension A1: label skew of the clients (Dirichlet alpha)", "",
              f"B3 with another Dirichlet alpha and everything else equal ({cfg['fl']['num_clients']} clients, {cfg['fl']['rounds']} rounds x {cfg['fl']['local_epochs']} local epochs, "
              "the same hyper-parameters, no DP, no SecAgg). Every seed draws its own partition, so the spread over seeds includes the partition draw. "
              "Macro-F1 on the real test split.", "", _table(rows), "",
              "Generator and training side of the same runs. alpha changes more than the label mix: with a skewed split the clients also differ in size, and a larger client "
              "takes more optimiser steps and carries more FedAvg weight (SPEC_DEVIATIONS 8.6), so the effective number of steps per round differs between rows.", "", _table(rows2), ""]

    if interp is not None:
        from ppfeddata.interpret_report import render
        L += render(interp, cfg)
    else:
        L += ["## 9. Interpretation (Phase 12)", "",
              "Not generated: the answers R1-R6, the red-flag checks and the recommended configuration need the test-set predictions of the runs (`ppfeddata aggregate` without `--no-interpret`).", ""]
    from ppfeddata import limitations
    L += ["## 10. Limitations that apply to every number above", "",
          "The list is written once (`limitations.py`) and also used by the README and the demo; numbers come from the split manifest, the feature schema, the config and `interpretation.json`.", ""]
    L += limitations.render_md(limitations.limitations(cfg, interp, summ)) + [""]
    L += ["## 11. Sources", "",
          "`results/runs.csv` (ledger), `results/summary.csv`, `results/interpretation.json` (section 9), `artifacts/<run>/preds/test.npz` (test predictions), `artifacts/<run>/rounds.jsonl` (round logs), `results/manifests/split_manifest_<mode>.json` and `feature_schema.json` (section 10), `configs/default.yaml`, `configs/best_cvae.yaml`, `configs/best_cvae_dp.yaml`, "
          "and the phase reports in `results/reports/` (g3_baseline, b2_cvae, b3_fl, m1_dp, m1_dp_tuned, m2_m3_secagg, g4_trial).", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def aggregate(cfg: dict[str, Any], out_dir: str | Path | None = None, dp_families_override: dict[str, str] | None = None,
              with_interpretation: bool = True, n_boot: int | None = None) -> dict[str, Any]:
    """Write summary.csv, the six figures, interpretation.json and final_report.md, and refresh the generated blocks of README.md. `out_dir` replaces `results/` (e.g. for a preview of a trial ledger;
    the README is then left alone).
    The interpretation (Phase 12) re-reads the test predictions of every run and runs a bootstrap, about half a minute; `with_interpretation=False` skips it."""
    root = Path(out_dir) if out_dir else Path("./results")
    df = annotate(load_ledger(cfg), cfg)
    summ = build_summary(df, cfg)
    root.mkdir(parents=True, exist_ok=True)
    summ.to_csv(root / "summary.csv", index=False)
    fam = dp_families_override or dp_families(set(summ["config"]))
    ents = entries(summ, fam)
    fd = root / "figures"
    figs = {"f1": fig_f1_by_config(summ, ents, fd / "f1_by_config.png"), "recall": fig_recall_rare(cfg, summ, ents, fd / "recall_rare_classes.png"),
            "up": fig_utility_privacy(summ, fam, fd / "utility_privacy.png"), "fid": fig_fidelity(summ, fam, fd / "fidelity_vs_eps.png"),
            "cost": fig_overhead(summ, ents, fd / "overhead.png"), "pareto": fig_pareto(summ, ents, fd / "pareto.png")}
    interp, interp_path = None, None
    if with_interpretation:
        from ppfeddata import interpret
        try:
            interp = interpret.interpret(cfg, summ, df, n_boot=n_boot)
            interp_path = str(interpret.write_json(interp, root / "interpretation.json"))
        except FileNotFoundError as e:                                  # no saved predictions / test split: the tables are still written
            logger.warning("interpretation skipped: %s", e)
    report = write_final_report(cfg, summ, df, figs, root / "reports" / "final_report.md", interp=interp)
    readme: list[str] = []
    if out_dir is None and interp is not None:                         # only the real results: a preview or a test must not rewrite the repository's README
        from ppfeddata import interpret, readme_gen
        try:                                                           # from the numbers as written to interpretation.json (8 decimals), so the README equals what is generated from that file
            readme = readme_gen.update_readme(readme_gen.README, cfg, interpret.to_jsonable(interp), summ)
        except (OSError, ValueError) as e:
            logger.warning("README blocks not updated: %s", e)
    return {"summary": str(root / "summary.csv"), "rows": int(len(summ)), "figures": [str(p) for p in figs.values()], "report": str(report), "interpretation": interp_path,
            "readme_blocks": readme}
