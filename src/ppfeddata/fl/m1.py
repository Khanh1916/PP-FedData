"""Phase 9: M1 = CVAE + FedAvg + client-side DP-SGD, for epsilon in `dp.epsilons`, plus the DP sanity runs and the report.

Every trained model is evaluated twice:
- `<name>-plain`: samples the plain decoder. Every number depends only on the DP-trained weights, so the epsilon covers it
  (post-processing). This is the HEADLINE variant of the utility-privacy curve.
- `<name>`: adds the per-class residual noise of SPEC_DEVIATIONS 7.1. Those 78 noise scales are computed from the pooled
  train data and are NOT covered by epsilon (SPEC_DEVIATIONS 8.8), so this variant is shown only as a comparison.
B3 is also re-evaluated as `B3-plain` so that the epsilon = infinity point of the curve uses the same generator.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ppfeddata.eval.baselines import _table, load_data
from ppfeddata.eval.runs import RunLedger, artifacts_dir, load_predictions, run_id, runs_csv_path
from ppfeddata.eval.stats import paired_bootstrap_diff
from ppfeddata.fl import core
from ppfeddata.fl.b3 import fl_extra, load_fl_model
from ppfeddata.fl.run import run_fl
from ppfeddata.models.b2 import PROTOCOLS, evaluate_generator, method_name

logger = logging.getLogger("ppfeddata.fl.m1")


def m1_name(eps: float, family: str = "M1") -> str:
    """`M1` = hyper-parameters of Phase 7 (tuned without DP); `M1d-t<trial>` = the DP-specific search (tune_dp.py)."""
    return f"{family}-eps{eps:g}"


def tuned_setup(cfg: dict[str, Any]) -> tuple[str, dict[str, Any], float]:
    """(family name, cvae hyper-parameters, clipping bound) of the candidate selected on validation in best_cvae_dp.yaml."""
    import yaml

    from ppfeddata.tune import load_best_cvae
    from ppfeddata.tune_dp import BEST_DP_PATH

    y = yaml.safe_load(Path(BEST_DP_PATH).read_text(encoding="utf-8"))
    if "selected" not in y:
        raise RuntimeError("no selected DP candidate: run `verify-dp` first")
    t = int(y["selected"]["trial"])
    c = next(c for c in y["candidates"] if c["trial"] == t)
    return f"M1d-t{t}", {**load_best_cvae(cfg), **c["cvae"]}, float(c["max_grad_norm"])


def dp_extra(run: dict[str, Any]) -> dict[str, Any]:
    d = run["summary"]["dp"]
    sig = [r["sigma"] for r in d["rows"]]
    return {"dp_target_eps": d["target_eps"], "dp_eps_max": d["eps_max"], "dp_eps_median": d["eps_median"], "dp_delta": d["delta"],
            "dp_sigma_min": min(sig), "dp_sigma_max": max(sig), "dp_clip": run["dp"]["max_grad_norm"],
            "dp_ratio_to_target": d["max_ratio_to_target"], "dp_steps_total": int(sum(r["steps"] for r in d["rows"]))}


def _all_done(ledger: RunLedger, prefixes: list[str], seed: int, mode: str) -> bool:
    return all(ledger.done(run_id(method_name(p, pr, c), seed, mode)) for p in prefixes for pr, c in PROTOCOLS)


def run_m1(cfg: dict[str, Any], eps_list: list[float] | None = None, seeds: list[int] | None = None, resume: bool = True,
           tuned: bool = False) -> None:
    eps_list = [float(e) for e in (cfg["dp"]["epsilons"] if eps_list is None else eps_list)]
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    data, schema = load_data(cfg)
    ledger, mode = RunLedger(runs_csv_path(cfg)), cfg["label_mode"]
    family, kw = "M1", {}
    if tuned:
        family, hp_, clip_ = tuned_setup(cfg)
        kw = {"hp": hp_, "max_grad_norm": clip_}
    for seed in seeds:                                              # the eps = infinity point, same generator as the DP runs
        if not (resume and _all_done(ledger, ["B3-plain"], seed, mode)):
            b3 = run_fl(cfg, seed, "B3", resume=True)
            evaluate_generator(cfg, load_fl_model(cfg, b3, schema), "B3-plain", seed, data, schema, ledger, fl_extra(b3), use_stats=False)
    for eps in eps_list:
        for seed in seeds:
            name = m1_name(eps, family)
            if resume and _all_done(ledger, [name, name + "-plain"], seed, mode):
                logger.info("skip %s seed %d", name, seed)
                continue
            run = run_fl(cfg, seed, name, target_eps=eps, resume=resume, **kw)
            d = run["summary"]["dp"]
            logger.info("%s seed %d: sigma %.3f-%.3f, epsilon max %.3f (target %g), %d rounds, %.0fs", name, seed,
                        min(r["sigma"] for r in d["rows"]), max(r["sigma"] for r in d["rows"]), d["eps_max"], eps,
                        run["summary"]["rounds_done"], run["summary"]["fl_total_s"])
            model, extra = load_fl_model(cfg, run, schema), {**fl_extra(run), **dp_extra(run)}
            evaluate_generator(cfg, model, name + "-plain", seed, data, schema, ledger, extra, use_stats=False)
            evaluate_generator(cfg, model, name, seed, data, schema, ledger, extra, use_stats=True)


# --------------------------------------------------------------------------------------------------
# DP-specific hyper-parameters: check the proxy's best candidates in real federated runs
# --------------------------------------------------------------------------------------------------
def candidate_name(trial: int, eps: float) -> str:
    return f"M1d-t{trial}-eps{eps:g}"


def verify_dp_candidates(cfg: dict[str, Any], trials: list[int], eps: float = 5.0, seed: int = 0, resume: bool = True) -> list[dict[str, Any]]:
    """One real FL run (seed `seed`, target `eps`) per candidate trial of the DP search; both utility and the test split are
    logged in the ledger, but the choice is made on VAL only (see `choose_dp_candidate`)."""
    import yaml

    from ppfeddata.tune_dp import BEST_DP_PATH, study_path

    data, schema = load_data(cfg)
    ledger = RunLedger(runs_csv_path(cfg))
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=f"cvae_dp_eps{eps:g}_{cfg['label_mode']}", storage=f"sqlite:///{study_path(cfg, eps).resolve().as_posix()}")
    from ppfeddata.tune import load_best_cvae
    from ppfeddata.tune_dp import hp_from_params
    base, out = load_best_cvae(cfg), []
    for t in trials:
        hp, clip = hp_from_params(base, study.trials[t].params)
        name = candidate_name(t, eps)
        if resume and _all_done(ledger, [name + "-plain"], seed, cfg["label_mode"]):
            logger.info("skip %s", name)
        else:
            run = run_fl(cfg, seed, name, target_eps=eps, hp=hp, max_grad_norm=clip, resume=resume)
            evaluate_generator(cfg, load_fl_model(cfg, run, schema), name + "-plain", seed, data, schema, ledger,
                               {**fl_extra(run), **dp_extra(run), "dp_trial": t}, use_stats=False)
        out.append({"trial": t, "name": name})
    return out


def choose_dp_candidate(cfg: dict[str, Any], trials: list[int], eps: float = 5.0, seed: int = 0) -> dict[str, Any]:
    """Pick by VAL macro-F1 of TSTR-rf (the search's fitness); ties within 0.01 go to the lower validation ELBO. Writes the
    winner into configs/best_cvae_dp.yaml (`selected`). The test split plays no part."""
    import yaml

    from ppfeddata.tune_dp import BEST_DP_PATH

    df = RunLedger(runs_csv_path(cfg)).frame()
    rows = []
    for t in trials:
        name = candidate_name(t, eps)
        r = df[(df["config"] == f"{name}-plain-TSTR-rf") & (df["seed"] == seed)]
        if len(r):
            rl = [x for x in core.read_round_log(artifacts_dir(cfg) / run_id(name, seed, cfg["label_mode"])) if x.get("val_loss") is not None]
            rows.append({"trial": t, "val_f1": float(r.iloc[0]["val_macro_f1"]), "val_elbo": float(rl[-1]["val_loss"])})
    best_f1 = max(r["val_f1"] for r in rows)
    pool = [r for r in rows if r["val_f1"] >= best_f1 - 0.01]
    win = min(pool, key=lambda r: r["val_elbo"])
    cfgy = yaml.safe_load(Path(BEST_DP_PATH).read_text(encoding="utf-8"))
    cfgy["verification"] = {"seed": seed, "eps": eps, "rule": "max val TSTR-rf macro-F1, ties within 0.01 -> lower val ELBO", "runs": rows}
    cfgy["selected"] = win
    Path(BEST_DP_PATH).write_text(yaml.safe_dump(cfgy, sort_keys=False), encoding="utf-8")
    return {"winner": win, "all": rows}


# --------------------------------------------------------------------------------------------------
# Sanity runs (seed 0)
# --------------------------------------------------------------------------------------------------
def run_dp_sanity(cfg: dict[str, Any], seed: int = 0, resume: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    b3 = run_fl(cfg, seed, "B3", resume=True)
    zero = run_fl(cfg, seed, "M1-sigma0", sigma=0.0, max_grad_norm=1e6, resume=resume)
    out["sigma0"] = {"fl_val_loss": zero["summary"]["final_val_loss"], "b3_val_loss": b3["summary"]["final_val_loss"]}
    out["sigma0"]["rel_diff"] = abs(out["sigma0"]["fl_val_loss"] - out["sigma0"]["b3_val_loss"]) / out["sigma0"]["b3_val_loss"]
    eps = float(cfg["dp"]["epsilons"][len(cfg["dp"]["epsilons"]) // 2])
    full = run_fl(cfg, seed, "M1-r20", rounds=20, target_eps=eps, resume=resume)
    rdir = Path(cfg["compute"]["artifacts_dir"]) / run_id("M1-res", seed, cfg["label_mode"])
    ck = core.load_checkpoint(rdir)
    if not (resume and ck is not None and ck["round"] >= 20 and (rdir / "final_state.pt").exists()):
        import shutil
        shutil.rmtree(rdir, ignore_errors=True)
        run_fl(cfg, seed, "M1-res", rounds=20, stop_after=10, target_eps=eps, resume=False)    # interrupted at round 10 of 20
    res = run_fl(cfg, seed, "M1-res", rounds=20, target_eps=eps, resume=True)
    sf = torch.load(Path(full["artifacts_dir"]) / full["run_id"] / "final_state.pt", weights_only=False)
    sr = torch.load(Path(res["artifacts_dir"]) / res["run_id"] / "final_state.pt", weights_only=False)
    ef, er = full["summary"]["dp"], res["summary"]["dp"]
    out["resume"] = {"eps_target": eps, "max_abs_diff": float(max((sf[k] - sr[k]).abs().max() for k in sf)),
                     "eps_full": ef["eps_max"], "eps_resumed": er["eps_max"],
                     "steps_full": [r["steps"] for r in ef["rows"]], "steps_resumed": [r["steps"] for r in er["rows"]],
                     "log_rows_full": full["summary"]["rounds_done"], "log_rows_resumed": res["summary"]["rounds_done"]}
    (artifacts_dir(cfg) / f"m1_sanity_{cfg['label_mode']}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def _ms(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.4f} ± {x.std():.4f}"


def _rounds_seconds(cfg: dict[str, Any], name: str, seed: int) -> list[float]:
    rl = [r for r in core.read_round_log(artifacts_dir(cfg) / run_id(name, seed, cfg["label_mode"])) if r["round"] > 1]
    return [r["round_seconds"] for r in rl]


def plot_utility_privacy(cfg: dict[str, Any], df, eps_list: list[float], out: str | Path, family: str = "M1") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    xs = list(range(len(eps_list) + 1))
    labels = [f"{e:g}" for e in eps_list] + ["no DP (B3)"]
    for ax, col, title in ((axes[0], "macro_f1", "test macro-F1"), (axes[1], "mia_auc_mean", "MIA AUC (0.5 = no signal)"),
                           (axes[2], "dcr_ratio_mean", "DCR ratio (1 = no copying)")):
        for config, style in (("TSTR-rf", "o-"), ("TSTR-mlp", "s--"), ("TAug-rf", "^-"), ("TAug-mlp", "v--")):
            if col != "macro_f1" and config != "TSTR-rf":
                continue
            m, sd = [], []
            for e in eps_list:
                v = df[df["config"] == f"{m1_name(e, family)}-plain-{config}"][col]
                m.append(v.mean()), sd.append(v.std(ddof=0))
            v = df[df["config"] == f"B3-plain-{config}"][col]
            m.append(v.mean()), sd.append(v.std(ddof=0))
            ax.errorbar(xs, m, yerr=sd, fmt=style, capsize=3, label=config if col == "macro_f1" else "plain decoder")
        ax.set_xticks(xs, labels)
        ax.set_xlabel("epsilon (worst client, packet level)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return Path(out)


def write_m1_report(cfg: dict[str, Any], out: str | Path | None = None, fig_dir: str | Path = "./results/figures",
                    tuned: bool = False) -> dict[str, Any]:
    family = tuned_setup(cfg)[0] if tuned else "M1"
    out = out or ("./results/reports/m1_dp_tuned.md" if tuned else "./results/reports/m1_dp.md")
    m1n = lambda e: m1_name(e, family)                            # noqa: E731  (all lookups below use this family)
    mode, dpc, th = cfg["label_mode"], cfg["dp"], cfg["thresholds"]
    df = RunLedger(runs_csv_path(cfg)).frame()
    df = df[df["label_mode"] == mode]
    data, schema = load_data(cfg)
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    k, nb = len(classes), int(cfg["eval"]["bootstrap"])
    y = data["test"]["y"]
    eps_list = [float(e) for e in dpc["epsilons"]]
    gate: dict[str, Any] = {}
    hp_note = ""
    if tuned:
        _, thp, tclip = tuned_setup(cfg)
        hp_note = (f" **DP-specific hyper-parameters** (`configs/best_cvae_dp.yaml`, trial {family.split('-t')[1]}, searched at eps 5 on a single-client proxy and "
                   f"confirmed in real FL runs on validation): latent {thp['latent_dim']}, hidden {thp['hidden']}, beta {thp['beta']:.3f}, lr {thp['lr']:.2e}, "
                   f"batch {thp['batch_size']}, clipping bound C = {tclip:.2f}.")
    L = [f"# Phase 9 - M1: CVAE + FedAvg + client-side DP-SGD ({mode}){' - DP-tuned hyper-parameters' if tuned else ''}", "",
         f"Auto-generated by `ppfeddata m1`. Opacus {_opacus_version()} (Poisson sampling, flat per-sample clipping C = {dpc['max_grad_norm']}, RDP accountant), "
         f"delta = {dpc['delta']}, {cfg['fl']['num_clients']} non-IID clients (alpha {cfg['fl']['dirichlet_alpha']}), {cfg['fl']['rounds']} rounds x {cfg['fl']['local_epochs']} local epochs, "
         f"same partition and seeds as B3. sigma_i is calibrated once per client for the whole run; epsilon is recomputed from the counted steps." + hp_note, "",
         "## What epsilon means here", "",
         "- **Record level (one packet), for the model weights**, against an observer who sees the updates of that one client. The configuration's epsilon is the **maximum over clients** (worst case); the median is shown next to it. No privacy amplification from secure aggregation is claimed.",
         "- **Packets of one TCP stream are strongly correlated.** By group privacy, protection of a whole attack session (many packets) is much weaker than the number suggests; no exact conversion is claimed. The partition is by row, so a stream can sit in several clients.",
         "- **The class label (the conditioning input) is not protected by DP-SGD.**",
         "- **Headline variant = `-plain`** (plain decoder): depends only on the DP-trained weights. The variant without the suffix adds per-class residual noise whose 78 scales are computed from the pooled train data and are **not** covered by epsilon; it is shown for comparison only (SPEC_DEVIATIONS 7.1, 8.8).",
         "- Hyper-parameters (`configs/best_cvae.yaml`) were tuned without DP on centralised data; they were not re-tuned for DP. The clipping bound is the config default, not tuned.", ""]

    rows = []
    for e in eps_list:
        for sd in sorted(int(s) for s in df[df["config"] == f"{m1n(e)}-plain-TSTR-rf"]["seed"].unique()):
            r = df[(df["config"] == f"{m1n(e)}-plain-TSTR-rf") & (df["seed"] == sd)].iloc[0]
            rows.append({"target eps": f"{e:g}", "seed": sd, "sigma (min-max over clients)": f"{r['dp_sigma_min']:.3f} - {r['dp_sigma_max']:.3f}",
                         "eps max": f"{r['dp_eps_max']:.3f}", "eps median": f"{r['dp_eps_median']:.3f}", "eps max / target": f"{r['dp_ratio_to_target']:.3f}",
                         "DP steps (all clients)": int(r["dp_steps_total"])})
    L += ["## Epsilon achieved (gate: eps_i <= 1.02 x target for every client)", "", _table(rows), ""]
    if rows:
        gate["epsilon_within_2pct"] = bool(all(float(r["eps max / target"]) <= 1.02 for r in rows))
        sg = [df[df["config"] == f"{m1n(e)}-plain-TSTR-rf"]["dp_sigma_max"].mean() for e in sorted(eps_list)]
        gate["sigma_increases_as_eps_decreases"] = bool(all(a > b for a, b in zip(sg, sg[1:])))      # eps ascending -> sigma descending
        # per-client table of the first configuration / seed
        e0, s0 = eps_list[len(eps_list) // 2], 0
        p = artifacts_dir(cfg) / run_id(m1n(e0), s0, mode) / "spec.json"
        if p.exists():
            from ppfeddata.fl import dp_utils
            from ppfeddata.fl.run import summarize
            spec = json.loads(p.read_text(encoding="utf-8"))
            t = summarize(spec)["dp"]
            L += [f"Per client, target eps {e0:g}, seed {s0} (clients differ in size, so sigma, q and eps_i differ):", "",
                  _table([{"client": r["client"], "rows": r["n"], "q": f"{r['q']:.4f}", "sigma": f"{r['sigma']:.3f}", "steps": r["steps"],
                           "eps_i": f"{r['epsilon']:.3f}"} for r in t["rows"]]), ""]

    configs = [f"B3-plain-{p}-{c}" for p, c in PROTOCOLS]
    allm = [f"{m1n(e)}-plain-{p}-{c}" for e in eps_list for p, c in PROTOCOLS]
    ref = [n for n in ("B0-rf", "B0-mlp", "B1b-rf", "B1b-mlp") if (df["config"] == n).any()]
    rows = []
    for n in ref + [c for c in configs + allm if (df["config"] == c).any()]:
        s = df[df["config"] == n].sort_values("seed")
        rows.append({"config": n, "runs": len(s), "macro-F1 (test)": _ms(s["macro_f1"]), "balanced acc": _ms(s["balanced_acc"]), "PR-AUC macro": _ms(s["pr_auc_macro"])})
    L += ["## Utility, headline (plain decoder; mean ± std over seeds)", "", _table(rows), ""]
    rows = [{"config": n, **{c: f"{df[df['config'] == n][f'recall_{c}'].mean():.3f}" for c in classes}}
            for n in ref + [c for c in configs + allm if (df["config"] == c).any()] if (df["config"] == n).any()]
    L += ["### Recall per class (test)", "", _table(rows), ""]

    # trend gate: TSTR macro-F1 should not increase as eps decreases
    trend = {}
    for clf in ("rf", "mlp"):
        vals = [df[df["config"] == f"{m1n(e)}-plain-TSTR-{clf}"]["macro_f1"].mean() for e in sorted(eps_list)]
        vals.append(df[df["config"] == f"B3-plain-TSTR-{clf}"]["macro_f1"].mean())
        trend[clf] = vals
    gate["tstr_f1_trend_monotone"] = bool(all(all(b >= a - 0.005 for a, b in zip(v, v[1:])) for v in trend.values()))
    L += ["### Trend with epsilon (TSTR macro-F1, plain decoder; eps " + ", ".join(f"{e:g}" for e in sorted(eps_list)) + ", then no DP)", "",
          "; ".join(f"{clf}: " + " -> ".join(f"{v:.4f}" for v in vals) for clf, vals in trend.items()) + ". "
          + ("Non-decreasing as the budget grows (within 0.005), as expected." if gate["tstr_f1_trend_monotone"] else
             "**Not monotone: all three DP levels sit at the same low level, about half of the non-DP value, and their differences are inside the seed std "
             "(up to 0.05). The expected trend is not visible in TSTR macro-F1 because it is already at a floor at eps = 10; the validation loss below does show it.**"), ""]

    # continuous quality measure: validation ELBO of the final global model (server-side simulation diagnostic)
    vrows, vmean = [], {}
    for e in sorted(eps_list) + [None]:
        pre = "B3" if e is None else m1n(e)
        vs = []
        for sd in cfg["seeds"]:
            rl = [r for r in core.read_round_log(artifacts_dir(cfg) / run_id(pre, sd, mode)) if r.get("val_loss") is not None]
            if rl:
                vs.append(rl[-1])
        if vs:
            vmean[e] = float(np.mean([v["val_loss"] for v in vs]))
            vrows.append({"eps": "no DP (B3)" if e is None else f"{e:g}", "val loss (per seed)": ", ".join(f"{v['val_loss']:.2f}" for v in vs),
                          "mean": f"{vmean[e]:.2f}", "numeric": f"{np.mean([v['val_recon_num'] for v in vs]):.2f}",
                          "binary": f"{np.mean([v['val_recon_bin'] for v in vs]):.2f}", "categorical": f"{np.mean([v['val_recon_cat'] for v in vs]):.2f}",
                          "KL": f"{np.mean([v['val_kl'] for v in vs]):.2f}"})
    order = [vmean[e] for e in sorted(eps_list) if e in vmean]
    gate["val_loss_monotone_in_eps"] = bool(len(order) == len(eps_list) and all(a >= b for a, b in zip(order, order[1:])) and vmean.get(None, 0) < min(order or [0]))
    L += ["## Validation loss (ELBO at the final beta) of the final global model", "",
          "Less noisy than macro-F1: it uses all 12,000 validation rows and no classifier. Lower is better; a simulation-side diagnostic, not available in a real federation.", "",
          _table(vrows), "",
          ("The loss grows as the budget shrinks and every DP level is far above the non-DP model: " if gate["val_loss_monotone_in_eps"] else "The loss does not grow monotonically as the budget shrinks: ")
          + "the DP-trained models are much worse generators than B3, and the drop from eps = 10 to eps = 1 is visible here even though TSTR macro-F1 (above) is already at its floor at eps = 10.", ""]
    L += ["## Paired difference vs B3-plain (test macro-F1; stratified bootstrap)", "",
          "Criterion: paired CI excludes 0 and |diff| > std of macro-F1 over seeds.", ""]
    rows = []
    for e in eps_list:
        for p, c in PROTOCOLS:
            n, base = f"{m1n(e)}-plain-{p}-{c}", f"B3-plain-{p}-{c}"
            for sd in sorted(int(s) for s in df[df["config"] == n]["seed"].unique()):
                ra, rb = run_id(n, sd, mode), run_id(base, sd, mode)
                if not ((df["run_id"] == ra).any() and (df["run_id"] == rb).any()):
                    continue
                d = paired_bootstrap_diff(y, load_predictions(cfg, ra)["y_pred"], load_predictions(cfg, rb)["y_pred"], k, nb, seed=sd)
                std = float(df[df["config"] == n]["macro_f1"].std(ddof=0))
                rows.append({"comparison": f"{n} - {base}", "seed": sd, "diff": f"{d['diff']:+.4f}", "95% CI": f"[{d['lo']:+.4f}, {d['hi']:+.4f}]",
                             "excludes 0": d["excludes_zero"], "|diff| > seed std": abs(d["diff"]) > std})
    L += [_table(rows), ""]

    rows = []
    for e in eps_list + [None]:
        pre = "B3-plain" if e is None else f"{m1n(e)}-plain"
        s = df[df["config"] == f"{pre}-TSTR-rf"].sort_values("seed")
        if len(s):
            rows.append({"eps": "no DP (B3)" if e is None else f"{e:g}", "C2ST AUC": _ms(s["c2st_auc_mean"]), "Wasserstein": _ms(s["wasserstein_mean"]),
                         "JS": _ms(s["js_mean"]), "dup rate": _ms(s["dup_rate"]), "DCR ratio": _ms(s["dcr_ratio_mean"]), "MIA AUC": _ms(s["mia_auc_mean"])})
    L += ["## Fidelity and empirical privacy of the synthetic set (plain decoder, mean over classes)", "",
          "The MIA only detects near-copies of training rows (positive control in b2_cvae.md: AUC 0.98 for exact copies, 0.67 at 0.05 sd of noise, 0.53 at 0.2 sd, and an over-fitted CVAE was not detected), so values near 0.5 mean 'no copying', not 'no leakage'. "
          "It is an empirical check next to, not a replacement for, the formal epsilon.", "", _table(rows), ""]
    rows = []
    for e in eps_list + [None]:
        for variant, suf in (("plain decoder", "-plain"), ("with residual noise", "")):
            pre = ("B3" if e is None else m1n(e)) + suf
            s = df[df["config"] == f"{pre}-TSTR-rf"]
            t = df[df["config"] == f"{pre}-TAug-rf"]
            if len(s) and len(t):
                rows.append({"eps": "no DP (B3)" if e is None else f"{e:g}", "generator": variant, "TSTR-rf": _ms(s["macro_f1"]), "TAug-rf": _ms(t["macro_f1"])})
    L += ["## Plain decoder vs residual noise (RF macro-F1, test)", "",
          "The with-noise rows use scales that are not covered by epsilon; the gap shows how much of the utility depends on that extra, non-DP statistic.", "", _table(rows), ""]

    # cost
    rows = []
    base = [x for sd in cfg["seeds"] for x in _rounds_seconds(cfg, "B3", sd)]
    for e in eps_list:
        xs = [x for sd in cfg["seeds"] for x in _rounds_seconds(cfg, m1n(e), sd)]
        if xs and base:
            rows.append({"eps": f"{e:g}", "median s/round (rounds 2+)": f"{np.median(xs):.2f}", "B3 median s/round": f"{np.median(base):.2f}",
                         "overhead (DP / plain)": f"{np.median(xs) / np.median(base):.2f}x"})
    L += ["## Cost (wall time per round, 5 clients in parallel on one machine)", "", _table(rows), ""]

    sp = artifacts_dir(cfg) / f"m1_sanity_{mode}.json"
    if sp.exists() and not tuned:
        s = json.loads(sp.read_text(encoding="utf-8"))
        z, r = s["sigma0"], s["resume"]
        gate["sigma0_matches_b3"] = bool(z["rel_diff"] < 0.05)
        gate["resume_identical"] = bool(r["max_abs_diff"] < 1e-5 and r["steps_full"] == r["steps_resumed"] and abs(r["eps_full"] - r["eps_resumed"]) < 1e-9
                                        and r["log_rows_full"] == r["log_rows_resumed"] == 20)
        L += ["## Sanity checks (seed 0)", "",
              f"- **sigma = 0, C = 1e6 vs B3:** validation loss {z['fl_val_loss']:.4f} vs {z['b3_val_loss']:.4f}, relative difference {z['rel_diff'] * 100:.2f} % (gate < 5 %): {'OK' if gate['sigma0_matches_b3'] else 'FAIL'}.",
              f"- **Resume (target eps {r['eps_target']:g}):** stopped at round 10 of a 20-round plan, then resumed: max |weight difference| {r['max_abs_diff']:.2e}; "
              f"DP step counters equal: {r['steps_full'] == r['steps_resumed']}; eps max {r['eps_resumed']:.4f} vs {r['eps_full']:.4f} for the uninterrupted run "
              f"({'OK' if gate['resume_identical'] else 'FAIL'}).", ""]
    if tuned:
        rows = []
        for e in sorted(eps_list):
            for label, fam in (("Phase 7 hyper-parameters (no DP tuning)", "M1"), (f"DP-tuned ({family})", family)):
                s_ = df[df["config"] == f"{m1_name(e, fam)}-plain-TSTR-rf"].sort_values("seed")
                if not len(s_):
                    continue
                ve = []
                for sd in s_["seed"]:
                    rl = [x for x in core.read_round_log(artifacts_dir(cfg) / run_id(m1_name(e, fam), int(sd), mode)) if x.get("val_loss") is not None]
                    ve.append(rl[-1]["val_loss"])
                rows.append({"eps": f"{e:g}", "hyper-parameters": label, "seeds": len(s_), "val TSTR-rf F1": _ms(s_["val_macro_f1"]),
                             "test TSTR-rf F1": _ms(s_["macro_f1"]), "val ELBO": f"{np.mean(ve):.2f}"})
        L += ["## Effect of the DP-specific tuning (plain decoder, RF TSTR)", "",
              "Same budgets, partitions and seeds; only the CVAE hyper-parameters and the clipping bound differ. The tuning used validation data only (the test split plays no part in it).", "",
              _table(rows), ""]
    if rows is not None and eps_list:
        fig = plot_utility_privacy(cfg, df, sorted(eps_list), Path(fig_dir) / ("utility_privacy_m1_tuned.png" if tuned else "utility_privacy_m1.png"), family)
        L += [f"![utility-privacy]({Path('../figures') / fig.name})", ""]
    L += ["## Gate Phase 9 checks", ""] + [f"- {k}: {'OK' if v else 'FAIL'}" for k, v in gate.items()] + [""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(L), encoding="utf-8")
    return gate


def _opacus_version() -> str:
    import opacus
    return opacus.__version__
