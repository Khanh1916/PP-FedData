"""Phase 8: B3 = CVAE trained with FedAvg over non-IID clients (no DP, no SecAgg), evaluated like B2, plus the gate sanity runs.

Sanity runs (seed 0, real data):
- `B3-k1`   : one client holding every row, vs the centralised B2 model (validation loss within 5 %)
- `B3-a100` : near-IID Dirichlet(100) vs the non-IID default, for the convergence curves
- `B3-r20` / `B3-res`: 20 rounds straight vs 10 rounds then resume to 20 (identical weights, log and step counters)
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
from ppfeddata.fl.run import run_fl
from ppfeddata.models.b2 import PROTOCOLS, evaluate_generator, method_name
from ppfeddata.models.cvae import build_layout
from ppfeddata.partition import label_table, load_partition, partition_path, plot_heatmap

logger = logging.getLogger("ppfeddata.fl.b3")


def load_fl_model(cfg: dict[str, Any], run: dict[str, Any], schema: dict[str, Any]):
    state = torch.load(Path(run["artifacts_dir"]) / run["run_id"] / "final_state.pt", weights_only=False)
    m = core.make_model(build_layout(schema), len(schema["label_map"]), run["hp"])
    m.load_state_dict(state)
    m.eval()
    return m


def fl_extra(run: dict[str, Any]) -> dict[str, Any]:
    s = run["summary"]
    return {"alpha": run["alpha"], "num_clients": run["num_clients"], "rounds": run["rounds"], "local_epochs": run["local_epochs"],
            "fl_total_s": s["fl_total_s"], "mean_round_s": s["mean_round_s"], "bytes_per_round": s["bytes_per_round"],
            "fl_server_peak_rss_gb": s["server_peak_rss_gb"], "fl_final_val_loss": s["final_val_loss"],
            "client_seconds_mean": s["client_seconds_mean"], "cvae_train_s": s["fl_total_s"],
            "cvae_epochs": run["rounds"] * run["local_epochs"], "cvae_val_loss": s["final_val_loss"]}


def run_b3(cfg: dict[str, Any], seeds: list[int] | None = None, resume: bool = True) -> None:
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    data, schema = load_data(cfg)
    ledger = RunLedger(runs_csv_path(cfg))
    mode = cfg["label_mode"]
    for seed in seeds:
        if resume and all(ledger.done(run_id(method_name("B3", p, c), seed, mode)) for p, c in PROTOCOLS):
            logger.info("skip B3 seed %d (all runs in the ledger)", seed)
            continue
        run = run_fl(cfg, seed, "B3", resume=resume)
        logger.info("B3 seed %d: %d rounds, %.0fs of rounds, final val loss %.4f", seed, run["summary"]["rounds_done"],
                    run["summary"]["fl_total_s"], run["summary"]["final_val_loss"])
        evaluate_generator(cfg, load_fl_model(cfg, run, schema), "B3", seed, data, schema, ledger, fl_extra(run))


# --------------------------------------------------------------------------------------------------
# Gate sanity runs
# --------------------------------------------------------------------------------------------------
def b2_val_loss(cfg: dict[str, Any], schema, data, seed: int = 0) -> float:
    from ppfeddata.models.cvae import CVAE
    from ppfeddata.models.train import eval_loss
    from ppfeddata.models.cvae import numpy_to_tensor

    ck = torch.load(artifacts_dir(cfg) / run_id("B2", seed, cfg["label_mode"]) / "model.pt", weights_only=False)
    c = ck["config"]
    m = CVAE(build_layout(schema), c["n_classes"], c["latent_dim"], tuple(c["hidden"]), c["layernorm"])
    m.load_state_dict(ck["state_dict"])
    return float(eval_loss(m, numpy_to_tensor(data["val"]["X"]), torch.as_tensor(data["val"]["y"], dtype=torch.long), float(ck["hp"]["beta"]))["loss"])


def effective_steps(sizes: list[int], batch_size: int, local_epochs: int) -> float:
    """Optimiser steps per round seen by the averaged model: sum_i w_i * steps_i with w_i = n_i / N (FedAvg weights)."""
    n = np.asarray(sizes, dtype=float)
    return float((n / n.sum() * np.ceil(n / batch_size) * local_epochs).sum())


def run_sanity(cfg: dict[str, Any], seed: int = 0, resume: bool = True) -> dict[str, Any]:
    data, schema = load_data(cfg)
    out: dict[str, Any] = {}
    one = run_fl(cfg, seed, "B3-k1", num_clients=1, rounds=int(cfg["fl"]["rounds"]), local_epochs=1, resume=resume)
    out["single_client"] = {"fl_val_loss": one["summary"]["final_val_loss"], "b2_val_loss": b2_val_loss(cfg, schema, data, seed)}
    out["single_client"]["rel_diff"] = abs(out["single_client"]["fl_val_loss"] - out["single_client"]["b2_val_loss"]) / out["single_client"]["b2_val_loss"]
    iid = run_fl(cfg, seed, "B3-a100", alpha=100.0, resume=resume)
    non = run_fl(cfg, seed, "B3", resume=resume)
    bs = int(non["hp"]["batch_size"])
    out["iid"] = {**iid["summary"], "effective_steps": effective_steps(iid["partition_sizes"], bs, iid["local_epochs"])}
    out["non_iid"] = {**non["summary"], "effective_steps": effective_steps(non["partition_sizes"], bs, non["local_epochs"])}
    # IID control with as many effective steps per round as the non-IID run (larger clients do more steps and weigh more)
    e = int(np.ceil(out["non_iid"]["effective_steps"] / max(effective_steps(iid["partition_sizes"], bs, 1), 1e-9)))
    matched = run_fl(cfg, seed, "B3-a100-matched", alpha=100.0, local_epochs=e, resume=resume)
    out["iid_matched"] = {**matched["summary"], "local_epochs": e,
                          "effective_steps": effective_steps(matched["partition_sizes"], bs, e)}
    full = run_fl(cfg, seed, "B3-r20", rounds=20, resume=resume)
    # interrupted run: first 10 rounds, then a second call with 20 rounds resumes from the round-10 checkpoint
    rdir = Path(cfg["compute"]["artifacts_dir"]) / run_id("B3-res", seed, cfg["label_mode"])
    ck = core.load_checkpoint(rdir)
    if not (resume and ck is not None and ck["round"] >= 20 and (rdir / "final_state.pt").exists()):
        import shutil
        shutil.rmtree(rdir, ignore_errors=True)
        run_fl(cfg, seed, "B3-res", rounds=10, resume=False)
    res = run_fl(cfg, seed, "B3-res", rounds=20, resume=True)
    s_full = torch.load(Path(full["artifacts_dir"]) / full["run_id"] / "final_state.pt", weights_only=False)
    s_res = torch.load(Path(res["artifacts_dir"]) / res["run_id"] / "final_state.pt", weights_only=False)
    out["resume"] = {"max_abs_diff": float(max((s_full[k] - s_res[k]).abs().max() for k in s_full)),
                     "log_rows_full": len([r for r in core.read_round_log(Path(full["artifacts_dir"]) / full["run_id"]) if r["round"] > 0]),
                     "log_rows_resumed": len([r for r in core.read_round_log(Path(res["artifacts_dir"]) / res["run_id"]) if r["round"] > 0]),
                     "steps_full": full["summary"]["client_steps"], "steps_resumed": res["summary"]["client_steps"]}
    (artifacts_dir(cfg) / f"b3_sanity_{cfg['label_mode']}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def _curve(cfg: dict[str, Any], name: str, seed: int = 0):
    rows = core.read_round_log(artifacts_dir(cfg) / run_id(name, seed, cfg["label_mode"]))
    pts = [(r["round"], r["val_loss"]) for r in rows if r.get("val_loss") is not None]
    return np.array(pts, dtype=float).reshape(-1, 2)


def partition_checks(cfg: dict[str, Any], seed: int, alpha: float, n_rows: int) -> dict[str, Any]:
    parts, meta = load_partition(partition_path(cfg, alpha, seed))
    cat = np.concatenate(parts)
    return {"disjoint": bool(len(np.unique(cat)) == len(cat)), "covers_train_pool": bool(len(cat) == n_rows and set(cat.tolist()) == set(range(n_rows))),
            "min_size": int(min(len(p) for p in parts)), "min_required": meta["min_client_size"], "draws": meta["draws"], "sizes": meta["sizes"]}


def plot_convergence(cfg: dict[str, Any], out: str | Path, extra_curves: dict[str, np.ndarray] | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name, label in (("B3", f"non-IID (alpha={cfg['fl']['dirichlet_alpha']})"), ("B3-a100", "near-IID (alpha=100)")):
        c = _curve(cfg, name)
        if len(c):
            ax.plot(c[:, 0], c[:, 1], marker="o", ms=3, label=label)
    for label, c in (extra_curves or {}).items():
        ax.plot(c[:, 0], c[:, 1], ls="--", label=label)
    ax.set_xlabel("round")
    ax.set_ylabel("validation loss (ELBO, final beta)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return Path(out)


def write_b3_report(cfg: dict[str, Any], out: str | Path = "./results/reports/b3_fl.md",
                    fig_dir: str | Path = "./results/figures") -> dict[str, Any]:
    mode, th, fl = cfg["label_mode"], cfg["thresholds"], cfg["fl"]
    df = RunLedger(runs_csv_path(cfg)).frame()
    df = df[df["label_mode"] == mode]
    data, schema = load_data(cfg)
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    k, nb = len(classes), int(cfg["eval"]["bootstrap"])
    y = data["test"]["y"]
    b3 = [method_name("B3", p, c) for p, c in PROTOCOLS if (df["config"] == method_name("B3", p, c)).any()]
    ref = [n for n in ("B0-rf", "B0-mlp", "B1a-rf", "B1b-rf", "B1b-mlp") if (df["config"] == n).any()]
    b2 = [method_name("B2", p, c) for p, c in PROTOCOLS if (df["config"] == method_name("B2", p, c)).any()]
    seeds = sorted(int(s) for s in df[df["config"].isin(b3)]["seed"].unique())
    gate: dict[str, Any] = {}
    L = [f"# Phase 8 - B3 CVAE + FedAvg, non-IID ({mode})", "",
         f"Auto-generated by `ppfeddata b3`. Flower {_flwr_version()} simulation on one machine (Ray backend), {fl['num_clients']} clients, "
         f"Dirichlet alpha = {fl['dirichlet_alpha']}, {fl['rounds']} rounds x {fl['local_epochs']} local epochs, FedAvg weighted by local rows, "
         "fresh Adam optimiser each round, KL warm-up over the global epoch count. Evaluation on the real test split exactly as B2. "
         f"Seeds {seeds}. No DP, no secure aggregation.", ""]

    rows = []
    for n in ref + b2 + b3:
        s = df[df["config"] == n].sort_values("seed")
        rows.append({"config": n, "runs": len(s), "macro-F1 (test)": _ms(s["macro_f1"]), "balanced acc": _ms(s["balanced_acc"]),
                     "PR-AUC macro": _ms(s["pr_auc_macro"])})
    L += ["## Utility (mean ± std over seeds)", "", _table(rows), ""]
    rows = [{"config": n, **{c: f"{df[df['config'] == n][f'recall_{c}'].mean():.3f}" for c in classes}} for n in ref + b2 + b3]
    L += ["## Recall per class (test)", "", _table(rows), ""]

    L += ["## Paired difference of B3 (test macro-F1, stratified bootstrap)", "",
          "Criterion: paired CI excludes 0 and |diff| > std of macro-F1 over seeds. B3 - B2 is the cost of federating the generator.", ""]
    rows = []
    for n in b3:
        clf = n.rsplit("-", 1)[1]
        proto = n.split("-")[1]
        for base in (method_name("B2", proto, clf), f"B0-{clf}", f"B1b-{clf}"):
            if base not in set(ref) | set(b2):
                continue
            for sd in seeds:
                ra, rb = run_id(n, sd, mode), run_id(base, sd, mode)
                if not ((df["run_id"] == ra).any() and (df["run_id"] == rb).any()):
                    continue
                d = paired_bootstrap_diff(y, load_predictions(cfg, ra)["y_pred"], load_predictions(cfg, rb)["y_pred"], k, nb, seed=sd)
                sd_std = float(df[df["config"] == n]["macro_f1"].std(ddof=0))
                rows.append({"comparison": f"{n} - {base}", "seed": sd, "diff": f"{d['diff']:+.4f}", "95% CI": f"[{d['lo']:+.4f}, {d['hi']:+.4f}]",
                             "excludes 0": d["excludes_zero"], "|diff| > seed std": abs(d["diff"]) > sd_std})
    L += [_table(rows), ""]

    rows = []
    for _, r in (df[df["config"] == b3[0]].sort_values("seed") if b3 else df.iloc[0:0]).iterrows():
        rows.append({"seed": int(r["seed"]), "C2ST AUC": f"{r['c2st_auc_mean']:.3f}", "Wasserstein": f"{r['wasserstein_mean']:.3f}",
                     "JS": f"{r['js_mean']:.3f}", "corr dist": f"{r['corr_dist_mean']:.3f}", "dup rate": f"{r['dup_rate']:.4f}",
                     "DCR ratio": f"{r['dcr_ratio_mean']:.3f}", "MIA AUC": f"{r['mia_auc_mean']:.3f}"})
    L += ["## Fidelity and privacy of the synthetic set (mean over classes)", "", _table(rows), ""]
    b2f = df[df["config"] == (b2[0] if b2 else "")]
    if len(b2f):
        L += [f"B2 for comparison (mean over seeds): C2ST {b2f['c2st_auc_mean'].mean():.3f}, Wasserstein {b2f['wasserstein_mean'].mean():.3f}, "
              f"DCR ratio {b2f['dcr_ratio_mean'].mean():.3f}, MIA AUC {b2f['mia_auc_mean'].mean():.3f}.", ""]

    rows = []
    for sd in seeds:
        rl = [r for r in core.read_round_log(artifacts_dir(cfg) / run_id("B3", sd, mode)) if r["round"] > 0]
        if not rl:
            continue
        secs = [r["round_seconds"] for r in rl]
        vl = [r["val_loss"] for r in rl if r.get("val_loss") is not None]
        rows.append({"seed": sd, "rounds": len(rl), "round 1 (incl. Ray start-up)": f"{secs[0]:.1f} s", "median s/round (rounds 2+)": f"{np.median(secs[1:]):.2f}",
                     "total FL time": f"{sum(secs):.0f} s", "mean client s/round": f"{np.mean([x for r in rl for x in r['client_seconds']]):.2f}",
                     "bytes/round": f"{int(rl[-1]['bytes']):,}", "server peak RAM GB": f"{max(r['peak_rss_gb'] for r in rl):.2f}",
                     "final val loss": f"{vl[-1]:.4f}"})
    L += ["## Cost of the federated run", "",
          "Bytes/round = model size x clients x 2 (download + upload), exact for the payload; the single-machine simulation measures compute but not network latency; "
          "peak RAM is the server process only (the Ray client workers are separate processes and are not included). Round 1 includes starting the Ray actors, "
          "so the median of the later rounds is the number to use for extrapolation.", "", _table(rows), ""]

    # partition checks + heatmap (seed 0)
    n_rows = len(data["train"]["y"])
    if seeds:
        pc = partition_checks(cfg, seeds[0], float(fl["dirichlet_alpha"]), n_rows)
        gate["partition"] = bool(pc["disjoint"] and pc["covers_train_pool"] and pc["min_size"] >= pc["min_required"])
        parts, _ = load_partition(partition_path(cfg, float(fl["dirichlet_alpha"]), seeds[0]))
        table = label_table(data["train"]["y"], parts, classes)
        fig = plot_heatmap(table, Path(fig_dir) / f"partition_alpha{fl['dirichlet_alpha']:g}_seed{seeds[0]}.png",
                           f"Dirichlet alpha={fl['dirichlet_alpha']}, seed {seeds[0]}: rows per client and class")
        L += ["## Partition (gate)", "",
              f"Seed {seeds[0]}, alpha {fl['dirichlet_alpha']}: client sizes {pc['sizes']} (minimum required {pc['min_required']}, {pc['draws']} Dirichlet draw(s)); "
              f"disjoint: {pc['disjoint']}; union = whole train pool ({n_rows} rows): {pc['covers_train_pool']}.", "",
              _table([{c: int(v) if c != "client" else int(v) for c, v in r.items()} for r in table.to_dict("records")]), "", f"![partition]({(Path('../figures') / fig.name).as_posix()})", ""]

    sp = artifacts_dir(cfg) / f"b3_sanity_{mode}.json"
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        sc, rs = s["single_client"], s["resume"]
        gate["single_client_within_5pct"] = bool(sc["rel_diff"] < 0.05)
        gate["resume_identical"] = bool(rs["max_abs_diff"] < 1e-5 and rs["log_rows_full"] == rs["log_rows_resumed"] == 20 and rs["steps_full"] == rs["steps_resumed"])
        curve_iid, curve_non = _curve(cfg, "B3-a100"), _curve(cfg, "B3")
        L += ["## Sanity checks (seed 0)", "",
              f"- **One client vs centralised B2:** validation loss {sc['fl_val_loss']:.4f} (FL, 1 client, {fl['rounds']} rounds x 1 epoch) vs {sc['b2_val_loss']:.4f} (B2); "
              f"relative difference {sc['rel_diff'] * 100:.2f} % (gate < 5 %): {'OK' if gate['single_client_within_5pct'] else 'FAIL'}.",
              f"- **Resume:** 10 rounds then resume to 20 vs 20 rounds straight: max |weight difference| {rs['max_abs_diff']:.2e}; log rows {rs['log_rows_resumed']} vs {rs['log_rows_full']}; "
              f"client step counters equal: {rs['steps_full'] == rs['steps_resumed']} ({'OK' if gate['resume_identical'] else 'FAIL'}).",
              ]
        if len(curve_iid) and len(curve_non):
            fi, fn = curve_iid[-1, 1], curve_non[-1, 1]
            gate["noniid_not_better"] = bool(fn >= fi * 0.999)
            L += [f"- **IID vs non-IID:** final validation loss {fi:.4f} (alpha=100) vs {fn:.4f} (alpha={fl['dirichlet_alpha']}); "
                  + ("non-IID is worse or equal, as expected." if gate["noniid_not_better"] else "**non-IID is better than IID, the opposite of the expectation.**")]
            if "iid_matched" in s:
                m = s["iid_matched"]
                gate["noniid_explained"] = bool(gate["noniid_not_better"] or m["final_val_loss"] <= fn * 1.05 or m["final_val_loss"] < fi)
                L += [f"  Explanation tested: with alpha={fl['dirichlet_alpha']} the clients differ a lot in size (see the partition table), a larger client takes more optimiser steps per "
                      f"epoch and also carries more FedAvg weight, so the averaged model gets {s['non_iid']['effective_steps']:.0f} effective steps per round against "
                      f"{s['iid']['effective_steps']:.0f} for near-IID equal-sized clients. An IID control with {m['local_epochs']} local epochs "
                      f"({m['effective_steps']:.0f} effective steps) ends at validation loss {m['final_val_loss']:.4f} (non-IID {fn:.4f}, IID with the standard local epochs {fi:.4f}). "
                      + ("The gap to the standard IID run is explained by the number of steps, not by the label skew." if gate["noniid_explained"] else
                         "The step count does not explain the gap; the cause is open."), ""]
        fig = plot_convergence(cfg, Path(fig_dir) / "fl_convergence.png")
        L += [f"![convergence]({(Path('../figures') / fig.name).as_posix()})", ""]          # forward slashes: GitHub does not read ..\figures
    L += ["## Gate Phase 8 checks", ""] + [f"- {k}: {'OK' if v else 'FAIL'}" for k, v in gate.items()] + [""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(L), encoding="utf-8")
    return gate


def _ms(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.4f} ± {x.std():.4f}"


def _flwr_version() -> str:
    import flwr
    return flwr.__version__
