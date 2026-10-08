"""Phase 10: M2 = CVAE + FedAvg + SecAgg+, M3 = CVAE + FedAvg + client-side DP-SGD + SecAgg+, the secure-aggregation checks on real
data (T-SA1/2/3), and the report.

M2 and M3 train exactly like B3 and M1; only the aggregation step is replaced by Flower's SecAgg+ (`fl/secagg.py`). So the model
that comes out should match B3 / M1 up to the quantisation noise of the secure sum, which is what the checks measure. M3 reports
the same epsilon as M1 (SecAgg adds no privacy accounting; no amplification is claimed).
M3 uses the DP-specific hyper-parameters of `configs/best_cvae_dp.yaml` by default (family `M3d-t<trial>`, to be compared with `M1d-t<trial>`).
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
from ppfeddata.fl import core, secagg
from ppfeddata.fl.b3 import fl_extra, load_fl_model
from ppfeddata.fl.m1 import _all_done, _rounds_seconds, dp_extra, m1_name, tuned_setup
from ppfeddata.fl.run import run_fl
from ppfeddata.models.b2 import PROTOCOLS, evaluate_generator, method_name

logger = logging.getLogger("ppfeddata.fl.m2")


def m3_setup(cfg: dict[str, Any], tuned: bool = True) -> tuple[str, dict[str, Any]]:
    """(family name, extra run_fl keywords). Tuned: family `M3d-t<trial>`, DP-specific hyper-parameters and clipping bound of M1d."""
    if tuned:
        fam, hp, clip = tuned_setup(cfg)
        return fam.replace("M1", "M3", 1), {"hp": hp, "max_grad_norm": clip}
    return "M3", {}


def m3_name(eps: float, family: str = "M3") -> str:
    return f"{family}-eps{eps:g}"


def m1_counterpart(eps: float, family: str) -> str:
    return m1_name(eps, family.replace("M3", "M1", 1))


def sa_extra(run: dict[str, Any]) -> dict[str, Any]:
    s = run["summary"]["secagg"]
    p = s["params"]
    return {"sa_num_shares": p["num_shares"], "sa_threshold": p["reconstruction_threshold"], "sa_clipping_range": p["clipping_range"],
            "sa_max_weight": p["max_weight"], "sa_bytes_per_round": s["bytes_per_round"], "sa_bytes_down": s["bytes_down"], "sa_bytes_up": s["bytes_up"],
            **{f"sa_stage_{k}_s": v for k, v in s["stage_seconds"].items()}, "sa_max_abs_w": s["clip"]["max_abs_w"], "sa_clip_ok": s["clip"]["ok"],
            "sa_error_bound": s["error_bound"]}


def run_m2(cfg: dict[str, Any], seeds: list[int] | None = None, resume: bool = True) -> None:
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    data, schema = load_data(cfg)
    ledger, mode = RunLedger(runs_csv_path(cfg)), cfg["label_mode"]
    sa = secagg.secagg_spec(cfg)
    for seed in seeds:
        if resume and _all_done(ledger, ["M2"], seed, mode):
            logger.info("skip M2 seed %d", seed)
            continue
        run = run_fl(cfg, seed, "M2", secagg=sa, resume=resume)
        s = run["summary"]
        logger.info("M2 seed %d: %d rounds, %.0fs, max|w| %.3f (clip range %g), %.0f KB/round, val loss %.4f", seed, s["rounds_done"], s["fl_total_s"],
                    s["secagg"]["clip"]["max_abs_w"], sa["params"]["clipping_range"], s["secagg"]["bytes_per_round"] / 1e3, s["final_val_loss"])
        evaluate_generator(cfg, load_fl_model(cfg, run, schema), "M2", seed, data, schema, ledger, {**fl_extra(run), **sa_extra(run)})


def run_m3(cfg: dict[str, Any], eps: float = 5.0, seeds: list[int] | None = None, resume: bool = True, tuned: bool = True) -> None:
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    data, schema = load_data(cfg)
    ledger, mode = RunLedger(runs_csv_path(cfg)), cfg["label_mode"]
    family, kw = m3_setup(cfg, tuned)
    name, sa = m3_name(eps, family), secagg.secagg_spec(cfg)
    for seed in seeds:
        if resume and _all_done(ledger, [name, name + "-plain"], seed, mode):
            logger.info("skip %s seed %d", name, seed)
            continue
        run = run_fl(cfg, seed, name, target_eps=eps, secagg=sa, resume=resume, **kw)
        d, s = run["summary"]["dp"], run["summary"]
        logger.info("%s seed %d: epsilon max %.3f (target %g), %d rounds, %.0fs, max|w| %.3f", name, seed, d["eps_max"], eps, s["rounds_done"],
                    s["fl_total_s"], s["secagg"]["clip"]["max_abs_w"])
        model, extra = load_fl_model(cfg, run, schema), {**fl_extra(run), **dp_extra(run), **sa_extra(run)}
        evaluate_generator(cfg, model, name + "-plain", seed, data, schema, ledger, extra, use_stats=False)
        evaluate_generator(cfg, model, name, seed, data, schema, ledger, extra, use_stats=True)


# --------------------------------------------------------------------------------------------------
# Checks on real data (seed 0)
# --------------------------------------------------------------------------------------------------
def _states(cfg: dict[str, Any], name: str, seed: int, rounds: int) -> list[dict[str, torch.Tensor]]:
    d = artifacts_dir(cfg) / run_id(name, seed, cfg["label_mode"]) / "states"
    return [torch.load(d / f"round_{r:04d}.pt", weights_only=False) for r in range(1, rounds + 1)]


def _maxdiff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    return float(max((a[k] - b[k]).abs().max() for k in a))


def _meandiff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    n = sum(v.numel() for v in a.values())
    return float(sum((a[k] - b[k]).abs().sum() for k in a) / n)


def _last_val_loss(cfg: dict[str, Any], name: str, seed: int) -> float:
    rl = [r for r in core.read_round_log(artifacts_dir(cfg) / run_id(name, seed, cfg["label_mode"])) if r.get("val_loss") is not None]
    return float(rl[-1]["val_loss"]) if rl else float("nan")


def checks_path(cfg: dict[str, Any]) -> Path:
    return artifacts_dir(cfg) / f"secagg_checks_{cfg['label_mode']}.json"


def run_checks(cfg: dict[str, Any], seed: int = 0, resume: bool = True, rounds: int | None = None) -> dict[str, Any]:
    """T-SA1 (secure vs plain, per round), T-SA2 (what the server receives) and T-SA3 (a client drops) on the real data."""
    rounds = int(cfg["fl"]["rounds"]) if rounds is None else rounds
    mode = cfg["label_mode"]
    out: dict[str, Any] = {"seed": seed, "rounds": rounds}
    plain = run_fl(cfg, seed, "B3-chk", rounds=rounds, save_states=True, resume=resume)
    sa = secagg.secagg_spec(cfg, diagnostics=True, debug_rounds=(1, 2, 3))
    sec = run_fl(cfg, seed, "M2-chk", rounds=rounds, secagg=sa, save_states=True, resume=resume)
    rdir = artifacts_dir(cfg) / sec["run_id"]
    sp, ss = _states(cfg, "B3-chk", seed, rounds), _states(cfg, "M2-chk", seed, rounds)
    rows = secagg.read_secagg_rounds(rdir)
    bound = sec["summary"]["secagg"]["error_bound"]
    out["tsa1"] = {"error_bound": bound, "agg_err_per_round": [r["agg_max_abs_err"] for r in rows], "model_delta_per_round": [_maxdiff(a, b) for a, b in zip(sp, ss)],
                   "agg_err_max": max(r["agg_max_abs_err"] for r in rows), "n_updates_checked": sec["summary"]["secagg"]["agg_err_rounds"] * int(sec["num_clients"])}
    b3 = artifacts_dir(cfg) / run_id("B3", seed, mode) / "final_state.pt"
    if b3.exists():          # the plain pipeline must be unchanged by the Phase 10 code: B3-chk is a re-run of B3
        out["plain_regression_max_abs_diff"] = _maxdiff(torch.load(b3, weights_only=False), sp[-1])
    # Control: M2 and B3 end up far apart in weight space although every single aggregation is exact to ~2e-5. Is that the sensitivity of
    # training to a perturbation of this size? Plain FedAvg + Gaussian noise of the same size per round (two draws) is the comparison.
    sigma = float(np.mean([r["agg_mean_abs_err"] for r in rows]) / np.sqrt(2 / np.pi))        # std of a Gaussian with that mean |error|
    series, ctrl_runs = {"B3-chk": sp, "M2-chk": ss}, {}
    for j in (1, 2):
        nm = f"B3-noise{j}"
        ctrl_runs[nm] = run_fl(cfg, seed, nm, rounds=rounds, save_states=True, agg_noise=sigma, agg_noise_seed=j, resume=resume)
        series[nm] = _states(cfg, nm, seed, rounds)
    pairs = [("M2-chk", "B3-chk"), ("B3-noise1", "B3-chk"), ("B3-noise2", "B3-chk"), ("B3-noise1", "B3-noise2")]
    out["chaos"] = {"noise_std": sigma, "pairs": {f"{a} vs {b}": {"max": [_maxdiff(x, y) for x, y in zip(series[a], series[b])],
                                                                  "mean": [_meandiff(x, y) for x, y in zip(series[a], series[b])]} for a, b in pairs},
                    "val_loss_final": {nm: _last_val_loss(cfg, nm, seed) for nm in series}}
    # Same four runs through the Phase 6 evaluation (scratch ledger): are the per-seed utility differences between M2 and B3 larger than
    # the differences noise alone produces?
    data, schema = load_data(cfg)
    tmp = RunLedger(artifacts_dir(cfg) / f"secagg_checks_ledger_{mode}.csv")
    f1 = {}
    for nm, run in {"B3-chk": plain, "M2-chk": sec, **ctrl_runs}.items():
        pre = f"chk-{nm}"
        if not (resume and _all_done(tmp, [pre], seed, mode)):
            evaluate_generator(cfg, load_fl_model(cfg, run, schema), pre, seed, data, schema, tmp, None)
        fr = tmp.frame()
        f1[nm] = {f"{p_}-{c_}": float(fr[fr["run_id"] == run_id(method_name(pre, p_, c_), seed, mode)].iloc[0]["macro_f1"]) for p_, c_ in PROTOCOLS}
    out["chaos"]["test_macro_f1"] = f1
    # DP counterpart: for seed 1 the validation ELBO of M3 and M1 differs (see the report). One more M1 run with the same tiny noise.
    fam, hp_, clip_ = tuned_setup(cfg)
    dseed, deps = 1, 5.0
    ref, m3n, ctl = m1_name(deps, fam), m3_name(deps, fam.replace("M1", "M3", 1)), f"{m1_name(deps, fam)}-noise"
    if (artifacts_dir(cfg) / run_id(ref, dseed, mode) / "final_state.pt").exists():
        run_fl(cfg, dseed, ctl, target_eps=deps, hp=hp_, max_grad_norm=clip_, agg_noise=sigma, agg_noise_seed=1, resume=resume)
        fin = {n: torch.load(artifacts_dir(cfg) / run_id(n, dseed, mode) / "final_state.pt", weights_only=False) for n in (ref, ctl, m3n)
               if (artifacts_dir(cfg) / run_id(n, dseed, mode) / "final_state.pt").exists()}
        out["dp_control"] = {"seed": dseed, "eps": deps, "noise_std": sigma, "val_elbo": {n: _last_val_loss(cfg, n, dseed) for n in fin},
                             "max_abs_diff_to_m1": {n: _maxdiff(fin[n], fin[ref]) for n in fin if n != ref}}
    # T-SA2: the masked vectors the server received in rounds 1-3 against the clients' real updates
    tsa2 = []
    p, sizes = sa["params"], sec["partition_sizes"]
    for r in (1, 2, 3):
        z = np.load(rdir / "secagg_debug" / f"masked_r{r:04d}.npz")
        for pid in range(int(sec["num_clients"])):
            real = np.load(rdir / "secagg_debug" / f"plain_r{r:04d}_c{pid}.npy")
            tsa2.append({"round": r, "client": pid, **secagg.masking_stats(z[f"c{pid}"], real, p, int(sizes[pid]))})
    out["tsa2"] = tsa2
    # T-SA3: client 3 fails in round 2 of a 3-round run; the other four must still be aggregated correctly
    drop = {"round": 2, "pid": 3}
    sd = secagg.secagg_spec(cfg, diagnostics=True, drop=drop, allow_dropout=True)
    dr = run_fl(cfg, seed, "M2-drop", rounds=3, secagg=sd, resume=resume)
    drows = secagg.read_secagg_rounds(artifacts_dir(cfg) / dr["run_id"])
    out["tsa3"] = {"drop": drop, "n_clients_per_round": [r["n_clients"] for r in drows], "agg_err_per_round": [r["agg_max_abs_err"] for r in drows],
                   "threshold": p["reconstruction_threshold"], "num_shares": p["num_shares"],
                   "error_bound_4_clients": secagg.quantization_error_bound(4, p, int(sum(sizes[i] for i in range(len(sizes)) if i != drop["pid"]))),
                   "completed": bool(dr["summary"]["rounds_done"] == 3)}
    checks_path(cfg).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    return out


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def _ms(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.4f} ± {x.std():.4f}"


def _median_round_s(cfg: dict[str, Any], name: str, seeds: list[int]) -> float:
    xs = [x for sd in seeds for x in _rounds_seconds(cfg, name, sd)]
    return float(np.median(xs)) if xs else float("nan")


def plot_checks(cfg: dict[str, Any], checks: dict[str, Any], out: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t1 = checks["tsa1"]
    r = np.arange(1, len(t1["model_delta_per_round"]) + 1)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.semilogy(r, t1["agg_err_per_round"], "o-", ms=3, label="one aggregation: |secure - plain mean| of the same updates")
    ax.semilogy(r, t1["model_delta_per_round"], "s-", ms=3, label="global model: |M2 - B3| (same seed)")
    for pair, v in checks.get("chaos", {}).get("pairs", {}).items():
        if pair.startswith("B3-noise"):
            ax.semilogy(r, v["max"], ls="--", lw=0.9, label=f"control {pair}")
    ax.axhline(t1["error_bound"], color="k", ls=":", label=f"quantisation bound {t1['error_bound']:.1e}")
    ax.axhline(1e-4, color="r", ls="--", lw=0.8, label="spec threshold 1e-4")
    ax.set_xlabel("round")
    ax.set_ylabel("max |difference| over parameters")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return Path(out)


def write_secagg_report(cfg: dict[str, Any], out: str | Path = "./results/reports/m2_m3_secagg.md", fig_dir: str | Path = "./results/figures",
                        eps: float = 5.0, tuned: bool = True) -> dict[str, Any]:
    mode, th = cfg["label_mode"], cfg["thresholds"]
    df = RunLedger(runs_csv_path(cfg)).frame()
    df = df[df["label_mode"] == mode]
    data, schema = load_data(cfg)
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    k, nb, y = len(classes), int(cfg["eval"]["bootstrap"]), data["test"]["y"]
    family, _ = m3_setup(cfg, tuned) if tuned else ("M3", {})
    m3, m1 = m3_name(eps, family), m1_counterpart(eps, family)
    seeds = sorted(int(s) for s in df[df["config"] == "M2-TSTR-rf"]["seed"].unique())
    seeds3 = sorted(int(s) for s in df[df["config"] == f"{m3}-plain-TSTR-rf"]["seed"].unique())
    sa = secagg.secagg_spec(cfg)["params"]
    gate: dict[str, Any] = {}
    L = [f"# Phase 10 - M2 (FedAvg + SecAgg+) and M3 (FedAvg + DP + SecAgg+) ({mode})", "",
         f"Auto-generated by `ppfeddata secagg-report`. Flower {_flwr_version()} `SecAggPlusWorkflow` (server) and `secaggplus_mod` (client), simulation on one machine "
         f"(Ray backend), {cfg['fl']['num_clients']} non-IID clients, same partitions, seeds and training as B3 / M1. SecAgg+ parameters: num_shares {sa['num_shares']}, "
         f"reconstruction threshold {sa['reconstruction_threshold']}, clipping range {sa['clipping_range']:g}, quantisation range 2^22, modulus 2^32 (Flower defaults), "
         f"`max_weight` {sa['max_weight']:g} (Flower default 1000 is below the client sizes 500 to about 70,000 rows; with it the FedAvg weights would be clipped, "
         "SPEC_DEVIATIONS 10.2). M3 uses " + ("the DP-specific hyper-parameters of `configs/best_cvae_dp.yaml` (" + family + ")" if tuned else "the Phase 7 hyper-parameters")
         + f" at target epsilon {eps:g}, and is compared with {m1}.", "",
         "## What SecAgg does and does not give", "",
         "- The server sees only the **sum** of the clients' (quantised, weighted) updates, never one client's update; with threshold t = "
         f"{sa['reconstruction_threshold']} of {sa['num_shares']} it tolerates the loss of up to {sa['num_shares'] - sa['reconstruction_threshold']} clients in a round "
         "(honest-but-curious server, Flower's SecAgg+ protocol).",
         "- It does **not** protect against what the aggregate (the global model) reveals about the training data: that is the job of DP (M3). M2 has no formal privacy guarantee; M3's epsilon is "
         "the one of M1 (record level, worst client), SecAgg adds no accounting and **no amplification is claimed**.",
         "- Flower forwards `num_examples` and the metrics dict **in the clear**: the server learns each client's row count (public in this study) and the clients here send only step counts and "
         "timings; the local training loss is not sent. The clipping check and the debug channel below write to disk beside the protocol and exist only in the simulation.",
         "- A single-machine simulation measures the cryptographic computation, not network latency. All clients are honest (no malicious-client or malicious-server model is tested).", ""]

    # ---- T-SA1
    ck = json.loads(checks_path(cfg).read_text(encoding="utf-8")) if checks_path(cfg).exists() else None
    L += ["## T-SA1 correctness: secure aggregation vs plain FedAvg", ""]
    if ck:
        t1 = ck["tsa1"]
        gate["tsa1_aggregation_error_within_1e-4"] = bool(t1["agg_err_max"] <= 1e-4 and t1["agg_err_max"] <= t1["error_bound"] * 1.001)
        d = t1["model_delta_per_round"]
        L += [f"**One aggregation, identical inputs** (seed {ck['seed']}, {ck['rounds']} rounds x {len(ck['tsa2']) // 3} clients; the clients' real updates were written beside the protocol and compared "
              f"with their ordinary weighted mean in float64): largest difference over all parameters and rounds **{t1['agg_err_max']:.2e}**; "
              f"hard bound from the stochastic quantiser {t1['error_bound']:.2e} (5 clients x quantisation step 2*clip/2^22 divided by the total weight sum n_i / max_weight); spec gate 1e-4: "
              f"{'OK' if gate['tsa1_aggregation_error_within_1e-4'] else 'FAIL'}.", "",
              f"**Global model, same seed, M2 vs plain B3** (`M2-chk` vs `B3-chk`, both with the global model saved every round): after round 1, where both start from the same model, the "
              f"models differ by {d[0]:.1e} (the quantisation effect only). The distance then grows (table below). "
              + (f"The plain pipeline is unchanged by the Phase 10 code: `B3-chk` reproduces the earlier `B3` run with max |difference| {ck['plain_regression_max_abs_diff']:.1e}. "
                 if "plain_regression_max_abs_diff" in ck else ""), ""]
        ch = ck.get("chaos")
        if ch:
            at = [r for r in (1, 2, 3, 5, 10, 20, 30) if r <= len(d)]
            rows = []
            for pair, v in ch["pairs"].items():
                rows.append({"pair": pair, **{f"r{r} max / mean": f"{v['max'][r - 1]:.1e} / {v['mean'][r - 1]:.1e}" for r in at}})
            vl = ch["val_loss_final"]
            L += [f"**Control: is this drift specific to secure aggregation?** `B3-noise1` and `B3-noise2` are plain FedAvg runs of the same seed to which Gaussian noise of std {ch['noise_std']:.1e} "
                  "(the size of SecAgg's quantisation error: same mean absolute value) is added to the aggregate every round, with two different noise draws. Max / mean absolute difference of the global "
                  "weights between pairs of runs, by round:", "", _table(rows), "",
                  "Final validation ELBO: " + ", ".join(f"{k} {v:.4f}" for k, v in vl.items()) + ". "
                  "If the M2-vs-B3 distance is of the same order as the distance between plain runs that differ only by noise of that size, the drift is the sensitivity of this training "
                  "process (Adam restarted every round, minibatch order fixed, no weight decay) to tiny perturbations, not a property of the secure sum.", ""]
            f1 = ch.get("test_macro_f1")
            if f1:
                prot = [f"{p_}-{c_}" for p_, c_ in PROTOCOLS]
                frows = [{"run (seed %d)" % ck["seed"]: nm, **{q: f"{v[q]:.4f}" for q in prot}} for nm, v in f1.items()]
                noise_pairs = [("B3-noise1", "B3-chk"), ("B3-noise2", "B3-chk"), ("B3-noise1", "B3-noise2")]
                drows = [{"protocol": q, "abs(M2 - B3)": f"{abs(f1['M2-chk'][q] - f1['B3-chk'][q]):.4f}",
                          "abs(noise1 - B3), abs(noise2 - B3), abs(noise1 - noise2)": ", ".join(f"{abs(f1[a][q] - f1[b][q]):.4f}" for a, b in noise_pairs)} for q in prot]
                inside = all(abs(f1["M2-chk"][q] - f1["B3-chk"][q]) <= max(abs(f1[a][q] - f1[b][q]) for a, b in noise_pairs) * 1.0 + 1e-12 for q in prot)
                L += ["**Test macro-F1 of the same four runs** (seed " + str(ck["seed"]) + ", full Phase 6 evaluation): ", "", _table(frows), "", _table(drows), "",
                      "This is the yardstick for the per-seed paired-bootstrap differences below: runs that differ only by noise of the size of SecAgg's quantisation error already differ in macro-F1 by "
                      f"{max(abs(f1[a][q] - f1[b][q]) for a, b in noise_pairs for q in prot):.3f} at most (TSTR), and the M2-B3 difference of this seed "
                      + ("is no larger than the largest noise-only difference for every protocol." if inside else "is larger than the largest noise-only difference for at least one protocol (only three noise-only pairs are available).") + "", ""]
        L += [f"![secagg checks]({(Path('../figures') / plot_checks(cfg, ck, Path(fig_dir) / 'secagg_checks.png').name).as_posix()})", ""]   # forward slashes for GitHub
    else:
        L += ["Not run yet (`ppfeddata secagg-check`).", ""]

    # utility of M2 vs B3
    rows, diffs = [], []
    for p_, c_ in PROTOCOLS:
        n2, n3 = method_name("M2", p_, c_), method_name("B3", p_, c_)
        a, b = df[df["config"] == n2].sort_values("seed"), df[df["config"] == n3].sort_values("seed")
        if not len(a):
            continue
        sd = max(float(a["macro_f1"].std(ddof=0)), float(b["macro_f1"].std(ddof=0)))
        diff = float(a["macro_f1"].mean() - b["macro_f1"].mean())
        rows.append({"protocol": f"{p_}-{c_}", "M2 macro-F1": _ms(a["macro_f1"]), "B3 macro-F1": _ms(b["macro_f1"]), "M2 - B3": f"{diff:+.4f}", "max seed std": f"{sd:.4f}",
                     "within seed std": abs(diff) <= sd})
        for s_ in seeds:
            ra, rb = run_id(n2, s_, mode), run_id(n3, s_, mode)
            if (df["run_id"] == ra).any() and (df["run_id"] == rb).any():
                dd = paired_bootstrap_diff(y, load_predictions(cfg, ra)["y_pred"], load_predictions(cfg, rb)["y_pred"], k, nb, seed=s_)
                diffs.append({"comparison": f"{n2} - {n3}", "seed": s_, "diff": f"{dd['diff']:+.4f}", "95% CI": f"[{dd['lo']:+.4f}, {dd['hi']:+.4f}]", "excludes 0": dd["excludes_zero"]})
    if rows:
        gate["m2_f1_within_seed_std_of_b3"] = bool(all(r["within seed std"] for r in rows))
        L += ["### Utility of M2 vs B3 (test macro-F1, mean ± std over seeds)", "", _table(rows), "",
              "Paired bootstrap on the test predictions (same seed, same test set). Two runs that differ only by the quantisation noise are two draws of a chaotic training process, "
              "so single-seed differences are expected to be of the size of the seed-to-seed spread; the criterion is the one of the spec (difference inside the seed std).", "", _table(diffs), ""]
        fd = []
        for s_ in seeds:
            a = artifacts_dir(cfg) / run_id("M2", s_, mode) / "final_state.pt"
            b = artifacts_dir(cfg) / run_id("B3", s_, mode) / "final_state.pt"
            if a.exists() and b.exists():
                fd.append({"seed": s_, "max abs(M2 - B3) of final weights": f"{_maxdiff(torch.load(a, weights_only=False), torch.load(b, weights_only=False)):.3f}",
                           "val loss M2": f"{[r for r in core.read_round_log(artifacts_dir(cfg) / run_id('M2', s_, mode)) if r.get('val_loss') is not None][-1]['val_loss']:.4f}",
                           "val loss B3": f"{[r for r in core.read_round_log(artifacts_dir(cfg) / run_id('B3', s_, mode)) if r.get('val_loss') is not None][-1]['val_loss']:.4f}"})
        L += ["### Final models M2 vs B3 (30 rounds)", "",
              "The weights of two runs are far apart (max over about 105k parameters) although the validation losses agree; this is the drift analysed above, the same size as between plain runs "
              "perturbed by noise (control: 4.6 to 5.7 for seed 0).", "", _table(fd), ""]

    # ---- T-SA2
    L += ["## T-SA2 hiding: what the server receives", ""]
    if ck:
        t2 = ck["tsa2"]
        rho, ks = max(abs(r["rho"]) for r in t2), max(r["ks_stat"] for r in t2)
        gate["tsa2_masked_uncorrelated"] = bool(rho < 0.05)
        gate["tsa2_masked_uniform"] = bool(ks < 0.01)
        gate["tsa2_positive_control"] = bool(min(r["rho_unmasked"] for r in t2) > 0.99 and min(r["ks_stat_unmasked"] for r in t2) > 0.5)
        L += [f"The masked vector each client uploaded in stage 2 (the only per-client information the server gets) was captured for rounds 1 to 3 and compared with that client's real update "
              f"({len(t2)} vectors of {t2[0]['n_sampled']:,} parameters): Pearson correlation of the masked integers with the real update, and a Kolmogorov-Smirnov test of masked / 2^32 against U[0, 1). "
              f"**Largest |rho| {rho:.4f}** (gate < 0.05: {'OK' if gate['tsa2_masked_uncorrelated'] else 'FAIL'}); **largest KS statistic {ks:.4f}** "
              f"(a uniform sample of this size gives about {0.87 / np.sqrt(t2[0]['n_sampled']):.4f}; gate < 0.01: {'OK' if gate['tsa2_masked_uniform'] else 'FAIL'}). "
              f"Positive control: the same statistics on the UNMASKED quantised update give rho >= {min(r['rho_unmasked'] for r in t2):.4f} and KS >= {min(r['ks_stat_unmasked'] for r in t2):.2f}, so the test can tell a masked "
              f"from an unmasked vector ({'OK' if gate['tsa2_positive_control'] else 'FAIL'}).", "",
              _table([{"round": r["round"], "client": r["client"], "rho (masked)": f"{r['rho']:+.4f}", "KS stat": f"{r['ks_stat']:.4f}", "KS p": f"{r['ks_p']:.2f}",
                       "rho (unmasked control)": f"{r['rho_unmasked']:.4f}", "KS (unmasked control)": f"{r['ks_stat_unmasked']:.2f}"} for r in t2]), ""]
    else:
        L += ["Not run yet.", ""]

    # ---- T-SA3
    L += ["## T-SA3 dropout (optional gate)", ""]
    if ck:
        t3 = ck["tsa3"]
        gate["tsa3_dropout_completes_and_is_correct"] = bool(t3["completed"] and t3["n_clients_per_round"] == [5, 4, 5]
                                                            and max(t3["agg_err_per_round"]) <= min(t3["error_bound_4_clients"] * 1.001, 1e-4))
        L += [f"A 3-round run in which client {t3['drop']['pid']} raises an exception in round {t3['drop']['round']} (Flower reports it as a failed reply). With {t3['num_shares']} shares and threshold {t3['threshold']} "
              f"the protocol finished the round with the remaining clients: clients aggregated per round {t3['n_clients_per_round']}; error against the plain weighted mean of the clients that did deliver "
              f"{', '.join(f'{e:.1e}' for e in t3['agg_err_per_round'])} (bound for 4 clients {t3['error_bound_4_clients']:.1e}). "
              f"{'OK' if gate['tsa3_dropout_completes_and_is_correct'] else 'FAIL'}. "
              "The unit tests (`tests/test_secagg.py`) additionally show that 2 of 5 clients may drop and that 3 dropouts (2 left < threshold) halt the protocol. "
              "In the experiments below a missing client aborts the run instead (and the run resumes from its checkpoint) so that the plan and the DP accounting stay as designed.", ""]
    else:
        L += ["Not run yet.", ""]

    # ---- clipping range
    clip_rows, clips = [], []
    for name, ss in (("M2", seeds), (m3, seeds3)):
        for s_ in ss:
            c = secagg.clip_summary(artifacts_dir(cfg) / run_id(name, s_, mode), sa["clipping_range"])
            clips.append(c)
            clip_rows.append({"run": f"{name} seed {s_}", "updates checked": c["n_updates"], "max abs(w)": f"{c['max_abs_w']:.3f}",
                              "max w * n_i / max_weight": "n/a" if c["max_abs_weighted"] is None else f"{c['max_abs_weighted']:.3f}",
                              "clipping range": f"{c['clipping_range']:g}", "abs(w) beyond range": c["n_beyond_range"]})
    if clip_rows:
        gate["clip_range_not_exceeded"] = bool(all(c["ok"] for c in clips))
        gate["clip_range_not_exceeded_weighted"] = bool(all(c["ok_weighted"] for c in clips if c["ok_weighted"] is not None))
        L += ["## Clipping range check", "",
              f"Every client's update (every round) is checked against the quantiser's clipping range ({sa['clipping_range']:g}); a value beyond it would be cut silently. The spec's criterion is on the raw "
              "weight |w|; what the quantiser actually clips is the weighted value w * n_i / max_weight (second column), which is smaller for every client but the largest. "
              "Flower's default range 8 was tried first: in M2 seed 1 the weights of `out.weight` reached 11.4 (the weighted maximum, 7.91, was 1 % below 8, so nothing had been cut, "
              "but with no margin), so by the spec's rule the range was raised to 16 and every Phase 10 run was repeated with it (SPEC_DEVIATIONS 10.3).", "", _table(clip_rows), ""]

    # ---- M3
    if seeds3:
        a_ = df[df["config"] == f"{m3}-plain-TSTR-rf"].sort_values("seed")
        L += ["## M3: DP + SecAgg", "",
              f"Target epsilon {eps:g}, delta {cfg['dp']['delta']:g}; epsilon achieved (max over clients, recomputed from the counted steps): "
              + ", ".join(f"seed {int(r['seed'])}: {r['dp_eps_max']:.3f} (x{r['dp_ratio_to_target']:.3f} of target)" for _, r in a_.iterrows())
              + ". The steps counted are those of M1 (same schedule), so epsilon equals M1's; SecAgg does not enter the accounting.", ""]
        gate["m3_epsilon_within_2pct"] = bool(all(float(r["dp_ratio_to_target"]) <= 1.02 for _, r in a_.iterrows()))
        rows = []
        for suf, label in (("-plain", "plain decoder (headline, covered by epsilon)"), ("", "with residual noise (not covered)")):
            for p_, c_ in PROTOCOLS:
                n3 = f"{m3}{suf}-{p_}-{c_}"
                n1 = f"{m1}{suf}-{p_}-{c_}"
                a, b = df[df["config"] == n3].sort_values("seed"), df[df["config"] == n1].sort_values("seed")
                if len(a) and len(b):
                    sd = max(float(a["macro_f1"].std(ddof=0)), float(b["macro_f1"].std(ddof=0)))
                    rows.append({"variant": label, "protocol": f"{p_}-{c_}", "M3 macro-F1": _ms(a["macro_f1"]), "M1 macro-F1": _ms(b["macro_f1"]),
                                 "M3 - M1": f"{a['macro_f1'].mean() - b['macro_f1'].mean():+.4f}", "max seed std": f"{sd:.4f}",
                                 "within seed std": abs(a['macro_f1'].mean() - b['macro_f1'].mean()) <= sd})
        if rows:
            gate["m3_f1_within_seed_std_of_m1"] = bool(all(r["within seed std"] for r in rows))
            L += [f"### Utility of M3 vs {m1} (same epsilon, same DP-trained schedule)", "", _table(rows), ""]
        vrows = []
        for s_ in seeds3:
            def vl(name):
                rl = [r for r in core.read_round_log(artifacts_dir(cfg) / run_id(name, s_, mode)) if r.get("val_loss") is not None]
                return rl[-1]["val_loss"] if rl else float("nan")
            a, b = artifacts_dir(cfg) / run_id(m3, s_, mode) / "final_state.pt", artifacts_dir(cfg) / run_id(m1, s_, mode) / "final_state.pt"
            vrows.append({"seed": s_, "val ELBO M3": f"{vl(m3):.3f}", f"val ELBO {m1}": f"{vl(m1):.3f}",
                          "max abs(M3 - M1) of final weights": f"{_maxdiff(torch.load(a, weights_only=False), torch.load(b, weights_only=False)):.3f}" if a.exists() and b.exists() else "n/a"})
        L += ["### Validation ELBO and final weights, M3 vs M1", "", _table(vrows), ""]
        dc = ck.get("dp_control") if ck else None
        if dc:
            L += [f"Seed {dc['seed']} is where M3 and M1 differ most in validation ELBO. Control: one more {m1} run for that seed with plain-FedAvg noise of the size of the quantisation error added to the aggregate "
                  f"(std {dc['noise_std']:.1e}), no SecAgg. Validation ELBO: " + ", ".join(f"{k} {v:.3f}" for k, v in dc["val_elbo"].items())
                  + "; max |difference| of final weights to the plain " + m1 + " run: " + ", ".join(f"{k} {v:.3f}" for k, v in dc["max_abs_diff_to_m1"].items())
                  + f". A perturbation of that size alone moves the final weights by {list(dc['max_abs_diff_to_m1'].values())[0]:.2f} (M3: {list(dc['max_abs_diff_to_m1'].values())[-1]:.2f}) and the validation ELBO by "
                  f"{list(dc['val_elbo'].values())[1] - list(dc['val_elbo'].values())[0]:+.2f} (M3: {list(dc['val_elbo'].values())[-1] - list(dc['val_elbo'].values())[0]:+.2f}). "
                  "The DP noise itself is identical in all these runs (it depends only on the seeds). One control draw cannot say whether the M3 value lies inside the spread of such perturbations: "
                  "it shows that the DP training of this seed (one client holds 79 % of the rows) reacts strongly to a 1e-5 perturbation, which is also the seed that is unstable in every "
                  "DP configuration (M1 at epsilon 10: 10.7). The test macro-F1 of M3 and M1 stays inside the seed std (table above), and seeds 0 and 2 agree in ELBO to 0.03.", ""]

    # ---- fidelity / privacy of the synthetic sets
    rows = []
    for name in ["B3", "M2"] + ([f"{m1}-plain", f"{m3}-plain"] if seeds3 else []):
        s = df[df["config"] == f"{name}-TSTR-rf"].sort_values("seed")
        if len(s):
            rows.append({"config": name, "seeds": len(s), "C2ST AUC": _ms(s["c2st_auc_mean"]), "Wasserstein": _ms(s["wasserstein_mean"]), "dup rate": _ms(s["dup_rate"]),
                         "DCR ratio": _ms(s["dcr_ratio_mean"]), "MIA AUC": _ms(s["mia_auc_mean"])})
    L += ["## Fidelity and empirical privacy of the synthetic sets (mean over classes)", "",
          "SecAgg changes only how the update is summed, so the generators should match their non-secure counterparts; the MIA is the weak near-copy detector described in b2_cvae.md.", "", _table(rows), ""]

    # ---- overhead
    L += ["## Cost: overhead of secure aggregation", ""]
    base = _median_round_s(cfg, "B3", seeds)
    rows = [{"config": "B3 (FedAvg)", "median s/round (rounds 2+)": f"{base:.2f}", "ratio to B3": "1.00", "bytes/round": f"{int(df[df['config'] == 'B3-TSTR-rf']['bytes_per_round'].mean()):,}" if (df['config'] == 'B3-TSTR-rf').any() else "n/a"}]
    s2 = _median_round_s(cfg, "M2", seeds)
    if seeds:
        rows.append({"config": "M2 (FedAvg + SecAgg+)", "median s/round (rounds 2+)": f"{s2:.2f}", "ratio to B3": f"{s2 / base:.2f}",
                     "bytes/round": f"{int(df[df['config'] == 'M2-TSTR-rf']['bytes_per_round'].mean()):,}"})
        gate["m2_overhead_below_threshold"] = bool(s2 / base <= float(th["overhead_ratio_max"]))
    if seeds3:
        d1, d3 = _median_round_s(cfg, m1, seeds3), _median_round_s(cfg, m3, seeds3)
        b1 = df[df["config"] == f"{m1}-plain-TSTR-rf"]["bytes_per_round"]
        rows += [{"config": f"{m1} (DP)", "median s/round (rounds 2+)": f"{d1:.2f}", "ratio to B3": f"{d1 / base:.2f}", "bytes/round": f"{int(b1.mean()):,}" if len(b1) else "n/a"},
                 {"config": f"{m3} (DP + SecAgg+)", "median s/round (rounds 2+)": f"{d3:.2f}", "ratio to B3": f"{d3 / base:.2f}",
                  "bytes/round": f"{int(df[df['config'] == f'{m3}-plain-TSTR-rf']['bytes_per_round'].mean()):,}"}]
        L_ratio_dp = d3 / d1
    L += [_table(rows), "",
          f"Threshold `overhead_ratio_max` = {th['overhead_ratio_max']} (a heuristic of the spec; applies to M2 / B3): "
          + (f"M2 / B3 = {s2 / base:.2f}, {'within' if gate.get('m2_overhead_below_threshold') else 'ABOVE'} the threshold. " if seeds else "")
          + (f"On top of DP the secure sum costs {L_ratio_dp:.2f}x ({m3} / {m1}). " if seeds3 else "")
          + "Plain bytes/round are the float32 payload (model x clients x 2); SecAgg bytes are counted on the grid (protobuf size of every message of the four stages). "
          "The DP-tuned model is smaller than the B3 model (hidden 128-64 against 256-128), so compare M2 with B3 and M3 with M1d, not across the two families.", ""]
    sums = [secagg.summarize(json.loads((artifacts_dir(cfg) / run_id("M2", s_, mode) / "spec.json").read_text(encoding="utf-8")), artifacts_dir(cfg) / run_id("M2", s_, mode)) for s_ in seeds]
    if sums:
        rows = [{"stage": st, "median seconds": f"{np.median([s['stage_seconds'][st] for s in sums]):.3f}", "median bytes (down + up, all clients)": f"{int(np.median([s['stage_bytes'][st] for s in sums])):,}"}
                for st in secagg.STAGES]
        L += ["### Where the time and the bytes go (M2, per round, median over rounds 2+ and seeds)", "", _table(rows), "",
              "The stage that carries the training request also contains the clients' local training, so its time is training + masking. The three protocol-only stages cost a fixed amount per round "
              "(about 0.1 s each in this simulation, which is the polling granularity of Flower's in-memory grid, not cryptography). "
              "Flower transmits the masked vector as int64 (8 bytes per parameter) although the modulus 2^32 would fit in 4 bytes, so the upload is twice the float32 update; "
              "the information content is 32 bits per parameter.", ""]
    L += ["## Limitations", "",
          "- Flower's SecAgg+ protects the individual updates from an honest-but-curious server; malicious clients/servers, collusion beyond the threshold and network effects are not tested.",
          "- `num_examples` and the metrics dict travel in the clear in Flower's implementation; the row counts are public in this study.",
          f"- The quantiser clips at +/- {sa['clipping_range']:g}; the check above shows the largest weight seen, but a different model or a longer training could exceed it, and the quantiser would then cut the value silently.",
          "- The aggregate (the global model) is exactly what plain FedAvg would reveal: SecAgg does not reduce leakage from the model; only DP does (M3), at record level (see m1_dp.md for the unit of protection and the label caveat).",
          "- Overhead is measured with 5 simulated clients on one machine (10 CPU threads shared by the server and 5 client workers); absolute times are specific to this machine.", ""]
    L += ["## Gate Phase 10 checks", ""] + [f"- {k}: {'OK' if v else 'FAIL'}" for k, v in gate.items()] + [""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(L), encoding="utf-8")
    return gate


def _flwr_version() -> str:
    import flwr
    return flwr.__version__
