"""Optimisation O2(a): M3-distributed = DP-FedSGD with the Gaussian noise split over the clients and summed by secure aggregation.

Why not M1/M3 with the noise split: there every client runs many local DP-SGD steps with Adam, so the noise of the other clients has gone
through non-linear training before the sum; the aggregate is no longer one Gaussian mechanism per record and distributed noise is not valid.
Here every round is ONE gradient step:

- each client k Poisson-samples its rows with the common rate q, computes per-sample gradients of the CVAE loss (class weights of O1(c)
  allowed), clips each to L2 norm C, sums them and adds N(0, sigma^2 C^2 / K) per coordinate (its share of the noise);
- secure aggregation returns only the sum over clients: sum of clipped gradients + N(0, sigma^2 C^2), the Gaussian mechanism of central
  DP-SGD on the union with sampling rate q (a record is in one client, the union of Poisson samples with rate q is a Poisson sample);
- the server divides by the public expected batch q * N and takes an Adam step (post-processing).

Accounting (`epsilon_report`): T rounds of the subsampled Gaussian with multiplier sigma, composed with the residual statistics release of
O1(a) whose noise is split the same way (`noise_share` 1/K). Trust assumption: the clients follow the protocol and SecAgg hides the
individual sums. If only h of the K clients add their share honestly (the others know and remove theirs), the effective multipliers are
sigma * sqrt(h / K) and sigma_stat * sqrt(h / K); the report gives epsilon for h = K and h = 1 (worst case).

Simulation: in-process, without Flower. SecAgg+ masks cancel exactly, so the sum is reproduced exactly; its stochastic quantisation is
reproduced too (each client sends its noisy sum divided by the public q * N, clipped to +-`clipping_range` and rounded to the grid of
`quantization_range` levels, as Flower's SecAgg+ does). The bytes per round are an ESTIMATE: the bytes per model parameter measured for M2
(Flower SecAgg+, `sa_bytes_per_round` / `cvae_params`) times the parameters of this model. Many more rounds than M1/M3 (one step each):
that is the price of distributed DP and it is reported.

Limitation (state it in the reports): the accounting is that of the continuous Gaussian sum. With the noise split, one client's message is
not DP by itself, so the SecAgg quantisation applied to each message is not plain post-processing of the noisy sum; a rigorous treatment uses
a discrete mechanism (distributed discrete Gaussian, Kairouz et al. 2021; Skellam, Agarwal et al. 2021). Here the quantisation step
(2 x 16 / 2^22 ~ 7.6e-6 per coordinate) is far below the per-client noise, and the fraction of clipped coordinates is recorded.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ppfeddata.fl import core, dp_utils
from ppfeddata.models.cvae import Layout, beta_at, loss_terms, numpy_to_tensor, one_hot
from ppfeddata.models.train import seed_torch

logger = logging.getLogger("ppfeddata.fl.dpfedsgd")


# --------------------------------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------------------------------
def calibrate(target_eps: float, delta: float, q: float, rounds: int, sigma_stat: float | None = None) -> float:
    return dp_utils.calibrate_noise(target_eps, delta, q, int(rounds), sigma_stat)


def epsilon_report(sigma: float, q: float, rounds: int, delta: float, num_clients: int, sigma_stat: float | None = None) -> dict[str, float]:
    """epsilon with every client honest (the target) and with a single honest client (noise sigma / sqrt(K) of its own share only)."""
    s1 = math.sqrt(1.0 / num_clients)
    return {"eps_all_honest": dp_utils.epsilon(sigma, q, rounds, delta, sigma_stat),
            "eps_one_honest": dp_utils.epsilon(sigma * s1, q, rounds, delta, None if not sigma_stat else sigma_stat * s1)}


# --------------------------------------------------------------------------------------------------
# One client's message
# --------------------------------------------------------------------------------------------------
def per_sample_clipped_sum(model, params: dict[str, torch.Tensor], X: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None, beta: float,
                           layout: Layout, n_classes: int, clip: float, chunk: int = 512) -> tuple[dict[str, torch.Tensor], int]:
    """Sum over the rows of the per-sample gradients clipped to L2 norm `clip` (flat over all parameters). Returns (sum, rows clipped)."""
    from torch.func import functional_call, grad, vmap

    def loss1(p, x, yo, wi):
        out, mu, logvar = functional_call(model, p, (x[None], yo[None]))
        return loss_terms(out, x[None], mu, logvar, beta, layout)["loss"] * wi

    g1 = vmap(grad(loss1), in_dims=(None, 0, 0, 0), randomness="different")
    total = {k: torch.zeros_like(v) for k, v in params.items()}
    n_clipped = 0
    for s in range(0, len(X), chunk):
        xb, yb = X[s:s + chunk], y[s:s + chunk]
        wb = torch.ones(len(xb)) if w is None else w[yb]
        g = g1(params, xb, one_hot(yb, n_classes), wb)
        norm = torch.sqrt(sum(v.reshape(len(xb), -1).pow(2).sum(1) for v in g.values()))
        f = torch.clamp(clip / (norm + 1e-12), max=1.0)
        n_clipped += int((norm > clip).sum())
        for k, v in g.items():
            total[k] += (v * f.view(-1, *([1] * (v.dim() - 1)))).sum(0)
    return total, n_clipped


def quantize(v: torch.Tensor, clipping_range: float, quantization_range: int, gen: torch.Generator) -> tuple[torch.Tensor, int]:
    """SecAgg+ encoding of one client's vector: clip to +-clipping_range, stochastic rounding to `quantization_range` levels; decoded back."""
    over = int((v.abs() > clipping_range).sum())
    c = v.clamp(-clipping_range, clipping_range)
    step = 2.0 * clipping_range / (quantization_range - 1)
    x = (c + clipping_range) / step
    low = torch.floor(x)
    up = (torch.rand(x.shape, generator=gen) < (x - low)).to(x.dtype)
    return (low + up) * step - clipping_range, over


# --------------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------------
def train(X: np.ndarray, y: np.ndarray, parts: list[np.ndarray], layout: Layout, n_classes: int, hp: dict[str, Any], rounds: int, q: float,
          sigma: float, clip: float, seed: int, secagg: dict[str, Any] | None = None, log_every: int = 100,
          Xv: np.ndarray | None = None, yv: np.ndarray | None = None) -> dict[str, Any]:
    """DP-FedSGD over `rounds` rounds. Returns the final state and the per-run diagnostics."""
    K, N = len(parts), int(sum(len(p) for p in parts))
    state = core.init_state(layout, n_classes, hp, seed)
    model = core.make_model(layout, n_classes, hp)
    model.load_state_dict(state)
    opt = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]))
    Xt, yt = numpy_to_tensor(X), torch.as_tensor(y, dtype=torch.long)
    cws = [core.class_weights(y[p], n_classes, float(hp.get("cw_power", 0.0))) for p in parts]
    beta, warm, denom = float(hp["beta"]), int(hp["beta_warmup_epochs"]), q * N
    sa = (secagg or {}).get("params")
    t0, n_rows, n_clipped, n_over, log = time.perf_counter(), 0, 0, 0, []
    for t in range(1, int(rounds) + 1):
        b = beta_at(t * q, beta, warm)                       # epochs seen so far = t * q
        params = {k: v.detach() for k, v in model.named_parameters()}
        agg = {k: torch.zeros_like(v) for k, v in params.items()}
        for k, idx in enumerate(parts):
            s = core.client_seed(seed, t, k)
            seed_torch(s)
            rng = np.random.default_rng(s)
            pick = idx[rng.random(len(idx)) < q]
            n_rows += len(pick)
            if len(pick):
                gsum, nc = per_sample_clipped_sum(model, params, Xt[pick], yt[pick], cws[k], b, layout, n_classes, clip)
                n_clipped += nc
            else:
                gsum = {kk: torch.zeros_like(v) for kk, v in params.items()}
            gn = torch.Generator().manual_seed(s + 1)
            for kk, v in gsum.items():
                msg = (v + float(sigma) * clip / math.sqrt(K) * torch.randn(v.shape, generator=gn)) / denom
                if sa:
                    msg, over = quantize(msg, float(sa["clipping_range"]), int(sa["quantization_range"]), gn)
                    n_over += over
                agg[kk] += msg
        opt.zero_grad(set_to_none=True)
        for kk, p in model.named_parameters():
            p.grad = agg[kk].clone()
        opt.step()
        if log_every and (t % log_every == 0 or t == rounds):
            row = {"round": t, "beta": b, "seconds": time.perf_counter() - t0}
            if Xv is not None:
                row["val_loss"] = core.val_loss(OrderedDict(model.state_dict()), Xv, yv, layout, n_classes, hp)["loss"]
            log.append(row)
            logger.info("DP-FedSGD round %d/%d: %s", t, rounds, {a: round(v, 4) for a, v in row.items() if a != "round"})
    n_params = int(sum(p.numel() for p in model.parameters()))
    return {"state": OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items()), "seconds": time.perf_counter() - t0,
            "rows_per_round": n_rows / rounds, "clipped_frac": n_clipped / max(n_rows, 1),
            "quant_clipped_frac": n_over / max(rounds * K * n_params, 1) if sa else 0.0, "n_params": n_params, "log": log}


def bytes_per_param_m2(cfg: dict[str, Any]) -> float:
    """Bytes per round per model parameter measured for M2 (Flower SecAgg+), from results/summary.csv; NaN when not there."""
    import pandas as pd

    p = Path(cfg["compute"]["runs_csv"]).parent / "summary.csv"
    if not p.exists():
        return math.nan
    s = pd.read_csv(p)
    r = s[s["config"] == "M2-TSTR-rf"]
    if not len(r) or "sa_bytes_per_round_mean" not in r or "cvae_params_mean" not in r:
        return math.nan
    return float(r["sa_bytes_per_round_mean"].iloc[0] / r["cvae_params_mean"].iloc[0])


def run(cfg: dict[str, Any], seed: int, name: str, hp: dict[str, Any], target_eps: float, rounds: int, batch_size: int, clip: float,
        stat_frac: float = 0.0, res_clip: float = 1.0, secagg: dict[str, Any] | None = None, resume: bool = True,
        data_dir: str | Path | None = None) -> dict[str, Any]:
    """One DP-FedSGD run on the same non-IID partition as M1/M3 (`fl.dirichlet_alpha`, `fl.num_clients`, run seed). `batch_size` is the
    expected total batch per round (q = batch_size / N). Writes `artifacts/<run_id>/final_state.pt` and `spec.json`; returns a run dict
    usable by `fl.b3.load_fl_model` and `fl.dp_stats.stats_for_run` (with `noise_share` 1/K)."""
    from ppfeddata.data.preprocess import processed_dir
    from ppfeddata.eval.runs import run_id
    from ppfeddata.models.cvae import build_layout
    from ppfeddata.partition import make_partition, partition_path

    fl, dcfg = cfg["fl"], cfg["dp"]
    ddir = Path(data_dir) if data_dir is not None else processed_dir(cfg)
    schema = json.loads((ddir / "feature_schema.json").read_text(encoding="utf-8"))
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    tr = np.load(ddir / "train.npz")
    X, y = tr["X"], tr["y"].astype(np.int64)
    alpha, K = float(fl["dirichlet_alpha"]), int(fl["num_clients"])
    parts, meta = make_partition(cfg, y, classes, alpha, seed, K)
    N, delta = int(sum(meta["sizes"])), float(dcfg["delta"])
    dp_utils.check_delta([N], delta)
    q = min(1.0, float(batch_size) / N)
    sigma_stat = dp_utils.stat_noise_multiplier(target_eps, stat_frac, delta) if stat_frac else None
    sigma = calibrate(target_eps, delta, q, rounds, sigma_stat)
    hp = {**hp, "batch_size": int(batch_size)}
    rid = run_id(name, seed, cfg["label_mode"])
    rdir = Path(cfg["compute"]["artifacts_dir"]) / rid
    rdir.mkdir(parents=True, exist_ok=True)
    dp = {"mode": "distributed (DP-FedSGD)", "target_eps": float(target_eps), "delta": delta, "max_grad_norm": float(clip), "sigma": float(sigma),
          "q": q, "rounds": int(rounds), "noise_share": 1.0 / K}
    if sigma_stat:
        dp.update(stat_frac=float(stat_frac), sigma_stat=float(sigma_stat), res_clip=float(res_clip))
    spec = {"run_id": rid, "name": name, "seed": int(seed), "num_clients": K, "alpha": alpha, "rounds": int(rounds), "local_epochs": 1, "hp": hp,
            "dp": dp, "secagg": secagg, "partition_path": str(partition_path(cfg, alpha, seed)), "partition_sizes": meta["sizes"],
            "artifacts_dir": str(cfg["compute"]["artifacts_dir"]), "label_mode": cfg["label_mode"], "data_dir": str(ddir)}
    done = rdir / "summary.json"
    if resume and (rdir / "final_state.pt").exists() and done.exists():
        old = json.loads((rdir / "spec.json").read_text(encoding="utf-8"))
        if old.get("dp", {}).get("sigma") == spec["dp"]["sigma"] and old.get("hp") == hp:
            return {**old, "summary": json.loads(done.read_text(encoding="utf-8"))}
    (rdir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    va = np.load(ddir / "val.npz")
    out = train(X, y, parts, build_layout(schema), len(classes), hp, rounds, q, sigma, clip, seed, secagg, Xv=va["X"], yv=va["y"].astype(np.int64))
    torch.save(out["state"], rdir / "final_state.pt")
    eps = epsilon_report(sigma, q, rounds, delta, K, sigma_stat)
    bpp = bytes_per_param_m2(cfg)
    summary = {"dp": {**eps, "eps_max": eps["eps_all_honest"], "sigma": float(sigma), "q": q, "rounds": int(rounds), "delta": delta,
                      "target_eps": float(target_eps), "sigma_stat": sigma_stat},
               "fl_total_s": out["seconds"], "mean_round_s": out["seconds"] / rounds, "rows_per_round": out["rows_per_round"],
               "clipped_frac": out["clipped_frac"], "quant_clipped_frac": out["quant_clipped_frac"], "n_params": out["n_params"],
               "bytes_per_round_est": bpp * out["n_params"] if not math.isnan(bpp) else None,
               "final_val_loss": out["log"][-1].get("val_loss") if out["log"] else None, "log": out["log"]}
    done.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("%s: eps %.3f (all honest) / %.1f (one honest), sigma %.3f, q %.4f, %d rounds, %.0fs", rid, eps["eps_all_honest"],
                eps["eps_one_honest"], sigma, q, rounds, out["seconds"])
    return {**spec, "summary": summary}
