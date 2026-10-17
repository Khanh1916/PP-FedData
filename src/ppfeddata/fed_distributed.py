"""Follow-up after round R: the directly federated detector with DISTRIBUTED DP, the fair competitor of FedDP-Marginal.

`fed_protected.py` gave the federated MLP DP-SGD at each client: every client adds the full noise to its own update, so secure aggregation does
not reduce it. FedDP-Marginal (MGs, MGd) splits its noise over the clients and lets SecAgg+ hide the shares, so its comparison with that classifier
mixed two things: the generator route and the noise split. Here the classifier gets the noise split too, with DP-FedSGD (as `fl/dpfedsgd.py` for
the CVAE): every round is ONE gradient step,

- each client Poisson-samples its rows with the common rate q = B / N, computes the per-sample gradients of the class-weighted cross-entropy
  (`torch.func`), clips each to L2 norm C, sums them and adds N(0, sigma^2 C^2 / K) per coordinate (its share of the noise);
- secure aggregation returns only the sum over the clients: the sum of clipped gradients + N(0, sigma^2 C^2), the Gaussian mechanism of central
  DP-SGD on the union with sampling rate q (a record is held by one client; the union of Poisson samples with rate q is a Poisson sample). SecAgg+
  is emulated in process: its masks cancel exactly, and the stochastic quantisation of every client's message is applied (`dpfedsgd.quantize`);
- the server divides by the public expected batch q N and takes an Adam step (post-processing).

Accounting: the RDP accountant of `fl/dp_utils.py` over T = rounds steps at rate q (delta of the config); the epsilon with a single honest client
(the others remove their shares: noise sigma / sqrt(K)) is reported too. Same limitation as `fl/dpfedsgd.py`: the accounting is that of the
continuous Gaussian sum while every share is quantised by SecAgg+ (a rigorous version uses a discrete mechanism, e.g. Skellam as MGs).

Labels `FedMLPddp<eps>`: the class-weighted MLP of `fed_protected.py` (the width chosen there), tuned on validation at EVERY epsilon, seed 0, as
FedDP-Marginal was (`SEARCH`: learning rate x clipping bound x learning-rate schedule, constant or cosine decay; expected batch and number of passes
over the data fixed); the model kept is the evaluation point (twice per pass) with the best validation macro-F1 (non-private selection, as everywhere
in the study). `FedMLPddp-nonoise` is the control: the setting chosen at epsilon 5 without noise (clipping kept), so the cost of DP and the cost of
the recipe (one clipped step per round) can be told apart. Compared (paired bootstrap, rule of R1 / R2) at the same epsilon with TSTR-MLP of
FedDP-Marginal (the framework configuration, Skellam through SecAgg+, and MGd, the Gaussian noise split simulated in process: the same mechanism
family as here) and with the local DP-SGD classifier of `fed_protected.py` (tuned at epsilon 5 only). Communication: rounds x the bytes per
parameter and round measured for M2 (Flower SecAgg+, all clients, up and down), an estimate, against the bytes FedDP-Marginal measured.

History: a first, compact search (epsilon 5 only, learning rate {0.001, 0.005, 0.02} x clipping bound {0.5, 2, 8} x batch {1024, 4096}) chose the
top learning rate and clipping bound of its grid; its results stay in the `compact` entry of the block and in SPEC_DEVIATIONS R.7, and its
epsilon-5 trials that lie in `SEARCH` are reused from the cache.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ppfeddata.eval.compare import PairedBootstrap, compare_runs, recall_of
from ppfeddata.eval.runs import load_predictions, run_id, save_predictions
from ppfeddata.eval.utility import compute_metrics
from ppfeddata.fed_classifier import class_weights, file_label, macro_f1, make_mlp, predict, summarise
from ppfeddata.fl import dp_utils
from ppfeddata.fl.dpfedsgd import bytes_per_param_m2, quantize
from ppfeddata.partition import make_partition
from ppfeddata.utils import config_hash, git_commit, set_seed

logger = logging.getLogger("ppfeddata.fed_distributed")

LABEL = "FedMLPddp"
NONOISE = f"{LABEL}-nonoise"
NODP_REF = "MGs-nodp-TSTR-mlp"                       # FedDP-Marginal without DP (round Q1), TSTR-MLP: the reference of the control without noise
EPOCHS = 30                                          # passes over the train rows (T = EPOCHS * N / B rounds), fixed; SEARCH is searched at every epsilon
# the compact search chose the top of lr {0.001, 0.005, 0.02} and C {0.5, 2, 8}, and batch 4096 lost in all 9 pairs: the grid goes on upwards,
# with a cosine decay of the learning rate against the oscillation of the validation curve
SEARCH = {"lr": (2.0e-2, 5.0e-2), "clip": (2.0, 8.0, 32.0), "sched": ("const", "cos"), "batch": (1024,)}
EPS_LIST = (1, 5, 10)
CHUNK = 1024                                         # rows per vmap call


def label(eps: float) -> str:
    return f"{LABEL}{eps:g}"


def lr_at(lr: float, t: int, rounds: int, sched: str = "const") -> float:
    """Learning rate of round t (1-based): constant, or cosine decay from lr to 0 over the rounds."""
    return float(lr) if sched == "const" else 0.5 * float(lr) * (1.0 + math.cos(math.pi * (t - 1) / rounds))


def rounds_for(n: int, batch: int, epochs: int | None = None) -> int:
    return max(1, int(round((EPOCHS if epochs is None else epochs) * n / batch)))


# --------------------------------------------------------------------------------------------------
# One client's share
# --------------------------------------------------------------------------------------------------
def clipped_sum(model: torch.nn.Module, params: dict[str, torch.Tensor], X: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, clip: float,
                chunk: int = CHUNK) -> tuple[dict[str, torch.Tensor], int]:
    """Sum over the rows of the per-sample gradients of w[y] * cross-entropy, each clipped to L2 norm `clip` (flat over all parameters).
    Returns (sum, rows clipped)."""
    from torch.func import functional_call, grad, vmap

    def loss1(p, x, yi, wi):
        return F.cross_entropy(functional_call(model, p, (x[None],)), yi[None]) * wi

    g1 = vmap(grad(loss1), in_dims=(None, 0, 0, 0))
    total = {k: torch.zeros_like(v) for k, v in params.items()}
    n_clipped = 0
    for s in range(0, len(X), chunk):
        xb, yb = X[s:s + chunk], y[s:s + chunk]
        g = g1(params, xb, yb, weight[yb])
        norm = torch.sqrt(sum(v.reshape(len(xb), -1).pow(2).sum(1) for v in g.values()))
        f = torch.clamp(clip / (norm + 1e-12), max=1.0)
        n_clipped += int((norm > clip).sum())
        for k, v in g.items():
            total[k] += (v * f.view(-1, *([1] * (v.dim() - 1)))).sum(0)
    return total, n_clipped


def noisy_share(v: torch.Tensor, sigma: float, clip: float, num_clients: int, denom: float, gen: torch.Generator) -> torch.Tensor:
    """One client's message for one tensor: its clipped sum plus its share N(0, sigma^2 C^2 / K) of the noise, divided by the public expected batch.
    The K shares add up to N(0, sigma^2 C^2): the noise of central DP-SGD."""
    return (v + float(sigma) * float(clip) / math.sqrt(num_clients) * torch.randn(v.shape, generator=gen)) / denom


def client_seed(seed: int, rnd: int, cid: int) -> int:
    return (seed * 1_000_003 + rnd * 1009 + cid) % (2 ** 31 - 1)


# --------------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------------
def train(data: dict[str, dict[str, np.ndarray]], k: int, parts: list[np.ndarray], hp: dict[str, Any], eps: float | None, delta: float, seed: int,
          secagg: dict[str, Any] | None, epochs: int | None = None) -> dict[str, Any]:
    """DP-FedSGD of the class-weighted MLP; hp = {lr, hidden, clip, batch, sched?}. `eps` None: the control without noise (clipping kept; epsilon
    reported as None). Returns the best state (validation macro-F1) and the diagnostics."""
    Xtr, ytr = torch.tensor(data["train"]["X"]), torch.tensor(data["train"]["y"], dtype=torch.long)
    Xva, yva = torch.tensor(data["val"]["X"]), data["val"]["y"].astype(np.int64)
    K, N = len(parts), int(sum(len(p) for p in parts))
    batch, clip, sched = int(hp["batch"]), float(hp["clip"]), str(hp.get("sched", "const"))
    q, T = min(1.0, batch / N), rounds_for(N, batch, epochs)
    sigma = 0.0 if eps is None else dp_utils.calibrate_noise(eps, delta, q, T)
    weight = class_weights(np.bincount(data["train"]["y"], minlength=k))        # from the clients' summed class counts, as FedMLPcwT
    set_seed(seed)
    model = make_mlp(Xtr.shape[1], k, tuple(hp["hidden"]))
    opt = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]))
    every = max(1, int(math.ceil(N / batch / 2)))                                 # evaluate twice per pass over the data
    sa = dict(secagg or {})
    best, best_f1, best_round, curve = None, -1.0, 0, []
    t0, n_rows, n_clipped, n_over = time.perf_counter(), 0, 0, 0
    for t in range(1, T + 1):
        params = {a: b.detach() for a, b in model.named_parameters()}
        agg = {a: torch.zeros_like(b) for a, b in params.items()}
        for cid, idx in enumerate(parts):
            s = client_seed(seed, t, cid)
            rng = np.random.default_rng(s)
            pick = torch.as_tensor(idx[rng.random(len(idx)) < q])
            n_rows += len(pick)
            if len(pick):
                gsum, nc = clipped_sum(model, params, Xtr[pick], ytr[pick], weight, clip)
                n_clipped += nc
            else:
                gsum = {a: torch.zeros_like(b) for a, b in params.items()}
            gn = torch.Generator().manual_seed(s + 1)
            for a, v in gsum.items():
                msg = noisy_share(v, sigma, clip, K, q * N, gn)
                if sa:
                    msg, over = quantize(msg, float(sa["clipping_range"]), int(sa["quantization_range"]), gn)
                    n_over += over
                agg[a] += msg
        opt.zero_grad(set_to_none=True)
        for a, p in model.named_parameters():
            p.grad = agg[a]
        for g in opt.param_groups:
            g["lr"] = lr_at(hp["lr"], t, T, sched)
        opt.step()
        if t % every == 0 or t == T:
            f1 = macro_f1(yva, predict(model, Xva)[0], k)
            curve.append((t, f1))
            if f1 > best_f1:
                best_f1, best_round, best = f1, t, {a: b.detach().clone() for a, b in model.state_dict().items()}
    n_params = int(sum(p.numel() for p in model.parameters()))
    spent = None if eps is None else dp_utils.epsilon(sigma, q, T, delta)
    eps_one = None if eps is None else float(dp_utils.epsilon(sigma / math.sqrt(K), q, T, delta))
    return {"state": best, "val_macro_f1": best_f1, "best_round": best_round, "curve": curve, "seconds": time.perf_counter() - t0, "rounds": T, "q": q,
            "sigma": float(sigma), "eps": spent, "eps_one_honest": eps_one, "rows_per_round": n_rows / T,
            "clipped_frac": n_clipped / max(n_rows, 1), "quant_clipped_frac": n_over / max(T * K * n_params, 1) if sa else 0.0, "n_params": n_params}


def secagg_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """The quantisation of Flower's SecAgg+ as configured (clipping range; 2^22 levels, Flower's default)."""
    s = cfg.get("secagg") or {}
    return {"clipping_range": float(s.get("clipping_range", 16.0)), "quantization_range": int(s.get("quantization_range", 2 ** 22))}


def run_one(lab: str, cfg: dict[str, Any], data: dict[str, dict[str, np.ndarray]], classes: list[str], seed: int, parts: list[np.ndarray], hp: dict[str, Any],
            eps: float | None, save: bool = True) -> dict[str, Any]:
    k = len(classes)
    out = train(data, k, parts, hp, eps, float(cfg["dp"]["delta"]), seed, secagg_params(cfg))
    model = make_mlp(data["train"]["X"].shape[1], k, tuple(hp["hidden"]))
    model.load_state_dict(out["state"])
    yte = data["test"]["y"].astype(np.int64)
    pred, proba = predict(model, torch.tensor(data["test"]["X"]))
    res = compute_metrics(yte, pred, proba, classes)
    rid = run_id(file_label(lab) + "-mlp", seed, cfg["label_mode"])
    info = {a: b for a, b in out.items() if a not in ("state", "curve")}
    r = {"seed": seed, "run_id": rid, "best_round": out["best_round"], "val_macro_f1": out["val_macro_f1"], "test_macro_f1": res["macro_f1"],
         "balanced_acc": res["balanced_acc"], "recall": {c: d["recall"] for c, d in res["per_class"].items()}, "val_curve": out["curve"], "dp": info}
    if save:
        save_predictions(cfg, rid, "test", pred, proba, meta={"run_id": rid, "config": lab, "seed": seed, "hp": {**hp, "hidden": list(hp["hidden"])}, "eps": eps,
                                                             "config_hash": config_hash(cfg), "git_commit": git_commit(), **info})
    logger.info("%s seed %d: test macro-F1 %.4f (val %.4f at round %d of %d), %s, %.0f s", lab, seed, res["macro_f1"], out["val_macro_f1"], out["best_round"],
                out["rounds"], "no noise" if eps is None else f"eps {out['eps']:.3f} (one honest client {out['eps_one_honest']:.1f})", out["seconds"])
    return r


def tune(cfg: dict[str, Any], data: dict[str, dict[str, np.ndarray]], classes: list[str], parts: list[np.ndarray], hidden: tuple[int, ...], cache,
         eps: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """`SEARCH` at epsilon `eps`, seed 0, validation macro-F1 only; every trial is cached as it ends. At epsilon 5 a constant-rate trial of the
    compact search (cache key without epsilon and schedule) is reused when it lies in `SEARCH`."""
    import itertools

    trials = []
    for lr, clip, sched, batch in itertools.product(SEARCH["lr"], SEARCH["clip"], SEARCH["sched"], SEARCH["batch"]):
        hp = {"lr": lr, "hidden": list(hidden), "clip": clip, "batch": batch, "sched": sched}
        key = f"trial:eps{eps:g}:{lr:g}:{clip:g}:{batch}:{sched}"
        t = cache.get(key)
        if t is None and float(eps) == 5.0 and sched == "const" and cache.get(f"trial:{lr:g}:{clip:g}:{batch}") is not None:
            t = cache.put(key, {**cache.get(f"trial:{lr:g}:{clip:g}:{batch}"), "hp": hp, "from_compact_search": True})
        if t is None:
            out = train(data, len(classes), parts, {**hp, "hidden": hidden}, eps, float(cfg["dp"]["delta"]), 0, secagg_params(cfg))
            t = cache.put(key, {"hp": hp, "val_macro_f1": out["val_macro_f1"], "best_round": out["best_round"], "rounds": out["rounds"],
                                "seconds": out["seconds"], "clipped_frac": out["clipped_frac"]})
            logger.info("tune eps %g %s: val macro-F1 %.4f (round %d of %d, %.0f s)", eps, hp, t["val_macro_f1"], t["best_round"], t["rounds"], t["seconds"])
        trials.append(t)
    best = max(trials, key=lambda x: x["val_macro_f1"])
    return {**best["hp"], "hidden": tuple(best["hp"]["hidden"])}, trials


def _hp_key(hp: dict[str, Any]) -> str:
    return f"{hp['lr']:g}:{hp['clip']:g}:{hp['batch']}:{hp.get('sched', 'const')}"


def compact_entry(prev: dict[str, Any] | None) -> dict[str, Any] | None:
    """The results of the compact search (epsilon 5 only, applied to every epsilon), kept when the block is rewritten by the per-epsilon search."""
    if not prev:
        return None
    if prev.get("compact"):
        return prev["compact"]
    if "per_eps" in (prev.get("tuning") or {}):
        return None
    return {"tuning": prev.get("tuning"), "classifiers": {k: {a: v.get(a) for a in ("macro_f1_mean", "macro_f1_std", "per_seed", "rare_recall_mean")}
                                                          for k, v in (prev.get("classifiers") or {}).items()},
            "comparisons": {k: {r: {"against": x["against"], "macro_f1": x["macro_f1"]} for r, x in c.items()} for k, c in (prev.get("comparisons") or {}).items()}}


# --------------------------------------------------------------------------------------------------
# Comparisons
# --------------------------------------------------------------------------------------------------
def against(eps: float) -> dict[str, str]:
    """What `FedMLPddp<eps>` is compared with: role -> run name (predictions under `artifacts/<name>_<seed>/preds`)."""
    from ppfeddata.fed_protected import TUNED
    from ppfeddata.tune_marginal import framework_name
    return {"framework": f"{framework_name(eps)}-TSTR-mlp", "MGd": f"MGd-eps{eps:g}-TSTR-mlp", "local DP": f"{TUNED}-dp{eps:g}-mlp"}


def compare(cfg: dict[str, Any], y_test: np.ndarray, classes: list[str], rare: list[str], seeds: list[int], n_boot: int, eps_list) -> dict[str, Any]:
    k, mode = len(classes), cfg["label_mode"]
    boot = PairedBootstrap(y_test, k, n_boot, 0, "rows")
    names = {label(e): file_label(label(e)) + "-mlp" for e in eps_list}
    for e in eps_list:
        names.update({v: v for v in against(e).values()})
    preds = {}
    for key, base in names.items():
        try:
            got = {(key, s): load_predictions(cfg, run_id(base, s, mode))["y_pred"] for s in seeds}
        except FileNotFoundError:
            continue
        preds.update(got)
    boot.add(preds)
    rare_metric = recall_of(*[classes.index(c) for c in rare])
    out: dict[str, Any] = {}
    for e in eps_list:
        a = label(e)
        if (a, seeds[0]) not in preds:
            continue
        out[a] = {}
        for role, b in against(e).items():
            if (b, seeds[0]) in preds:
                ka, kb = [(a, s) for s in seeds], [(b, s) for s in seeds]
                out[a][role] = {"against": b, "macro_f1": compare_runs(boot, ka, kb), "rare_recall": compare_runs(boot, ka, kb, rare_metric)}
    return out


def compare_nonoise(cfg: dict[str, Any], y_test: np.ndarray, classes: list[str], rare: list[str], seeds: list[int], n_boot: int) -> dict[str, Any] | None:
    """The control without noise against FedDP-Marginal without DP (TSTR-MLP): the two routes with no DP at all; None when either is missing."""
    mode = cfg["label_mode"]
    try:
        preds = {(n, s): load_predictions(cfg, run_id(b, s, mode))["y_pred"] for n, b in ((NONOISE, file_label(NONOISE) + "-mlp"), (NODP_REF, NODP_REF)) for s in seeds}
    except FileNotFoundError:
        return None
    boot = PairedBootstrap(y_test, len(classes), n_boot, 0, "rows")
    boot.add(preds)
    ka, kb = [(NONOISE, s) for s in seeds], [(NODP_REF, s) for s in seeds]
    return {"against": NODP_REF, "macro_f1": compare_runs(boot, ka, kb), "rare_recall": compare_runs(boot, ka, kb, recall_of(*[classes.index(c) for c in rare]))}


def references(cfg: dict[str, Any], rare: list[str], eps_list, protected: dict[str, Any]) -> dict[str, Any]:
    """Macro-F1, rare-class recall and bytes of the runs compared with (from results/summary.csv and the protected block of fed_classifier.json)."""
    import pandas as pd

    p = Path(cfg["compute"]["runs_csv"]).parent / "summary.csv"
    summ = pd.read_csv(p).set_index("config") if p.exists() else pd.DataFrame()
    out = {}
    for e in eps_list:
        for role, name in against(e).items():
            if name in summ.index:
                r = summ.loc[name]
                b, h = r.get("bytes_per_round_mean"), r.get("dp_eps_one_honest_mean")
                out[name] = {"macro_f1_mean": float(r["macro_f1_mean"]), "macro_f1_std": float(r["macro_f1_std"]),
                             "rare_recall_mean": float(np.mean([r[f"recall_{c}_mean"] for c in rare])),
                             "bytes_total": float(b * r["rounds_mean"]) if pd.notna(b) and pd.notna(r.get("rounds_mean")) else None,
                             "eps_one_honest": float(h) if pd.notna(h) else None}
            elif role == "local DP":
                v = (protected.get("classifiers") or {}).get(name[:-len("-mlp")])
                if v:
                    out[name] = {"macro_f1_mean": v["macro_f1_mean"], "macro_f1_std": v["macro_f1_std"], "rare_recall_mean": v["rare_recall_mean"], "bytes_total": None}
    if NODP_REF in summ.index:
        r = summ.loc[NODP_REF]
        out[NODP_REF] = {"macro_f1_mean": float(r["macro_f1_mean"]), "macro_f1_std": float(r["macro_f1_std"]),
                         "rare_recall_mean": float(np.mean([r[f"recall_{c}_mean"] for c in rare])), "bytes_total": None, "eps_one_honest": None}
    return out


def run_all(cfg: dict[str, Any], seeds: list[int] | None = None, n_boot: int | None = None, out: str | Path | None = None, eps_list=EPS_LIST) -> dict[str, Any]:
    """Tune at every epsilon (seed 0), run every epsilon for every seed, then the control without noise (the setting of epsilon 5, else of the
    first epsilon), and write the block `distributed` to results/fed_classifier.json (the compact search kept under `compact`)."""
    from ppfeddata.eval.baselines import load_data
    from ppfeddata.fed_protected import Cache
    from ppfeddata.interpret import to_jsonable

    seeds = [int(s) for s in (seeds if seeds is not None else cfg["seeds"])]
    data, schema = load_data(cfg)
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    q = cfg.get("quota", {})
    rare = [c for c in classes if c in q and q[c][0] <= 5000] or classes[-4:]
    p = Path(out) if out else Path(cfg["compute"]["runs_csv"]).parent / "fed_classifier.json"
    res = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    hidden = tuple(((res.get("tuning") or {}).get("non_dp") or {}).get("chosen", {}).get("hidden") or (256, 128))   # the width chosen for FedMLPcwT
    cache = Cache(cfg, "fed_distributed_cache.json")
    parts0, _ = make_partition(cfg, data["train"]["y"], classes, seed=0)
    compact = compact_entry(res.get("distributed"))
    hps, per_eps = {}, {}
    for e in eps_list:                                                   # every search first, so that a crash in the final runs keeps them all
        hps[e], trials = tune(cfg, data, classes, parts0, hidden, cache, float(e))
        per_eps[f"{e:g}"] = {"chosen": {**hps[e], "hidden": list(hps[e]["hidden"])}, "trials": trials}
        logger.info("tuned (epsilon %g, seed 0): %s", e, hps[e])
    hp0 = hps[5] if 5 in hps else hps[eps_list[0]]
    runs: dict[str, list[dict[str, Any]]] = {label(e): [] for e in eps_list}
    runs[NONOISE] = []
    for seed in seeds:
        parts, _ = make_partition(cfg, data["train"]["y"], classes, seed=seed)
        for e in eps_list:
            key, h = f"run:{label(e)}:{seed}:{_hp_key(hps[e])}", hps[e]
            old = f"run:{label(e)}:{seed}:{h['lr']:g}:{h['clip']:g}:{h['batch']}" if h.get("sched", "const") == "const" else None   # a run of the compact search
            runs[label(e)].append(cache.get(key) or (old and cache.get(old)) or cache.put(key, run_one(label(e), cfg, data, classes, seed, parts, h, float(e))))
    for seed in seeds:
        parts, _ = make_partition(cfg, data["train"]["y"], classes, seed=seed)
        key = f"run:{NONOISE}:{seed}:{_hp_key(hp0)}"
        runs[NONOISE].append(cache.get(key) or cache.put(key, run_one(NONOISE, cfg, data, classes, seed, parts, hp0, None)))
    cl = {}
    for lab, rs in runs.items():
        cl[lab] = summarise(rs, classes, rare)
        d = [r["dp"] for r in rs]
        noisy = lab != NONOISE
        cl[lab]["dp"] = {"target_eps": float(lab[len(LABEL):]) if noisy else None, "eps_max_over_seeds": max(x["eps"] for x in d) if noisy else None,
                         "eps_one_honest": max(x["eps_one_honest"] for x in d) if noisy else None, "sigma": d[0]["sigma"], "q": d[0]["q"], "rounds": d[0]["rounds"],
                         "clipped_frac": float(np.mean([x["clipped_frac"] for x in d])), "quant_clipped_frac": float(max(x["quant_clipped_frac"] for x in d)),
                         "seconds_mean": float(np.mean([x["seconds"] for x in d]))}
    cl[NONOISE]["hp"] = {**hp0, "hidden": list(hp0["hidden"])}
    n_params = rs[0]["dp"]["n_params"]
    bpp = bytes_per_param_m2(cfg)
    rounds = max(cl[label(e)]["dp"]["rounds"] for e in eps_list)
    res["distributed"] = {"classifiers": cl, "tuning": {"per_eps": per_eps, "search": {a: list(b) for a, b in SEARCH.items()}, "seed": 0, "epochs": EPOCHS,
                                                         "selected_on": "validation macro-F1"}, "compact": compact,
                          "against": {label(e): against(e) for e in eps_list}, "references": references(cfg, rare, eps_list, res.get("protected") or {}),
                          "comparisons": compare(cfg, data["test"]["y"].astype(np.int64), classes, rare, seeds, int(n_boot or cfg["eval"]["bootstrap"]), eps_list),
                          "nonoise_comparison": compare_nonoise(cfg, data["test"]["y"].astype(np.int64), classes, rare, seeds, int(n_boot or cfg["eval"]["bootstrap"])),
                          "n_params": n_params, "bytes_per_param_round_m2": None if math.isnan(bpp) else bpp,
                          "bytes_total_est": None if math.isnan(bpp) else bpp * n_params * rounds, "delta": float(cfg["dp"]["delta"]), "seeds": seeds,
                          "num_clients": len(parts0), "config_hash": config_hash(cfg), "git_commit": git_commit()}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    logger.info("wrote %s", p)
    return res
