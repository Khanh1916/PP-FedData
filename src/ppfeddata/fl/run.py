"""Phase 8: launch one federated run (Flower simulation, single machine) and summarise it.

Each run executes in its own process (`python -m ppfeddata.fl.run --spec <json>`): Ray is started and stopped per run and
the peak RAM of the run is not mixed with the caller's. The run description is written to `artifacts/<run_id>/spec.json`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ppfeddata.data.preprocess import processed_dir
from ppfeddata.eval.runs import run_id
from ppfeddata.fl import core, dp_utils
from ppfeddata.partition import make_partition, partition_path
from ppfeddata.tune import load_best_cvae

logger = logging.getLogger("ppfeddata.fl.run")


def build_spec(cfg: dict[str, Any], seed: int, name: str = "B3", alpha: float | None = None, num_clients: int | None = None,
               rounds: int | None = None, local_epochs: int | None = None, resume: bool = True,
               data_dir: str | Path | None = None, hp: dict[str, Any] | None = None,
               torch_threads: int = 2, target_eps: float | None = None, sigma: float | None = None,
               max_grad_norm: float | None = None, delta: float | None = None, stop_after: int | None = None,
               secagg: dict[str, Any] | None = None, save_states: bool = False, agg_noise: float = 0.0,
               agg_noise_seed: int = 0, stat_frac: float = 0.0, res_clip: float = 1.0) -> dict[str, Any]:
    """Partition the train pool (stored on disk) and describe the run."""
    fl = cfg["fl"]
    ddir = Path(data_dir) if data_dir is not None else processed_dir(cfg)
    schema = json.loads((ddir / "feature_schema.json").read_text(encoding="utf-8"))
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    y = np.load(ddir / "train.npz")["y"]
    alpha = float(fl["dirichlet_alpha"] if alpha is None else alpha)
    k = int(fl["num_clients"] if num_clients is None else num_clients)
    parts, meta = make_partition(cfg, y, classes, alpha, seed, k)
    pp = partition_path(cfg, alpha, seed)
    if k != int(fl["num_clients"]):
        pp = pp.with_name(pp.stem + f"_k{k}.json")
    rid = run_id(name, seed, cfg["label_mode"])
    rounds_, le_ = int(fl["rounds"] if rounds is None else rounds), int(fl["local_epochs"] if local_epochs is None else local_epochs)
    hp_ = dict(hp if hp is not None else load_best_cvae(cfg))
    dp = None
    if target_eps is not None or sigma is not None:
        dcfg = cfg["dp"]
        delta_ = float(dcfg["delta"] if delta is None else delta)
        clip = float(dcfg["max_grad_norm"] if max_grad_norm is None else max_grad_norm)
        # O1(a): part of epsilon for the DP residual statistics (dp_stats.py), composed with the training
        sigma_stat = dp_utils.stat_noise_multiplier(target_eps, stat_frac, delta_) if (stat_frac and target_eps is not None) else None
        if sigma is not None:                                   # explicit noise (sanity runs, e.g. sigma = 0)
            sigmas = [float(sigma)] * k
        else:
            sigmas = dp_utils.calibrate_clients(meta["sizes"], target_eps, delta_, int(hp_["batch_size"]), rounds_, le_, sigma_stat)
        dp = {"target_eps": None if target_eps is None else float(target_eps), "delta": delta_, "max_grad_norm": clip, "sigmas": sigmas}
        if sigma_stat:
            dp.update(stat_frac=float(stat_frac), sigma_stat=float(sigma_stat), res_clip=float(res_clip))
    return {"run_id": rid, "dp": dp, "secagg": secagg, "save_states": bool(save_states), "agg_noise": float(agg_noise),
            "agg_noise_seed": int(agg_noise_seed),
            "stop_after": None if stop_after is None else int(stop_after), "name": name, "seed": int(seed), "num_clients": k, "alpha": alpha,
            "rounds": rounds_, "local_epochs": le_, "hp": hp_, "data_dir": str(ddir), "partition_path": str(pp),
            "artifacts_dir": str(cfg["compute"]["artifacts_dir"]), "checkpoint_every": int(cfg["compute"]["checkpoint_every_rounds"]),
            "eval_every": 5, "torch_threads": int(torch_threads), "resume": bool(resume), "label_mode": cfg["label_mode"],
            "partition_sizes": meta["sizes"]}


def write_spec(spec: dict[str, Any]) -> Path:
    p = Path(spec["artifacts_dir"]) / spec["run_id"] / "spec.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return p


def run_simulation_from_spec(spec_path: str | Path) -> None:
    """Runs inside the child process (or directly in tests)."""
    from flwr.simulation import run_simulation

    from ppfeddata.fl import app

    os.environ[app.SPEC_ENV] = str(Path(spec_path).resolve())
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")
    sp = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    run_simulation(server_app=app.server_app, client_app=app.secagg_client_app if sp.get("secagg") else app.client_app,
                   num_supernodes=int(sp["num_clients"]),
                   backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0},
                                   "init_args": {"include_dashboard": False, "logging_level": "ERROR"}})


def summarize(spec: dict[str, Any], wall_s: float | None = None) -> dict[str, Any]:
    """Per-run numbers from the round log."""
    rdir = Path(spec["artifacts_dir"]) / spec["run_id"]
    rows = [r for r in core.read_round_log(rdir) if r["round"] > 0]
    vals = [r for r in rows if r.get("val_loss") is not None]
    secs = [r["round_seconds"] for r in rows]
    dp_info = None
    if spec.get("dp") and rows:
        dp = spec["dp"]
        dp_info = dp_utils.epsilon_table(spec["partition_sizes"], dp["sigmas"], rows[-1].get("client_dp_steps", {}),
                                         int(spec["hp"]["batch_size"]), dp["delta"], dp["target_eps"], dp.get("sigma_stat"))
    sa_info = None
    if spec.get("secagg") and rows:
        from ppfeddata.fl import secagg
        sa_info = secagg.summarize(spec, rdir)
    return {"dp": dp_info, "secagg": sa_info, "rounds_done": len(rows), "fl_total_s": float(sum(secs)), "mean_round_s": float(np.mean(secs)) if secs else float("nan"),
            "bytes_per_round": int(rows[-1]["bytes"]) if rows else 0, "server_peak_rss_gb": max([r["peak_rss_gb"] for r in rows], default=0.0),
            "final_val_loss": float(vals[-1]["val_loss"]) if vals else float("nan"),
            "client_seconds_mean": float(np.mean([s for r in rows for s in r["client_seconds"]])) if rows else float("nan"),
            "wall_s": wall_s, "client_steps": rows[-1]["client_steps"] if rows else {}}


def run_fl(cfg: dict[str, Any], seed: int, name: str = "B3", in_process: bool = False, max_retries: int = 2, **kw) -> dict[str, Any]:
    """Run (or resume) one FL run; returns the spec plus the summary. Skips a run whose final state already exists."""
    spec = build_spec(cfg, seed, name, **kw)
    rdir = Path(spec["artifacts_dir"]) / spec["run_id"]
    old = rdir / "spec.json"
    if spec["resume"] and spec["dp"] and old.exists() and core.load_checkpoint(rdir) is not None:
        prev = json.loads(old.read_text(encoding="utf-8")).get("dp")
        if prev and not np.allclose(prev["sigmas"], spec["dp"]["sigmas"]):
            raise RuntimeError(f"{spec['run_id']}: sigma was calibrated for a different plan (rounds/epsilon/delta changed); "
                               "resuming would break the privacy accounting. Use stop_after to cut a run short instead.")
    t0 = time.perf_counter()
    ck = core.load_checkpoint(rdir) if spec["resume"] else None
    finished = (rdir / "final_state.pt").exists() and ck is not None and int(ck["round"]) >= spec["rounds"]
    path = old if finished and old.exists() else write_spec(spec)      # a finished run keeps the spec it was run with
    if not finished:
        if in_process:
            run_simulation_from_spec(path)
        else:
            env = {**os.environ, "PYTHONUTF8": "1"}
            for attempt in range(max_retries + 1):
                # a retry resumes from the last checkpoint (exact: seeds depend only on run seed, round and client)
                if attempt:
                    spec["resume"] = True
                    path = write_spec(spec)
                    logger.warning("%s: attempt %d failed, resuming from the checkpoint", spec["run_id"], attempt)
                with open(rdir / "fl.log", "a", encoding="utf-8") as log:
                    r = subprocess.run([sys.executable, "-m", "ppfeddata.fl.run", "--spec", str(path)], stdout=log,
                                       stderr=subprocess.STDOUT, env=env)
                if r.returncode == 0:
                    break
            if r.returncode != 0:
                raise RuntimeError(f"FL run {spec['run_id']} failed (exit {r.returncode}) after {max_retries + 1} attempts; see {rdir / 'fl.log'}")
    if not (rdir / "final_state.pt").exists():
        raise RuntimeError(f"FL run {spec['run_id']} did not produce final_state.pt; see {rdir}")
    return {**spec, "summary": summarize(spec, time.perf_counter() - t0)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    run_simulation_from_spec(ap.parse_args(argv).spec)


if __name__ == "__main__":
    main()
