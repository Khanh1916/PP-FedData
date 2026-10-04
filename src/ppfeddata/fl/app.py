"""Phase 8: the Flower app (ServerApp + ClientApp) around `fl/core.py`. Built for Flower 1.39 (message API; the template
was obtained with `flwr new @flwrlabs/quickstart-pytorch` and the model/data replaced by the CVAE and the partitions).

The run description is read from the JSON file named by the environment variable PPFEDDATA_FL_SPEC, which both the
ServerApp (this process) and the ClientApps (Ray workers, which inherit the environment) read.
"""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from ppfeddata.eval.overhead import bytes_per_round, peak_rss_gb
from ppfeddata.fl import core
from ppfeddata.models.cvae import build_layout

SPEC_ENV = "PPFEDDATA_FL_SPEC"


@lru_cache(maxsize=4)
def load_spec(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _spec() -> dict[str, Any]:
    return load_spec(os.environ[SPEC_ENV])


@lru_cache(maxsize=2)
def _data(data_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(Path(data_dir) / f"{split}.npz")
    return z["X"], z["y"]


@lru_cache(maxsize=2)
def _layout_and_k(data_dir: str):
    schema = json.loads((Path(data_dir) / "feature_schema.json").read_text(encoding="utf-8"))
    return build_layout(schema), len(schema["label_map"])


@lru_cache(maxsize=2)
def _partition(path: str) -> list[np.ndarray]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return [np.asarray(p, dtype=np.int64) for p in d["parts"]]


# --------------------------------------------------------------------------------------------------
# ClientApp
# --------------------------------------------------------------------------------------------------
client_app = ClientApp()


@client_app.train()
def train(msg: Message, context: Context) -> Message:
    sp = _spec()
    torch.set_num_threads(int(sp.get("torch_threads", 2)))
    pid = int(context.node_config["partition-id"])
    X, y = _data(sp["data_dir"], "train")
    idx = _partition(sp["partition_path"])[pid]
    layout, k = _layout_and_k(sp["data_dir"])
    state = msg.content["arrays"].to_torch_state_dict()
    rnd = int(msg.content["config"]["round"])
    dp = sp.get("dp")
    if dp:
        out = core.local_train_dp(state, X[idx], y[idx], layout, k, sp["hp"], int(sp["local_epochs"]), rnd, int(sp["seed"]), pid,
                                  float(dp["sigmas"][pid]), float(dp["max_grad_norm"]))
    else:
        out = core.local_train(state, X[idx], y[idx], layout, k, sp["hp"], int(sp["local_epochs"]), rnd, int(sp["seed"]), pid)
    metrics = MetricRecord({"num-examples": out["n"], "train_loss": out["loss"], "steps": out["steps"],
                            "train_seconds": out["seconds"], "partition-id": pid})
    return Message(content=RecordDict({"arrays": ArrayRecord(out["state"]), "metrics": metrics}), reply_to=msg)


# --------------------------------------------------------------------------------------------------
# ServerApp
# --------------------------------------------------------------------------------------------------
def require_all_replies(replies: list, expected: int, server_round: int) -> None:
    """Flower carries on with whatever replies arrive. For this study a round with a missing or failed client is not the
    experiment that was planned (and would silently change the DP accounting), so stop; `run_fl` resumes from the checkpoint."""
    ok = [m for m in replies if not m.has_error()]
    if len(ok) != expected:
        raise RuntimeError(f"round {server_round}: {len(ok)} of {expected} clients replied; aborting so the run can be resumed")


class RoundStrategy(FedAvg):
    """FedAvg weighted by the number of local rows, with the absolute round number added to the train config and the
    per-client step counters / timings kept for the round log and the checkpoint."""

    def __init__(self, offset: int, num_clients: int, steps0: dict[int, int] | None = None, dp: bool = False):
        super().__init__(fraction_train=1.0, fraction_evaluate=0.0, min_train_nodes=num_clients,
                         min_evaluate_nodes=0, min_available_nodes=num_clients)
        self.offset = offset
        self.num_clients = num_clients
        self.steps = {int(k): int(v) for k, v in (steps0 or {}).items()}
        self.dp = dp
        self.round_t0 = time.perf_counter()
        self.last: dict[str, Any] = {}

    def configure_train(self, server_round, arrays, config, grid):
        self.round_t0 = time.perf_counter()
        cfg = ConfigRecord({**dict(config), "round": self.offset + server_round})
        return super().configure_train(server_round, arrays, cfg, grid)

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        require_all_replies(replies, self.num_clients, self.offset + server_round)
        # replies arrive in completion order; summing in a fixed (client-id) order makes the float32 average reproducible
        replies = sorted(replies, key=lambda m: int(m.content["metrics"]["partition-id"]))
        secs, losses, ns = [], [], []
        for m in replies:
            if m.has_error():
                continue
            mt = m.content["metrics"]
            pid = int(mt["partition-id"])
            self.steps[pid] = self.steps.get(pid, 0) + int(mt["steps"])
            secs.append(float(mt["train_seconds"]))
            losses.append(float(mt["train_loss"]))
            ns.append(int(mt["num-examples"]))
        arrays, metrics = super().aggregate_train(server_round, replies)
        self.last = {"client_seconds": secs, "train_loss": float(np.average(losses, weights=ns)) if ns else float("nan"),
                     "n_replies": len(secs), "agg_done": time.perf_counter()}
        return arrays, metrics


def make_evaluate_fn(sp: dict[str, Any], strategy: RoundStrategy, rdir: Path):
    layout, k = _layout_and_k(sp["data_dir"])
    Xv, yv = _data(sp["data_dir"], "val")
    total, every, ck_every = int(sp["rounds"]), int(sp["eval_every"]), int(sp["checkpoint_every"])

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        rnd = strategy.offset + server_round
        state = arrays.to_torch_state_dict()
        evaluated = rnd % every == 0 or rnd == total or server_round == 0
        val = core.val_loss(state, Xv, yv, layout, k, sp["hp"]) if evaluated else None
        if server_round == 0:
            if strategy.offset == 0:
                core.append_round_log(rdir, {"round": 0, "val_loss": val["loss"], "val_recon_num": val["recon_num"],
                                             "val_recon_bin": val["recon_bin"], "val_recon_cat": val["recon_cat"], "val_kl": val["kl"]})
            return MetricRecord({"val_loss": val["loss"]})
        size = core.model_bytes_of({n: v.numpy() for n, v in state.items()})
        row = {"round": rnd, "round_seconds": time.perf_counter() - strategy.round_t0,
               "client_seconds": strategy.last.get("client_seconds", []), "train_loss": strategy.last.get("train_loss"),
               "bytes": bytes_per_round(size, int(sp["num_clients"])), "peak_rss_gb": peak_rss_gb(),
               "client_steps": dict(strategy.steps)}
        if sp.get("dp"):
            row["client_dp_steps"] = dict(strategy.steps)          # every counted step of a DP client is a DP step
        if val is not None:
            row.update({"val_loss": val["loss"], "val_recon_num": val["recon_num"], "val_recon_bin": val["recon_bin"],
                        "val_recon_cat": val["recon_cat"], "val_kl": val["kl"]})
        core.append_round_log(rdir, row)
        if rnd % ck_every == 0 or rnd == total:
            core.save_checkpoint(rdir, state, rnd, strategy.steps, dp_steps=strategy.steps if sp.get("dp") else None)
        return MetricRecord({"val_loss": val["loss"]} if val is not None else {"evaluated": 0.0})

    return evaluate


server_app = ServerApp()


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    sp = _spec()
    rdir = Path(sp["artifacts_dir"]) / sp["run_id"]
    layout, k = _layout_and_k(sp["data_dir"])
    ck = core.load_checkpoint(rdir) if sp.get("resume") else None
    if ck is not None and int(ck["round"]) < int(sp["rounds"]):
        offset, state, steps0 = int(ck["round"]), ck["state"], ck["client_steps"]
        core.truncate_round_log(rdir, offset)
    elif ck is not None:
        return                                           # nothing left to do
    else:
        offset, steps0 = 0, {}
        state = core.init_state(layout, k, sp["hp"], int(sp["seed"]))
        (rdir / "rounds.jsonl").unlink(missing_ok=True)
    strategy = RoundStrategy(offset, int(sp["num_clients"]), steps0, bool(sp.get("dp")))
    last = int(sp["stop_after"]) if sp.get("stop_after") else int(sp["rounds"])    # stop_after = simulated interruption
    result = strategy.start(grid=grid, initial_arrays=ArrayRecord(state), num_rounds=max(last - offset, 0),
                            train_config=ConfigRecord({"lr": float(sp["hp"]["lr"])}),
                            evaluate_fn=make_evaluate_fn(sp, strategy, rdir))
    rdir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(result.arrays.to_torch_state_dict()), rdir / "final_state.pt")
