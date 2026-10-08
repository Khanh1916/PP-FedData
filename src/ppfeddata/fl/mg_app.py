"""Optimisation O2(b): the three releases of FedDP-Marginal (`models/marginal.py`) through Flower's real SecAgg+ (simulation engine).

One Flower round per release (fine, pairs, edges). The server sends the public information of the release in the round's config (the
coarse maps after `fine`, the tree after `pairs`); every client computes its count tables, adds its share of the Skellam (or Gaussian)
noise, flattens them and answers through `secaggplus_mod`. Encoding chosen so that the integers pass exactly:

- every client reports `num_examples = max_weight`, so Flower's weight factor is exactly 1 and the server's weighted average is sum / K
  (multiplied back by K here);
- `quantization_range = 2 x clipping_range`: the quantisation step is 1, an integer value is mapped to an integer and Flower's stochastic
  rounding never moves it; `clipping_range` = 2^20 bounds |count + noise| (counts are at most the client size; it is checked);
- the masked vectors go as uint32 (`secagg.set_compact`, modulus 2^32 > K x 2^21).

Diagnostics (off-protocol): each client also writes its noisy vector to the run directory, and the server checks that the secure sum equals
their exact sum (`mg_rounds.jsonl`: bytes up / down per stage, seconds, max |difference|). The model is saved as `mg_model.pkl`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("ppfeddata.fl.mg_app")

SPEC_ENV = "PPFEDDATA_MG_SPEC"
CLIP = float(2 ** 20)


def secagg_params(cfg: dict[str, Any]) -> dict[str, Any]:
    from ppfeddata.fl import secagg

    return secagg.secagg_spec(cfg, compact=True, clipping_range=CLIP, quantization_range=int(2 * CLIP), modulus_range=2 ** 32)


@lru_cache(maxsize=1)
def _spec() -> dict[str, Any]:
    return json.loads(Path(os.environ[SPEC_ENV]).read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _client_data(pid: int):
    from ppfeddata.fl.dp_stats import load_parts
    from ppfeddata.models import marginal as mg

    sp = _spec()
    schema = json.loads((Path(sp["data_dir"]) / "feature_schema.json").read_text(encoding="utf-8"))
    tr = np.load(Path(sp["data_dir"]) / "train.npz")
    idx = load_parts(sp["partition_path"])[pid]
    attrs = mg.attributes(schema)
    F = mg.discretise_fine(tr["X"][idx], attrs, float(schema["settings"]["clip_sigma"]))
    return F, tr["y"][idx].astype(np.int64), attrs, len(schema["label_map"])


def client_vector(sp: dict[str, Any], pid: int, payload: dict[str, Any]) -> np.ndarray:
    """One client's flattened noisy tables of a release."""
    from ppfeddata.models import marginal as mg

    F, y, attrs, k = _client_data(pid)
    stage = payload["stage"]
    maps = [np.asarray(m, dtype=np.int64) for m in payload["maps"]] if payload.get("maps") else None
    edges = [tuple(e) for e in payload["edges"]] if payload.get("edges") else None
    tabs = mg.client_tables(stage, F, y, attrs, k, maps, edges)
    rng = np.random.default_rng([int(sp["seed"]), 911, mg.STAGES.index(stage), int(pid)])
    vec = np.concatenate([(t + mg.client_noise(t.shape, float(payload["scale"]), float(sp["noise_share"]), rng, sp["mechanism"])).ravel()
                          for t in tabs])
    if np.abs(vec).max() >= CLIP:
        raise RuntimeError(f"client {pid}, {stage}: |value| {np.abs(vec).max():.0f} >= clipping range {CLIP:.0f}")
    return vec


# --------------------------------------------------------------------------------------------------
# ClientApp
# --------------------------------------------------------------------------------------------------
from flwr.app import Context, Message  # noqa: E402
from flwr.client.mod import secaggplus_mod  # noqa: E402
from flwr.clientapp import ClientApp  # noqa: E402
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters  # noqa: E402
from flwr.compat.common import recorddict_compat as compat  # noqa: E402

client_app = ClientApp()


@client_app.train(mods=[secaggplus_mod])
def train(msg: Message, context: Context) -> Message:
    from ppfeddata.fl import secagg

    sp = _spec()
    secagg.set_compact(True)
    pid = int(context.node_config["partition-id"])
    fitins = compat.recorddict_to_fitins(msg.content, keep_input=True)
    payload = json.loads(str(fitins.config["payload"]))
    t0 = time.perf_counter()
    vec = client_vector(sp, pid, payload)
    np.save(Path(sp["artifacts_dir"]) / sp["run_id"] / f"client{pid}_{payload['stage']}.npy", vec)        # diagnostic, off-protocol
    fitres = FitRes(status=Status(code=Code.OK, message="OK"), parameters=ndarrays_to_parameters([vec]),
                    num_examples=int(sp["secagg"]["params"]["max_weight"]),
                    metrics={"partition-id": pid, "steps": 0, "train_seconds": float(time.perf_counter() - t0)})
    return Message(content=compat.fitres_to_recorddict(fitres, keep_input=True), reply_to=msg)


# --------------------------------------------------------------------------------------------------
# ServerApp
# --------------------------------------------------------------------------------------------------
from flwr.serverapp import Grid, ServerApp  # noqa: E402

server_app = ServerApp()


def _strategy_class():
    from flwr.common import FitIns
    from flwr.server.strategy import Strategy

    class ReleaseStrategy(Strategy):
        """Asks every client for its noisy tables of the current release; receives the secure (average) result back."""

        def __init__(self, num_clients: int):
            self.num_clients, self.payload, self.aggregated, self.pids = num_clients, "{}", None, []

        def initialize_parameters(self, client_manager):
            return None

        def configure_fit(self, server_round, parameters, client_manager):
            self.aggregated = None
            clients = sorted(client_manager.all().values(), key=lambda c: c.node_id)
            return [(c, FitIns(parameters, {"payload": self.payload})) for c in clients]

        def aggregate_fit(self, server_round, results, failures):
            if len(results) != self.num_clients or failures:
                raise RuntimeError(f"{len(results)} of {self.num_clients} clients delivered ({len(failures)} failures)")
            self.pids = sorted(int(fr.metrics["partition-id"]) for _, fr in results)
            self.aggregated = results[0][1].parameters
            return self.aggregated, {}

        def configure_evaluate(self, server_round, parameters, client_manager):
            return []

        def aggregate_evaluate(self, server_round, results, failures):
            return None, {}

        def evaluate(self, server_round, parameters):
            return None

    return ReleaseStrategy


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.compat.common import recorddict_compat as rc
    from flwr.server import LegacyContext, ServerConfig, SimpleClientManager
    from flwr.server.compat.grid_client_proxy import GridClientProxy
    from flwr.server.workflow import SecAggPlusWorkflow
    from flwr.server.workflow.constant import MAIN_CONFIGS_RECORD, MAIN_PARAMS_RECORD, Key
    from flwr.app import ConfigRecord

    from ppfeddata.fl import secagg
    from ppfeddata.models import marginal as mg

    sp = _spec()
    secagg.set_compact(True)
    rdir = Path(sp["artifacts_dir"]) / sp["run_id"]
    schema = json.loads((Path(sp["data_dir"]) / "feature_schema.json").read_text(encoding="utf-8"))
    attrs, k, clip = mg.attributes(schema), len(schema["label_map"]), float(schema["settings"]["clip_sigma"])
    A, K = len(attrs), int(sp["num_clients"])
    scales = mg.noise_scales(float(sp["eps"]), float(sp["delta"]), A, sp["split"], sp["mechanism"])
    p = sp["secagg"]["params"]
    strategy = _strategy_class()(K)
    tap = secagg.TapGrid(grid)
    lc = LegacyContext(context=context, config=ServerConfig(num_rounds=1), strategy=strategy, client_manager=SimpleClientManager())
    for nid in sorted(grid.get_node_ids()):
        lc.client_manager.register(GridClientProxy(node_id=nid, grid=tap, run_id=grid.run.run_id))
    cfgrec = ConfigRecord()
    lc.state.config_records[MAIN_CONFIGS_RECORD] = cfgrec
    wf = SecAggPlusWorkflow(num_shares=p["num_shares"], reconstruction_threshold=p["reconstruction_threshold"], max_weight=p["max_weight"],
                            clipping_range=p["clipping_range"], quantization_range=p["quantization_range"], modulus_range=p["modulus_range"])
    (rdir / "mg_rounds.jsonl").unlink(missing_ok=True)
    out: dict[str, list[np.ndarray]] = {}
    maps = edges = W = None
    n_cells = 0
    for i, stage in enumerate(mg.STAGES, start=1):
        strategy.payload = json.dumps({"stage": stage, "scale": scales[stage], "maps": [m.tolist() for m in maps] if maps is not None else None,
                                       "edges": [list(e) for e in edges] if edges is not None else None})
        cfgrec[Key.CURRENT_ROUND] = i
        lc.state.array_records[MAIN_PARAMS_RECORD] = rc.parameters_to_arrayrecord(ndarrays_to_parameters([np.zeros(1, np.float32)]), True)
        tap.reset()
        t0 = time.perf_counter()
        wf(tap, lc)
        secs = time.perf_counter() - t0
        if strategy.aggregated is None:
            raise RuntimeError(f"{stage}: secure aggregation halted")
        total = np.rint(parameters_to_ndarrays(strategy.aggregated)[0] * len(strategy.pids))
        exact = sum(np.load(rdir / f"client{pid}_{stage}.npy") for pid in strategy.pids)      # off-protocol check
        shapes = [t.shape for t in mg.client_tables(stage, np.zeros((1, A), np.int64), np.zeros(1, np.int64), attrs, k, maps, edges)]
        tabs, pos = [], 0
        for sh in shapes:
            n = int(np.prod(sh))
            tabs.append(total[pos:pos + n].reshape(sh))
            pos += n
        n_cells += pos
        out[stage] = tabs
        row = {"stage": stage, "round": i, "n_values": int(pos), "max_abs_diff_vs_exact_sum": float(np.abs(total - exact).max()),
               "bytes_up": int(sum(s["bytes_up"] for s in tap.stages)), "bytes_down": int(sum(s["bytes_down"] for s in tap.stages)),
               "seconds": secs, "stage_seconds": {s["stage"]: s["seconds"] for s in tap.stages}}
        with open(rdir / "mg_rounds.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        if stage == "fine":
            maps = mg.coarse_maps(tabs, attrs, int(sp["bins"]))
        elif stage == "pairs":
            W = mg.mi_matrix(tabs, A)
            edges = mg.max_spanning_tree(W)
    model = mg.assemble(attrs, k, clip, out["fine"], maps, edges, out["edges"], scales, float(sp["eps"]), float(sp["delta"]), float(sp["noise_share"]),
                        sp["mechanism"], int(sp["bins"]), sp["split"], K, n_cells, mi=W)
    model.info["eps_by_honest_clients"] = mg.eps_honest_curve(scales, A, float(sp["delta"]), K, sp["mechanism"])
    with open(rdir / "mg_model.pkl", "wb") as fh:
        pickle.dump(model, fh)


# --------------------------------------------------------------------------------------------------
# Launcher (one process per run, as fl/run.py)
# --------------------------------------------------------------------------------------------------
def run(cfg: dict[str, Any], seed: int, name: str, eps: float, bins: int, split, mechanism: str = "skellam", resume: bool = True) -> dict[str, Any]:
    """FedDP-Marginal with distributed noise through Flower SecAgg+; returns the model and the per-stage measurements."""
    from ppfeddata.data.preprocess import processed_dir
    from ppfeddata.eval.runs import run_id
    from ppfeddata.partition import make_partition, partition_path

    ddir = processed_dir(cfg)
    schema = json.loads((ddir / "feature_schema.json").read_text(encoding="utf-8"))
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    y = np.load(ddir / "train.npz")["y"]
    alpha, K = float(cfg["fl"]["dirichlet_alpha"]), int(cfg["fl"]["num_clients"])
    make_partition(cfg, y, classes, alpha, seed, K)
    rid = run_id(name, seed, cfg["label_mode"])
    rdir = Path(cfg["compute"]["artifacts_dir"]) / rid
    rdir.mkdir(parents=True, exist_ok=True)
    sp = {"run_id": rid, "seed": int(seed), "eps": float(eps), "delta": float(cfg["dp"]["delta"]), "bins": int(bins), "split": list(split),
          "mechanism": mechanism, "noise_share": 1.0 / K, "num_clients": K, "data_dir": str(ddir), "partition_path": str(partition_path(cfg, alpha, seed)),
          "artifacts_dir": str(cfg["compute"]["artifacts_dir"]), "secagg": secagg_params(cfg), "label_mode": cfg["label_mode"]}
    model_p, spec_p = rdir / "mg_model.pkl", rdir / "spec.json"
    same = spec_p.exists() and json.loads(spec_p.read_text(encoding="utf-8")) == sp
    if not (resume and same and model_p.exists()):
        spec_p.write_text(json.dumps(sp, indent=2), encoding="utf-8")
        with open(rdir / "fl.log", "a", encoding="utf-8") as log:
            r = subprocess.run([sys.executable, "-m", "ppfeddata.fl.mg_app", "--spec", str(spec_p)], stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONUTF8": "1"})
        if r.returncode != 0 or not model_p.exists():
            raise RuntimeError(f"FedDP-Marginal run {rid} failed (exit {r.returncode}); see {rdir / 'fl.log'}")
    with open(model_p, "rb") as fh:
        model = pickle.load(fh)
    rounds = [json.loads(line) for line in (rdir / "mg_rounds.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"spec": sp, "model": model, "rounds": rounds, "bytes_total": int(sum(r["bytes_up"] + r["bytes_down"] for r in rounds)),
            "seconds": float(sum(r["seconds"] for r in rounds)), "max_abs_diff": float(max(r["max_abs_diff_vs_exact_sum"] for r in rounds))}


def main_cli(argv: list[str] | None = None) -> None:
    from flwr.simulation import run_simulation

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args(argv)
    os.environ[SPEC_ENV] = str(Path(a.spec).resolve())
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")
    sp = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    from ppfeddata.fl import mg_app as m          # by its package name: under `-m` this module is `__main__`, which Ray actors cannot import
    run_simulation(server_app=m.server_app, client_app=m.client_app, num_supernodes=int(sp["num_clients"]),
                   backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}, "init_args": {"include_dashboard": False, "logging_level": "ERROR"}})


if __name__ == "__main__":
    main_cli()
