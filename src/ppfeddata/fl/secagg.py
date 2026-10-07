"""Phase 10: secure aggregation (SecAgg+) for the federated CVAE, using Flower's own implementation (Flower 1.39).

Server: for every round the UNCHANGED `flwr.server.workflow.SecAggPlusWorkflow` runs its four stages (setup, share keys,
collect masked vectors, unmask) through the simulation grid and hands the weighted average to a thin legacy `Strategy`
(`SecAggStrategy`), which only counts steps and refuses to go on with a missing client. The surrounding round loop, the
validation, the round log and the checkpoints are the same code as for plain FedAvg (`fl/app.py`).
Client: `secaggplus_mod` wraps the same local training (plain or DP-SGD); the mod quantises, masks and uploads.

What this module adds on top of Flower, and why:
- `TapGrid`: a transparent wrapper around the grid that counts bytes and seconds per stage and (debug runs only) keeps the
  masked vectors the server receives, so that "the server sees uniform noise" can be tested (T-SA2).
- clip check: each client appends max|w| of its update to a file that is NOT part of the protocol (simulation diagnostic),
  because values beyond `clipping_range` would be clipped silently by the quantiser.
- diagnostic channel (debug runs only): each client also dumps its plain update to disk so that the server process can
  compare the secure aggregate with an ordinary weighted mean of the same updates (T-SA1). The aggregation itself never
  reads these files.
- Flower's quantiser uses stochastic rounding drawn from numpy's GLOBAL RNG; the client re-seeds that RNG from
  (run seed, round, client) just before the mod runs, so a secure run is reproducible and resumable like a plain one.

Not covered: network latency (single machine), a malicious server, and anything beyond what Flower's SecAgg+ gives.
Flower sends `num_examples` and the metrics dict in the clear; the clients here put only non-data-dependent numbers there.
"""
from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("ppfeddata.fl.secagg")

DEFAULTS: dict[str, Any] = {"num_shares": 5, "reconstruction_threshold": 3, "clipping_range": 16.0, "max_weight": 100000.0,
                            "quantization_range": 2 ** 22, "modulus_range": 2 ** 32}      # Flower's own defaults, except max_weight and clipping_range
RESEED_XOR = 0x5EC466


def secagg_spec(cfg: dict[str, Any], diagnostics: bool = False, debug_rounds: tuple[int, ...] = (), drop: dict[str, int] | None = None,
                allow_dropout: bool = False, compact: bool = False, **overrides) -> dict[str, Any]:
    """Run description for the SecAgg part of a spec. Flower's defaults for `quantization_range` and `modulus_range` are kept.
    Two parameters differ from Flower's defaults, both decided by checks (SPEC_DEVIATIONS 10.2, 10.3): `max_weight` must exceed
    the largest client (default 1000 is below the client sizes, 500 to ~70,000 rows; it is a public upper bound for n_i), and
    `clipping_range` is 16 instead of 8 because the weights of the trained CVAE reach 11.4. `compact` (O2): send the masked vectors as
    uint16 / uint32 instead of int64 (`set_compact`); needs modulus_range 2^16 or 2^32."""
    params = {**DEFAULTS, **{k: v for k, v in cfg.get("secagg", {}).items() if k in DEFAULTS}, **overrides}
    params["clipping_range"] = float(params["clipping_range"])
    params["max_weight"] = float(params["max_weight"])
    for k in ("num_shares", "reconstruction_threshold", "quantization_range", "modulus_range"):
        params[k] = int(params[k])
    if compact:
        check_compact(params, int(cfg["fl"]["num_clients"]))
    return {"params": params, "diagnostics": bool(diagnostics), "debug_rounds": sorted(int(r) for r in debug_rounds),
            "drop": drop, "allow_dropout": bool(allow_dropout), "compact": bool(compact)}


# --------------------------------------------------------------------------------------------------
# Compact encoding of the masked vectors (optimisation O2, bandwidth)
# --------------------------------------------------------------------------------------------------
# Flower's SecAgg+ client sends the masked vector as int64 whatever the modulus (8 bytes per parameter). When the modulus is 2^16 or 2^32,
# the values fit in uint16 / uint32: the client casts them after its final `mod`, and the server's sum of the uint arrays wraps around
# exactly modulo 2^16 / 2^32, which is the arithmetic of the protocol. NumPy 2 refuses `uint16 % 65536` (the Python int is out of range for
# the dtype), so the server-side `mod` casts to int64 first. Nothing else of the protocol changes. Enabled per run (`secagg.compact`).
COMPACT_DTYPES = {2 ** 16: np.uint16, 2 ** 32: np.uint32}
_ORIGINAL: dict[str, Any] = {}


def _client_mod(params, mod_range):
    out = _ORIGINAL["client"](params, mod_range)
    dt = COMPACT_DTYPES.get(int(mod_range))
    return [np.asarray(o).astype(dt) for o in out] if dt is not None else out


def _server_mod(params, mod_range):
    return _ORIGINAL["server"]([np.asarray(p).astype(np.int64) for p in params], mod_range)


def set_compact(on: bool) -> None:
    """Install (or remove) the compact encoding in this process (the client mod and the server workflow)."""
    import importlib

    # the packages re-export functions with the modules' names, so take the modules themselves
    cm = importlib.import_module("flwr.client.mod.secure_aggregation.secaggplus_mod")
    sw = importlib.import_module("flwr.server.workflow.secure_aggregation.secaggplus_workflow")
    _ORIGINAL.setdefault("client", cm.parameters_mod)
    _ORIGINAL.setdefault("server", sw.parameters_mod)
    cm.parameters_mod = _client_mod if on else _ORIGINAL["client"]
    sw.parameters_mod = _server_mod if on else _ORIGINAL["server"]


def check_compact(params: dict[str, Any], n_clients: int) -> None:
    """The modulus must be a compact width and hold the sum of the clients' quantised values (each < quantization_range, plus the
    quantised weight factor) without wrapping."""
    m = int(params["modulus_range"])
    if m not in COMPACT_DTYPES:
        raise ValueError(f"compact encoding needs modulus_range 2^16 or 2^32, got {m}")
    if n_clients * int(params["quantization_range"]) >= m:
        raise ValueError(f"{n_clients} clients x quantization_range {params['quantization_range']} >= modulus {m}: the sum would wrap")


def quantization_error_bound(n_clients: int, params: dict[str, Any], total_rows: int) -> float:
    """Hard bound on |secure average - exact weighted average| per coordinate. Each client's stochastic rounding errs by less
    than one quantisation step `2*clip/quantization_range` on its weighted value, the errors add over clients, and the sum is
    divided by the total weight `sum_i n_i / max_weight` (the weights themselves are rounded to 2^-22, a negligible term)."""
    step = 2.0 * params["clipping_range"] / params["quantization_range"]
    return float(n_clients * step / (total_rows / params["max_weight"]))


# --------------------------------------------------------------------------------------------------
# state <-> arrays
# --------------------------------------------------------------------------------------------------
def state_to_arrays(state: dict[str, torch.Tensor], dtype=np.float64) -> list[np.ndarray]:
    """float64: the quantiser works on float arrays, and float32 would add a rounding error of its own near the range limits."""
    return [v.detach().cpu().numpy().astype(dtype) for v in state.values()]


def arrays_to_state(arrays: list[np.ndarray], template: dict[str, torch.Tensor]) -> OrderedDict:
    if len(arrays) != len(template):
        raise ValueError(f"{len(arrays)} arrays for a model with {len(template)} tensors")
    return OrderedDict((k, torch.as_tensor(np.asarray(a), dtype=v.dtype).reshape(v.shape).clone()) for (k, v), a in zip(template.items(), arrays))


def flat(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(a).ravel() for a in arrays])


# --------------------------------------------------------------------------------------------------
# client-side simulation diagnostics (off the protocol path)
# --------------------------------------------------------------------------------------------------
def debug_dir(rdir: Path) -> Path:
    return rdir / "secagg_debug"


def client_diagnostics(sp: dict[str, Any], rdir: Path, rnd: int, pid: int, arrays: list[np.ndarray], n_rows: int | None = None) -> None:
    """Called by the client with its plain update. (1) max|w| for the clipping-range check, and the maximum of the WEIGHTED value
    w * n_i / max_weight, which is what the quantiser actually clips; (2) in debug runs, the update itself."""
    sa = sp["secagg"]
    rdir.mkdir(parents=True, exist_ok=True)
    mx = float(max(np.abs(a).max() for a in arrays))
    line = {"round": rnd, "pid": pid, "max_abs": mx}
    if n_rows is not None and sa.get("params"):
        line["max_abs_weighted"] = mx * float(n_rows) / float(sa["params"]["max_weight"])
    with open(rdir / f"clip_check_c{pid}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    if sa.get("diagnostics"):
        d = debug_dir(rdir)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"plain_r{rnd:04d}_c{pid}.npy", flat(arrays).astype(np.float32))


def clip_summary(rdir: Path, clipping_range: float) -> dict[str, Any]:
    """Largest |w| any client produced in any round (last write per (round, client) wins, a re-run overwrites a stale line).
    `ok` is the spec's conservative criterion (every raw |w| below the range); `ok_weighted` is what actually matters, since the
    quantiser clips the weighted value w * n_i / max_weight (None when the files carry no weighted maxima)."""
    best: dict[tuple[int, int], dict[str, float]] = {}
    for f in sorted(rdir.glob("clip_check_c*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                best[(r["round"], r["pid"])] = r
    vals = np.array([r["max_abs"] for r in best.values()], dtype=float)
    wvals = np.array([r["max_abs_weighted"] for r in best.values() if "max_abs_weighted" in r], dtype=float)
    return {"n_updates": int(len(vals)), "max_abs_w": float(vals.max()) if len(vals) else float("nan"), "clipping_range": float(clipping_range),
            "max_abs_weighted": float(wvals.max()) if len(wvals) else None, "ok_weighted": bool(wvals.max() < clipping_range) if len(wvals) else None,
            "n_beyond_range": int((vals >= clipping_range).sum()), "ok": bool(len(vals) and vals.max() < clipping_range)}


def reset_client_files(rdir: Path, from_round: int) -> None:
    """On resume, drop diagnostic lines of rounds that will be re-run."""
    for f in rdir.glob("clip_check_c*.jsonl"):
        keep = [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip() and json.loads(ln)["round"] <= from_round]
        f.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8")


# --------------------------------------------------------------------------------------------------
# server side
# --------------------------------------------------------------------------------------------------
def content_bytes(msg) -> int:
    """Size of the message content on the wire (protobuf serialisation, the format Flower sends over gRPC)."""
    from flwr.common.serde import recorddict_to_proto

    return int(recorddict_to_proto(msg.content).ByteSize())


class TapGrid:
    """Wraps the simulation grid. Everything is forwarded; each `send_and_receive` call is one protocol stage, for which the
    bytes sent / received and the seconds are kept. With `capture=True` the masked vectors of stage 2 are kept per client."""

    def __init__(self, grid, capture: bool = False):
        self._grid = grid
        self.capture = capture
        self.stages: list[dict[str, Any]] = []
        self.masked: dict[int, np.ndarray] = {}

    def __getattr__(self, name):
        return getattr(self._grid, name)

    def reset(self) -> None:
        self.stages, self.masked = [], {}

    def send_and_receive(self, messages, *args, **kwargs):
        from flwr.common import bytes_to_ndarray
        from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS, Key, Stage

        messages = list(messages)
        stage = str(messages[0].content.config_records[RECORD_KEY_CONFIGS][Key.STAGE]) if messages else "?"
        down = sum(content_bytes(m) for m in messages)
        t0 = time.perf_counter()
        replies = list(self._grid.send_and_receive(messages, *args, **kwargs))
        dt = time.perf_counter() - t0
        up = sum(content_bytes(m) for m in replies if not m.has_error())
        self.stages.append({"stage": stage, "seconds": dt, "bytes_down": int(down), "bytes_up": int(up), "n_replies": int(sum(not m.has_error() for m in replies))})
        if self.capture and stage == Stage.COLLECT_MASKED_VECTORS:
            for m in replies:
                if m.has_error():
                    continue
                pid = int(m.content.config_records["fitres.metrics"]["partition-id"])
                self.masked[pid] = np.concatenate([bytes_to_ndarray(b).ravel() for b in m.content.config_records[RECORD_KEY_CONFIGS][Key.MASKED_PARAMETERS]])
        return replies


def make_strategy_class():
    """The legacy `Strategy` is imported lazily so that importing this module does not pull in Flower's server package."""
    from flwr.common import FitIns
    from flwr.server.strategy import Strategy

    class SecAggStrategy(Strategy):
        """Feeds `SecAggPlusWorkflow`: asks every client for an update and receives the (already averaged) result back.
        Same bookkeeping attributes as `app.RoundStrategy`, so that `app.make_evaluate_fn` serves both."""

        def __init__(self, offset: int, num_clients: int, steps0: dict[int, int] | None = None, dp: bool = False, allow_dropout: bool = False):
            self.offset, self.num_clients, self.dp, self.allow_dropout = offset, num_clients, dp, allow_dropout
            self.steps = {int(k): int(v) for k, v in (steps0 or {}).items()}
            self.round_t0 = time.perf_counter()
            self.last: dict[str, Any] = {}
            self.aggregated = None
            self.n_by_pid: dict[int, int] = {}

        def initialize_parameters(self, client_manager):
            return None

        def configure_fit(self, server_round, parameters, client_manager):
            self.round_t0 = time.perf_counter()
            self.aggregated = None
            clients = sorted(client_manager.all().values(), key=lambda c: c.node_id)
            return [(c, FitIns(parameters, {"round": self.offset + server_round})) for c in clients]

        def aggregate_fit(self, server_round, results, failures):
            if (len(results) != self.num_clients or failures) and not self.allow_dropout:
                raise RuntimeError(f"round {self.offset + server_round}: {len(results)} of {self.num_clients} clients delivered "
                                   f"({len(failures)} failures); aborting so the run can be resumed")
            results = sorted(results, key=lambda r: int(r[1].metrics["partition-id"]))
            secs = []
            self.n_by_pid = {}
            for _, fr in results:
                pid = int(fr.metrics["partition-id"])
                self.steps[pid] = self.steps.get(pid, 0) + int(fr.metrics["steps"])
                secs.append(float(fr.metrics["train_seconds"]))
                self.n_by_pid[pid] = int(fr.num_examples)
            # every FitRes carries the same aggregated parameters (the workflow overwrites them); train_loss is not sent (data dependent)
            self.aggregated = results[0][1].parameters
            self.last = {"client_seconds": secs, "train_loss": None, "n_replies": len(secs), "agg_done": time.perf_counter()}
            return self.aggregated, {}

        def configure_evaluate(self, server_round, parameters, client_manager):
            return []

        def aggregate_evaluate(self, server_round, results, failures):
            return None, {}

        def evaluate(self, server_round, parameters):
            return None

    return SecAggStrategy


def run_rounds(sp: dict[str, Any], grid, context, strategy, state: OrderedDict, offset: int, last: int, evaluate, rdir: Path) -> OrderedDict:
    """Rounds offset+1 .. last of secure aggregation. `evaluate(server_round, ArrayRecord)` is `app.make_evaluate_fn(...)`."""
    from flwr.app import ArrayRecord, ConfigRecord
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.compat.common import recorddict_compat as compat
    from flwr.server import LegacyContext, ServerConfig, SimpleClientManager
    from flwr.server.compat.grid_client_proxy import GridClientProxy
    from flwr.server.workflow import SecAggPlusWorkflow
    from flwr.server.workflow.constant import MAIN_CONFIGS_RECORD, MAIN_PARAMS_RECORD, Key

    sa = sp["secagg"]
    p = sa["params"]
    tap = TapGrid(grid, capture=bool(sa.get("diagnostics")))
    lc = LegacyContext(context=context, config=ServerConfig(num_rounds=max(last - offset, 0)), strategy=strategy, client_manager=SimpleClientManager())
    for nid in sorted(grid.get_node_ids()):
        lc.client_manager.register(GridClientProxy(node_id=nid, grid=tap, run_id=grid.run.run_id))
    cfg = ConfigRecord()
    lc.state.config_records[MAIN_CONFIGS_RECORD] = cfg
    wf = SecAggPlusWorkflow(num_shares=p["num_shares"], reconstruction_threshold=p["reconstruction_threshold"], max_weight=p["max_weight"],
                            clipping_range=p["clipping_range"], quantization_range=p["quantization_range"], modulus_range=p["modulus_range"])
    template = OrderedDict((k, v.clone()) for k, v in state.items())
    evaluate(0, ArrayRecord(state))
    total_rows = None
    for i in range(1, last - offset + 1):
        rnd = offset + i
        cfg[Key.CURRENT_ROUND] = i
        # download in float32 (the model's own dtype, nothing is lost); the clients answer in float64 so the quantiser sees exact values
        lc.state.array_records[MAIN_PARAMS_RECORD] = compat.parameters_to_arrayrecord(ndarrays_to_parameters(state_to_arrays(state, np.float32)), True)
        tap.reset()
        wf(tap, lc)
        if strategy.aggregated is None:
            raise RuntimeError(f"round {rnd}: secure aggregation halted (see the Flower log); aborting so the run can be resumed")
        aggregate = parameters_to_ndarrays(strategy.aggregated)
        state = arrays_to_state(aggregate, template)
        strategy.round_bytes = int(sum(s["bytes_down"] + s["bytes_up"] for s in tap.stages))      # read by app.make_evaluate_fn
        evaluate(i, ArrayRecord(state))
        # ---- off-protocol bookkeeping: not part of the timed round
        total_rows = sum(strategy.n_by_pid.values())
        row = {"round": rnd, "stages": tap.stages, "bytes_down": int(sum(s["bytes_down"] for s in tap.stages)),
               "bytes_up": int(sum(s["bytes_up"] for s in tap.stages)), "stage_seconds": {s["stage"]: s["seconds"] for s in tap.stages},
               "n_clients": len(strategy.n_by_pid)}
        if sa.get("diagnostics"):
            row.update(verify_round(rdir, rnd, strategy.n_by_pid, flat(aggregate), keep=(rnd in sa.get("debug_rounds", [])), masked=tap.masked))
        with open(rdir / "secagg_rounds.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    return state


def verify_round(rdir: Path, rnd: int, n_by_pid: dict[int, int], aggregate: np.ndarray, keep: bool, masked: dict[int, np.ndarray]) -> dict[str, Any]:
    """Compare the secure aggregate (float64, before it is cast to the model's float32) with the plain weighted mean of the very
    same client updates, read from the diagnostic files."""
    d = debug_dir(rdir)
    pids = sorted(n_by_pid)
    files = [d / f"plain_r{rnd:04d}_c{pid}.npy" for pid in pids]
    ups = [np.load(f).astype(np.float64) for f in files]
    w = np.array([n_by_pid[pid] for pid in pids], dtype=np.float64)
    ref = np.average(np.stack(ups), axis=0, weights=w)
    got = aggregate
    out = {"agg_max_abs_err": float(np.abs(got - ref).max()), "agg_mean_abs_err": float(np.abs(got - ref).mean())}
    if keep and masked:
        np.savez_compressed(d / f"masked_r{rnd:04d}.npz", **{f"c{pid}": v for pid, v in masked.items()})
    if not keep:
        for f in files:
            f.unlink(missing_ok=True)
    return out


def read_secagg_rounds(rdir: Path) -> list[dict[str, Any]]:
    p = rdir / "secagg_rounds.jsonl"
    if not p.exists():
        return []
    rows = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["round"]] = r                     # a re-run round replaces the stale line
    return [rows[k] for k in sorted(rows)]


def truncate_secagg_rounds(rdir: Path, upto: int) -> None:
    rows = [r for r in read_secagg_rounds(rdir) if r["round"] <= upto]
    (rdir / "secagg_rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


STAGES = ("setup", "share_keys", "collect_masked_vectors", "unmask")


def summarize(spec: dict[str, Any], rdir: Path) -> dict[str, Any]:
    """Per-run numbers of the secure aggregation: bytes and seconds per stage (median over rounds 2+, round 1 includes starting Ray),
    the clipping-range check and, for debug runs, the aggregation error against the plain weighted mean."""
    p = spec["secagg"]["params"]
    rows = read_secagg_rounds(rdir)
    late = [r for r in rows if r["round"] > 1] or rows
    med = lambda xs: float(np.median(xs)) if len(xs) else float("nan")      # noqa: E731
    errs = [r["agg_max_abs_err"] for r in rows if "agg_max_abs_err" in r]
    rows_per_client = spec.get("partition_sizes", [])
    return {"params": p, "n_rounds": len(rows), "clip": clip_summary(rdir, p["clipping_range"]),
            "bytes_per_round": med([r["bytes_down"] + r["bytes_up"] for r in late]), "bytes_down": med([r["bytes_down"] for r in late]),
            "bytes_up": med([r["bytes_up"] for r in late]),
            "stage_seconds": {s: med([r["stage_seconds"].get(s, np.nan) for r in late]) for s in STAGES},
            "stage_bytes": {s: med([sum(x["bytes_down"] + x["bytes_up"] for x in r["stages"] if x["stage"] == s) for r in late]) for s in STAGES},
            "agg_max_abs_err": float(max(errs)) if errs else None, "agg_err_rounds": len(errs),
            "error_bound": quantization_error_bound(int(spec["num_clients"]), p, int(sum(rows_per_client))) if rows_per_client else None}


# --------------------------------------------------------------------------------------------------
# T-SA2: what does the server see?
# --------------------------------------------------------------------------------------------------
def masking_stats(masked: np.ndarray, plain: np.ndarray, params: dict[str, Any], n_rows: int, sample: int = 200_000, seed: int = 0) -> dict[str, float]:
    """`masked`: the vector the server received (first entry = the weight factor, then the parameters), `plain`: the client's real
    update (flat). Pearson correlation of the masked integers with the real update, and a Kolmogorov-Smirnov test of masked / modulus
    against U[0, 1). Positive control: the same two statistics for the UNMASKED quantised vector, which must correlate ~1 and is
    not uniform, so the test can tell the difference."""
    from scipy import stats

    mod = float(params["modulus_range"])
    m = masked[1:].astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(m), size=min(sample, len(m)), replace=False)
    rho = float(np.corrcoef(m[idx], plain[idx].astype(np.float64))[0, 1])
    ks = stats.kstest(m[idx] / mod, "uniform")
    ratio = n_rows / params["max_weight"]
    q_ratio = round(ratio * params["quantization_range"])
    x = np.clip(plain.astype(np.float64) * (q_ratio / params["quantization_range"]), -params["clipping_range"], params["clipping_range"])
    q = (x + params["clipping_range"]) * (params["quantization_range"] / (2 * params["clipping_range"]))
    rho_q = float(np.corrcoef(q[idx], plain[idx].astype(np.float64))[0, 1])
    ks_q = stats.kstest(q[idx] / mod, "uniform")
    return {"rho": rho, "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue), "rho_unmasked": rho_q, "ks_stat_unmasked": float(ks_q.statistic),
            "n_sampled": int(len(idx)), "min": float(m.min()), "max": float(m.max()), "mod": mod}
