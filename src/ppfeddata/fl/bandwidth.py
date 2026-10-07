"""Optimisation O2 (bandwidth): M2 (and optionally M3o at one epsilon) with a compact SecAgg+ encoding of the masked vectors.

Levels (`LEVELS`): `c32` = modulus 2^32 sent as uint32 (quantisation 2^22 levels unchanged, so the aggregate is the same as M2 and only the
bytes change); `c16` = modulus 2^16 sent as uint16, quantisation 2^13 levels (5 clients x 2^13 < 2^16, so the sum cannot wrap; the step
is 2 x 16 / 2^13 ~ 0.004 on each weighted parameter, so the aggregate is coarser and utility may drop: that is what the sweep measures).
Each run is the full Flower SecAgg+ path with the per-round check of the sum (`secagg.verify_round`). Configs: `M2-c32`, `M2-c16`,
`M3o-t<n>-eps<e>-c16`, ... in the run ledger.
"""
from __future__ import annotations

import logging
from typing import Any

from ppfeddata.eval.baselines import load_data
from ppfeddata.eval.runs import RunLedger, runs_csv_path
from ppfeddata.fl import secagg
from ppfeddata.fl.b3 import fl_extra, load_fl_model
from ppfeddata.fl.m1 import _all_done
from ppfeddata.fl.m2 import sa_extra
from ppfeddata.fl.run import run_fl
from ppfeddata.models.b2 import evaluate_generator

logger = logging.getLogger("ppfeddata.fl.bandwidth")

LEVELS = {"c32": {"modulus_range": 2 ** 32, "quantization_range": 2 ** 22}, "c16": {"modulus_range": 2 ** 16, "quantization_range": 2 ** 13}}


def run_m2_levels(cfg: dict[str, Any], levels: list[str] | None = None, seeds: list[int] | None = None, resume: bool = True) -> list[str]:
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    data, schema = load_data(cfg)
    ledger, mode = RunLedger(runs_csv_path(cfg)), cfg["label_mode"]
    names = []
    for lv in levels or list(LEVELS):
        name = f"M2-{lv}"
        names.append(name)
        sa = secagg.secagg_spec(cfg, compact=True, **LEVELS[lv])
        for seed in seeds:
            if resume and _all_done(ledger, [name], seed, mode):
                logger.info("skip %s seed %d", name, seed)
                continue
            run = run_fl(cfg, seed, name, secagg=sa, resume=resume)
            s = run["summary"]
            logger.info("%s seed %d: %.0f KB/round (M2: int64), max|w| %.3f, val loss %.4f, %.0fs", name, seed, s["secagg"]["bytes_per_round"] / 1e3,
                        s["secagg"]["clip"]["max_abs_w"], s["final_val_loss"], s["fl_total_s"])
            evaluate_generator(cfg, load_fl_model(cfg, run, schema), name, seed, data, schema, ledger,
                               {**fl_extra(run), **sa_extra(run), "sa_compact": lv, "sa_modulus_bits": LEVELS[lv]["modulus_range"].bit_length() - 1})
    return names


def run_m3o_level(cfg: dict[str, Any], eps: float, level: str = "c16", seeds: list[int] | None = None, resume: bool = True) -> str:
    """The O1 winner at `eps` with SecAgg+ (M3o) and the compact encoding `level`; same evaluation as `tune_dp_full.run_final`."""
    from pathlib import Path

    import yaml

    from ppfeddata.fl import dp_stats
    from ppfeddata.fl.m1 import dp_extra
    from ppfeddata.tune import load_best_cvae
    from ppfeddata.tune_dp_full import BEST_PATH, final_name, run_kwargs

    s = yaml.safe_load(Path(BEST_PATH).read_text(encoding="utf-8"))["searches"][f"eps{eps:g}"]
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    data, schema = load_data(cfg)
    ledger, mode = RunLedger(runs_csv_path(cfg)), cfg["label_mode"]
    name = f"{final_name('M3', s['best_trial'], eps)}-{level}"
    kw = {**run_kwargs(load_best_cvae(cfg), s["params"], eps), "secagg": secagg.secagg_spec(cfg, compact=True, **LEVELS[level])}
    for seed in seeds:
        if resume and _all_done(ledger, [name, name + "-plain"], seed, mode):
            continue
        run = run_fl(cfg, seed, name, resume=resume, **kw)
        model = load_fl_model(cfg, run, schema)
        extra = {**fl_extra(run), **dp_extra(run), **sa_extra(run), "sa_compact": level, "dp_trial": int(s["best_trial"]),
                 "dp_stat_frac": float(s["params"]["stat_frac"]), "dp_sigma_stat": float(run["dp"]["sigma_stat"]), "dp_res_clip": float(s["params"]["res_clip"]),
                 "cw_power": float(s["params"]["cw_power"]), "fl_rounds": int(s["params"]["rounds"]), "fl_local_epochs": int(s["params"]["local_epochs"])}
        st = dp_stats.stats_for_run(model, run, data["train"]["X"], data["train"]["y"])
        evaluate_generator(cfg, model, name + "-plain", seed, data, schema, ledger, extra, use_stats=False)
        evaluate_generator(cfg, model, name, seed, data, schema, ledger, {**extra, "residual_noise_dp": True}, stats=st)
    return name
