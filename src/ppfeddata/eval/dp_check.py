"""Phase 12 red flag: epsilon of a DP run recomputed WITHOUT Opacus.

The Sampled Gaussian mechanism (Poisson subsampling with rate q, noise multiplier sigma) has Renyi-DP at integer order a (Mironov, Talwar and
Zhang, 2019)

    A_a = sum_{k=0..a} C(a, k) (1 - q)^(a - k) q^k exp((k^2 - k) / (2 sigma^2)),      RDP(a) = log(A_a) / (a - 1)   per step,

and T steps compose additively. The conversion to (eps, delta) is the improved bound of Balle et al. (2020), the same one Opacus applies:

    eps = T * RDP(a) + log((a - 1) / a) - (log(delta) + log(a)) / (a - 1),         minimised over the order a.

Only integer orders are used here (Opacus also tries fractional orders below 11), so the value is the same or slightly larger than the one
Opacus reports; a large gap in either direction means the step counters, the noise multipliers or the accountant are wrong.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import gammaln, logsumexp, xlog1py, xlogy

ORDERS = tuple(range(2, 257))


def log_a_int(q: float, sigma: float, alpha: int) -> float:
    """log A_alpha of the Sampled Gaussian mechanism, summed in log space."""
    k = np.arange(alpha + 1, dtype=float)
    logs = (gammaln(alpha + 1) - gammaln(k + 1) - gammaln(alpha - k + 1) + xlog1py(alpha - k, -q) + xlogy(k, q) + (k * k - k) / (2.0 * sigma ** 2))
    return float(logsumexp(logs))


def rdp_int(q: float, sigma: float, alpha: int) -> float:
    return log_a_int(q, sigma, alpha) / (alpha - 1)


def epsilon_independent(sigma: float, q: float, steps: int, delta: float, orders: Iterable[int] = ORDERS, sigma_stat: float | None = None) -> float:
    """eps of `steps` Sampled-Gaussian steps; 0 steps -> 0, sigma = 0 -> infinity. `sigma_stat` adds one Gaussian release without
    subsampling (RDP a / (2 sigma_stat^2), the residual statistics of optimisation O1)."""
    if steps <= 0 and not sigma_stat:
        return 0.0
    if (steps > 0 and sigma <= 0) or (sigma_stat is not None and sigma_stat <= 0):
        return float("inf")
    best = float("inf")
    for a in orders:
        rdp = (steps * rdp_int(q, sigma, int(a)) if steps > 0 else 0.0) + (a / (2.0 * sigma_stat ** 2) if sigma_stat else 0.0)
        eps = rdp + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        best = min(best, eps)
    return max(best, 0.0)


def loader_steps_per_epoch(n: int, batch_size: int) -> int:
    """len(DPDataLoader) of Opacus: int(1 / sample_rate) with sample_rate = 1 / ceil(n / batch_size). The float division can land just below
    the integer (1 / (1 / 99) = 98.99999999999999), so a few client sizes run one step per epoch fewer than ceil(n / batch_size)."""
    return int(1.0 / (1.0 / math.ceil(n / batch_size)))


def recompute_run(spec: Mapping[str, Any], counted_steps: Mapping[Any, int] | None, reported_eps_max: float) -> dict[str, Any]:
    """Recompute the epsilon of one DP run from its spec (client sizes, noise multipliers, batch size, rounds, local epochs, delta).

    `eps_counted` uses the step counters the run logged (what the report uses); the counters are checked against the plan
    (rounds x local epochs x ceil(n_i / B)) and against the length Opacus' Poisson loader really has (`loader_steps_per_epoch`).
    The configuration's epsilon is the maximum over clients. The sampling rate is the loader's: 1 / ceil(n_i / B).
    """
    dp = spec["dp"]
    sizes, sigmas, bs = spec["partition_sizes"], dp["sigmas"], int(spec["hp"]["batch_size"])
    delta, rounds, epochs = float(dp["delta"]), int(spec["rounds"]), int(spec["local_epochs"])
    counted_steps = counted_steps or {}
    ss = dp.get("sigma_stat")
    rows = []
    for i, (n, s) in enumerate(zip(sizes, sigmas)):
        per_epoch = math.ceil(int(n) / bs)
        q = 1.0 / per_epoch
        planned = rounds * epochs * per_epoch
        loader = rounds * epochs * loader_steps_per_epoch(int(n), bs)
        counted = int(counted_steps.get(str(i), counted_steps.get(i, 0)))
        rows.append({"client": i, "n": int(n), "sigma": float(s), "q": q, "planned_steps": planned, "loader_steps": loader, "counted_steps": counted,
                     "eps_planned": epsilon_independent(float(s), q, planned, delta, sigma_stat=ss), "eps_counted": epsilon_independent(float(s), q, counted, delta, sigma_stat=ss)})
    eps = max(r["eps_counted"] for r in rows)
    rel = (eps - float(reported_eps_max)) / float(reported_eps_max) if reported_eps_max else float("nan")
    return {"rows": rows, "eps_independent": float(eps), "eps_reported": float(reported_eps_max), "rel_diff": float(rel),
            "steps_match_plan": all(r["planned_steps"] == r["counted_steps"] for r in rows),
            "steps_match_loader": all(r["loader_steps"] == r["counted_steps"] for r in rows), "target_eps": dp.get("target_eps"), "delta": delta}
