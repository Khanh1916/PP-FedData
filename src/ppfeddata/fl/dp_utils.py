"""Phase 9: privacy accounting for client-side DP-SGD (Opacus 1.6.0, RDP accountant).

Per client i with n_i rows and batch size B: Opacus' Poisson loader uses q_i = 1 / ceil(n_i / B) and int(1 / q_i) steps per
epoch (checked in the tests against a real `DPDataLoader`). int(1 / q_i) is ceil(n_i / B) except when the float division lands
just below the integer (1 / (1 / 99) = 98.99999999999999, so n_i = 50501, B = 512 runs 98 steps per epoch, not 99; 141 values of
ceil(n_i / B) below 2000 are affected, see `eval/dp_check.py`). sigma_i is calibrated ONCE for the whole run from
(target epsilon, delta, q_i, planned steps T_i = rounds x local_epochs x ceil(n_i / B)), so for such a client it is calibrated for
one step per epoch more than is taken (slightly conservative); epsilon never uses the plan, only the counted steps. A fresh PrivacyEngine is built every
round, which resets Opacus' own accountant, so Opacus' per-round `get_epsilon` is never used: the final epsilon is
recomputed here from the step counters (sigma_i, q_i, total steps), which are stored in the checkpoint, so a resumed run
reports exactly the epsilon of an uninterrupted one.

Meaning of epsilon (state this in every report): record-level (packet) DP for the training of the model weights, for an
observer who sees the updates of that one client. Packets of one TCP stream are strongly correlated, so a whole attack
session is protected far less than the number suggests (group privacy). The class label (the conditioning input) is not
protected by DP-SGD. epsilon of a configuration = max over clients (worst case); the median is reported next to it.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


STAT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + list(range(64, 1025, 8))   # Opacus' defaults + high orders


def steps_per_epoch(n: int, batch_size: int) -> int:
    return int(math.ceil(n / batch_size))


def sample_rate(n: int, batch_size: int) -> float:
    return 1.0 / steps_per_epoch(n, batch_size)


def planned_steps(n: int, batch_size: int, rounds: int, local_epochs: int) -> int:
    return int(rounds * local_epochs * steps_per_epoch(n, batch_size))


def check_delta(sizes: Sequence[int], delta: float) -> None:
    """delta must be smaller than 1/n_i for every client (otherwise 'delta' allows releasing a record outright)."""
    bad = [int(n) for n in sizes if delta >= 1.0 / n]
    if bad:
        raise ValueError(f"delta={delta} is not < 1/n_i for clients with n_i={bad}")


def epsilon(sigma: float, q: float, steps: int, delta: float, sigma_stat: float | None = None) -> float:
    """RDP-accountant epsilon of `steps` Poisson-subsampled Gaussian steps. 0 steps -> 0; sigma = 0 -> infinity.
    `sigma_stat` (optimisation O1): one more Gaussian release with noise multiplier `sigma_stat` and no subsampling (the per-class
    residual statistics of `dp_stats.py`), composed in RDP with the training."""
    hist = [(float(sigma), float(q), int(steps))] if steps > 0 else []
    if sigma_stat:
        hist.append((float(sigma_stat), 1.0, 1))
    if not hist:
        return 0.0
    if min(h[0] for h in hist) <= 0:
        return float("inf")
    acc = RDPAccountant()
    acc.history = hist
    if sigma_stat:          # a cheap single release needs high orders (with Opacus' default orders <= 63, epsilon cannot go below ~0.12)
        return float(acc.get_epsilon(delta=delta, alphas=STAT_ALPHAS))
    return float(acc.get_epsilon(delta=delta))


def calibrate_sigma(target_eps: float, delta: float, n: int, batch_size: int, rounds: int, local_epochs: int,
                    sigma_stat: float | None = None) -> float:
    if not sigma_stat:
        return float(get_noise_multiplier(target_epsilon=float(target_eps), target_delta=float(delta), sample_rate=sample_rate(n, batch_size),
                                          steps=planned_steps(n, batch_size, rounds, local_epochs), accountant="rdp"))
    q, T = sample_rate(n, batch_size), planned_steps(n, batch_size, rounds, local_epochs)
    if epsilon(1e6, q, T, delta, sigma_stat) > target_eps:
        raise ValueError(f"the statistics release alone (sigma_stat {sigma_stat}) exceeds epsilon {target_eps}")
    lo, hi = 0.05, 1.0
    while epsilon(hi, q, T, delta, sigma_stat) > target_eps:
        lo, hi = hi, hi * 2
    for _ in range(60):                      # smallest sigma (to 1e-3 relative) whose composed epsilon is <= target
        mid = (lo + hi) / 2
        if epsilon(mid, q, T, delta, sigma_stat) > target_eps:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3 * hi:
            break
    return float(hi)


def stat_noise_multiplier(target_eps: float, frac: float, delta: float) -> float:
    """Noise multiplier of the single Gaussian release that ALONE would cost `frac * target_eps` (at the same delta); the training
    noise is then calibrated so that the composition reaches `target_eps` (so the split is defined before composition)."""
    target = float(frac) * float(target_eps)
    if target <= 0:
        raise ValueError("frac * target_eps must be > 0")
    lo, hi = 0.01, 1.0
    while epsilon(0.0, 1.0, 0, delta, hi) > target:          # Opacus' own calibration steps sigma by 0.01 x 10 here: too coarse
        lo, hi = hi, hi * 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if epsilon(0.0, 1.0, 0, delta, mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4 * hi:
            break
    return float(hi)


def calibrate_clients(sizes: Sequence[int], target_eps: float, delta: float, batch_size: int, rounds: int,
                      local_epochs: int, sigma_stat: float | None = None) -> list[float]:
    """One sigma per client so that each client's epsilon over the planned run (plus the statistics release) is <= target_eps."""
    check_delta(sizes, delta)
    return [calibrate_sigma(target_eps, delta, int(n), batch_size, rounds, local_epochs, sigma_stat) for n in sizes]


def epsilon_table(sizes: Sequence[int], sigmas: Sequence[float], dp_steps: dict[Any, int], batch_size: int, delta: float,
                  target_eps: float | None = None, sigma_stat: float | None = None) -> dict[str, Any]:
    """Per-client epsilon from the counted steps (and the statistics release, if any); the configuration's epsilon is the maximum over clients."""
    rows = []
    for i, (n, s) in enumerate(zip(sizes, sigmas)):
        steps = int(dp_steps.get(i, dp_steps.get(str(i), 0)))
        e = epsilon(s, sample_rate(int(n), batch_size), steps, delta, sigma_stat)
        rows.append({"client": i, "n": int(n), "sigma": float(s), "q": sample_rate(int(n), batch_size), "steps": steps, "epsilon": e})
    eps = [r["epsilon"] for r in rows]
    out = {"rows": rows, "eps_max": float(max(eps)), "eps_median": float(np.median(eps)), "delta": float(delta)}
    if sigma_stat:
        out["sigma_stat"] = float(sigma_stat)
    if target_eps is not None:
        out["target_eps"] = float(target_eps)
        out["max_ratio_to_target"] = float(max(eps) / target_eps)
        out["within_2pct"] = bool(max(eps) <= 1.02 * target_eps)
    return out
