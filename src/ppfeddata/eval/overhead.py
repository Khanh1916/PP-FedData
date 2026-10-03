"""Phase 6: cost measurements - time per round and in total, bytes per round, peak RAM.

A single-machine simulation measures the compute part of cryptographic cost but not network latency; bytes per round
are computed analytically from the model size (and the SecAgg expansion factor), which is exact for the payload.
"""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np


def peak_rss_gb(previous: float = 0.0) -> float:
    try:
        import psutil
        return max(previous, psutil.Process().memory_info().rss / 1e9)
    except ImportError:
        return previous


class Timer:
    """Context manager: `with Timer() as t: ...; t.seconds`."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return False


class RoundMeter:
    """Collects the wall time of each FL round and the peak RAM seen while running."""

    def __init__(self):
        self.round_seconds: list[float] = []
        self.peak_gb = 0.0

    def record(self, seconds: float) -> None:
        self.round_seconds.append(float(seconds))
        self.peak_gb = peak_rss_gb(self.peak_gb)

    def summary(self) -> dict[str, float]:
        r = np.asarray(self.round_seconds)
        return {"rounds": int(len(r)), "total_s": float(r.sum()), "mean_round_s": float(r.mean()) if len(r) else 0.0,
                "peak_rss_gb": self.peak_gb}


def model_bytes(arrays: Iterable[np.ndarray]) -> int:
    return int(sum(np.asarray(a).nbytes for a in arrays))


def bytes_per_round(model_size_bytes: int, n_clients: int, expansion: float = 1.0) -> int:
    """Payload of one round: every client downloads the global model and uploads an update of the same size.
    `expansion` > 1 models SecAgg (masked, quantised integers are larger than float32 weights)."""
    return int(model_size_bytes * n_clients * 2 * expansion)


def overhead_ratio(seconds_protected: float, seconds_plain: float) -> float:
    return float(seconds_protected / seconds_plain) if seconds_plain > 0 else float("inf")
