"""Optimisation O1(a): the per-class residual std of the generator, computed under DP so that the residual-noise variant is valid.

Without DP (Phase 7, SPEC_DEVIATIONS 7.1) the decoder means get Gaussian noise with the per-class std of the reconstruction residual,
estimated on the POOLED train split: that statistic is outside epsilon, which is why every DP number of the study used the plain decoder.

Here each client, after training, takes the final global model (already DP, so using it is post-processing) and its own rows:

    r_i = x_i[num] - decoder_mean(encoder_mean(x_i, y_i), y_i)        residual of record i (numeric columns of the encoded space)
    r_i <- r_i * min(1, C / ||r_i||_2)                                clipped to L2 norm C (`res_clip`)
    S_c = sum_{i in class c} r_i * r_i   (vector),   N_c = #{i in class c}

and releases (S_c, N_c) for every class with Gaussian noise. One record is in one class only and changes its class's block by
(r_i * r_i, 1), whose L2 norm is at most sqrt(C^4 + 1) (since ||r * r||_2 <= ||r||_2^2 <= C^2): that is the sensitivity of the whole
release under add/remove of one record. Noise std = sigma_stat * sqrt(C^4 + 1) on every coordinate. The server sums the noisy releases
of the clients and sets std_c = sqrt(max(S_c, 0) / max(N_c, 1)), capped at C (a clipped residual has no coordinate above C): post-processing.

Accounting: one Gaussian release (no subsampling) per client, composed in RDP with that client's DP-SGD (`dp_utils.epsilon(...,
sigma_stat)`); `sigma_stat` is set so that the release alone would cost `stat_frac` of the target epsilon and the training noise is
calibrated so that the composition reaches the target (`dp_utils.calibrate_sigma(..., sigma_stat)`).
Local DP (M1, M3-local): each client adds its full noise. The variance of the summed noise is then num_clients times that of one
client; the distributed variant of O2 lets each client add a share.
The class label is not protected (as in the rest of the study); the counts N_c are noised anyway, so the class sizes of a client are
not released in the clear by this step.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ppfeddata.models.cvae import one_hot
from ppfeddata.models.generate import GenStats


def sensitivity(res_clip: float) -> float:
    return float(np.sqrt(float(res_clip) ** 4 + 1.0))


@torch.no_grad()
def clipped_residuals(model, X: np.ndarray, y: np.ndarray, res_clip: float, batch: int = 8192) -> np.ndarray:
    """Residual of every record on the numeric columns at the posterior mean, clipped to L2 norm `res_clip`."""
    n, k = model.layout.n_num, model.n_classes
    model.eval()
    Xt, yt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
    mean = []
    for i in range(0, len(Xt), batch):
        xb, yb = Xt[i:i + batch], one_hot(yt[i:i + batch], k)
        mu, _ = model.encode(xb, yb)
        mean.append(model.decode(mu, yb)[:, :n].numpy())
    r = X[:, :n].astype(np.float64) - (np.vstack(mean).astype(np.float64) if mean else np.zeros((0, n)))
    norm = np.linalg.norm(r, axis=1, keepdims=True)
    return r * np.minimum(1.0, float(res_clip) / np.maximum(norm, 1e-12))


def client_release(r: np.ndarray, y: np.ndarray, n_classes: int, sigma_stat: float, res_clip: float, rng: np.random.Generator,
                   noise_share: float = 1.0) -> dict[str, np.ndarray]:
    """Noisy (S_c, N_c) of one client. `noise_share` scales the noise variance (1 = the client alone covers its epsilon; O2 uses
    1 / num_clients for the distributed variant)."""
    n = r.shape[1]
    S, N = np.zeros((n_classes, n)), np.zeros(n_classes)
    for c in range(n_classes):
        m = y == c
        S[c], N[c] = (r[m] ** 2).sum(axis=0), float(m.sum())
    sd = float(sigma_stat) * sensitivity(res_clip) * float(np.sqrt(noise_share))
    return {"S": S + rng.normal(0.0, sd, S.shape), "N": N + rng.normal(0.0, sd, N.shape)}


def aggregate_std(releases: Sequence[dict[str, np.ndarray]], res_clip: float) -> np.ndarray:
    S = sum(r["S"] for r in releases)
    N = sum(r["N"] for r in releases)
    return np.minimum(np.sqrt(np.maximum(S, 0.0) / np.maximum(N, 1.0)[:, None]), float(res_clip))


def private_residual_std(model, X: np.ndarray, y: np.ndarray, parts: Sequence[np.ndarray], sigma_stat: float, res_clip: float,
                         seed: int, noise_share: float = 1.0) -> np.ndarray:
    """Run the release on every client's rows (`parts` = row indices of each client in the train split) and aggregate.
    The noise of client i comes from a generator seeded by (seed, i), so the result is reproducible."""
    k = model.n_classes
    rel = []
    for i, idx in enumerate(parts):
        idx = np.asarray(idx, dtype=np.int64)
        r = clipped_residuals(model, X[idx], y[idx], res_clip)
        rel.append(client_release(r, y[idx], k, sigma_stat, res_clip, np.random.default_rng([int(seed), 7919, int(i)]), noise_share))
    return aggregate_std(rel, res_clip)


def load_parts(partition_path: str | Path) -> list[np.ndarray]:
    return [np.asarray(v, dtype=np.int64) for v in json.loads(Path(partition_path).read_text(encoding="utf-8"))["parts"]]


def stats_for_run(model, run: dict[str, Any], X: np.ndarray, y: np.ndarray, noise_share: float = 1.0) -> GenStats | None:
    """GenStats with the DP residual std of a finished run whose spec has `dp.sigma_stat` (None otherwise). Written next to the run
    (`dp_stats.json`) so the numbers can be checked."""
    dp = run.get("dp") or {}
    if not dp.get("sigma_stat"):
        return None
    std = private_residual_std(model, X, y, load_parts(run["partition_path"]), float(dp["sigma_stat"]), float(dp["res_clip"]),
                               int(run["seed"]), noise_share)
    out = Path(run["artifacts_dir"]) / run["run_id"] / "dp_stats.json"
    out.write_text(json.dumps({"sigma_stat": float(dp["sigma_stat"]), "res_clip": float(dp["res_clip"]), "stat_frac": dp.get("stat_frac"),
                               "noise_share": float(noise_share), "residual_std": std.tolist()}), encoding="utf-8")
    return GenStats({}, std)
