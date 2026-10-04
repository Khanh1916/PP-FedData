"""Phase 8: the federated algorithm itself, with no Flower import (so it is unit-testable and reused by the Flower app).

- `local_train`: one client's work in one round (fresh Adam, `local_epochs` epochs over its own rows).
- `fedavg_numpy`: reference weighted average used in the tests to check Flower's aggregation.
- `Checkpoint`: global weights + round + RNG state + per-client step counters (+ DP step counters in Phase 9) + round log.

Seeds: the client's randomness depends only on (run seed, absolute round, client id), so a resumed run reproduces an
uninterrupted one.
"""
from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ppfeddata.models.cvae import CVAE, Layout, beta_at, numpy_to_tensor
from ppfeddata.models.train import eval_loss, fit_epoch, seed_torch


def make_model(layout: Layout, n_classes: int, hp: dict[str, Any]) -> CVAE:
    return CVAE(layout, n_classes, int(hp["latent_dim"]), tuple(hp["hidden"]), bool(hp.get("layernorm", False)))


def init_state(layout: Layout, n_classes: int, hp: dict[str, Any], seed: int) -> OrderedDict:
    """Common initial weights for the run (same seed -> same start for every method)."""
    seed_torch(seed)
    return OrderedDict((k, v.clone()) for k, v in make_model(layout, n_classes, hp).state_dict().items())


def client_seed(seed: int, round_idx: int, client_id: int) -> int:
    return (int(seed) * 1_000_003 + int(round_idx) * 1009 + int(client_id)) % (2 ** 31 - 1)


def local_train(state: dict[str, torch.Tensor], X: np.ndarray, y: np.ndarray, layout: Layout, n_classes: int,
                hp: dict[str, Any], local_epochs: int, round_idx: int, seed: int, client_id: int) -> dict[str, Any]:
    """`round_idx` is the absolute round (1-based). The KL warm-up runs over the global epoch count
    (round-1) * local_epochs + e, so FL and centralised training see the same beta schedule."""
    s = client_seed(seed, round_idx, client_id)
    seed_torch(s)
    model = make_model(layout, n_classes, hp)
    model.load_state_dict(state)
    opt = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]))
    Xt, yt = numpy_to_tensor(X), torch.as_tensor(y, dtype=torch.long)
    rng = np.random.default_rng(s)
    bs, beta, warm = int(hp["batch_size"]), float(hp["beta"]), int(hp["beta_warmup_epochs"])
    t0 = time.perf_counter()
    last: dict[str, float] = {}
    for e in range(local_epochs):
        last = fit_epoch(model, opt, Xt, yt, bs, beta_at((round_idx - 1) * local_epochs + e, beta, warm), rng)
    steps = local_epochs * int(np.ceil(len(X) / bs))
    return {"state": OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items()), "n": int(len(X)),
            "loss": float(last.get("loss", float("nan"))), "steps": int(steps), "seconds": time.perf_counter() - t0}


def local_train_dp(state: dict[str, torch.Tensor], X: np.ndarray, y: np.ndarray, layout: Layout, n_classes: int,
                   hp: dict[str, Any], local_epochs: int, round_idx: int, seed: int, client_id: int, sigma: float,
                   max_grad_norm: float) -> dict[str, Any]:
    """DP-SGD version of `local_train` (Opacus, Poisson sampling, flat per-sample clipping to `max_grad_norm`, Gaussian
    noise `sigma * max_grad_norm`, then the usual Adam update). A new PrivacyEngine is built every round (its accountant is
    NOT used; epsilon is recomputed from the step counter in dp_utils). `steps` counts every batch the loader yielded,
    including empty Poisson batches, which is what the accountant assumes."""
    import warnings

    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader, TensorDataset

    from ppfeddata.models.cvae import loss_terms, one_hot

    s = client_seed(seed, round_idx, client_id)
    seed_torch(s)
    model = make_model(layout, n_classes, hp)
    model.load_state_dict(state)
    opt = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]))
    bs, beta, warm = int(hp["batch_size"]), float(hp["beta"]), int(hp["beta_warmup_epochs"])
    g_sample, g_noise = torch.Generator().manual_seed(s), torch.Generator().manual_seed(s + 1)
    loader = DataLoader(TensorDataset(numpy_to_tensor(X), torch.as_tensor(y, dtype=torch.long)), batch_size=bs, shuffle=True,
                        generator=g_sample)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # "Secure RNG turned off" and the full-backward-hook notice
        gs, dopt, dl = PrivacyEngine().make_private(module=model, optimizer=opt, data_loader=loader, noise_multiplier=float(sigma),
                                                    max_grad_norm=float(max_grad_norm), poisson_sampling=True,
                                                    noise_generator=g_noise)
        steps, last = 0, float("nan")
        for e in range(local_epochs):
            gs.train()
            b = beta_at((round_idx - 1) * local_epochs + e, beta, warm)
            for xb, yb in dl:
                steps += 1
                if len(xb) == 0:
                    continue
                out, mu, logvar = gs(xb, one_hot(yb, n_classes))
                loss = loss_terms(out, xb, mu, logvar, b, layout)["loss"]
                dopt.zero_grad(set_to_none=True)
                loss.backward()
                dopt.step()
                last = float(loss.detach())
    return {"state": OrderedDict((k, v.detach().clone()) for k, v in gs._module.state_dict().items()), "n": int(len(X)),
            "loss": last, "steps": int(steps), "seconds": time.perf_counter() - t0}


def fedavg_numpy(states: list[dict[str, Any]], weights: list[float]) -> dict[str, np.ndarray]:
    """sum_i w_i * theta_i / sum_i w_i in float64, per tensor."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return {k: sum(wi * np.asarray(s[k], dtype=np.float64) for wi, s in zip(w, states)) for k in states[0]}


def val_loss(state: dict[str, torch.Tensor], Xv: np.ndarray, yv: np.ndarray, layout: Layout, n_classes: int,
             hp: dict[str, Any]) -> dict[str, float]:
    """ELBO of the global model on the validation split at the final beta (server-side simulation diagnostic)."""
    model = make_model(layout, n_classes, hp)
    model.load_state_dict(state)
    return eval_loss(model, numpy_to_tensor(Xv), torch.as_tensor(yv, dtype=torch.long), float(hp["beta"]))


def model_bytes_of(state: dict[str, Any]) -> int:
    return int(sum(np.asarray(v).nbytes for v in state.values()))


# --------------------------------------------------------------------------------------------------
# Checkpoints and the round log
# --------------------------------------------------------------------------------------------------
def run_dir(artifacts_dir: str | Path, rid: str) -> Path:
    return Path(artifacts_dir) / rid


def save_checkpoint(rdir: Path, state: dict[str, torch.Tensor], round_idx: int, steps: dict[int, int],
                    extra: dict[str, Any] | None = None, dp_steps: dict[int, int] | None = None) -> Path:
    """Atomic write of `ckpt/round_{r}.pt` and of the pointer `ckpt/latest.json`; keeps the two newest checkpoints."""
    d = rdir / "ckpt"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"round_{round_idx:04d}.pt"
    payload = {"state": dict(state), "round": int(round_idx), "client_steps": {int(k): int(v) for k, v in steps.items()},
               "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
               "dp_steps": {int(k): int(v) for k, v in (dp_steps or {}).items()}, **(extra or {})}
    torch.save(payload, f.with_suffix(".tmp"))
    os.replace(f.with_suffix(".tmp"), f)
    (d / "latest.json").write_text(json.dumps({"file": f.name, "round": int(round_idx)}), encoding="utf-8")
    for old in sorted(d.glob("round_*.pt"))[:-2]:
        old.unlink()
    return f


def load_checkpoint(rdir: Path) -> dict[str, Any] | None:
    p = rdir / "ckpt" / "latest.json"
    if not p.exists():
        return None
    f = rdir / "ckpt" / json.loads(p.read_text(encoding="utf-8"))["file"]
    return torch.load(f, weights_only=False) if f.exists() else None


def append_round_log(rdir: Path, row: dict[str, Any]) -> None:
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "rounds.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def read_round_log(rdir: Path, upto: int | None = None) -> list[dict[str, Any]]:
    p = rdir / "rounds.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if upto is None or r["round"] <= upto]


def truncate_round_log(rdir: Path, upto: int) -> None:
    """On resume, drop log lines newer than the checkpoint (the rounds after it are re-run)."""
    rows = read_round_log(rdir, upto)
    (rdir / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
