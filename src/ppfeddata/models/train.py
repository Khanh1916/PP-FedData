"""Phase 7: CVAE training loop (centralised B2). Phase 8 reuses `fit_epoch` for the local FL epochs."""
from __future__ import annotations

import copy
import logging
import time
from typing import Any

import numpy as np
import torch

from ppfeddata.models.cvae import CVAE, Layout, beta_at, loss_terms, numpy_to_tensor, one_hot

logger = logging.getLogger("ppfeddata.models.train")

TERMS = ("loss", "recon_num", "recon_bin", "recon_cat", "kl")


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def sampler_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-row weights that make every class equally likely in a batch (cvae.class_balanced_sampler)."""
    cnt = np.maximum(np.bincount(y, minlength=n_classes), 1)
    return (1.0 / cnt)[y]


def fit_epoch(model: CVAE, opt: torch.optim.Optimizer, X: torch.Tensor, y: torch.Tensor, batch_size: int, beta: float,
              rng: np.random.Generator, weights: np.ndarray | None = None) -> dict[str, float]:
    """One pass over the data (drop-last off). `weights` -> sample rows with replacement by these probabilities."""
    model.train()
    n = len(X)
    if weights is None:
        order = rng.permutation(n)
    else:
        order = rng.choice(n, size=n, replace=True, p=weights / weights.sum())
    acc = {k: 0.0 for k in TERMS}
    for i in range(0, n, batch_size):
        idx = torch.as_tensor(order[i:i + batch_size])
        xb, yb = X[idx], y[idx]
        out, mu, logvar = model(xb, one_hot(yb, model.n_classes))
        t = loss_terms(out, xb, mu, logvar, beta, model.layout)
        opt.zero_grad(set_to_none=True)
        t["loss"].backward()
        opt.step()
        for k in TERMS:
            acc[k] += float(t[k].detach()) * len(idx)
    return {k: v / n for k, v in acc.items()}


@torch.no_grad()
def eval_loss(model: CVAE, X: torch.Tensor, y: torch.Tensor, beta: float, batch_size: int = 4096) -> dict[str, float]:
    """ELBO-style loss with a fixed noise seed, so epochs are comparable."""
    model.eval()
    acc = {k: 0.0 for k in TERMS}
    with torch.random.fork_rng():
        torch.manual_seed(12345)
        for i in range(0, len(X), batch_size):
            xb, yb = X[i:i + batch_size], y[i:i + batch_size]
            out, mu, logvar = model(xb, one_hot(yb, model.n_classes))
            t = loss_terms(out, xb, mu, logvar, beta, model.layout)
            for k in TERMS:
                acc[k] += float(t[k]) * len(xb)
    return {k: v / len(X) for k, v in acc.items()}


def train_cvae(X: np.ndarray, y: np.ndarray, Xv: np.ndarray | None, yv: np.ndarray | None, n_classes: int,
               layout: Layout, hp: dict[str, Any], seed: int, epochs: int | None = None, patience: int | None = None,
               hidden: tuple[int, ...] | None = None) -> tuple[CVAE, list[dict[str, float]]]:
    """Train a CVAE. `hp` carries the `cvae` config keys (latent_dim, hidden, beta, beta_warmup_epochs, lr, batch_size,
    epochs, class_balanced_sampler). Early stopping on the validation loss at the final beta; it only counts after the
    beta warm-up, and the best weights are restored. Returns (model, history)."""
    seed_torch(seed)
    epochs = int(epochs if epochs is not None else hp["epochs"])
    model = CVAE(layout, n_classes, int(hp["latent_dim"]), tuple(hidden or hp["hidden"]), bool(hp.get("layernorm", False)))
    opt = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]))
    Xt, yt = numpy_to_tensor(X), torch.as_tensor(y, dtype=torch.long)
    Xvt = numpy_to_tensor(Xv) if Xv is not None else None
    yvt = torch.as_tensor(yv, dtype=torch.long) if yv is not None else None
    rng = np.random.default_rng(seed)
    w = sampler_weights(y, n_classes) if hp.get("class_balanced_sampler") else None
    warm, beta = int(hp["beta_warmup_epochs"]), float(hp["beta"])
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        t0 = time.perf_counter()
        tr = fit_epoch(model, opt, Xt, yt, int(hp["batch_size"]), beta_at(ep, beta, warm), rng, w)
        row = {"epoch": ep, "beta": beta_at(ep, beta, warm), "seconds": time.perf_counter() - t0,
               **{f"train_{k}": v for k, v in tr.items()}}
        if Xvt is not None:
            va = eval_loss(model, Xvt, yvt, beta)
            row.update({f"val_{k}": v for k, v in va.items()})
            if ep >= warm and va["loss"] < best - 1e-6:
                best, bad, best_state = va["loss"], 0, copy.deepcopy(model.state_dict())
            elif ep >= warm:
                bad += 1
        hist.append(row)
        if patience is not None and bad >= patience:
            logger.info("early stop at epoch %d (best val loss %.4f)", ep, best)
            break
    if best_state is not None and patience is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, hist
