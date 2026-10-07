"""Phase 7: conditional VAE over the encoded feature space (data/processed/<mode>).

The encoded layout is `[numeric][binary][na_flag][categorical groups]` (Phase 4), so the decoder emits one vector of
width D whose slices are: numeric means (MSE), binary + is_na logits (BCE), one logit group per categorical column (CE).

No BatchNorm (incompatible with Opacus in Phase 9): only `nn.Linear`, `ReLU` and an optional `LayerNorm`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Layout:
    """Slices of the encoded vector, derived from `feature_schema.json`."""
    D: int
    n_num: int                       # [0, n_num)               numeric (Gaussian, MSE)
    n_bin: int                       # [n_num, n_num + n_bin)   binary and is_na flags (Bernoulli, BCE)
    groups: tuple[tuple[int, int], ...]   # (start, width) per categorical column (softmax, CE)

    @property
    def cat_start(self) -> int:
        return self.n_num + self.n_bin


def build_layout(schema: dict[str, Any]) -> Layout:
    """Check that the blocks are contiguous and in the order numeric < binary/na_flag < categorical."""
    order = {"numeric": 0, "binary": 1, "na_flag": 1, "categorical": 2}
    pos, last, n = 0, 0, {0: 0, 1: 0, 2: 0}
    groups: list[tuple[int, int]] = []
    for b in schema["blocks"]:
        if b["start"] != pos:
            raise ValueError(f"block {b['name']} starts at {b['start']}, expected {pos}")
        r = order[b["type"]]
        if r < last:
            raise ValueError(f"block {b['name']} ({b['type']}) is out of order")
        last, pos = r, pos + b["width"]
        n[r] += b["width"]
        if r == 2:
            groups.append((b["start"], b["width"]))
    if pos != schema["n_features"]:
        raise ValueError(f"blocks cover {pos} columns, schema says {schema['n_features']}")
    return Layout(D=pos, n_num=n[0], n_bin=n[1], groups=tuple(groups))


def _mlp(sizes: list[int], layernorm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(a, b))
        if layernorm:
            layers.append(nn.LayerNorm(b))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class CVAE(nn.Module):
    """`[x; onehot(y)] -> hidden[0] -> hidden[1] -> (mu, logvar)`; `[z; onehot(y)] -> reversed hidden -> D`."""

    def __init__(self, layout: Layout, n_classes: int, latent_dim: int = 16, hidden: tuple[int, ...] = (128, 64),
                 layernorm: bool = False):
        super().__init__()
        self.layout, self.n_classes, self.latent_dim = layout, int(n_classes), int(latent_dim)
        self.hidden = tuple(int(h) for h in hidden)
        h = list(self.hidden)
        self.enc = _mlp([layout.D + n_classes] + h, layernorm)
        self.mu = nn.Linear(h[-1], latent_dim)
        self.logvar = nn.Linear(h[-1], latent_dim)
        self.dec = _mlp([latent_dim + n_classes] + h[::-1], layernorm)
        self.out = nn.Linear(h[0], layout.D)

    def encode(self, x: torch.Tensor, y_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.enc(torch.cat([x, y_onehot], dim=1))
        return self.mu(e), self.logvar(e)

    def decode(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        return self.out(self.dec(torch.cat([z, y_onehot], dim=1)))

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor):
        mu, logvar = self.encode(x, y_onehot)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, y_onehot), mu, logvar

    def config(self) -> dict[str, Any]:
        return {"latent_dim": self.latent_dim, "hidden": list(self.hidden), "n_classes": self.n_classes,
                "layernorm": any(isinstance(m, nn.LayerNorm) for m in self.modules())}


def one_hot(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    return F.one_hot(y.long(), n_classes).float()


def recon_terms(out: torch.Tensor, x: torch.Tensor, layout: Layout) -> dict[str, torch.Tensor]:
    """Per-sample reconstruction loss of each block family (shape [B])."""
    n, b = layout.n_num, layout.n_bin
    num = ((out[:, :n] - x[:, :n]) ** 2).sum(1)
    bce = F.binary_cross_entropy_with_logits(out[:, n:n + b], x[:, n:n + b], reduction="none").sum(1)
    cat = torch.zeros_like(num)
    for s, w in layout.groups:
        cat = cat + F.cross_entropy(out[:, s:s + w], x[:, s:s + w].argmax(1), reduction="none")
    return {"num": num, "bin": bce, "cat": cat}


def loss_terms(out, x, mu, logvar, beta: float, layout: Layout, weights: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """loss = mean over samples of (recon_num + recon_bin + recon_cat + beta * KL), each sample times `weights` when given
    (class weights of optimisation O1; under DP-SGD the weighted per-sample gradient is still clipped, so epsilon is unchanged)."""
    r = recon_terms(out, x, layout)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    total = r["num"] + r["bin"] + r["cat"] + beta * kl
    return {"loss": (total * weights).mean() if weights is not None else total.mean(), "recon_num": r["num"].mean(), "recon_bin": r["bin"].mean(),
            "recon_cat": r["cat"].mean(), "kl": kl.mean()}


def beta_at(epoch: int, beta: float, warmup_epochs: int) -> float:
    """Linear warm-up of the KL weight: 0 at epoch 0, `beta` from epoch `warmup_epochs` on."""
    if warmup_epochs <= 0:
        return float(beta)
    return float(beta) * min(1.0, epoch / warmup_epochs)


def hidden_from_width(width: int) -> tuple[int, int]:
    """Optuna searches one width; the second layer is half of it (128 -> (128, 64), the config default)."""
    return int(width), max(8, int(width) // 2)


def numpy_to_tensor(a: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)
