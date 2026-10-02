"""Core utilities: seeding, config loading, hashing, logging."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Set deterministic seed for python, numpy, and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config. Falls back to configs/default.yaml if path is None."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    else:
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 hex digest (first 12 chars) of a config dict."""
    serialized = json.dumps(cfg, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the ppfeddata root logger."""
    logger = logging.getLogger("ppfeddata")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
