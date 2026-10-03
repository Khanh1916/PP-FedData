"""Run ledger (results/runs.csv) and per-run artifacts (predictions) with resume support."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def runs_csv_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg.get("compute", {}).get("runs_csv", "./results/runs.csv"))


def artifacts_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["compute"]["artifacts_dir"])


def run_id(config: str, seed: int, label_mode: str | None = None) -> str:
    """`{config}_{seed}`; the label mode is part of the id for 11-class runs so the two modes never collide."""
    return f"{config}_{seed}" if label_mode in (None, "6class") else f"{config}_{label_mode}_{seed}"


class RunLedger:
    """One row per finished run. A run that is already in the ledger is skipped on resume."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def frame(self) -> pd.DataFrame:
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()

    def done(self, rid: str) -> bool:
        df = self.frame()
        return bool(len(df) and (df["run_id"] == rid).any())

    def append(self, row: dict[str, Any]) -> None:
        df = self.frame()
        df = pd.concat([df[df["run_id"] != row["run_id"]] if len(df) else df, pd.DataFrame([row])], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.path)


def save_predictions(cfg: dict[str, Any], rid: str, split: str, y_pred: np.ndarray, proba: np.ndarray | None,
                     meta: dict[str, Any] | None = None) -> Path:
    d = artifacts_dir(cfg) / rid / "preds"
    d.mkdir(parents=True, exist_ok=True)
    arrays = {"y_pred": y_pred.astype(np.int16)}
    if proba is not None:
        arrays["proba"] = proba.astype(np.float32)
    np.savez_compressed(d / f"{split}.npz", **arrays)
    if meta is not None:
        (d.parent / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return d / f"{split}.npz"


def load_predictions(cfg: dict[str, Any], rid: str, split: str = "test") -> dict[str, np.ndarray]:
    return dict(np.load(artifacts_dir(cfg) / rid / "preds" / f"{split}.npz"))
