"""Phase 4: fit-on-train preprocessing, feature schema and processed arrays.

Column roles come from configs/feature_decisions.yaml (approved at G1):
  numeric      -> fill 0 -> [is_na flag] -> log1p if skewed -> standardise -> clip to +-clip_sigma
  binary       -> 0/1 (True/False, Set/Not set, 1/0), empty -> 0 [+ is_na flag]
  categorical  -> top-K by train frequency + OTHER + NONE, one-hot
  diagnostic   -> numeric columns excluded from the core feature set; exported separately (X_diag) so that the
                  Phase 5 leakage checks can measure them.
Multi-value cells use the first element (multi_policy = first_only).

All statistics come from the training split only. `transform` never changes the fitted state.
Core layout (contiguous groups, convenient for the CVAE heads): [numeric][binary][na_flag][categorical groups].
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import skew

from ppfeddata.data.harmonize import SHARED_COLUMNS, canonical_token
from ppfeddata.data.parse_multi import first_values
from ppfeddata.data.split_sample import interim_dir

logger = logging.getLogger("ppfeddata.data.preprocess")

TIME_COLS = {"time_delta_from_previous_displayed_frame", "irtt", "time_since_first_frame_in_this_tcp_stream"}
_BIN = {"set": 1.0, "true": 1.0, "1": 1.0, "not set": 0.0, "false": 0.0, "0": 0.0}
_STD_FLOOR = 1e-12


def load_decisions(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_columns(decisions: dict[str, Any]) -> tuple[list[tuple[str, str]], list[str]]:
    """(core [(column, action)], diagnostic [column]) in the raw column order."""
    core, diag = [], []
    for col in SHARED_COLUMNS:
        d = decisions.get(col)
        if not d:
            continue
        if d.get("diagnostic"):
            diag.append(col)
        elif d["action"] in ("numeric", "binary", "categorical"):
            core.append((col, d["action"]))
    return core, diag


# --------------------------------------------------------------------------------------------------
# Column parsers
# --------------------------------------------------------------------------------------------------
def parse_numeric(series: pd.Series) -> pd.Series:
    """First element as float; empty or non-numeric -> NaN."""
    return pd.to_numeric(first_values(series), errors="coerce").astype("float64")


def parse_binary(series: pd.Series) -> pd.Series:
    first = first_values(series)
    return first.map(lambda t: _BIN.get(str(t).lower()) if t is not None else None).astype("float64")


def parse_categorical(series: pd.Series, col: str) -> pd.Series:
    """Canonical token per row; NaN = NONE (empty cell), '' = unmappable (-> OTHER)."""
    first = first_values(series)
    uniq = {t: canonical_token(col, t) for t in first.dropna().unique()}
    return first.map(lambda t: np.nan if t is None else (uniq[t] if uniq[t] is not None else ""))


# --------------------------------------------------------------------------------------------------
# Preprocessor
# --------------------------------------------------------------------------------------------------
class Preprocessor:
    def __init__(self, decisions: dict[str, Any], pcfg: dict[str, Any]):
        self.core_spec, self.diag_cols = select_columns(decisions)
        self.top_k = int(pcfg["categorical_top_k"])
        self.skew_thr = float(pcfg["log1p_skew_threshold"])
        self.clip = float(pcfg["clip_sigma"])
        lo, hi = pcfg.get("na_flag_range", [0.005, 0.995])
        self.na_lo, self.na_hi = float(lo), float(hi)
        self.ultra_fix = bool(pcfg.get("ultra_sparse_fix", True))
        # Per-column multiplier applied before log1p (e.g. seconds -> milliseconds for gaps below 1 s, where
        # log1p is almost the identity and the 5-sigma clip would merge the tail). Raw units are restored on inverse.
        self.scales = {k: float(v) for k, v in (pcfg.get("numeric_scale") or {}).items()}
        self.params: dict[str, dict[str, Any]] = {}
        self.core_blocks: list[dict[str, Any]] = []
        self.diag_blocks: list[dict[str, Any]] = []
        self.fitted = False

    # ---- fit ---------------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.params = {}
        for col, action in self.core_spec:
            self.params[col] = self._fit_column(df[col], col, action)
        for col in self.diag_cols:
            self.params[col] = self._fit_column(df[col], col, "numeric")
        self.core_blocks = self._layout([c for c, _ in self.core_spec])
        self.diag_blocks = self._layout(self.diag_cols)
        self.fitted = True
        return self

    def _fit_column(self, s: pd.Series, col: str, action: str) -> dict[str, Any]:
        if action == "categorical":
            tok = parse_categorical(s, col)
            counts = tok[tok != ""].dropna().value_counts()
            counts = counts.sort_index(kind="stable").sort_values(ascending=False, kind="stable")
            top = [str(t) for t in counts.index[: self.top_k]]
            return {"type": "categorical", "categories": top + ["OTHER", "NONE"],
                    "train_counts": {**{t: int(counts[t]) for t in top},
                                     "OTHER": int((tok == "").sum() + counts.iloc[self.top_k:].sum()),
                                     "NONE": int(tok.isna().sum())},
                    "na_rate": float(tok.isna().mean()), "has_flag": False}
        if action == "binary":
            raw = parse_binary(s)
            na_rate = float(raw.isna().mean())
            return {"type": "binary", "na_rate": na_rate, "has_flag": self.na_lo < na_rate < self.na_hi,
                    "p_one": float(raw.dropna().mean()) if raw.notna().any() else 0.0}
        scale = self.scales.get(col, 1.0)
        raw = parse_numeric(s) * scale
        na = raw.isna()
        na_rate = float(na.mean())
        nonneg_clip = col in TIME_COLS
        v = raw.fillna(0.0).to_numpy()
        n_neg = int((v < 0).sum()) if nonneg_clip else 0
        if nonneg_clip:
            v = np.maximum(v, 0.0)
        # Ultra-sparse column (NA rate >= upper flag bound, but not all NA): filling with 0 and standardising over
        # every row would put all applicable rows beyond the clip bound and merge distinct values. Instead the
        # statistics come from the applicable rows only, NA rows sit at 0 after standardising, and an explicit
        # <col>_is_na flag carries the NA information (deviation from spec, switch: preprocess.ultra_sparse_fix).
        ultra = bool(self.ultra_fix and self.na_hi <= na_rate < 1.0)
        basis = v[~na.to_numpy()] if ultra else v
        sk = float(skew(basis)) if np.std(basis) > _STD_FLOOR else 0.0
        use_log = bool(sk > self.skew_thr and basis.min() >= 0)
        t = np.log1p(basis) if use_log else basis
        std = float(np.std(t))
        applicable = v[~na.to_numpy()]
        return {"type": "numeric", "na_rate": na_rate, "scale": scale,
                "has_flag": bool(self.na_lo < na_rate < self.na_hi or ultra), "ultra_sparse": ultra,
                "log1p": use_log, "skew": sk, "mean": float(np.mean(t)),
                "std": std if std > _STD_FLOOR else 1.0, "constant": bool(std <= _STD_FLOOR),
                "nonneg_clip": nonneg_clip, "n_negative_in_train": n_neg,
                "raw_min": float(applicable.min() / scale) if len(applicable) else 0.0,
                "raw_max": float(applicable.max() / scale) if len(applicable) else 0.0,
                "is_integer": bool(len(applicable) and np.all(applicable / scale == np.round(applicable / scale)))}

    def _layout(self, columns: list[str]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        pos = 0

        def add(name, typ, width, column, **extra):
            nonlocal pos
            blocks.append({"name": name, "type": typ, "column": column, "start": pos, "width": width, **extra})
            pos += width

        for typ in ("numeric", "binary"):
            for c in columns:
                if self.params[c]["type"] == typ:
                    add(c, typ, 1, c)
        for c in columns:
            if self.params[c]["type"] in ("numeric", "binary") and self.params[c]["has_flag"]:
                add(f"{c}_is_na", "na_flag", 1, c)
        for c in columns:
            if self.params[c]["type"] == "categorical":
                add(c, "categorical", len(self.params[c]["categories"]), c,
                    categories=self.params[c]["categories"])
        return blocks

    # ---- transform ---------------------------------------------------------------------------
    def _check(self):
        if not self.fitted:
            raise RuntimeError("Preprocessor is not fitted")

    def _numeric_z(self, p: dict[str, Any], raw: pd.Series) -> np.ndarray:
        v = raw.fillna(0.0).to_numpy()
        if p["nonneg_clip"] or p["log1p"]:
            v = np.maximum(v, 0.0)
        t = np.log1p(v) if p["log1p"] else v
        z = np.clip((t - p["mean"]) / p["std"], -self.clip, self.clip)
        if p.get("ultra_sparse"):
            z = np.where(raw.isna().to_numpy(), 0.0, z)
        return z

    def _transform(self, df: pd.DataFrame, blocks: list[dict[str, Any]]) -> np.ndarray:
        self._check()
        out = np.zeros((len(df), sum(b["width"] for b in blocks)), dtype=np.float32)
        cache: dict[str, Any] = {}
        for b in blocks:
            col, p = b["column"], self.params[b["column"]]
            if b["type"] == "numeric":
                cache[col] = parse_numeric(df[col]) * p["scale"]
                out[:, b["start"]] = self._numeric_z(p, cache[col])
            elif b["type"] == "binary":
                cache[col] = parse_binary(df[col])
                out[:, b["start"]] = cache[col].fillna(0.0).to_numpy()
            elif b["type"] == "na_flag":
                out[:, b["start"]] = cache[col].isna().to_numpy()
            else:
                tok = parse_categorical(df[col], col)
                cats = {c: i for i, c in enumerate(p["categories"])}
                idx = tok.map(lambda t: cats["NONE"] if pd.isna(t) else cats.get(t, cats["OTHER"]))
                out[np.arange(len(df)), b["start"] + idx.to_numpy(dtype=int)] = 1.0
        if not np.isfinite(out).all():
            raise ValueError("non-finite values in transformed features")
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._transform(df, self.core_blocks)

    def transform_diagnostic(self, df: pd.DataFrame) -> np.ndarray:
        return self._transform(df, self.diag_blocks)

    # ---- inverse -----------------------------------------------------------------------------
    def inverse_transform_block(self, values: np.ndarray, block: dict[str, Any]) -> np.ndarray:
        """Decode one block (values = the block's columns). Numeric -> raw units; binary/na_flag -> 0/1;
        categorical -> category label. Clipped values cannot be recovered."""
        p = self.params[block["column"]]
        if block["type"] == "numeric":
            t = values[:, 0].astype("float64") * p["std"] + p["mean"]
            return (np.expm1(t) if p["log1p"] else t) / p["scale"]
        if block["type"] in ("binary", "na_flag"):
            return (values[:, 0] >= 0.5).astype("float64")
        cats = np.array(p["categories"], dtype=object)
        return cats[np.argmax(values, axis=1)]

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        """Raw-unit table. NaN where the na_flag is 1 or the category is NONE; columns without a flag keep the
        fill value 0 for 'not applicable'."""
        self._check()
        cols: dict[str, np.ndarray] = {}
        flags = {b["column"]: b for b in self.core_blocks if b["type"] == "na_flag"}
        for b in self.core_blocks:
            if b["type"] == "na_flag":
                continue
            v = self.inverse_transform_block(X[:, b["start"]:b["start"] + b["width"]], b)
            if b["type"] == "categorical":
                v = np.where(v == "NONE", None, v)
            elif b["column"] in flags:
                fb = flags[b["column"]]
                v = np.where(X[:, fb["start"]] >= 0.5, np.nan, v)
            cols[b["column"]] = v
        return pd.DataFrame(cols)

    # ---- reporting ---------------------------------------------------------------------------
    def audit(self, df: pd.DataFrame) -> dict[str, Any]:
        """Counts for the gate report (does not touch fitted state)."""
        self._check()
        rep: dict[str, Any] = {"unparsed": {}, "clipped_sigma": {}, "negative_clipped": {}}
        for col, action in self.core_spec + [(c, "numeric") for c in self.diag_cols]:
            if action == "categorical":
                continue
            first = first_values(df[col])
            parsed = parse_numeric(df[col]) if action == "numeric" else parse_binary(df[col])
            n_unp = int((first.notna() & parsed.isna()).sum())
            if n_unp:
                rep["unparsed"][col] = n_unp
            if action == "numeric":
                p = self.params[col]
                raw = parse_numeric(df[col]) * p["scale"]
                v = raw.fillna(0.0).to_numpy()
                if p["nonneg_clip"] or p["log1p"]:
                    n_neg = int((v < 0).sum())
                    if n_neg:
                        rep["negative_clipped"][col] = n_neg
                    v = np.maximum(v, 0.0)
                z = ((np.log1p(v) if p["log1p"] else v) - p["mean"]) / p["std"]
                if p.get("ultra_sparse"):
                    z = z[raw.notna().to_numpy()]
                n_clip = int((np.abs(z) > self.clip).sum())
                if n_clip:
                    rep["clipped_sigma"][col] = n_clip
        return rep

    def schema(self) -> dict[str, Any]:
        self._check()
        n_flags = sum(b["type"] == "na_flag" for b in self.core_blocks)
        return {
            "n_features": int(sum(b["width"] for b in self.core_blocks)),
            "n_na_flags": int(n_flags),
            "settings": {"categorical_top_k": self.top_k, "log1p_skew_threshold": self.skew_thr,
                         "clip_sigma": self.clip, "na_flag_range": [self.na_lo, self.na_hi],
                         "ultra_sparse_fix": self.ultra_fix, "numeric_scale": self.scales,
                         "multi_policy": "first_only"},
            "blocks": self.core_blocks,
            "diagnostic": {"n_features": int(sum(b["width"] for b in self.diag_blocks)),
                           "blocks": self.diag_blocks},
            "params": self.params,
        }


# --------------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------------
def label_map(cfg: dict[str, Any]) -> dict[str, int]:
    keys = list(cfg["quota_11class"] if cfg["label_mode"] == "11class" else cfg["quota"])
    return {k: i for i, k in enumerate(keys)}


def processed_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["work_dir"]) / "processed" / cfg["label_mode"]


def run_preprocess(cfg: dict[str, Any]) -> dict[str, Any]:
    src = interim_dir(cfg)
    dec_path = Path(cfg.get("harmonize", {}).get("decisions_path", "./configs/feature_decisions.yaml"))
    decisions = load_decisions(dec_path)
    lm = label_map(cfg)
    data = {s: pd.read_parquet(src / f"{s}.parquet") for s in ("train", "val", "test")}

    pre = Preprocessor(decisions, cfg["preprocess"]).fit(data["train"])      # fit on train ONLY
    out = processed_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)

    dist, audit = {}, {}
    for split, df in data.items():
        unknown = set(df["label"]) - set(lm)
        if unknown:
            raise KeyError(f"labels not in label map: {unknown}")
        X, Xd = pre.transform(df), pre.transform_diagnostic(df)
        y = df["label"].map(lm).to_numpy(dtype=np.int64)
        np.savez_compressed(out / f"{split}.npz", X=X, X_diag=Xd, y=y,
                            group_id=df["group_id"].to_numpy(dtype=str),
                            stream_id=df["stream_id"].to_numpy(dtype=str),
                            class11=df["class11"].to_numpy(dtype=str))
        dist[split] = {k: int(v) for k, v in df["label"].value_counts().reindex(list(lm)).fillna(0).items()}
        audit[split] = pre.audit(df)

    schema = pre.schema()
    schema["label_map"] = lm
    schema["label_mode"] = cfg["label_mode"]
    schema["class_distribution"] = dist
    schema["audit"] = audit
    with open(out / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    with open(out / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(lm, f, indent=2)
    joblib.dump(pre, out / "preprocessor.joblib")
    logger.info("D=%d, is_na flags=%d, diagnostic=%d; wrote %s", schema["n_features"], schema["n_na_flags"],
                schema["diagnostic"]["n_features"], out)
    return schema
