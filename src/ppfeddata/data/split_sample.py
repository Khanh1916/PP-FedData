"""Phase 3: group-aware train/val/test split and streaming uniform sampling.

Pipeline
  1. Scan (pass A): for every non-empty file read only the filter columns, `Stream index` and, for Normal,
     `Capture_ID`. Result per file: eligibility mask (row filters from feature_decisions.yaml) and stream index.
  2. Plan: per sub-class, split units (groups, or blocks when a sub-class has < 3 groups) into test/val/train,
     then draw an exact-size uniform random sample without replacement per unit.
  3. Extract (pass B): stream each file once and keep only the chosen rows.
  4. Verify and write `{train,val,test}.parquet` + `split_manifest.json`.

Uniform sampling by pre-drawn row indices is equivalent to reservoir sampling because the exact row count of
every file is known from the inventory; it needs a single streaming pass and is fully deterministic.
Raw files are only read.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ppfeddata.data.harmonize import COLUMN_RENAME, SHARED_COLUMNS
from ppfeddata.utils import config_hash

logger = logging.getLogger("ppfeddata.data.split_sample")

_CHUNK = 100_000
SPLITS = ("train", "val", "test")
_SNAKE_TO_ORIG = {v: k for k, v in COLUMN_RENAME.items()}
_META_COLS = ["label", "class6", "class11", "split", "data_source", "group_id", "stream_id", "source_file", "row_idx"]


class QuotaError(RuntimeError):
    """Raised when a val/test class falls below the minimum size; the user must decide (spec Phase 3)."""


# --------------------------------------------------------------------------------------------------
# Quotas
# --------------------------------------------------------------------------------------------------
def subclass_quotas(cfg: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    """[train, val, test] per sub-class (class11 name) for the configured label_mode."""
    if cfg["label_mode"] == "11class":
        return {k: tuple(v) for k, v in cfg["quota_11class"].items()}
    out: dict[str, tuple[int, int, int]] = {}
    for scen, q in cfg["quota"].items():
        if scen == "NORMAL":
            out["NORMAL"] = tuple(q)
        else:  # split the scenario quota evenly between DoS and DDoS (DoS gets the extra row if odd)
            out[f"{scen}_DoS"] = tuple((x + 1) // 2 for x in q)
            out[f"{scen}_DDoS"] = tuple(x // 2 for x in q)
    return out


def allocate(quota: int, caps: list[int]) -> list[int]:
    """Spread `quota` as evenly as possible over units without exceeding each unit's capacity."""
    alloc = [0] * len(caps)
    active = [i for i, c in enumerate(caps) if c > 0]
    remaining = quota
    while remaining > 0 and active:
        base, extra = divmod(remaining, len(active))
        for j, i in enumerate(active):
            alloc[i] += min(base + (1 if j < extra else 0), caps[i] - alloc[i])
        remaining = quota - sum(alloc)
        active = [i for i in active if alloc[i] < caps[i]]
    return alloc


# --------------------------------------------------------------------------------------------------
# Pass A: scan
# --------------------------------------------------------------------------------------------------
@dataclass
class FileScan:
    mask: np.ndarray      # bool, row passes the row filters
    stream: np.ndarray    # int32 stream index, -1 when missing
    n_rows: int


def load_row_filters(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Row filters approved at G1, read from feature_decisions.yaml (`_row_filters.all_sources`)."""
    path = Path(cfg.get("harmonize", {}).get("decisions_path", "./configs/feature_decisions.yaml"))
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: the row filters approved at G1 live there. Run `harmonize --decisions-only` "
            "or pass filters explicitly; refusing to sample without them.")
    with open(path, encoding="utf-8") as f:
        dec = yaml.safe_load(f) or {}
    return dict((dec.get("_row_filters") or {}).get("all_sources") or {})


def _scan_file(path: Path, source: str, expected_group: str, filters: dict[str, list[str]]) -> FileScan:
    usecols = {"Stream index"} | {_SNAKE_TO_ORIG[c] for c in filters}
    if source == "normal":
        usecols.add("Capture_ID")
    masks, streams, captures = [], [], set()
    for chunk in pd.read_csv(path, usecols=sorted(usecols), chunksize=_CHUNK, dtype=str, low_memory=False):
        m = np.ones(len(chunk), dtype=bool)
        for col, allowed in filters.items():
            m &= chunk[_SNAKE_TO_ORIG[col]].isin(allowed).to_numpy()
        masks.append(m)
        streams.append(pd.to_numeric(chunk["Stream index"], errors="coerce").fillna(-1).astype(np.int32).to_numpy())
        if source == "normal":
            captures.update(chunk["Capture_ID"].dropna().unique())
    if source == "normal":
        expected = f"Pcap Files/{expected_group}.pcap"
        if captures != {expected}:
            raise RuntimeError(f"{path.name}: Capture_ID values {sorted(captures)} != inventory group {expected!r}")
    mask = np.concatenate(masks) if masks else np.zeros(0, bool)
    stream = np.concatenate(streams) if streams else np.zeros(0, np.int32)
    return FileScan(mask, stream, len(mask))


def scan_files(files: pd.DataFrame, filters: dict[str, list[str]], cache_dir: Path | None) -> dict[str, FileScan]:
    """Pass A with an on-disk cache keyed by (file, size, filters)."""
    fhash = hashlib.md5(json.dumps(filters, sort_keys=True).encode()).hexdigest()[:8]
    out: dict[str, FileScan] = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(files.itertuples(index=False)):
        key = f"{zlib.crc32(str(r.path).encode()):08x}_{int(r.bytes)}_{fhash}.npz"
        cpath = cache_dir / key if cache_dir is not None else None
        if cpath is not None and cpath.exists():
            z = np.load(cpath)
            scan = FileScan(z["mask"], z["stream"], int(z["mask"].shape[0]))
        else:
            if i % 25 == 0:
                logger.info("Scan %d/%d: %s", i + 1, len(files), r.filename)
            scan = _scan_file(Path(r.path), r.source, r.group_id, filters)
            if cpath is not None:
                np.savez_compressed(cpath, mask=scan.mask, stream=scan.stream)
        if scan.n_rows != int(r.rows):
            raise RuntimeError(f"{r.filename}: scanned {scan.n_rows} rows, inventory says {r.rows}")
        out[str(r.path)] = scan
    return out


# --------------------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------------------
@dataclass
class Unit:
    unit_id: str                      # group_id, or "<group>#blkNN" for block-split sub-classes
    group: str                        # original group (file/capture)
    paths: list[str]
    lo: int | None = None             # row range inside paths[0] for block units
    hi: int | None = None
    n_elig: int = 0


def _rng(*parts: Any) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(str(p).encode()) for p in parts])


def _drop_cross_block_streams(sc: FileScan, bounds: list[tuple[int, int]]) -> tuple[FileScan, dict[str, int]]:
    """Make rows of streams that have eligible rows in more than one block ineligible.

    A gap of `gap_rows` cannot separate a TCP stream that lives for the whole capture; leaving such streams in
    would put the same stream in train and test.
    """
    block = np.full(sc.n_rows, -1, dtype=np.int16)
    for b, (lo, hi) in enumerate(bounds):
        block[lo:hi] = b
    ok = sc.mask & (block >= 0) & (sc.stream >= 0)
    pairs = pd.DataFrame({"s": sc.stream[ok], "b": block[ok]}).drop_duplicates()
    nb = pairs.groupby("s")["b"].size()
    cross = nb[nb > 1].index.to_numpy()
    kill = ok & np.isin(sc.stream, cross)
    stats = {"cross_block_streams": int(len(cross)), "rows_removed": int(kill.sum()),
             "eligible_before": int(sc.mask.sum())}
    return FileScan(sc.mask & ~kill, sc.stream, sc.n_rows), stats


def build_units(sub_files: pd.DataFrame, scans: dict[str, FileScan], n_min_groups: int,
                n_blocks: int, gap_rows: int, drop_cross: bool = True) -> tuple[list[Unit], str, dict[str, int]]:
    """Group units, or block units (with gap) when the sub-class has fewer than `n_min_groups` groups."""
    groups = sorted(sub_files["group_id"].unique())
    units: list[Unit] = []
    if len(groups) >= n_min_groups:
        for g in groups:
            paths = sub_files[sub_files["group_id"] == g].sort_values("filename")["path"].astype(str).tolist()
            units.append(Unit(g, g, paths, n_elig=int(sum(scans[p].mask.sum() for p in paths))))
        return units, "group", {}
    extra = {"cross_block_streams": 0, "rows_removed": 0, "eligible_before": 0}
    for g in groups:
        paths = sub_files[sub_files["group_id"] == g].sort_values("filename")["path"].astype(str).tolist()
        if len(paths) != 1:
            raise RuntimeError(f"block split expects one file per group, got {len(paths)} for {g}")
        n = scans[paths[0]].n_rows
        size = n // n_blocks
        bounds = []
        for b in range(n_blocks):
            lo = b * size
            hi = (b + 1) * size if b < n_blocks - 1 else n
            bounds.append((lo, hi - gap_rows if b < n_blocks - 1 else hi))   # gap_rows dropped between blocks
        if drop_cross:
            scans[paths[0]], st = _drop_cross_block_streams(scans[paths[0]], bounds)
            for k, v in st.items():
                extra[k] += v
        else:
            extra["eligible_before"] += int(scans[paths[0]].mask.sum())
        for b, (lo, hi_eff) in enumerate(bounds):
            u = Unit(f"{g}#blk{b:02d}", g, paths, lo, hi_eff)
            u.n_elig = int(scans[paths[0]].mask[lo:hi_eff].sum())
            units.append(u)
    return units, "block", extra


def assign_units(sub: str, units: list[Unit], quota: tuple[int, int, int], n_test: int, n_val: int,
                 seed: int) -> tuple[dict[str, list[Unit]], list[str]]:
    """Choose test, then val units (random order, skipping units too small for the quota); rest is train."""
    notes: list[str] = []
    order = list(_rng(seed, sub, "assign").permutation(len(units)))
    chosen: dict[str, list[Unit]] = {"test": [], "val": [], "train": []}
    used: set[int] = set()
    max_eval = max(len(units) - 1, 0)
    for split, k, q in (("test", n_test, quota[2]), ("val", n_val, quota[1])):
        k = min(k, max_eval - len(used))
        for i in order:
            if len(chosen[split]) == k:
                break
            if i in used:
                continue
            if units[i].n_elig >= q:
                chosen[split].append(units[i])
                used.add(i)
            else:
                notes.append(f"{sub}: unit {units[i].unit_id} skipped for {split} "
                             f"(eligible {units[i].n_elig} < quota {q})")
        if len(chosen[split]) < k:  # no unit is large enough: take the largest remaining ones and reduce quota later
            rest = sorted((i for i in range(len(units)) if i not in used), key=lambda i: -units[i].n_elig)
            for i in rest[: k - len(chosen[split])]:
                chosen[split].append(units[i])
                used.add(i)
            notes.append(f"{sub}: no unit large enough for {split} quota {q}; using the largest available")
    chosen["train"] = [units[i] for i in range(len(units)) if i not in used]
    return chosen, notes


def _unit_positions(unit: Unit, scans: dict[str, FileScan], cap: int | None, rng: np.random.Generator):
    """Eligible (path index, row) positions of a unit, optionally thinned to `cap` rows per stream."""
    f_idx, rows, streams = [], [], []
    for j, p in enumerate(unit.paths):
        sc = scans[p]
        m = sc.mask
        if unit.lo is not None:
            m = np.zeros_like(sc.mask)
            m[unit.lo:unit.hi] = sc.mask[unit.lo:unit.hi]
        r = np.flatnonzero(m).astype(np.int32)
        rows.append(r)
        f_idx.append(np.full(len(r), j, dtype=np.int32))
        streams.append(sc.stream[r])
    f_idx, rows, streams = np.concatenate(f_idx), np.concatenate(rows), np.concatenate(streams)
    if cap is not None and len(rows):
        perm = rng.permutation(len(rows))
        s = streams[perm]
        rank = pd.Series(s).groupby(s).cumcount().to_numpy()
        keep = perm[(rank < cap) | (s < 0)]
        f_idx, rows = f_idx[keep], rows[keep]
    return f_idx, rows


@dataclass
class Plan:
    selections: dict[str, list[tuple[np.ndarray, str, str, str]]]   # path -> [(rows, sub, split, unit_id)]
    manifest_units: dict[str, Any]
    warnings: list[str]
    requested: dict[str, dict[str, int]]
    actual: dict[str, dict[str, int]]


def plan_sampling(files: pd.DataFrame, scans: dict[str, FileScan], cfg: dict[str, Any]) -> Plan:
    sp = cfg["split"]
    seed = int(sp.get("split_seed", 0))
    cap = (cfg.get("train_sampling") or {}).get("max_rows_per_stream")
    quotas = subclass_quotas(cfg)
    missing = set(quotas) - set(files["class11"].unique())
    if missing:
        raise KeyError(f"quota defined for sub-classes without files: {sorted(missing)}")
    plan = Plan({}, {}, [], {}, {})
    for sub, quota in quotas.items():
        sub_files = files[files["class11"] == sub]
        units, kind, extra = build_units(
            sub_files, scans, int(sp.get("min_groups_for_group_split", 3)), sp["fallback_block_split"]["n_blocks"],
            sp["fallback_block_split"]["gap_rows"], bool(sp.get("drop_cross_block_streams", True)))
        assign, notes = assign_units(sub, units, quota, sp["test_groups_per_subclass"],
                                     sp["val_groups_per_subclass"], seed)
        plan.warnings += notes
        plan.requested[sub] = dict(zip(SPLITS, quota))
        plan.actual[sub] = {}
        plan.manifest_units[sub] = {"split_kind": kind, "n_units": len(units), **({"cross_block": extra} if extra else {}),
                                    "units": {}}
        for split, q in zip(SPLITS, quota):
            us = assign[split]
            thin = cap if split == "train" else None
            # Stream thinning (A5) needs the positions to know the capacity; otherwise capacity = eligible rows.
            cache = {u.unit_id: _unit_positions(u, scans, thin, _rng(seed, sub, split, u.unit_id, "thin"))
                     for u in us} if thin is not None else {}
            caps = [len(cache[u.unit_id][1]) if thin is not None else u.n_elig for u in us]
            if sum(caps) < q:
                plan.warnings.append(f"{sub}/{split}: quota reduced {q} -> {sum(caps)} (not enough eligible rows)")
            alloc = allocate(min(q, sum(caps)), caps)
            plan.actual[sub][split] = int(sum(alloc))
            plan.manifest_units[sub]["units"][split] = []
            for u, k in zip(us, alloc):
                f_idx, rows = cache.get(u.unit_id) or _unit_positions(u, scans, None, None)
                pick = np.sort(_rng(seed, sub, split, u.unit_id, "pick").choice(len(rows), size=k, replace=False))
                for j, p in enumerate(u.paths):
                    sel = rows[pick][f_idx[pick] == j]
                    if len(sel):
                        plan.selections.setdefault(p, []).append((np.sort(sel), sub, split, u.unit_id))
                cache.pop(u.unit_id, None)
                plan.manifest_units[sub]["units"][split].append(
                    {"unit_id": u.unit_id, "group": u.group, "eligible_rows": int(u.n_elig), "sampled": int(k),
                     **({"row_range": [u.lo, u.hi]} if u.lo is not None else {})})
    return plan


# --------------------------------------------------------------------------------------------------
# Pass B: extract
# --------------------------------------------------------------------------------------------------
def _peak_rss_gb(prev: float) -> float:
    try:
        import psutil
        return max(prev, psutil.Process().memory_info().rss / 1e9)
    except ImportError:
        return prev


def extract(files: pd.DataFrame, plan: Plan, scans: dict[str, FileScan]) -> tuple[pd.DataFrame, float]:
    meta = files.set_index(files["path"].astype(str))
    pieces: list[pd.DataFrame] = []
    peak = 0.0
    for n_done, (path, sels) in enumerate(plan.selections.items()):
        rows = np.concatenate([s[0] for s in sels])
        tag = np.concatenate([np.full(len(s[0]), t, dtype=np.int32) for t, s in enumerate(sels)])
        o = np.argsort(rows)
        rows, tag = rows[o], tag[o]
        parts, offset, seen = [], 0, 0
        for chunk in pd.read_csv(path, chunksize=_CHUNK, dtype=str, low_memory=False):
            lo, hi = np.searchsorted(rows, [offset, offset + len(chunk)])
            if hi > lo:
                part = chunk.iloc[rows[lo:hi] - offset].copy()
                part["row_idx"] = rows[lo:hi]
                part["_tag"] = tag[lo:hi]
                parts.append(part)
            offset += len(chunk)
            seen += len(chunk)
            peak = _peak_rss_gb(peak)
        if seen != scans[path].n_rows:
            raise RuntimeError(f"{path}: row count changed between passes ({seen} != {scans[path].n_rows})")
        df = pd.concat(parts, ignore_index=True)
        df = df.rename(columns=COLUMN_RENAME).drop(columns=["label", "capture_id"], errors="ignore")
        info = meta.loc[path]
        tags = pd.DataFrame([{"sub": s[1], "split": s[2], "group_id": s[3]} for s in sels])
        t = tags.iloc[df.pop("_tag").to_numpy()].reset_index(drop=True)
        df["split"], df["group_id"] = t["split"], t["group_id"]
        df["class11"], df["class6"] = info["class11"], info["class6"]
        df["data_source"], df["source_file"] = info["source"], info["filename"]
        df["stream_id"] = info["group_id"] + ":" + df["stream_index"].fillna("NA")
        pieces.append(df)
        if n_done % 50 == 0:
            logger.info("Extract %d/%d files", n_done + 1, len(plan.selections))
    return pd.concat(pieces, ignore_index=True), peak


# --------------------------------------------------------------------------------------------------
# Verification and output
# --------------------------------------------------------------------------------------------------
def verify(df: pd.DataFrame, plan: Plan, cfg: dict[str, Any], filters: dict[str, list[str]]) -> dict[str, Any]:
    """Gate checks. Raises AssertionError / QuotaError; returns facts for the manifest."""
    sp = cfg["split"]
    # no group (or block) in more than one split
    n_splits_per_group = df.groupby("group_id")["split"].nunique()
    assert (n_splits_per_group == 1).all(), f"groups in several splits: {n_splits_per_group[n_splits_per_group > 1].index.tolist()}"
    # each raw row sampled at most once
    assert not df.duplicated(["source_file", "row_idx"]).any(), "a raw row was sampled twice"
    # row filters hold
    for col, allowed in filters.items():
        assert df[col].isin(allowed).all(), f"row filter violated for {col}"
    # counts equal the plan (and the plan equals the quota unless a warning says it was reduced)
    counts = df.groupby(["class11", "split"]).size()
    for sub, per in plan.actual.items():
        for split, n in per.items():
            assert counts.get((sub, split), 0) == n, f"{sub}/{split}: {counts.get((sub, split), 0)} != planned {n}"
    # label sizes for val/test
    label_counts = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    for split in ("val", "test"):
        small = label_counts[label_counts[split] < sp.get("min_eval_rows", 1000)]
        if len(small):
            raise QuotaError(f"{split} classes below {sp.get('min_eval_rows', 1000)} rows: "
                             f"{small[split].to_dict()} -- stop and ask the user")
    # stream overlap across splits (matters for block-split sub-classes)
    overlap = {}
    st = df.groupby("stream_id")["split"].nunique()
    shared = st[st > 1].index
    if len(shared):
        by_sub = df[df["stream_id"].isin(shared)].groupby("class11")["stream_id"].nunique()
        overlap = {k: int(v) for k, v in by_sub.items()}
    if sp.get("drop_cross_block_streams", True):
        assert not overlap, f"streams shared across splits despite drop_cross_block_streams: {overlap}"
    return {"label_counts": {k: {s: int(v) for s, v in r.items()} for k, r in label_counts.iterrows()},
            "streams_in_multiple_splits": overlap}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _versions() -> dict[str, str]:
    import sys

    import pyarrow
    out = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
           "pyarrow": pyarrow.__version__}
    return out


def _git_dirty() -> bool | None:
    try:
        return bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"],
                                   capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def interim_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["work_dir"]) / "interim" / cfg["label_mode"]


def publish_manifest(cfg: dict[str, Any], manifest_path: Path) -> Path | None:
    """Copy the manifest to `paths.shared_manifest_dir` (tracked by git, unlike data/interim).

    The manifest holds group ids, counts and file hashes only - no data rows, no local paths.
    """
    d = cfg["paths"].get("shared_manifest_dir")
    if not d:
        return None
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    dest = Path(d) / f"split_manifest_{m['label_mode']}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def run_sample(cfg: dict[str, Any], files: pd.DataFrame | None = None,
               filters: dict[str, list[str]] | None = None, use_cache: bool = True) -> dict[str, Any]:
    work = Path(cfg["paths"]["work_dir"])
    if files is None:
        files = pd.read_csv(work / "inventory" / "files.csv")
    files = files[files["rows"] > 0].reset_index(drop=True)
    filters = load_row_filters(cfg) if filters is None else filters
    sp = cfg["split"]
    logger.info("label_mode=%s split_seed=%s filters=%s", cfg["label_mode"], sp.get("split_seed", 0), filters)

    needed = set(subclass_quotas(cfg))
    files = files[files["class11"].isin(needed)].reset_index(drop=True)
    scans = scan_files(files, filters, work / "interim" / "_scan" if use_cache else None)
    plan = plan_sampling(files, scans, cfg)
    for w in plan.warnings:
        logger.warning(w)
    df, peak = extract(files, plan, scans)

    labcol = "class11" if cfg["label_mode"] == "11class" else "class6"
    df["label"] = df[labcol]
    seed = int(sp.get("split_seed", 0))
    df = df.iloc[_rng(seed, "shuffle").permutation(len(df))].reset_index(drop=True)
    cols = [c for c in SHARED_COLUMNS if c in df.columns] + _META_COLS
    df = df[cols]

    facts = verify(df, plan, cfg, filters)
    max_gb = float(sp.get("max_peak_gb", 4.0))
    assert peak < max_gb, f"peak memory {peak:.2f} GB >= {max_gb} GB"

    out = interim_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    hashes, sizes = {}, {}
    for split in SPLITS:
        p = out / f"{split}.parquet"
        d = df[df["split"] == split].reset_index(drop=True)
        d.to_parquet(p, index=False, engine="pyarrow", compression="snappy")
        hashes[split], sizes[split] = _sha256(p), int(len(d))
    manifest = {
        "label_mode": cfg["label_mode"], "split_seed": seed, "config_hash": config_hash(cfg),
        "git_commit": _git_commit(), "git_dirty": _git_dirty(),
        "versions": _versions(), "row_filters": filters,
        "max_rows_per_stream": (cfg.get("train_sampling") or {}).get("max_rows_per_stream"),
        "quota_requested": plan.requested, "quota_actual": plan.actual, "rows": sizes,
        "parquet_sha256": hashes, "peak_rss_gb": round(peak, 3), "warnings": plan.warnings,
        "subclasses": plan.manifest_units, **facts,
    }
    with open(out / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    shared = publish_manifest(cfg, out / "split_manifest.json")
    if shared is not None:
        logger.info("Shareable manifest copy: %s", shared)
    logger.info("Wrote %s (rows=%s, peak=%.2f GB)", out, sizes, peak)
    return manifest
