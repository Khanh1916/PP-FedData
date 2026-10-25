"""Provenance of the raw data: is the Normal CSV that the group extracted from PCAP comparable to the attack CSV of the dataset's authors?

The Kaggle release of DoS/DDoS-MQTT-IoT (alaaalatram/dosddos-mqtt-iot) has 362 PCAP files and 49 CSV files: the authors' CSV of 48 attack
captures (one more file is empty) and no CSV for the 49 Normal captures, which the group extracted from PCAP with TShark
(`tools/extract_normal_pcap.py`). The checks, written to results/data_provenance.json and results/reports/data_provenance.md:

0. **Release** (`kaggle`, needs the network): every file of the Kaggle release against the local copy (path and size, from the public
   listing API).
1. **Method** (`method`): the extraction procedure of the group, run on the PCAP of every attack capture that has an official CSV, against
   that CSV: number of rows, and per column the share of equal values, raw (string) and after the parsers of `preprocess.py` (what the
   models see). The procedure is the script's own `extract_pcap`, loaded from its file, so it is the one that produced the Normal CSV.
2. **Normal CSV** (`normal_csv`): per Normal capture, the parts hold the frame numbers 1..N exactly once, N equals the rows recorded at
   extraction, the label is Normal and the capture id is the PCAP's path.
3. **Normal PCAP** (`normal_pcap`, when the PCAP files are present): N equals the number of packets of the PCAP (`capinfos`), and a fresh
   extraction of a few captures gives byte-identical CSV parts.
4. **The authors' export settings** (`recipe`): the same script with two TShark options added, two-pass dissection (`-2`, as the
   Wireshark GUI) and a display filter (`-Y`; the first of AUTHOR_FILTERS that gives the official frame numbers), against the official
   CSV: which filter, and the column table again. It tells which differences of check 1 come from these export settings.
5. **Weight on the results** (`impact`): the broker-sent rows of the sampled Normal data; the saved test predictions of every 6-class
   run re-scored on the client-sent test rows only (the rows the authors' filter would keep), and RF retrained without the broker-sent
   rows.

Only counts, shares, scores and hashes are written; no data row.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("ppfeddata.extraction_check")

KAGGLE_DATASET = "alaaalatram/dosddos-mqtt-iot"
# display filters tried in this order for the authors' export (`{broker}` = the broker's address)
AUTHOR_FILTERS = ("ip.dst=={broker}", "ip")
# configurations shown in the report of the impact check (all are in the JSON); the names are those of the first study, whose saved
# predictions the impact check scored (section 5 of the report); round R has no broker-sent rows to weigh
HEADLINE = r"^((B0|B1a|B1b)-(rf|mlp)|(B3|M2|M1-eps5|MGs-eps5|MGs-eps10|MGr-eps1|MGs-nodp|MGp-nodp)-TSTR-(rf|mlp))$"
META = ("Label", "Capture_ID")
# columns the pipeline drops but that are numbers / flags (compared as such); the rest of the 33 follow feature_decisions.yaml
EXTRA_NUMERIC = ("no", "epoch_time", "time_since_reference_or_first_frame", "stream_index", "frame_length_on_wire_1")
EXTRA_BINARY = ("clean_session_flag_1",)
TEXT = ("source", "info")


# --------------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------------
def load_extractor(path: str | Path):
    """The group's extraction script as a module (its FIELDS, find_tshark, extract_pcap)."""
    spec = importlib.util.spec_from_file_location("extract_normal_pcap", Path(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _key(stem: str) -> str:
    """CSV and PCAP names of one capture differ in '-' / '_' and spaces (SYN_DDoS-AD_1.csv, SYN-DDoS-AD_1.pcap, 'BF1_DDoS_AD_13 .pcap')."""
    return re.sub(r"[\s]+", "", stem).replace("-", "_").lower()


def attack_pairs(root: Path) -> list[dict[str, Any]]:
    """Every official attack CSV with its PCAP: [{scenario, attack_type, csv, pcap (or None), empty}]."""
    out = []
    for csv_path in sorted(root.glob("*/*/CSV Files/*.csv")):
        pdir = csv_path.parent.parent / "Pcap Files"
        pcaps = {_key(p.stem): p for p in pdir.glob("*.pcap")} if pdir.exists() else {}
        out.append({"scenario": csv_path.parts[-4], "attack_type": csv_path.parts[-3], "csv": csv_path,
                    "pcap": pcaps.get(_key(csv_path.stem)), "empty": csv_path.stat().st_size == 0})
    return out


def capinfos_packets(capinfos: str, pcap: Path) -> int:
    out = subprocess.run([capinfos, "-c", "-M", str(pcap)], capture_output=True, text=True, check=True).stdout
    m = re.search(r"Number of packets:\s*([\d,]+)", out)
    if not m:
        raise RuntimeError(f"capinfos gave no packet count for {pcap}: {out[:200]}")
    return int(m.group(1).replace(",", ""))


def read_csvs(paths: list[Path]) -> pd.DataFrame:
    """CSV files as strings, as the pipeline reads them (duplicate headers get pandas' '.1' suffix)."""
    frames = [pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig") for p in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def with_options(make_command, options: tuple[str, ...]):
    """The script's own TShark command with extra options placed after `-n` (before `-r`)."""
    def command(tshark, pcap, mqtt_port):
        c = make_command(tshark, pcap, mqtt_port)
        if c[1] != "-n":
            raise RuntimeError(f"unexpected TShark command of the extraction script: {c[:4]}")
        return c[:2] + list(options) + c[2:]
    return command


def extract(extractor, tshark: str, pcap: Path, out_dir: Path, max_rows: int = 100_000, mqtt_port: int = 1883,
            options: tuple[str, ...] = ()) -> list[Path]:
    """Run the script's own extract_pcap on one PCAP into `out_dir` (fresh); returns the CSV parts in order. `options` are extra TShark
    options (check 4); the script itself is not changed, its command builder is wrapped for this call only."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    own = extractor.make_tshark_command
    if options:
        extractor.make_tshark_command = with_options(own, options)
    try:
        res = extractor.extract_pcap(tshark, pcap, pcap.parent, out_dir, max_rows, mqtt_port, True)
    finally:
        extractor.make_tshark_command = own
    if str(res.get("Status")) != "OK":
        raise RuntimeError(f"extraction of {pcap.name} failed: {res}")
    return sorted(out_dir.glob("*_part[0-9][0-9][0-9].csv"))


def kaggle_listing(dataset: str = KAGGLE_DATASET, timeout: int = 60) -> list[tuple[str, int]]:
    """(path, bytes) of every file of the Kaggle release, from the public listing API (no login)."""
    import urllib.parse
    import urllib.request

    files, token = [], None
    for _ in range(100):
        q = f"https://www.kaggle.com/api/v1/datasets/list/{dataset}?pageSize=200" + (f"&pageToken={urllib.parse.quote(token)}" if token else "")
        with urllib.request.urlopen(q, timeout=timeout) as r:
            j = json.load(r)
        files += [(f["name"], int(f["totalBytes"])) for f in j.get("datasetFiles", [])]
        token = j.get("nextPageToken")
        if not token:
            break
    return files


def _kind(name: str) -> str:
    return ("Normal " if "NormalData" in Path(name).parts else "attack ") + (Path(name).suffix.lower().lstrip(".") or "other")


def match_listing(listing: list[tuple[str, int]], root: Path) -> dict[str, Any]:
    """The release against the local copy. Listed paths start with the dataset folder, which is `root` here (any local name)."""
    by_kind: dict[str, dict[str, int]] = {}
    issues = []
    for name, size in listing:
        p = root.joinpath(*Path(name).parts[1:])
        st = "missing" if not p.exists() else "same size" if p.stat().st_size == size else "size differs"
        by_kind.setdefault(_kind(name), {"same size": 0, "size differs": 0, "missing": 0})[st] += 1
        if st != "same size":
            issues.append({"file": name, "status": st})
    listed = {Path(*Path(n).parts[1:]).as_posix() for n, _ in listing}
    extra: dict[str, int] = {}
    for p in root.rglob("*"):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and p.suffix.lower() in (".csv", ".pcap") and rel not in listed:
            folder = str(Path(rel).parent.as_posix())
            extra[folder] = extra.get(folder, 0) + 1
    return {"dataset": KAGGLE_DATASET, "files": len(listing), "bytes": int(sum(s for _, s in listing)), "by_kind": by_kind,
            "issues": issues[:50], "local_csv_pcap_not_in_release": dict(sorted(extra.items()))}


# --------------------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------------------
def roles(decisions: dict[str, Any], columns: list[str]) -> dict[str, str]:
    """numeric / binary / categorical / text per snake_case column."""
    out = {}
    for c in columns:
        a = (decisions.get(c) or {}).get("action")
        out[c] = (a if a in ("numeric", "binary", "categorical") else
                  "numeric" if c in EXTRA_NUMERIC else "binary" if c in EXTRA_BINARY else "text")
    return out


def _parsed(series: pd.Series, col: str, role: str) -> pd.Series:
    from ppfeddata.data.parse_multi import first_values
    from ppfeddata.data.preprocess import parse_binary, parse_categorical, parse_numeric

    if role == "numeric":
        return parse_numeric(series)
    if role == "binary":
        return parse_binary(series)
    if role == "categorical":
        return parse_categorical(series, col)
    return first_values(series) if col != "info" else series.astype(str).str.strip()


def _equal(a: pd.Series, b: pd.Series, role: str) -> np.ndarray:
    if role in ("numeric", "binary"):
        x, y = a.to_numpy(dtype="float64"), b.to_numpy(dtype="float64")
        both_nan = np.isnan(x) & np.isnan(y)
        with np.errstate(invalid="ignore"):
            close = np.abs(x - y) <= 1e-6 * np.maximum(1.0, np.abs(x))
        return both_nan | close
    x, y = a.astype(object).where(a.notna(), None).to_numpy(), b.astype(object).where(b.notna(), None).to_numpy()
    return np.array([u == v for u, v in zip(x, y)], dtype=bool)


def compare(official: pd.DataFrame, ours: pd.DataFrame, decisions: dict[str, Any]) -> dict[str, Any]:
    """Per column: share of rows with the same value, raw and after the pipeline's parsers. Rows are aligned by frame number."""
    from ppfeddata.data.harmonize import COLUMN_RENAME

    off = official.rename(columns=COLUMN_RENAME)
    our = ours.drop(columns=[c for c in META if c in ours]).rename(columns=COLUMN_RENAME)
    out: dict[str, Any] = {"rows_official": int(len(off)), "rows_ours": int(len(our))}
    a = off.set_index(pd.to_numeric(off["no"], errors="coerce"))
    b = our.set_index(pd.to_numeric(our["no"], errors="coerce"))
    common = a.index.intersection(b.index)
    out.update(rows_common=int(len(common)), only_official=int(len(a.index.difference(b.index))), only_ours=int(len(b.index.difference(a.index))))
    a, b = a.loc[common], b.loc[common]
    cols = [c for c in a.columns if c in b.columns]
    out["columns"] = {}
    bad = np.zeros(len(common), dtype=bool)          # rows with a difference in a column the models use
    for c, role in roles(decisions, cols).items():
        raw = (a[c].astype(str).str.strip().to_numpy() == b[c].astype(str).str.strip().to_numpy())
        eq = _equal(_parsed(a[c], c, role).reset_index(drop=True), _parsed(b[c], c, role).reset_index(drop=True), role)
        used = (decisions.get(c) or {}).get("action") in ("numeric", "binary", "categorical")
        out["columns"][c] = {"role": role, "used": used,
                             "raw_match": float(raw.mean()) if len(raw) else float("nan"), "match": float(eq.mean()) if len(eq) else float("nan"),
                             "mismatch_rows": int((~eq).sum())}
        if used:
            bad |= ~eq
    retx = a["info"].astype(str).str.contains("Retransmission", regex=False).to_numpy() if "info" in a else np.zeros(len(common), dtype=bool)
    out.update(rows_used_mismatch=int(bad.sum()), rows_used_mismatch_retransmission=int((bad & retx).sum()))
    return out


# --------------------------------------------------------------------------------------------------
# The three checks
# --------------------------------------------------------------------------------------------------
_FROM_1883 = r"^\s*1883\s*\S+\s*\d+"
_IPV4 = r"^\d{1,3}(\.\d{1,3}){3}$"


def broker_of(df: pd.DataFrame) -> str | None:
    """The broker: the most frequent source of packets whose Info starts with 'srcport 1883 → …'."""
    s = df.loc[df["Info"].astype(str).str.match(_FROM_1883), "Source"]
    return None if s.empty else str(s.value_counts().index[0])


def population(df: pd.DataFrame, broker: str | None) -> pd.Series:
    """Who sent each packet: 'broker', 'client' (any other IPv4 host) or 'non-IP' (switch frames: STP, CDP, LOOP)."""
    src = df["Source"].astype(str)
    return pd.Series(np.where(src == broker, "broker", np.where(src.str.match(_IPV4), "client", "non-IP")), index=df.index)


def kept_population(official: pd.DataFrame, ours: pd.DataFrame) -> dict[str, Any]:
    """Which packets of the PCAP the authors' CSV keeps, by sender, and whether its time delta is the gap to the previous KEPT packet
    (the sign of a display filter at export: 'Time delta from previous displayed frame')."""
    broker = broker_of(ours)
    kept = pd.to_numeric(ours["No."], errors="coerce").isin(set(pd.to_numeric(official["No."], errors="coerce")))
    pop = population(ours, broker)
    t = pd.to_numeric(official["Epoch Time"], errors="coerce").to_numpy()
    d = pd.to_numeric(official["Time delta from previous displayed frame"], errors="coerce").to_numpy()
    gap = np.r_[0.0, np.diff(t)]
    return {"broker": broker, "by_sender": {k: {"packets": int((pop == k).sum()), "kept": int(((pop == k) & kept).sum())} for k in ("client", "broker", "non-IP")},
            "delta_is_gap_to_previous_kept": float(np.mean(np.abs(d - gap) < 1e-6)) if len(t) else float("nan")}


def check_method(root: Path, extractor, tshark: str, capinfos: str, decisions: dict[str, Any], work: Path,
                 only: list[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for p in attack_pairs(root):
        name = p["csv"].stem
        if only and name not in only:
            continue
        r: dict[str, Any] = {"capture": name, "scenario": p["scenario"], "attack_type": p["attack_type"], "pcap": p["pcap"].name if p["pcap"] else None}
        if p["empty"] or p["pcap"] is None:
            r["status"] = "empty official CSV" if p["empty"] else "no PCAP"
            rows.append(r)
            continue
        r["packets"] = capinfos_packets(capinfos, p["pcap"])
        parts = extract(extractor, tshark, p["pcap"], work / "method" / _key(name))
        off, our = read_csvs([p["csv"]]), read_csvs(parts)
        r.update(compare(off, our, decisions), kept=kept_population(off, our), status="compared")
        shutil.rmtree(work / "method" / _key(name), ignore_errors=True)
        logger.info("method %s: %d official rows, %d extracted, %d packets; feature columns min match %.4f", name, r["rows_official"], r["rows_ours"],
                    r["packets"], min(v["match"] for v in r["columns"].values() if v["used"]))
        rows.append(r)
    return rows


def check_normal_csv(root: Path) -> list[dict[str, Any]]:
    d = root / "NormalData" / "CSV_Extracted_Split"
    summary = {r["PCAP"]: r for r in csv.DictReader(open(d / "normal_extraction_summary.csv", encoding="utf-8-sig"))} \
        if (d / "normal_extraction_summary.csv").exists() else {}
    out = []
    for marker in sorted(d.glob("*.done.json")):
        m = json.loads(marker.read_text(encoding="utf-8"))
        prefix = marker.name[:-len(".done.json")]
        parts = sorted(d.glob(f"{prefix}_part[0-9][0-9][0-9].csv"))
        no, labels, caps, frames = [], set(), set(), []
        for p in parts:
            df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig", usecols=["No.", "Source", "Protocol", "Info", *META])
            no.append(pd.to_numeric(df["No."], errors="coerce").to_numpy())
            labels |= set(df["Label"].unique())
            caps |= set(df["Capture_ID"].unique())
            frames.append(df[["Source", "Protocol", "Info"]])
        n = np.concatenate(no) if no else np.array([])
        s = summary.get(m["pcap"], {})
        allf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Source", "Protocol", "Info"])
        broker = broker_of(allf)
        pop = population(allf, broker)
        tcp_mqtt = allf["Protocol"].isin(["TCP", "MQTT"])            # the row filter of the pipeline (Phase 3)
        out.append({"pcap": m["pcap"], "parts": len(parts), "parts_recorded": int(m["parts"]), "rows": int(len(n)), "rows_recorded": int(m["rows"]),
                    "rows_summary": int(s["Rows"]) if s.get("Rows") else None, "frames_1_to_n": bool(len(n) and np.array_equal(n, np.arange(1, len(n) + 1))),
                    "labels": sorted(labels), "capture_ids": sorted(caps), "broker": broker,
                    "by_sender": {k: int((pop == k).sum()) for k in ("client", "broker", "non-IP")},
                    "by_sender_tcp_mqtt": {k: int(((pop == k) & tcp_mqtt).sum()) for k in ("client", "broker", "non-IP")}})
    return out


def check_normal_pcap(root: Path, extractor, tshark: str, capinfos: str, normal_csv: list[dict[str, Any]], work: Path,
                      reextract: list[str]) -> dict[str, Any]:
    nd = root / "NormalData"
    out: dict[str, Any] = {"packets": [], "reextract": []}
    for r in normal_csv:
        pcap = nd / r["pcap"]
        if not pcap.exists():
            out["packets"].append({"pcap": r["pcap"], "present": False})
            continue
        k = capinfos_packets(capinfos, pcap)
        out["packets"].append({"pcap": r["pcap"], "present": True, "bytes": pcap.stat().st_size, "packets": k, "rows": r["rows"], "equal": k == r["rows"]})
    for name in reextract:
        pcap = nd / "Pcap Files" / name
        if not pcap.exists():
            continue
        prefix = extractor.prefix_for(pcap, nd)
        old = sorted((nd / "CSV_Extracted_Split").glob(f"{prefix}_part[0-9][0-9][0-9].csv"))
        new_dir = work / "reextract" / prefix
        if new_dir.exists():
            shutil.rmtree(new_dir)
        new_dir.mkdir(parents=True)
        res = extractor.extract_pcap(tshark, pcap, nd, new_dir, 100_000, 1883, True)
        new = sorted(new_dir.glob(f"{prefix}_part[0-9][0-9][0-9].csv"))
        h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()       # noqa: E731
        same = len(old) == len(new) and all(h(a) == h(b) for a, b in zip(old, new))
        out["reextract"].append({"pcap": f"Pcap Files/{name}", "status": str(res.get("Status")), "parts_old": len(old), "parts_new": len(new),
                                 "byte_identical": bool(same), "sha256_first_part": h(new[0])[:16] if new else None})
        shutil.rmtree(new_dir, ignore_errors=True)
        logger.info("re-extract %s: %d parts, byte-identical %s", name, len(new), same)
    return out


def check_normal_authors(root: Path, decisions: dict[str, Any], broker: str, new_name: str,
                         old_name: str = "CSV_Extracted_Split") -> list[dict[str, Any]]:
    """The Normal CSV extracted again with the authors' settings (`extract-normal`) against the group's first extraction, per capture:
    its frames are a subset of the first one (none missing), which senders it keeps, whether its time delta is the gap to the previous
    kept packet, and per column the share of equal values on the common frames. The first extraction is read part by part (memory)."""
    nd = root / "NormalData"
    new_dir, old_dir = nd / new_name, nd / old_name
    out = []
    for marker in sorted(new_dir.glob("*.done.json")):
        m = json.loads(marker.read_text(encoding="utf-8"))
        prefix = marker.name[:-len(".done.json")]
        new = read_csvs(sorted(new_dir.glob(f"{prefix}_part[0-9][0-9][0-9].csv")))
        new_no = pd.to_numeric(new["No."], errors="coerce").to_numpy() if len(new) else np.array([])
        t = pd.to_numeric(new.get("Epoch Time", pd.Series(dtype=str)), errors="coerce").to_numpy()
        d = pd.to_numeric(new.get("Time delta from previous displayed frame", pd.Series(dtype=str)), errors="coerce").to_numpy()
        gap = np.r_[0.0, np.diff(t)] if len(t) else t
        r: dict[str, Any] = {"pcap": m["pcap"], "rows_new": int(len(new)), "rows_recorded": int(m["rows"]),
                             "frames_increasing": bool(len(new_no) == 0 or np.all(np.diff(new_no) > 0)),
                             "labels": sorted(new["Label"].unique()) if len(new) else [], "capture_ids": sorted(new["Capture_ID"].unique()) if len(new) else [],
                             "delta_is_gap_to_previous_kept": float(np.mean(np.abs(d - gap) < 1e-6)) if len(d) else float("nan")}
        by_sender = {k: {"packets": 0, "kept": 0} for k in ("client", "broker", "non-IP")}
        acc: dict[str, dict[str, Any]] = {}
        rows_old = common = missing = 0
        for p in sorted(old_dir.glob(f"{prefix}_part[0-9][0-9][0-9].csv")):
            old = read_csvs([p])
            rows_old += len(old)
            ono = pd.to_numeric(old["No."], errors="coerce").to_numpy()
            pop = population(old, broker).to_numpy()
            kept = np.isin(ono, new_no)
            for k in by_sender:
                by_sender[k]["packets"] += int((pop == k).sum())
                by_sender[k]["kept"] += int(((pop == k) & kept).sum())
            sel = new[(new_no >= ono.min()) & (new_no <= ono.max())]
            if not len(sel):
                continue
            c = compare(sel, old, decisions)
            common += c["rows_common"]
            missing += c["only_official"]
            for col, v in c["columns"].items():
                a = acc.setdefault(col, {"role": v["role"], "used": v["used"], "n": 0, "match": 0.0, "raw": 0.0})
                a["n"] += c["rows_common"]
                a["match"] += v["match"] * c["rows_common"]
                a["raw"] += v["raw_match"] * c["rows_common"]
        r.update(rows_old=rows_old, rows_common=common, frames_missing_in_old=missing, by_sender=by_sender,
                 columns={col: {"role": a["role"], "used": a["used"], "match": a["match"] / a["n"], "raw_match": a["raw"] / a["n"]}
                          for col, a in acc.items() if a["n"]})
        logger.info("normal_authors %s: %d rows (first extraction %d), %d not found there", m["pcap"], len(new), rows_old, missing)
        out.append(r)
    return out


def broker_from(out: dict[str, Any]) -> str | None:
    """The broker's address as found by checks 1 and 2 (most frequent)."""
    seen = [r["kept"]["broker"] for r in out.get("method") or [] if (r.get("kept") or {}).get("broker")]
    seen += [r["broker"] for r in out.get("normal_csv") or [] if r.get("broker")]
    return max(set(seen), key=seen.count) if seen else None


def check_recipe(root: Path, extractor, tshark: str, decisions: dict[str, Any], work: Path, broker: str,
                 only: list[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for p in attack_pairs(root):
        name = p["csv"].stem
        if (only and name not in only) or p["empty"] or p["pcap"] is None:
            continue
        off = read_csvs([p["csv"]])
        off_no = pd.to_numeric(off["No."], errors="coerce").to_numpy()
        r: dict[str, Any] = {"capture": name, "tried": [], "filter": None}
        d = work / "recipe" / _key(name)
        for flt in AUTHOR_FILTERS:
            f = flt.format(broker=broker)
            our = read_csvs(extract(extractor, tshark, p["pcap"], d, options=("-2", "-Y", f)))
            same = len(our) == len(off) and np.array_equal(pd.to_numeric(our["No."], errors="coerce").to_numpy(), off_no)
            r["tried"].append({"filter": f, "rows": int(len(our)), "same_frames": bool(same)})
            if same:
                r.update(compare(off, our, decisions), filter=f)
                break
        shutil.rmtree(d, ignore_errors=True)
        logger.info("recipe %s: filter %s, %s", name, r["filter"], "frames identical" if r["filter"] else "no filter gives the official frames")
        rows.append(r)
    return rows


def normal_time_delta(normal_dir: Path, broker: str, captures: tuple[str, ...]) -> list[dict[str, Any]]:
    """Time delta of the client-sent Normal rows (IPv4 source other than the broker, TCP or MQTT) as extracted (gap to the previous packet of
    the PCAP) and as the authors' filter would give it (gap to the previous kept packet, recomputed from the epoch times of the kept rows)."""
    out = []
    for stem in captures:
        parts = sorted(normal_dir.glob(f"*{stem}_part[0-9][0-9][0-9].csv"))
        if not parts:
            continue
        df = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig",
                                    usecols=["Epoch Time", "Source", "Protocol", "Time delta from previous displayed frame"]) for p in parts], ignore_index=True)
        kept = df["Source"].str.match(_IPV4) & df["Source"].ne(broker) & df["Protocol"].isin(["TCP", "MQTT"])
        now = pd.to_numeric(df.loc[kept, "Time delta from previous displayed frame"], errors="coerce")
        t = pd.to_numeric(df.loc[kept, "Epoch Time"], errors="coerce").to_numpy()
        auth = pd.Series(np.r_[0.0, np.diff(t)])
        q = lambda s: [float(s.quantile(p)) for p in (0.25, 0.5, 0.75)]       # noqa: E731
        out.append({"capture": stem, "rows": int(len(df)), "client_rows": int(kept.sum()), "quartiles_s_now": q(now), "quartiles_s_authors": q(auth),
                    "median_ratio": float(auth.median() / now.median())})
    return out


def check_impact(cfg: dict[str, Any], broker: str, seeds: tuple[int, ...] = (0, 1, 2),
                 delta_captures: tuple[str, ...] = ("NormalData1", "NormalData31", "NormalData50")) -> dict[str, Any]:
    from scipy.stats import spearmanr

    from ppfeddata.data.preprocess import parse_binary, parse_numeric, processed_dir
    from ppfeddata.data.split_sample import interim_dir
    from ppfeddata.eval.utility import confusion, macro_f1_from_confusion, make_classifier

    c6 = {**cfg, "label_mode": "6class"}
    I = {sp: pd.read_parquet(interim_dir(c6) / f"{sp}.parquet", columns=["source", "class6", "stream_id", "irtt", "syn"]) for sp in ("train", "test")}
    P = {sp: np.load(processed_dir(c6) / f"{sp}.npz", allow_pickle=True) for sp in ("train", "test")}
    lab = json.loads((processed_dir(c6) / "label_map.json").read_text(encoding="utf-8"))
    names = sorted(lab, key=lab.get)
    for sp in I:                                   # the interim rows (which keep the sender) are the processed rows, in the same order
        if not (len(I[sp]) == len(P[sp]["y"]) and np.array_equal(I[sp]["class6"].map(lab).to_numpy(), P[sp]["y"])
                and np.array_equal(I[sp]["stream_id"].astype(str).to_numpy(), P[sp]["stream_id"].astype(str))):
            raise RuntimeError(f"{sp}: interim and processed rows differ; run sample and preprocess again")
    normal, y = lab["NORMAL"], P["test"]["y"]
    br = I["test"]["source"].eq(broker).to_numpy()
    cl = ~br
    mf = lambda yt, yp: macro_f1_from_confusion(confusion(yt, yp, len(names)))       # noqa: E731
    share = {sp: {c: float(I[sp].loc[I[sp]["class6"] == c, "source"].eq(broker).mean()) for c in names} for sp in I}
    tr = I["train"]
    syn = parse_binary(tr["syn"]).eq(1).to_numpy()
    no_irtt = parse_numeric(tr["irtt"]).isna().to_numpy()
    irtt_syn = {c: {"syn_rows": int((syn & (tr["class6"] == c).to_numpy()).sum()),
                    "irtt_missing": float(no_irtt[syn & (tr["class6"] == c).to_numpy()].mean()) if (syn & (tr["class6"] == c).to_numpy()).any() else float("nan")}
                for c in names}
    root = Path(cfg["paths"]["raw_root"])
    deltas = normal_time_delta(root / cfg["paths"].get("normal_csv_dir", "NormalData/CSV_Extracted_Split"), broker, delta_captures)
    ledger = pd.read_csv(cfg["compute"]["runs_csv"], low_memory=False, usecols=["run_id", "config", "seed", "label_mode", "macro_f1"])
    ledger = ledger[ledger["label_mode"].eq("6class")].drop_duplicates("run_id", keep="last")
    art = Path(cfg["compute"]["artifacts_dir"])
    runs, skipped = [], {"no saved predictions": 0, "other test rows": 0}
    for r in ledger.itertuples():
        f = art / r.run_id / "preds" / "test.npz"
        if not f.exists():
            skipped["no saved predictions"] += 1
            continue
        pred = np.load(f)["y_pred"]
        if len(pred) != len(y) or not np.isclose(mf(y, pred), r.macro_f1, atol=1e-6):
            skipped["other test rows"] += 1            # e.g. the separate worlds of `sensitivity`
            continue
        runs.append({"config": r.config, "seed": int(r.seed), "macro_f1": mf(y, pred), "macro_f1_client": mf(y[cl], pred[cl]),
                     "normal_recall_broker": float(np.mean(pred[(y == normal) & br] == normal)),
                     "normal_recall_client": float(np.mean(pred[(y == normal) & cl] == normal))})
    df = pd.DataFrame(runs)
    by = df.groupby("config").agg(seeds=("seed", "size"), macro_f1=("macro_f1", "mean"), macro_f1_client=("macro_f1_client", "mean"),
                                  normal_recall_broker=("normal_recall_broker", "mean"), normal_recall_client=("normal_recall_client", "mean"))
    shift = by["macro_f1_client"] - by["macro_f1"]
    a, b = by["macro_f1"].to_numpy(), by["macro_f1_client"].to_numpy()
    i, j = np.triu_indices(len(by), 1)
    flipped = np.sign(a[i] - a[j]) != np.sign(b[i] - b[j])
    Xtr, ytr, Xte = P["train"]["X"], P["train"]["y"], P["test"]["X"]
    keep = ~I["train"]["source"].eq(broker).to_numpy()
    rf = []
    for s in seeds:
        full = make_classifier("rf", cfg, s).fit(Xtr, ytr).predict(Xte)
        client = make_classifier("rf", cfg, s).fit(Xtr[keep], ytr[keep]).predict(Xte)
        rf.append({"seed": s, "train_all_test_all": mf(y, full), "train_all_test_client": mf(y[cl], full[cl]),
                   "train_client_test_client": mf(y[cl], client[cl])})
    logger.info("impact: %d runs re-scored, %d configurations, Spearman %.3f", len(df), len(by), spearmanr(a, b).statistic)
    return {"broker": broker, "broker_share": share, "irtt_on_syn_rows": irtt_syn, "normal_time_delta": deltas,
            "test_rows": int(len(y)), "test_rows_client": int(cl.sum()),
            "train_rows_client": int(keep.sum()), "runs_scored": int(len(df)), "runs_skipped": skipped, "configs": int(len(by)),
            "spearman": float(spearmanr(a, b).statistic), "pairs": int(len(i)), "pairs_flipped": int(flipped.sum()),
            "largest_gap_flipped": float(np.abs(a[i] - a[j])[flipped].max()) if flipped.any() else 0.0,
            "shift": {"mean": float(shift.mean()), "min": float(shift.min()), "max": float(shift.max())},
            "by_config": by.reset_index().to_dict("records"), "rf_retrain": rf}


def tool_versions(tshark: str) -> dict[str, str]:
    first = lambda cmd: subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0]      # noqa: E731
    return {"tshark": first([tshark, "--version"])}


STAGES = ("kaggle", "method", "normal_csv", "normal_pcap", "recipe", "impact", "normal_authors")


def run(cfg: dict[str, Any], extractor_path: str | Path, stages: tuple[str, ...] = STAGES,
        only: list[str] | None = None, reextract: list[str] | None = None) -> dict[str, Any]:
    import yaml

    from ppfeddata.interpret import to_jsonable

    root = Path(cfg["paths"]["raw_root"])
    ext = load_extractor(extractor_path)
    tshark = ext.find_tshark(None)
    capinfos = str(Path(tshark).with_name("capinfos.exe" if tshark.lower().endswith(".exe") else "capinfos"))
    decisions = yaml.safe_load(Path("configs/feature_decisions.yaml").read_text(encoding="utf-8"))
    work = Path(cfg["compute"]["artifacts_dir"]) / "_extraction_check"
    res_dir = Path(cfg["compute"]["runs_csv"]).parent
    p = res_dir / "data_provenance.json"
    out = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    out.update(extractor=Path(extractor_path).as_posix(), versions=tool_versions(tshark), extraction_command=" ".join(
        ["tshark", "-n", "-r", "<pcap>", "-d", "tcp.port==1883,mqtt", "-T", "fields", "-E", "header=n", "-E", "separator=,", "-E", "quote=d",
         "-E", "escape=y", "-E", "occurrence=f"] + [f"-e {f}" for _, f in ext.FIELDS]))
    if "kaggle" in stages:
        out["kaggle"] = match_listing(kaggle_listing(), root)
    if "method" in stages:
        out["method"] = check_method(root, ext, tshark, capinfos, decisions, work, only)
    if "normal_csv" in stages or "normal_pcap" in stages:
        out["normal_csv"] = check_normal_csv(root)
    if "normal_pcap" in stages:
        out["normal_pcap"] = check_normal_pcap(root, ext, tshark, capinfos, out["normal_csv"], work, reextract or [])
    if "recipe" in stages or "impact" in stages or "normal_authors" in stages:
        broker = broker_from(out)
        if broker is None:
            raise RuntimeError("the broker's address is not known yet: run the method or normal_csv stage first")
        if "recipe" in stages:
            out["recipe"] = check_recipe(root, ext, tshark, decisions, work, broker, only)
        if "impact" in stages:
            out["impact"] = check_impact(cfg, broker)
        if "normal_authors" in stages:
            from ppfeddata.data.normal_extract import OUT_DIR
            out["normal_authors"] = check_normal_authors(root, decisions, broker, OUT_DIR)
    p.write_text(json.dumps(to_jsonable(out), indent=1), encoding="utf-8")
    (res_dir / "reports" / "data_provenance.md").write_text(render(out), encoding="utf-8")
    return out


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def _pct(x: Any, d: int = 2) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{100 * v:.{d}f} %"


def _column_means(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Per column, (match, raw match) weighted by the rows in both, over the compared captures."""
    tot = sum(r["rows_common"] for r in rows)
    cols = list(rows[0]["columns"]) if rows else []
    return {c: (sum(r["columns"][c]["match"] * r["rows_common"] for r in rows if c in r["columns"]) / tot,
                sum(r["columns"][c]["raw_match"] * r["rows_common"] for r in rows if c in r["columns"]) / tot) for c in cols}


def render(out: dict[str, Any]) -> str:
    L = ["# Provenance of the raw data", "",
         "Generated by `ppfeddata check-extraction` (module `data/extraction_check.py`); do not edit. Counts, shares, scores and hashes only, no data row.", "",
         "The Kaggle release (alaaalatram/dosddos-mqtt-iot) has the authors' CSV for 48 attack captures (one more file is empty) and no CSV for the "
         "Normal captures, which the group extracted from PCAP with `tools/extract_normal_pcap.py`. "
         f"Tools: {', '.join(f'{k} {v}'.rstrip('.') for k, v in (out.get('versions') or {}).items())}.", ""]
    if out.get("normal_authors"):
        L += ["**Since round R** the study uses the Normal traffic extracted with the authors' export settings (section 6). Sections 2, 3 and 5 describe the "
              "first extraction (every packet, one pass), kept in `NormalData/CSV_Extracted_Split` as the reference; section 5 measured its weight on the "
              "results of the first study.", ""]
    kg = out.get("kaggle")
    if kg:
        L += ["## 0. The local files against the Kaggle release", "",
              f"Release `{kg['dataset']}`: {kg['files']} files, {kg['bytes'] / 1e9:.2f} GB (public listing API: path and size of every file).", "",
              "| files | same path and size | size differs | missing locally |", "|---|---|---|---|"]
        L += [f"| {k} | {v['same size']} | {v['size differs']} | {v['missing']} |" for k, v in sorted(kg["by_kind"].items())]
        extra = kg.get("local_csv_pcap_not_in_release") or {}
        if extra:
            L += ["", "Local CSV / PCAP files that are not in the release (the group's own outputs): "
                  + "; ".join(f"`{k}` {v}" for k, v in extra.items()) + "."]
        if kg.get("issues"):
            L += ["", "Differences: " + "; ".join(f"{i['file']} ({i['status']})" for i in kg["issues"]) + "."]
        L.append("")
    m = [r for r in out.get("method") or [] if r.get("status") == "compared"]
    if out.get("method"):
        L += ["## 1. The extraction procedure against the authors' CSV (attack captures)", "",
              "The group's procedure, run on the PCAP of each attack capture that has an official CSV, compared with that CSV row by row "
              "(aligned by frame number). *Match* = share of rows with the same value after the parsers of `preprocess.py` (multi-value cells "
              "reduced to their first element, display labels mapped to codes, numbers compared to 1e-6), i.e. what the models see; *raw* = the "
              "same string.", "",
              "| capture | packets (capinfos) | official rows | extracted rows | rows in both | lowest match, feature columns |", "|---|---|---|---|---|---|"]
        for r in out["method"]:
            if r.get("status") != "compared":
                L.append(f"| {r['capture']} | - | - | - | - | {r.get('status')} |")
                continue
            used = [v["match"] for v in r["columns"].values() if v["used"]]
            L.append(f"| {r['capture']} | {r['packets']:,} | {r['rows_official']:,} | {r['rows_ours']:,} | {r['rows_common']:,} | {_pct(min(used)) if used else 'n/a'} |")
        if m and all("kept" in r for r in m):
            tot = {k: sum(r["kept"]["by_sender"][k]["packets"] for r in m) for k in ("client", "broker", "non-IP")}
            kept = {k: sum(r["kept"]["by_sender"][k]["kept"] for r in m) for k in ("client", "broker", "non-IP")}
            dg = [r["kept"]["delta_is_gap_to_previous_kept"] for r in m]
            brokers = sorted({r["kept"]["broker"] for r in m if r["kept"]["broker"]})
            L += ["", "**Which packets the authors' CSV keeps** (all captures above; broker = the source of the packets sent from port 1883: "
                  + ", ".join(brokers) + "):", "",
                  "| sender | packets in the PCAP | kept in the authors' CSV | share kept |", "|---|---|---|---|"]
            L += [f"| {k} | {tot[k]:,} | {kept[k]:,} | {_pct(kept[k] / tot[k] if tot[k] else float('nan'), 1)} |" for k in ("client", "broker", "non-IP")]
            L += ["", f"In the authors' CSV the time delta equals the gap to the previous kept packet in {_pct(min(dg), 2)} to {_pct(max(dg), 2)} of the rows "
                  "of each capture: the CSV was exported with a display filter, so 'Time delta from previous displayed frame' is measured within the kept packets."]
        if m:
            tot = sum(r["rows_common"] for r in m)
            L += ["", f"Per column over the {len(m)} captures ({tot:,} rows in both):", "",
                  "| column | role | used by the models | match | raw string equal |", "|---|---|---|---|---|"]
            for c, (mt, rw) in _column_means(m).items():
                L.append(f"| `{c}` | {m[0]['columns'][c]['role']} | {'yes' if m[0]['columns'][c]['used'] else 'no'} | {_pct(mt, 3)} | {_pct(rw, 1)} |")
    nc = out.get("normal_csv") or []
    if nc:
        ok = [r for r in nc if r["frames_1_to_n"] and r["rows"] == r["rows_recorded"] and r["parts"] == r["parts_recorded"]
              and r["labels"] == ["Normal"] and r["capture_ids"] == [r["pcap"]]]
        L += ["", "## 2. The Normal CSV", "",
              f"{len(nc)} captures, {sum(r['rows'] for r in nc):,} rows in {sum(r['parts'] for r in nc)} parts. Frame numbers 1..N exactly once, N = the rows "
              f"recorded at extraction, label Normal and capture id = the PCAP path: **{len(ok)} of {len(nc)}** captures."]
        bad = [r["pcap"] for r in nc if r not in ok]
        if bad:
            L += ["", "Not passing: " + ", ".join(bad) + "."]
        if all("by_sender" in r for r in nc):
            s_all = {k: sum(r["by_sender"][k] for r in nc) for k in ("client", "broker", "non-IP")}
            s_tm = {k: sum(r["by_sender_tcp_mqtt"][k] for r in nc) for k in ("client", "broker", "non-IP")}
            n_all, n_tm = sum(s_all.values()), sum(s_tm.values())
            L += ["", "**Sender of the Normal packets** (the extraction has no display filter, so it keeps every packet of the PCAP):", "",
                  "| sender | all rows | share | rows with protocol TCP or MQTT (the pipeline's row filter) | share |", "|---|---|---|---|---|"]
            L += [f"| {k} | {s_all[k]:,} | {_pct(s_all[k] / n_all, 1)} | {s_tm[k]:,} | {_pct(s_tm[k] / n_tm, 1)} |" for k in ("client", "broker", "non-IP")]
    npk = out.get("normal_pcap") or {}
    if npk.get("packets"):
        pr = [r for r in npk["packets"] if r.get("present")]
        eq = [r for r in pr if r["equal"]]
        L += ["", "## 3. The Normal PCAP", "",
              f"PCAP files present: {len(pr)} of {len(npk['packets'])}; rows = packets of the PCAP (capinfos): **{len(eq)} of {len(pr)}** "
              f"({sum(r['packets'] for r in pr):,} packets, {sum(r['rows'] for r in pr):,} rows)."]
        if npk.get("reextract"):
            L += ["", "| re-extracted | parts (before / now) | byte-identical |", "|---|---|---|"]
            L += [f"| {r['pcap']} | {r['parts_old']} / {r['parts_new']} | {'yes' if r['byte_identical'] else 'NO'} |" for r in npk["reextract"]]
    rc = out.get("recipe") or []
    if rc:
        L += ["", "## 4. The authors' export settings, reproduced", "",
              "The same script with two TShark options added: `-2` (two-pass dissection, as the Wireshark GUI exports) and `-Y <display filter>`, "
              f"the first of {', '.join('`' + f + '`' for f in AUTHOR_FILTERS)} (broker = `{broker_from(out)}`) that gives exactly the frame numbers of the "
              "official CSV.", "", "| display filter | captures | frame numbers identical to the official CSV |", "|---|---|---|"]
        filters: dict[str, list[str]] = {}
        for r in rc:
            filters.setdefault(r["filter"] or "none of them", []).append(r["capture"])
        for f, caps in filters.items():
            names = f" ({', '.join(caps)})" if len(caps) <= 3 else ""
            L.append(f"| `{f}` | {len(caps)}{names} | {len(caps) if f != 'none of them' else 0} |")
        ok = [r for r in rc if r["filter"]]
        if ok and m:
            c1, c4 = _column_means([r for r in m if r["capture"] in {o["capture"] for o in ok}]), _column_means(ok)
            L += ["", f"Per column over these {len(ok)} captures ({sum(r['rows_common'] for r in ok):,} rows), with the group's command (check 1) "
                  "and with the authors' settings:", "",
                  "| column | used by the models | match, group's command | match, authors' settings |", "|---|---|---|---|"]
            L += [f"| `{c}` | {'yes' if ok[0]['columns'][c]['used'] else 'no'} | {_pct(c1[c][0], 3) if c in c1 else 'n/a'} | {_pct(c4[c][0], 3)} |" for c in c4]
            if all("rows_used_mismatch" in r for r in ok):
                n, bad, rt = sum(r["rows_common"] for r in ok), sum(r["rows_used_mismatch"] for r in ok), sum(r["rows_used_mismatch_retransmission"] for r in ok)
                L += ["", f"With the authors' settings, rows with a difference in at least one column used by the models: {bad:,} of {n:,} "
                      f"({_pct(bad / n, 3)}); {rt:,} of them ({_pct(rt / bad if bad else float('nan'), 1)}) are TCP retransmissions "
                      "(official Info contains 'Retransmission')."]
    im = out.get("impact")
    if im:
        sh = im["broker_share"]
        L += ["", "## 5. Weight of the broker-sent Normal rows on the results", "",
              f"Share of rows sent by the broker (`{im['broker']}`) in the sampled data of the pipeline (6 classes):", "",
              "| class | train | test |", "|---|---|---|"]
        L += [f"| {c} | {_pct(sh['train'][c], 1)} | {_pct(sh['test'][c], 1)} |" for c in sh["train"]]
        if im.get("irtt_on_syn_rows"):
            ir = im["irtt_on_syn_rows"]
            L += ["", "SYN packets of the train split whose iRTT is empty (one-pass TShark leaves it empty on the SYN; a two-pass export fills it when the "
                  "handshake completes later in the capture):", "", "| class | SYN rows | iRTT empty |", "|---|---|---|"]
            L += [f"| {c} | {v['syn_rows']:,} | {_pct(v['irtt_missing'], 1)} |" for c, v in ir.items()]
        if im.get("normal_time_delta"):
            L += ["", "Time delta of the client-sent Normal rows, quartiles in ms: as extracted (gap to the previous packet of the PCAP) and as the authors' "
                  "filter would give it (gap to the previous client-sent packet):", "",
                  "| capture | client-sent rows | as extracted | authors' filter | ratio of the medians |", "|---|---|---|---|---|"]
            L += [f"| {r['capture']} | {r['client_rows']:,} | {' / '.join(f'{1e3 * v:.2f}' for v in r['quartiles_s_now'])} | "
                  f"{' / '.join(f'{1e3 * v:.2f}' for v in r['quartiles_s_authors'])} | {r['median_ratio']:.2f} |" for r in im["normal_time_delta"]]
        s = im["shift"]
        L += ["", f"Test rows sent by a client (the rows that a display filter on the broker as destination keeps): {im['test_rows_client']:,} of {im['test_rows']:,}. "
              f"The saved test predictions of {im['runs_scored']} runs ({im['configs']} configurations) scored again on these rows only "
              f"(not scored: {', '.join(f'{v} {k}' for k, v in im['runs_skipped'].items())}):", "",
              f"- macro-F1 on the client-sent rows minus macro-F1 on all rows, per configuration: mean {s['mean']:+.3f}, from {s['min']:+.3f} to {s['max']:+.3f};",
              f"- Spearman rank correlation of the configurations between the two scores: {im['spearman']:.3f}; of the {im['pairs']:,} pairs of "
              f"configurations, {im['pairs_flipped']:,} change order, none of them further apart than {im['largest_gap_flipped']:.3f} in macro-F1 on all rows.", "",
              "| configuration | seeds | macro-F1, all test rows | macro-F1, client-sent rows | NORMAL recall, broker-sent rows | NORMAL recall, client-sent rows |",
              "|---|---|---|---|---|---|"]
        L += [f"| {r['config']} | {r['seeds']} | {r['macro_f1']:.3f} | {r['macro_f1_client']:.3f} | {r['normal_recall_broker']:.3f} | {r['normal_recall_client']:.3f} |"
              for r in im["by_config"] if re.match(HEADLINE, r["config"])]
        L += ["", f"RF trained without the broker-sent rows ({im['train_rows_client']:,} train rows left), macro-F1:", "",
              "| seed | trained on all rows, all test rows | trained on all rows, client-sent test rows | trained on client-sent rows, client-sent test rows |",
              "|---|---|---|---|"]
        L += [f"| {r['seed']} | {r['train_all_test_all']:.3f} | {r['train_all_test_client']:.3f} | {r['train_client_test_client']:.3f} |" for r in im["rf_retrain"]]
    na = out.get("normal_authors") or []
    if na:
        ok = [r for r in na if r["rows_new"] == r["rows_recorded"] and r["frames_increasing"] and r["frames_missing_in_old"] == 0
              and r["labels"] == ["Normal"] and r["capture_ids"] == [r["pcap"]]]
        tot = {k: sum(r["by_sender"][k]["packets"] for r in na) for k in ("client", "broker", "non-IP")}
        kept = {k: sum(r["by_sender"][k]["kept"] for r in na) for k in ("client", "broker", "non-IP")}
        dg = [r["delta_is_gap_to_previous_kept"] for r in na if r["rows_new"]]
        L += ["", "## 6. The Normal traffic extracted again with the authors' settings", "",
              "`extract-normal`: the same script with `-2` and `-Y ip.dst==<broker>` (module `data/normal_extract.py`), into its own folder; the first "
              f"extraction is kept. {len(na)} captures, {sum(r['rows_new'] for r in na):,} rows (first extraction: {sum(r['rows_old'] for r in na):,}). "
              f"Rows = rows recorded, frame numbers increasing, every frame also in the first extraction, label Normal, capture id = the PCAP path: "
              f"**{len(ok)} of {len(na)}** captures.", "",
              "| sender (first extraction) | packets | kept | share kept |", "|---|---|---|---|"]
        L += [f"| {k} | {tot[k]:,} | {kept[k]:,} | {_pct(kept[k] / tot[k] if tot[k] else float('nan'), 1)} |" for k in ("client", "broker", "non-IP")]
        if dg:
            L += ["", f"Time delta equal to the gap to the previous kept packet: {_pct(min(dg))} to {_pct(max(dg))} of the rows of each capture."]
        cols = [r for r in na if r.get("columns") and r["rows_common"]]
        if cols:
            n = sum(r["rows_common"] for r in cols)
            names = list(cols[0]["columns"])
            L += ["", f"Per column, the new rows against the same frames of the first extraction ({n:,} rows):", "",
                  "| column | used by the models | match | raw string equal |", "|---|---|---|---|"]
            for c in names:
                mt = sum(r["columns"][c]["match"] * r["rows_common"] for r in cols if c in r["columns"]) / n
                rw = sum(r["columns"][c]["raw_match"] * r["rows_common"] for r in cols if c in r["columns"]) / n
                L.append(f"| `{c}` | {'yes' if cols[0]['columns'][c]['used'] else 'no'} | {_pct(mt, 3)} | {_pct(rw, 1)} |")
    if out.get("extraction_command"):
        L += ["", "## Extraction command", "", "```", out["extraction_command"], "```"]
    return "\n".join(L) + "\n"
