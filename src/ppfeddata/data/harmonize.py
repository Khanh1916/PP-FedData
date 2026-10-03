"""Harmonize schema, presence matrix, format audit, and feature decisions."""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ppfeddata.utils import load_config

logger = logging.getLogger("ppfeddata.data.harmonize")

_CHUNK = 50_000

# === Column name mapping ===
# Original column names → snake_case. Duplicates get _1/_2 suffix by POSITION.
COLUMN_RENAME = {
    "No.": "no",
    "Message Type": "message_type",
    "QoS Level": "qos_level_1",
    "QoS Level.1": "qos_level_2",
    "Requested QoS": "requested_qos",
    "Epoch Time": "epoch_time",
    "Protocol": "protocol",
    "Source": "source",
    "Frame length on the wire": "frame_length_on_wire_1",
    "Time delta from previous displayed frame": "time_delta_from_previous_displayed_frame",
    "Time since reference or first frame": "time_since_reference_or_first_frame",
    "Frame length on the wire.1": "frame_length_on_wire_2",
    "Stream index": "stream_index",
    "iRTT": "irtt",
    "Time since first frame in this TCP stream": "time_since_first_frame_in_this_tcp_stream",
    "TCP Segment Len": "tcp_segment_len",
    "Calculated window size": "calculated_window_size",
    "Syn": "syn",
    "Reset": "reset",
    "Acknowledgment": "acknowledgment",
    "Clean Session Flag": "clean_session_flag_1",
    "Keep Alive": "keep_alive",
    "User Name Length": "user_name_length",
    "Password Length": "password_length",
    "Retain": "retain",
    "Clean Session Flag.1": "clean_session_flag_2",
    "Will Retain": "will_retain",
    "Will Flag": "will_flag",
    "Will Message Length": "will_message_length",
    "Will Topic Length": "will_topic_length",
    "Topic Length": "topic_length",
    "Msg Len": "msg_len",
    "Info": "info",
    # Normal-only columns
    "Label": "label",
    "Capture_ID": "capture_id",
}

# The 33 shared column names (original) in expected order
SHARED_COLUMNS_ORIG = [
    "No.", "Message Type", "QoS Level", "QoS Level.1", "Requested QoS",
    "Epoch Time", "Protocol", "Source", "Frame length on the wire",
    "Time delta from previous displayed frame",
    "Time since reference or first frame", "Frame length on the wire.1",
    "Stream index", "iRTT", "Time since first frame in this TCP stream",
    "TCP Segment Len", "Calculated window size", "Syn", "Reset",
    "Acknowledgment", "Clean Session Flag", "Keep Alive",
    "User Name Length", "Password Length", "Retain", "Clean Session Flag.1",
    "Will Retain", "Will Flag", "Will Message Length", "Will Topic Length",
    "Topic Length", "Msg Len", "Info",
]

# Snake_case versions of the 33 shared columns
SHARED_COLUMNS = [COLUMN_RENAME[c] for c in SHARED_COLUMNS_ORIG]


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case with positional suffixes for duplicates."""
    return df.rename(columns=COLUMN_RENAME)


# === Format classification via regex ===
def classify_value(val: str) -> str:
    """Classify a single cell value into a format category."""
    if pd.isna(val) or str(val).strip() == "":
        return "empty"
    s = str(val).strip()
    # Check for comma-separated (multi-value)
    if "," in s:
        return "multi_value"
    # Boolean patterns
    if s in ("True", "False"):
        return "true_false"
    if s in ("Set", "Not set"):
        return "set_notset"
    # Numeric patterns
    if re.fullmatch(r"-?\d+", s):
        return "integer"
    if re.fullmatch(r"-?\d+\.\d+", s):
        return "float"
    # Hex pattern (e.g. 0x800f)
    if re.fullmatch(r"0x[0-9a-fA-F]+", s):
        return "hex"
    # Everything else is a string
    return "string"


def format_audit_chunk(chunk: pd.DataFrame, source: str,
                       accum: dict[str, dict[str, int]]) -> None:
    """Update format audit accumulator with one chunk using vectorized ops."""
    for col in chunk.columns:
        key = (col, source)
        if key not in accum:
            accum[key] = defaultdict(int)
        s = chunk[col].astype(str).str.strip()
        n = len(s)
        is_na = chunk[col].isna() | (s == "") | (s == "nan")
        accum[key]["empty"] += int(is_na.sum())
        valid = s[~is_na]
        if valid.empty:
            continue
        has_comma = valid.str.contains(",", na=False)
        accum[key]["multi_value"] += int(has_comma.sum())
        rest = valid[~has_comma]
        if rest.empty:
            continue
        is_tf = rest.isin(["True", "False"])
        accum[key]["true_false"] += int(is_tf.sum())
        rest = rest[~is_tf]
        is_sn = rest.isin(["Set", "Not set"])
        accum[key]["set_notset"] += int(is_sn.sum())
        rest = rest[~is_sn]
        is_int = rest.str.fullmatch(r"-?\d+", na=False)
        accum[key]["integer"] += int(is_int.sum())
        rest = rest[~is_int]
        is_flt = rest.str.fullmatch(r"-?\d+\.\d+", na=False)
        accum[key]["float"] += int(is_flt.sum())
        rest = rest[~is_flt]
        is_hex = rest.str.fullmatch(r"0x[0-9a-fA-F]+", na=False)
        accum[key]["hex"] += int(is_hex.sum())
        rest = rest[~is_hex]
        accum[key]["string"] += int(len(rest))


def presence_matrix_chunk(chunk: pd.DataFrame, class11: str,
                          null_counts: dict[str, dict[str, int]],
                          total_counts: dict[str, int]) -> None:
    """Update null counts and totals per class for presence matrix."""
    n = len(chunk)
    total_counts[class11] = total_counts.get(class11, 0) + n
    if class11 not in null_counts:
        null_counts[class11] = defaultdict(int)
    for col in chunk.columns:
        nulls = int(chunk[col].isna().sum())
        null_counts[class11][col] += nulls


def build_presence_matrix(null_counts: dict, total_counts: dict,
                          columns: list[str]) -> pd.DataFrame:
    """Build presence matrix: fraction of non-null values per column × class."""
    classes = sorted(total_counts.keys())
    data = {}
    for cls in classes:
        total = total_counts[cls]
        row = {}
        for col in columns:
            nulls = null_counts.get(cls, {}).get(col, 0)
            row[col] = 1.0 - (nulls / total) if total > 0 else 0.0
        data[cls] = row
    df = pd.DataFrame(data).T
    df.index.name = "class11"
    return df


def time_feature_percentiles_chunk(
    chunk: pd.DataFrame,
    source: str,
    protocol_col: str,
    time_cols: list[str],
    accum: dict[str, list[float]],
) -> None:
    """Collect time feature values for percentile comparison.
    
    Only collect from TCP-only rows (no MQTT) for fair comparison.
    We sample to avoid memory blowup.
    """
    # Filter TCP-only rows (not MQTT)
    mask = chunk[protocol_col].str.upper().eq("TCP") if protocol_col in chunk.columns else pd.Series([True] * len(chunk))
    tcp_rows = chunk.loc[mask]
    if tcp_rows.empty:
        return
    # Sample up to 1000 rows per chunk to keep memory bounded
    sample = tcp_rows.sample(min(1000, len(tcp_rows)), random_state=42)
    for col in time_cols:
        if col not in sample.columns:
            continue
        key = (col, source)
        if key not in accum:
            accum[key] = []
        vals = pd.to_numeric(sample[col], errors="coerce").dropna().tolist()
        accum[key].extend(vals)


def build_time_feature_audit(accum: dict) -> pd.DataFrame:
    """Build time feature audit: percentiles per (column, source)."""
    percentiles = [1, 25, 50, 75, 99]
    rows = []
    for (col, source), vals in sorted(accum.items()):
        arr = np.array(vals)
        if len(arr) == 0:
            continue
        row = {"column": col, "source": source, "n_samples": len(arr)}
        for p in percentiles:
            row[f"p{p}"] = float(np.percentile(arr, p))
        rows.append(row)
    return pd.DataFrame(rows)


# === Categorical token audit, duplicate-pair audit, header audit ===
TOKEN_AUDIT_COLS = [
    "protocol", "message_type", "qos_level_1", "qos_level_2", "requested_qos",
    "syn", "reset", "acknowledgment", "clean_session_flag_2", "retain",
    "will_retain", "will_flag",
]

# (first, second) columns that share a TShark field in the extraction script.
DUPLICATE_PAIRS = [
    ("frame_length_on_wire_1", "frame_length_on_wire_2"),
    ("clean_session_flag_1", "clean_session_flag_2"),
]


def token_audit_chunk(chunk: pd.DataFrame, source: str,
                      accum: dict[tuple[str, str], dict[str, int]]) -> None:
    """Count individual value tokens (multi-value cells are split on commas)."""
    for col in TOKEN_AUDIT_COLS:
        if col not in chunk.columns:
            continue
        s = chunk[col].dropna()
        if s.empty:
            continue
        toks = s.str.split(",").explode().str.strip()
        toks = toks[toks != ""]
        bucket = accum.setdefault((col, source), defaultdict(int))
        for tok, n in toks.value_counts().items():
            bucket[tok] += int(n)


def duplicate_pairs_chunk(chunk: pd.DataFrame, source: str,
                          accum: dict[tuple[str, str, str], dict[str, int]]) -> None:
    """Compare columns that are exported twice from the same TShark field."""
    for a, b in DUPLICATE_PAIRS:
        if a not in chunk.columns or b not in chunk.columns:
            continue
        bucket = accum.setdefault((a, b, source), defaultdict(int))
        na, nb = chunk[a].notna(), chunk[b].notna()
        both = na & nb
        bucket["rows"] += len(chunk)
        bucket["both"] += int(both.sum())
        bucket["equal"] += int((chunk.loc[both, a] == chunk.loc[both, b]).sum())
        bucket["only_first"] += int((na & ~nb).sum())
        bucket["only_second"] += int((~na & nb).sum())


def check_column_order(files_df: pd.DataFrame) -> pd.DataFrame:
    """Read only the header of each non-empty file and compare with the expected 33 columns."""
    rows = []
    for _, r in files_df.iterrows():
        if r["empty"]:
            continue
        cols = list(pd.read_csv(r["path"], nrows=0).columns)
        expected = list(SHARED_COLUMNS_ORIG)
        if r["source"] == "normal":
            expected = expected + ["Label", "Capture_ID"]
        rows.append({"path": r["path"], "source": r["source"], "n_cols": len(cols),
                     "matches_expected": cols == expected})
    return pd.DataFrame(rows)


def tokens_to_df(accum: dict) -> pd.DataFrame:
    rows = [{"column": col, "source": src, "token": tok, "count": n}
            for (col, src), d in accum.items() for tok, n in d.items()]
    return (pd.DataFrame(rows).sort_values(["column", "source", "count"],
                                           ascending=[True, True, False])
            .reset_index(drop=True))


def dups_to_df(accum: dict) -> pd.DataFrame:
    rows = [{"first": a, "second": b, "source": src, **dict(d)}
            for (a, b, src), d in accum.items()]
    return pd.DataFrame(rows)


# Default roles come from the spec (section Phase 2, step 4); evidence-based rules below may override them.
_DROP_SPEC = {
    "no": "Row number, not a feature",
    "epoch_time": "Absolute timestamp, not a feature",
    "time_since_reference_or_first_frame": "Depends on capture start",
    "stream_index": "Not a feature; kept as metadata stream_id = (group_id, stream_index)",
    "source": "Address of the sender, would identify the host",
    "info": "Free-text description, not structured",
    "label": "Target label (Normal-only column), not a feature",
    "capture_id": "Grouping metadata (Normal-only column), not a feature",
}
_NUMERIC = ["frame_length_on_wire_2", "time_delta_from_previous_displayed_frame", "irtt",
            "time_since_first_frame_in_this_tcp_stream", "tcp_segment_len",
            "calculated_window_size", "keep_alive", "user_name_length", "password_length",
            "will_message_length", "will_topic_length", "topic_length", "msg_len"]
_BINARY = ["syn", "reset", "acknowledgment", "clean_session_flag_2", "retain",
           "will_retain", "will_flag"]
_CATEGORICAL = ["protocol", "message_type", "qos_level_1", "qos_level_2", "requested_qos"]
_MULTI = ["message_type", "msg_len", "qos_level_1", "retain", "topic_length"]
_TIME_COLS = ["time_delta_from_previous_displayed_frame", "irtt",
              "time_since_first_frame_in_this_tcp_stream"]

# Wireshark display labels (Attack CSVs) -> numeric MQTT codes (Normal export). The label tables are the
# MQTT 3.1.1 message types / QoS levels; the audit (token_audit.csv) checks that the observed tokens are covered.
MSGTYPE_LABELS = {
    "Connect Command": "1", "Connect Ack": "2", "Publish Message": "3", "Publish Ack": "4",
    "Publish Received": "5", "Publish Release": "6", "Publish Complete": "7",
    "Subscribe Request": "8", "Subscribe Ack": "9", "Unsubscribe Request": "10",
    "Unsubscribe Ack": "11", "Ping Request": "12", "Ping Response": "13", "Disconnect Req": "14",
}
QOS_LABELS = {
    "At most once delivery (Fire and Forget)": "0",
    "At least once delivery (Acknowledged deliver)": "1",
    "Exactly once delivery (Assured Delivery)": "2",
    "Reserved": "3",
}
VALUE_MAPS: dict[str, dict[str, str]] = {
    "message_type": MSGTYPE_LABELS,
    "qos_level_1": QOS_LABELS,
    "qos_level_2": QOS_LABELS,
    "requested_qos": QOS_LABELS,
}
_BOOL_MAP = {"Set": "1", "True": "1", "Not set": "0", "False": "0"}


def canonical_token(col: str, tok: str) -> str | None:
    """Map one raw token to its canonical code. Returns None if it cannot be mapped unambiguously.

    Attack CSVs contain a few truncated or glued cells (e.g. "Publish Re", "Publish ReceivedP"): a token
    that starts with a full label maps to it; a truncated token maps only if it is a prefix of exactly one
    label (length >= 4); otherwise it is unmapped.
    """
    tok = str(tok).strip()
    if col in _BINARY or col in ("syn", "reset", "acknowledgment"):
        return _BOOL_MAP.get(tok)
    labels = VALUE_MAPS.get(col)
    if labels is None:
        return tok
    if tok in labels:
        return labels[tok]
    if tok.isdigit():
        return tok
    full = [k for k in labels if tok.startswith(k)]
    if full:
        return labels[max(full, key=len)]
    if len(tok) >= 4:
        cand = [k for k in labels if k.startswith(tok)]
        if len(cand) == 1:
            return labels[cand[0]]
    return None


def _canon(col: str, tok: str) -> str:
    c = canonical_token(col, tok)
    return c if c is not None else f"UNMAPPED:{tok}"


# Columns that only have a value when a parent MQTT event occurs: child -> (parent column, event token or None).
# If Normal exports the parent event, an empty child in Normal means "Normal never uses it", not an extraction gap.
PARENT_EVENT = {
    "will_message_length": ("will_flag", None),
    "will_topic_length": ("will_flag", None),
    "requested_qos": ("message_type", "8"),   # 8 = SUBSCRIBE
}


def _parent_event_in_normal(col: str, presence_df: pd.DataFrame, token_df: pd.DataFrame,
                            present_min: float, dec: dict) -> bool:
    """True (and records evidence) if the parent event of `col` is exported for Normal."""
    if col not in PARENT_EVENT:
        return False
    parent, event = PARENT_EVENT[col]
    if event is None:
        n = float(presence_df.loc["NORMAL", parent])
        ok = n >= present_min
        ev = f"parent column {parent} is present in {n:.3f} of Normal rows"
    else:
        t = token_df[(token_df["column"] == parent) & (token_df["source"] == "normal")]
        cnt = int(t.loc[t["token"].astype(str) == event, "count"].sum())
        ok = cnt > 0
        ev = f"Normal has {cnt} packets with {parent}={event}"
    if ok:
        dec[col]["genuine_absence"] = (
            f"Empty in Normal because Normal rarely/never triggers the parent event ({ev}); "
            f"not treated as an extraction artifact.")
    return ok


def apply_user_overrides(dec: dict, overrides: dict) -> None:
    """Apply human (G1) decisions. A SUSPECT flag that is overridden is kept as `suspect_reasons_resolved`."""
    for col, ov in overrides.items():
        if col not in dec:
            raise KeyError(f"user_overrides refers to unknown column {col!r}")
        d = dec[col]
        if d.get("suspect"):
            d["suspect_reasons_resolved"] = d.pop("suspect_reasons")
            d["suspect"] = False
            if "action_if_kept" in d:
                d["action"] = d.pop("action_if_kept")
        d.update(ov)


def generate_feature_decisions(
    presence_df: pd.DataFrame,
    format_df: pd.DataFrame,
    time_df: pd.DataFrame,
    token_df: pd.DataFrame,
    dup_df: pd.DataFrame,
    hcfg: dict[str, Any],
) -> dict[str, Any]:
    """Build feature_decisions content. Every flag below is derived from the audit outputs."""
    diff_thr = hcfg.get("suspect_presence_diff", 0.95)
    absent_max = hcfg.get("absent_presence_max", 0.001)
    present_min = hcfg.get("present_presence_min", 0.01)
    time_ratio_thr = hcfg.get("time_ratio_suspect", 10.0)
    token_share = hcfg.get("token_min_share", 0.001)
    token_min_count = hcfg.get("token_min_count", 100)

    attack_classes = [c for c in presence_df.index if c != "NORMAL"]
    dec: dict[str, Any] = {}

    def put(col, action, reason, **extra):
        dec[col] = {"action": action, "reason": reason, **extra}

    def suspect(col, why):
        d = dec[col]
        d.setdefault("suspect_reasons", []).append(why)
        d["suspect"] = True
        if d["action"] != "drop":
            d["action_if_kept"] = d["action"]
            d["action"] = "drop"

    for c, r in _DROP_SPEC.items():
        put(c, "drop", r)
    for c in _NUMERIC:
        put(c, "numeric", c.replace("_", " "), multi=c in _MULTI)
    for c in _BINARY:
        put(c, "binary", c.replace("_", " "), multi=c in _MULTI)
    for c in _CATEGORICAL:
        put(c, "categorical", c.replace("_", " "), multi=c in _MULTI)
    put("frame_length_on_wire_1", "numeric", "frame length (first copy)")
    put("clean_session_flag_1", "binary", "clean session flag (first copy)")

    # Rule: empty in every class -> drop (spec: requested_qos if empty > 99.9% in every class).
    for col in presence_df.columns:
        if col in dec and dec[col]["action"] != "drop" and presence_df[col].max() <= absent_max:
            dec[col].update(action="drop", reason=f"Empty in >= {1 - absent_max:.1%} of rows in every class")
    if dec["requested_qos"]["action"] != "drop":
        dec["requested_qos"]["note"] = (
            "Spec auto-drop rule (empty >99.9% in every class) does not hold: "
            f"max presence {presence_df['requested_qos'].max():.3f}")

    # Rule: duplicate export of one TShark field -> drop the first copy when the data proves redundancy.
    for _, r in dup_df[dup_df["source"] == "attack"].iterrows():
        if r["both"] > 0 and r["equal"] == r["both"] and r["only_first"] == 0:
            dec[r["first"]].update(
                action="drop", duplicate_of=r["second"],
                reason=(f"Duplicate of {r['second']}: equal in {int(r['equal'])}/{int(r['both'])} Attack rows "
                        f"where both are present, never present without it; empty in Normal."))

    evaluated = [c for c in presence_df.columns
                 if c in dec and c not in _DROP_SPEC and not dec[c].get("duplicate_of")
                 and presence_df[c].max() > absent_max]

    # Rule: column absent in one source but present in the other.
    for col in evaluated:
        normal = presence_df.loc["NORMAL", col]
        atk = presence_df.loc[attack_classes, col]
        if normal <= absent_max and atk.max() >= present_min and _parent_event_in_normal(
                col, presence_df, token_df, present_min, dec):
            pass  # genuine: Normal exports the parent event but never exercises this function
        elif normal <= absent_max and atk.max() >= present_min:
            where = [k for k, v in atk.items() if v >= present_min]
            suspect(col, f"Empty in NORMAL ({normal:.4f}) but present in Attack classes {where} "
                         f"(max {atk.max():.3f}).")
        elif atk.max() <= absent_max and normal >= present_min:
            suspect(col, f"Present in NORMAL ({normal:.3f}) but empty in all Attack classes.")

    # Rule (spec): presence differs by more than diff_thr between NORMAL and any Attack class.
    for col in evaluated:
        normal = presence_df.loc["NORMAL", col]
        for k in attack_classes:
            d = abs(normal - presence_df.loc[k, col])
            if d > diff_thr:
                suspect(col, f"Presence differs by {d:.3f} between NORMAL ({normal:.3f}) and {k} "
                             f"({presence_df.loc[k, col]:.3f}).")
                break

    # Rule: Normal has no multi-value cells while Attack has (TShark occurrence=f in the Normal export).
    fm = format_df.set_index(["column", "source"])
    has_multi_artifact = False
    for col in evaluated:
        if (col, "normal") in fm.index and (col, "attack") in fm.index:
            n_m = fm.loc[(col, "normal"), "multi_value_pct"]
            a_m = fm.loc[(col, "attack"), "multi_value_pct"]
            if n_m == 0 and a_m > token_share:
                dec[col]["multi_artifact"] = (
                    f"multi-value cells: Normal {n_m:.4%} vs Attack {a_m:.4%}. The Normal export keeps the "
                    f"first occurrence only, so counts/sums over multi-value cells are not comparable.")
                dec[col]["multi_policy"] = "first_only"
                has_multi_artifact = True
    if has_multi_artifact:
        dec["n_mqtt_msgs"] = {
            "action": "drop", "suspect": True, "derived_from": "message_type",
            "reason": "Number of MQTT messages per packet (spec: first+count).",
            "suspect_reasons": ["Multi-value cells exist only in Attack, so the count is <= 1 in Normal by construction."],
        }

    # Rule: time-feature distributions differ strongly between sources on TCP-only packets.
    for col in _TIME_COLS:
        n = time_df[(time_df["column"] == col) & (time_df["source"] == "normal")]
        a = time_df[(time_df["column"] == col) & (time_df["source"] == "attack")]
        if n.empty or a.empty:
            continue
        ratios = {}
        for p in ("p25", "p50", "p75", "p99"):
            x, y = float(n.iloc[0][p]), float(a.iloc[0][p])
            if x > 0 and y > 0:
                ratios[p] = max(x, y) / min(x, y)
        dec[col]["time_ratios"] = {k: round(v, 2) for k, v in ratios.items()}
        if ratios and max(ratios.values()) > time_ratio_thr:
            worst = max(ratios, key=ratios.get)
            suspect(col, f"Normal vs Attack on TCP-only packets differ x{ratios[worst]:.1f} at {worst} "
                         f"(Normal {float(n.iloc[0][worst]):.6g}, Attack {float(a.iloc[0][worst]):.6g}).")

    # Rule: categorical/binary token sets differ between sources after canonicalisation.
    if not token_df.empty:
        tk = token_df.copy()
        tk["token"] = [_canon(c, str(t)) for c, t in zip(tk["column"], tk["token"])]
        unm = tk[tk["token"].str.startswith("UNMAPPED:")]
        for col, g in unm.groupby("column"):
            if col in dec:
                tot = int(token_df.loc[token_df["column"] == col, "count"].sum())
                dec[col]["unmapped_tokens"] = {
                    "count": int(g["count"].sum()), "share": round(float(g["count"].sum()) / tot, 8),
                    "examples": [t[len("UNMAPPED:"):] for t in g.sort_values("count", ascending=False)["token"].head(5)],
                }
        tk = tk[~tk["token"].str.startswith("UNMAPPED:")]
        tk = tk.groupby(["column", "source", "token"], as_index=False)["count"].sum()
        for col in tk["column"].unique():
            if col not in dec or col not in evaluated:
                continue
            t = tk[tk["column"] == col]
            totals = t.groupby("source")["count"].sum()
            piv = t.pivot_table(index="token", columns="source", values="count", fill_value=0)
            for src in ("normal", "attack"):
                if src not in piv.columns:
                    piv[src] = 0
            tn, ta = max(totals.get("normal", 1), 1), max(totals.get("attack", 1), 1)
            only_n = piv[(piv["normal"] > 0) & (piv["attack"] == 0)]
            only_a = piv[(piv["attack"] > 0) & (piv["normal"] == 0)]
            if len(only_n) or len(only_a):
                dec[col]["category_mismatch"] = {
                    "normal_only": {str(k): int(v) for k, v in only_n["normal"].items()},
                    "attack_only": {str(k): int(v) for k, v in only_a["attack"].items()},
                }
            # Only Normal-only values are flagged: Attack-only values (e.g. will_flag=1 in WILL attacks) are
            # attack behaviour, whereas protocols never seen in Attack (STP, LOOP, ...) are capture-environment
            # noise that would identify Normal rows.
            n_share = float(only_n["normal"].sum()) / tn
            if n_share >= token_share and float(only_n["normal"].sum()) >= token_min_count:
                suspect(col, f"{n_share:.3%} of Normal tokens have values never seen in Attack: "
                             f"{ {str(k): int(v) for k, v in only_n['normal'].items()} }.")

    # Representation differences resolved by the parser (listed for G1, not SUSPECT by themselves).
    for col in evaluated:
        if (col, "normal") in fm.index and (col, "attack") in fm.index:
            fn, fa = fm.loc[(col, "normal")], fm.loc[(col, "attack")]
            if (fn["true_false"] > 0 and fa["set_notset"] > 0) or (fn["set_notset"] > 0 and fa["true_false"] > 0):
                dec[col]["format_note"] = "Normal uses True/False, Attack uses Set/Not set; parser maps both to 0/1."
        if col in VALUE_MAPS:
            dec[col]["value_map"] = dict(VALUE_MAPS[col])
            dec[col]["format_note"] = "Normal uses numeric codes, Attack uses text; parser maps text to code."
    apply_user_overrides(dec, hcfg.get("user_overrides") or {})
    if hcfg.get("row_filters"):
        dec["_row_filters"] = {"normal_only": hcfg["row_filters"],
                               "note": "Applied in Phase 3 sampling to Normal rows only."}
    if hcfg.get("g1_log"):
        dec["_g1"] = hcfg["g1_log"]
    return dec


STAT_OUTPUTS = {
    "presence": "presence_matrix.csv",
    "format": "format_audit.csv",
    "time": "time_feature_audit.csv",
    "tokens": "token_audit.csv",
    "dups": "duplicate_pairs.csv",
    "header": "header_check.csv",
}


def compute_stats(cfg: dict[str, Any], stages: set[str]) -> None:
    """Stream the raw CSVs once, computing only the requested stages. Raw data is read-only."""
    work_dir = Path(cfg["paths"]["work_dir"])
    out_dir = work_dir / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    files_df = pd.read_csv(out_dir / "files.csv")

    if "header" in stages:
        hdr = check_column_order(files_df)
        hdr.to_csv(out_dir / STAT_OUTPUTS["header"], index=False)
        bad = hdr[~hdr["matches_expected"]]
        logger.info("Header check: %d files, %d mismatched", len(hdr), len(bad))
        if len(bad):
            raise RuntimeError(f"{len(bad)} files have unexpected column order, e.g. {bad.iloc[0]['path']}")

    stream_stages = stages & {"presence", "format", "time", "tokens", "dups"}
    if not stream_stages:
        return

    null_counts: dict[str, dict[str, int]] = {}
    total_counts: dict[str, int] = {}
    format_accum: dict = {}
    time_accum: dict = {}
    token_accum: dict = {}
    dup_accum: dict = {}
    time_cols = ["time_delta_from_previous_displayed_frame", "irtt",
                 "time_since_first_frame_in_this_tcp_stream"]
    # Only read the columns a stage needs when presence/format are not requested.
    usecols = None
    if not (stream_stages & {"presence", "format", "time"}):
        needed = set(TOKEN_AUDIT_COLS) | {c for p in DUPLICATE_PAIRS for c in p}
        needed_orig = [o for o, sn in COLUMN_RENAME.items() if sn in needed]
        usecols = needed_orig

    n_files = len(files_df)
    for idx, row in files_df.iterrows():
        fpath = Path(row["path"])
        if row["empty"]:
            continue
        if idx % 20 == 0:
            logger.info("Processing file %d/%d: %s", idx + 1, n_files, fpath.name)
        for chunk in pd.read_csv(fpath, chunksize=_CHUNK, dtype=str, low_memory=False,
                                 usecols=usecols):
            chunk = rename_columns(chunk)
            if "presence" in stream_stages:
                presence_matrix_chunk(chunk, row["class11"], null_counts, total_counts)
            if "format" in stream_stages:
                audit_cols = [c for c in SHARED_COLUMNS if c in chunk.columns]
                format_audit_chunk(chunk[audit_cols], row["source"], format_accum)
            if "time" in stream_stages:
                time_feature_percentiles_chunk(chunk, row["source"], "protocol", time_cols, time_accum)
            if "tokens" in stream_stages:
                token_audit_chunk(chunk, row["source"], token_accum)
            if "dups" in stream_stages:
                duplicate_pairs_chunk(chunk, row["source"], dup_accum)

    if "presence" in stream_stages:
        build_presence_matrix(null_counts, total_counts, SHARED_COLUMNS).to_csv(
            out_dir / STAT_OUTPUTS["presence"])
    if "format" in stream_stages:
        fmt_rows = []
        format_types = ["empty", "integer", "float", "hex", "multi_value",
                        "true_false", "set_notset", "string"]
        for (col, source), counts in sorted(format_accum.items()):
            total = sum(counts.values())
            r = {"column": col, "source": source, "total_sampled": total}
            for ft in format_types:
                r[ft] = counts.get(ft, 0)
                r[f"{ft}_pct"] = counts.get(ft, 0) / total if total > 0 else 0
            fmt_rows.append(r)
        pd.DataFrame(fmt_rows).to_csv(out_dir / STAT_OUTPUTS["format"], index=False)
    if "time" in stream_stages:
        build_time_feature_audit(time_accum).to_csv(out_dir / STAT_OUTPUTS["time"], index=False)
    if "tokens" in stream_stages:
        tokens_to_df(token_accum).to_csv(out_dir / STAT_OUTPUTS["tokens"], index=False)
    if "dups" in stream_stages:
        dups_to_df(dup_accum).to_csv(out_dir / STAT_OUTPUTS["dups"], index=False)


def plot_presence_heatmap(presence_df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = presence_df.T
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(data.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=60, ha="right")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index, fontsize=8)
    ax.set_title("Presence matrix: fraction of non-empty cells (column x sub-class)")
    fig.colorbar(im, ax=ax, label="non-empty fraction")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_g1_report(decisions: dict, presence_df: pd.DataFrame, cfg: dict[str, Any], path: Path) -> None:
    """Human-readable G1 summary. Every number is read from the audit outputs / decisions."""
    inv = Path(cfg["paths"]["work_dir"]) / "inventory"
    hdr = pd.read_csv(inv / STAT_OUTPUTS["header"])
    dups = pd.read_csv(inv / STAT_OUTPUTS["dups"])
    L = ["# G1 - Feature review (auto-generated by `ppfeddata harmonize`)", ""]
    L.append(f"- Header check: {len(hdr)} non-empty files, {int((~hdr['matches_expected']).sum())} with unexpected column order "
             f"(Attack: 33 columns; Normal: 33 + Label + Capture_ID).")
    L.append(f"- Presence matrix: {presence_df.shape[1]} columns x {presence_df.shape[0]} sub-classes "
             f"(`data/inventory/presence_matrix.csv`, heatmap `results/figures/presence_heatmap.png`).")
    L.append("")
    L.append("## Duplicate-export columns")
    L.append("| first | second | source | both present | equal | only first | only second |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in dups.iterrows():
        L.append(f"| {r['first']} | {r['second']} | {r['source']} | {int(r['both'])} | {int(r['equal'])} | "
                 f"{int(r['only_first'])} | {int(r['only_second'])} |")
    L.append("")
    L.append("## SUSPECT columns (default action = drop; user decides)")
    for col, d in decisions.items():
        if not col.startswith("_") and d.get("suspect"):
            L.append(f"- **{col}** (would be `{d.get('action_if_kept', 'drop')}` if kept)")
            for r in d["suspect_reasons"]:
                L.append(f"  - {r}")
    if not any(d.get("suspect") for d in decisions.values() if isinstance(d, dict)):
        L.append("- none open")
    L.append("")
    L.append("## Resolved by G1 decisions")
    for col, d in decisions.items():
        if col.startswith("_"):
            continue
        if "g1_decision" in d or "genuine_absence" in d:
            L.append(f"- **{col}** -> `{d['action']}`" + (" (diagnostic column)" if d.get("diagnostic") else ""))
            for why in d.get("suspect_reasons_resolved", []):
                L.append(f"  - flag: {why}")
            if "genuine_absence" in d:
                L.append(f"  - {d['genuine_absence']}")
            if "g1_decision" in d:
                L.append(f"  - decision: {d['g1_decision']}")
    if "_row_filters" in decisions:
        L.append(f"- Normal row filter for Phase 3: {decisions['_row_filters']['normal_only']}")
    L.append("")
    L.append("## Not SUSPECT but needs the parser (Phase 4) to handle")
    for col, d in decisions.items():
        if col.startswith("_"):
            continue
        notes = [d[k] for k in ("format_note", "multi_artifact") if k in d]
        if d.get("unmapped_tokens"):
            u = d["unmapped_tokens"]
            notes.append(f"{u['count']} tokens ({u['share']:.2e} of all) cannot be mapped, e.g. {u['examples']}")
        if notes and not d.get("suspect"):
            L.append(f"- **{col}** (`{d['action']}`"
                     + (f", multi_policy={d['multi_policy']}" if "multi_policy" in d else "") + ")")
            for n in notes:
                L.append(f"  - {n}")
    L.append("")
    L.append("## Time-feature ratios Normal vs Attack (TCP-only packets, max/min of percentiles)")
    for col, d in decisions.items():
        if not col.startswith("_") and "time_ratios" in d:
            L.append(f"- {col}: {d['time_ratios']}")
    L.append("")
    L.append("Thresholds used are heuristics in `configs/default.yaml -> harmonize` (spec mark: needs verification).")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def run_harmonize(cfg: dict[str, Any], recompute: bool = False,
                  decisions_only: bool = False) -> dict[str, Any]:
    """Phase 2 pipeline. Stats stages are cached on disk; pass recompute=True to redo them."""
    work_dir = Path(cfg["paths"]["work_dir"])
    out_dir = work_dir / "inventory"
    if not decisions_only:
        stages = set(STAT_OUTPUTS) if recompute else {
            k for k, f in STAT_OUTPUTS.items() if not (out_dir / f).exists()}
        if stages:
            logger.info("Computing stages: %s", sorted(stages))
            compute_stats(cfg, stages)
        else:
            logger.info("All stat outputs cached; use --recompute to redo.")

    presence_df = pd.read_csv(out_dir / STAT_OUTPUTS["presence"], index_col=0)
    format_df = pd.read_csv(out_dir / STAT_OUTPUTS["format"])
    time_df = pd.read_csv(out_dir / STAT_OUTPUTS["time"])
    token_df = pd.read_csv(out_dir / STAT_OUTPUTS["tokens"], keep_default_na=False)
    dup_df = pd.read_csv(out_dir / STAT_OUTPUTS["dups"])

    hcfg = cfg.get("harmonize", {})
    plot_presence_heatmap(presence_df, Path(hcfg.get("figures_dir", "./results/figures"))
                          / "presence_heatmap.png")

    decisions = generate_feature_decisions(presence_df, format_df, time_df, token_df, dup_df, hcfg)
    decisions_path = Path(hcfg.get("decisions_path", "./configs/feature_decisions.yaml"))
    with open(decisions_path, "w", encoding="utf-8") as f:
        yaml.dump(decisions, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info("Saved %s", decisions_path)

    write_g1_report(decisions, presence_df, cfg,
                    Path(hcfg.get("reports_dir", "./results/reports")) / "g1_feature_review.md")
    suspects = [(c, " | ".join(i["suspect_reasons"])) for c, i in decisions.items()
                if isinstance(i, dict) and i.get("suspect")]
    return {"presence_df": presence_df, "decisions": decisions, "suspects": suspects}
