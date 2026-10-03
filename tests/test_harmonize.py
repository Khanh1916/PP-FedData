"""Tests for harmonize module."""
import pandas as pd
import numpy as np
import pytest

from ppfeddata.data.harmonize import (
    COLUMN_RENAME,
    SHARED_COLUMNS,
    SHARED_COLUMNS_ORIG,
    classify_value,
    rename_columns,
)


class TestColumnMapping:
    """Tests for column name mapping."""

    def test_33_shared_columns(self):
        """Should have exactly 33 shared columns."""
        assert len(SHARED_COLUMNS) == 33
        assert len(SHARED_COLUMNS_ORIG) == 33

    def test_rename_produces_snake_case(self):
        """All renamed columns should be snake_case."""
        import re
        for orig, snake in COLUMN_RENAME.items():
            assert re.fullmatch(r"[a-z][a-z0-9_]*", snake), (
                f"{orig!r} -> {snake!r} is not snake_case"
            )

    def test_dot1_maps_to_2(self):
        """Columns with .1 suffix should map to _2."""
        assert COLUMN_RENAME["QoS Level.1"] == "qos_level_2"
        assert COLUMN_RENAME["Frame length on the wire.1"] == "frame_length_on_wire_2"
        assert COLUMN_RENAME["Clean Session Flag.1"] == "clean_session_flag_2"

    def test_first_occurrence_maps_to_1(self):
        """First occurrence of duplicate columns should map to _1."""
        assert COLUMN_RENAME["QoS Level"] == "qos_level_1"
        assert COLUMN_RENAME["Frame length on the wire"] == "frame_length_on_wire_1"
        assert COLUMN_RENAME["Clean Session Flag"] == "clean_session_flag_1"

    def test_rename_df(self):
        """rename_columns should work on a DataFrame."""
        df = pd.DataFrame({
            "No.": [1], "Message Type": ["x"],
            "QoS Level": [0], "QoS Level.1": [1],
        })
        renamed = rename_columns(df)
        assert "no" in renamed.columns
        assert "message_type" in renamed.columns
        assert "qos_level_1" in renamed.columns
        assert "qos_level_2" in renamed.columns


class TestClassifyValue:
    """Tests for regex-based format classification."""

    def test_empty(self):
        assert classify_value("") == "empty"
        assert classify_value(None) == "empty"
        assert classify_value(float("nan")) == "empty"
        assert classify_value("  ") == "empty"

    def test_integer(self):
        assert classify_value("42") == "integer"
        assert classify_value("0") == "integer"
        assert classify_value("-1") == "integer"

    def test_float(self):
        assert classify_value("3.14") == "float"
        assert classify_value("0.000000000") == "float"
        assert classify_value("-0.5") == "float"

    def test_hex(self):
        assert classify_value("0x800f") == "hex"
        assert classify_value("0xFF") == "hex"

    def test_true_false(self):
        assert classify_value("True") == "true_false"
        assert classify_value("False") == "true_false"

    def test_set_notset(self):
        assert classify_value("Set") == "set_notset"
        assert classify_value("Not set") == "set_notset"

    def test_multi_value(self):
        assert classify_value("Publish Message,Publish Message") == "multi_value"
        assert classify_value("40,41") == "multi_value"
        assert classify_value("Not set,Not set") == "multi_value"

    def test_string(self):
        assert classify_value("Connect Command") == "string"
        assert classify_value("TCP") == "string"
        assert classify_value("192.168.1.1") == "string"


class TestTimeColumnsNonNegative:
    """Time columns should not have negative values in well-formed data."""

    def test_non_negative_times(self):
        """Time delta values parsed as float should be >= 0."""
        vals = ["0.000000000", "2.001520000", "0.352245000", "0.002479000"]
        for v in vals:
            assert float(v) >= 0


class TestPresenceMatrix:
    """Test presence matrix computation."""

    def test_build_presence_matrix(self):
        from ppfeddata.data.harmonize import build_presence_matrix
        null_counts = {
            "A": {"col1": 20, "col2": 0},
            "B": {"col1": 0, "col2": 50},
        }
        total_counts = {"A": 100, "B": 100}
        df = build_presence_matrix(null_counts, total_counts, ["col1", "col2"])
        assert df.loc["A", "col1"] == pytest.approx(0.8)
        assert df.loc["A", "col2"] == pytest.approx(1.0)
        assert df.loc["B", "col1"] == pytest.approx(1.0)
        assert df.loc["B", "col2"] == pytest.approx(0.5)


class TestCanonicalToken:
    """Text labels (Attack) and numeric codes (Normal) must map to the same canonical value."""

    def test_message_type_text_and_code(self):
        from ppfeddata.data.harmonize import canonical_token
        assert canonical_token("message_type", "Publish Message") == "3"
        assert canonical_token("message_type", "3") == "3"
        assert canonical_token("message_type", "Connect Command") == "1"

    def test_truncated_and_glued_cells(self):
        from ppfeddata.data.harmonize import canonical_token
        assert canonical_token("message_type", "Publish ReceivedP") == "5"   # glued suffix
        assert canonical_token("qos_level_1", "Exactly o") == "2"           # unique prefix
        assert canonical_token("message_type", "Publish Re") is None        # ambiguous prefix
        assert canonical_token("message_type", "P") is None                 # too short

    def test_qos_labels(self):
        from ppfeddata.data.harmonize import canonical_token
        assert canonical_token("qos_level_2", "At most once delivery (Fire and Forget)") == "0"
        assert canonical_token("qos_level_1", "2") == "2"

    def test_binary_forms_agree(self):
        from ppfeddata.data.harmonize import canonical_token
        assert canonical_token("syn", "Set") == canonical_token("syn", "True") == "1"
        assert canonical_token("syn", "Not set") == canonical_token("syn", "False") == "0"


class TestAuditChunks:
    def test_token_audit_splits_multi_values(self):
        from collections import defaultdict
        from ppfeddata.data.harmonize import token_audit_chunk
        chunk = pd.DataFrame({"message_type": ["Publish Message,Publish Message", "40", None, ""]})
        acc: dict = {}
        token_audit_chunk(chunk, "attack", acc)
        d = dict(acc[("message_type", "attack")])
        assert d == {"Publish Message": 2, "40": 1}

    def test_duplicate_pairs(self):
        from ppfeddata.data.harmonize import duplicate_pairs_chunk
        chunk = pd.DataFrame({
            "frame_length_on_wire_1": ["60", "70", None, "80"],
            "frame_length_on_wire_2": ["60", "70", "90", None],
        })
        acc: dict = {}
        duplicate_pairs_chunk(chunk, "attack", acc)
        d = acc[("frame_length_on_wire_1", "frame_length_on_wire_2", "attack")]
        assert (d["both"], d["equal"], d["only_first"], d["only_second"]) == (2, 2, 1, 1)

    def test_header_check_detects_mismatch(self, tmp_path):
        from ppfeddata.data.harmonize import SHARED_COLUMNS_ORIG, check_column_order
        good, bad = tmp_path / "a.csv", tmp_path / "b.csv"
        pd.DataFrame(columns=SHARED_COLUMNS_ORIG).to_csv(good, index=False)
        pd.DataFrame(columns=list(reversed(SHARED_COLUMNS_ORIG))).to_csv(bad, index=False)
        files = pd.DataFrame({"path": [str(good), str(bad)], "source": ["attack", "attack"],
                              "empty": [False, False]})
        res = check_column_order(files)
        assert res["matches_expected"].tolist() == [True, False]


def _synthetic_inputs():
    """Tiny presence/format/time/token/dup tables exercising every decision rule."""
    cols = ["syn", "frame_length_on_wire_1", "frame_length_on_wire_2", "requested_qos", "protocol",
            "message_type", "time_since_first_frame_in_this_tcp_stream", "irtt"]
    pres = pd.DataFrame(
        {
            "syn": [1.0, 1.0], "frame_length_on_wire_1": [0.0, 1.0], "frame_length_on_wire_2": [1.0, 1.0],
            "requested_qos": [0.0, 0.2], "protocol": [1.0, 1.0], "message_type": [0.4, 0.4],
            "time_since_first_frame_in_this_tcp_stream": [1.0, 1.0], "irtt": [0.6, 0.6],
        },
        index=["NORMAL", "A_DoS"],
    )
    fmt = pd.DataFrame([
        {"column": "message_type", "source": "normal", "multi_value_pct": 0.0, "true_false_pct": 0, "set_notset_pct": 0,
         "true_false": 0, "set_notset": 0},
        {"column": "message_type", "source": "attack", "multi_value_pct": 0.05, "true_false_pct": 0, "set_notset_pct": 0,
         "true_false": 0, "set_notset": 0},
        {"column": "syn", "source": "normal", "multi_value_pct": 0.0, "true_false": 5, "set_notset": 0},
        {"column": "syn", "source": "attack", "multi_value_pct": 0.0, "true_false": 0, "set_notset": 5},
    ])
    time = pd.DataFrame([
        {"column": c, "source": s, "n_samples": 10, "p1": 0, "p25": a, "p50": a, "p75": a, "p99": a}
        for c, vals in {"time_since_first_frame_in_this_tcp_stream": (1.0, 500.0), "irtt": (1.0, 2.0)}.items()
        for s, a in zip(("normal", "attack"), vals)
    ])
    tokens = pd.DataFrame([
        {"column": "protocol", "source": "normal", "token": "TCP", "count": 9000},
        {"column": "protocol", "source": "normal", "token": "STP", "count": 1000},
        {"column": "protocol", "source": "attack", "token": "TCP", "count": 10000},
        {"column": "message_type", "source": "normal", "token": "3", "count": 10},
        {"column": "message_type", "source": "attack", "token": "Publish Message", "count": 10},
    ])
    dups = pd.DataFrame([
        {"first": "frame_length_on_wire_1", "second": "frame_length_on_wire_2", "source": "attack",
         "rows": 10, "both": 10, "equal": 10, "only_first": 0, "only_second": 0},
    ])
    return pres, fmt, time, tokens, dups


class TestFeatureDecisions:
    def _run(self):
        from ppfeddata.data.harmonize import generate_feature_decisions
        pres, fmt, time, tokens, dups = _synthetic_inputs()
        return generate_feature_decisions(pres, fmt, time, tokens, dups, {})

    def test_duplicate_first_copy_dropped_with_evidence(self):
        d = self._run()
        assert d["frame_length_on_wire_1"]["action"] == "drop"
        assert d["frame_length_on_wire_1"]["duplicate_of"] == "frame_length_on_wire_2"
        assert d["frame_length_on_wire_2"]["action"] == "numeric"

    def test_requested_qos_not_auto_dropped_as_empty(self):
        d = self._run()
        assert "does not hold" in d["requested_qos"]["note"]
        assert d["requested_qos"]["suspect"] is True  # absent in Normal, present in Attack

    def test_time_feature_ratio_flag(self):
        d = self._run()
        assert d["time_since_first_frame_in_this_tcp_stream"]["suspect"] is True
        assert not d["irtt"].get("suspect")

    def test_normal_only_category_flagged_attack_text_code_not(self):
        d = self._run()
        assert d["protocol"]["suspect"] is True
        assert "STP" in d["protocol"]["category_mismatch"]["normal_only"]
        # text label in Attack vs numeric code in Normal agree after canonicalisation
        assert not d["message_type"].get("suspect")
        assert "category_mismatch" not in d["message_type"]

    def test_multi_artifact_policy_and_derived_count_dropped(self):
        d = self._run()
        assert d["message_type"]["multi_policy"] == "first_only"
        assert d["n_mqtt_msgs"]["action"] == "drop"

    def test_binary_format_note(self):
        assert "True/False" in self._run()["syn"]["format_note"]


class TestG1Overrides:
    def test_parent_event_makes_absence_genuine(self):
        from ppfeddata.data.harmonize import generate_feature_decisions
        pres, fmt, time, tokens, dups = _synthetic_inputs()
        pres["will_flag"] = [0.075, 0.1]
        pres["will_message_length"] = [0.0, 0.11]
        d = generate_feature_decisions(pres, fmt, time, tokens, dups, {})
        assert not d["will_message_length"].get("suspect")
        assert "genuine_absence" in d["will_message_length"]
        assert d["will_message_length"]["action"] == "numeric"

    def test_no_parent_event_still_flagged(self):
        d = TestFeatureDecisions()._run()
        assert d["requested_qos"]["suspect"] is True   # no Subscribe tokens in the synthetic Normal data

    def test_tiny_normal_only_counts_not_flagged(self):
        from ppfeddata.data.harmonize import generate_feature_decisions
        pres, fmt, time, tokens, dups = _synthetic_inputs()
        d = generate_feature_decisions(pres, fmt, time, tokens, dups, {"token_min_count": 5000})
        assert not d["protocol"].get("suspect")        # STP count 1000 < 5000

    def test_override_resolves_suspect_and_keeps_reasons(self):
        from ppfeddata.data.harmonize import generate_feature_decisions
        pres, fmt, time, tokens, dups = _synthetic_inputs()
        cfg = {"user_overrides": {"protocol": {"action": "categorical", "g1_decision": "keep"},
                                  "time_since_first_frame_in_this_tcp_stream":
                                      {"action": "drop", "diagnostic": True}},
               "row_filters": {"protocol": ["TCP", "MQTT"]}}
        d = generate_feature_decisions(pres, fmt, time, tokens, dups, cfg)
        assert d["protocol"]["action"] == "categorical" and d["protocol"]["suspect"] is False
        assert d["protocol"]["suspect_reasons_resolved"]
        assert d["time_since_first_frame_in_this_tcp_stream"]["diagnostic"] is True
        assert d["_row_filters"]["all_sources"] == {"protocol": ["TCP", "MQTT"]}

    def test_unknown_override_column_raises(self):
        from ppfeddata.data.harmonize import apply_user_overrides
        with pytest.raises(KeyError):
            apply_user_overrides({"a": {"action": "drop"}}, {"zzz": {"action": "drop"}})
