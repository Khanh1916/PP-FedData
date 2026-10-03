"""Tests for Phase 4: parse_multi and Preprocessor."""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from ppfeddata.data.harmonize import SHARED_COLUMNS  # noqa: E402
from ppfeddata.data.parse_multi import count_values, first_values, parse_multi  # noqa: E402
from ppfeddata.data.preprocess import (  # noqa: E402
    Preprocessor, label_map, load_decisions, processed_dir, run_preprocess,
)
from ppfeddata.utils import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = load_decisions(ROOT / "configs" / "feature_decisions.yaml")
PCFG = {"categorical_top_k": 10, "log1p_skew_threshold": 2.0, "clip_sigma": 5}

MSG_TEXT = {"1": "Connect Command", "3": "Publish Message", "4": "Publish Ack"}


def make_df(n=2000, seed=0, shift=0.0):
    """Raw-looking string table with every harmonised column; `shift` moves the numeric distributions."""
    rng = np.random.default_rng(seed)
    d = {c: [None] * n for c in SHARED_COLUMNS}
    d["frame_length_on_wire_2"] = rng.integers(54, 1500, n).astype(str)
    d["time_delta_from_previous_displayed_frame"] = [f"{x:.9f}" for x in rng.exponential(0.002 + shift, n)]
    d["time_delta_from_previous_displayed_frame"][0] = "-0.000001"              # negative as in the real data
    d["tcp_segment_len"] = rng.choice(["0", "0", "0", "34", "1460"], n)
    d["calculated_window_size"] = (rng.lognormal(10 + shift, 2, n)).astype(int).astype(str)
    d["time_since_first_frame_in_this_tcp_stream"] = [f"{x:.6f}" for x in rng.exponential(1.0, n)]
    d["irtt"] = [None if rng.random() < 0.4 else f"{x:.6f}" for x in rng.exponential(0.003, n)]
    d["keep_alive"] = [None if rng.random() < 0.9 else str(rng.choice([60, 3600])) for _ in range(n)]
    d["password_length"] = [None if rng.random() < 0.5 else "8" for _ in range(n)]       # constant when present
    d["user_name_length"] = [None if rng.random() < 0.0005 else "7" for _ in range(n)]    # NA rate 0.05%
    d["will_topic_length"] = [None if i % 1000 else "15" for i in range(n)]                # NA rate 99.9%
    d["topic_length"] = [None if rng.random() < 0.8 else rng.choice(["23", "23,23", "40,41"]) for _ in range(n)]
    d["msg_len"] = d["topic_length"]
    d["will_message_length"] = [None] * n
    d["syn"] = rng.choice(["Set", "Not set", "True", "False"], n)
    d["reset"] = rng.choice(["Not set", "False"], n)
    d["acknowledgment"] = rng.choice(["Set", "True"], n)
    d["clean_session_flag_2"] = [None if rng.random() < 0.85 else rng.choice(["Set", "True", "Not set"]) for _ in range(n)]
    d["retain"] = [None if rng.random() < 0.9 else rng.choice(["Not set", "False", "Not set,Not set"]) for _ in range(n)]
    d["will_retain"] = d["will_flag"] = [None] * n
    d["protocol"] = rng.choice(["TCP", "MQTT", "MQTT", "STP"], n, p=[0.5, 0.3, 0.19, 0.01])
    mt = rng.choice(["1", "3", "4"], n)
    d["message_type"] = [None if rng.random() < 0.5 else (MSG_TEXT[m] if rng.random() < 0.5 else m) for m in mt]
    d["qos_level_1"] = [None if rng.random() < 0.9 else rng.choice(["0", "1", "2", "Exactly once delivery (Assured Delivery)"]) for _ in range(n)]
    d["qos_level_2"] = [None if rng.random() < 0.9 else "At most once delivery (Fire and Forget)" for _ in range(n)]
    d["requested_qos"] = [None] * n
    return pd.DataFrame(d)


@pytest.fixture(scope="module")
def train_val():
    return make_df(3000, 0), make_df(1500, 1, shift=0.5)


@pytest.fixture(scope="module")
def pre(train_val):
    return Preprocessor(DECISIONS, PCFG).fit(train_val[0])


class TestParseMulti:
    def test_examples(self):
        assert parse_multi("Publish Message,Publish Message") == ["Publish Message", "Publish Message"]
        assert parse_multi("40,41") == ["40", "41"]
        assert parse_multi("  7 ") == ["7"]
        assert parse_multi("1,2,3") == ["1", "2", "3"]

    @pytest.mark.parametrize("cell", ["", "   ", None, float("nan"), "nan", ",", pd.NA])
    def test_empty(self, cell):
        assert parse_multi(cell) == []

    def test_first_and_count(self):
        s = pd.Series(["40,41", None, "7", ""])
        assert first_values(s).tolist() == ["40", None, "7", None]
        assert count_values(s).tolist() == [2, 0, 1, 0]


class TestTransform:
    def test_shape_dtype_finite(self, pre, train_val):
        for df in train_val:
            X = pre.transform(df)
            assert X.dtype == np.float32 and X.shape == (len(df), pre.schema()["n_features"])
            assert np.isfinite(X).all()

    def test_layout_contiguous_and_grouped(self, pre):
        pos, order = 0, []
        for b in pre.core_blocks:
            assert b["start"] == pos
            pos += b["width"]
            order.append(b["type"])
        assert pos == pre.schema()["n_features"]
        rank = {"numeric": 0, "binary": 1, "na_flag": 2, "categorical": 3}
        assert [rank[t] for t in order] == sorted(rank[t] for t in order)

    def test_one_hot_groups_sum_to_one(self, pre, train_val):
        X = pre.transform(train_val[1])
        for b in pre.core_blocks:
            if b["type"] == "categorical":
                s = X[:, b["start"]:b["start"] + b["width"]].sum(axis=1)
                assert np.all(s == 1.0), b["name"]

    def test_clip_bounds(self, pre, train_val):
        X = pre.transform(train_val[1])
        for b in pre.core_blocks:
            if b["type"] == "numeric":
                assert np.abs(X[:, b["start"]]).max() <= PCFG["clip_sigma"] + 1e-6

    def test_binary_forms_agree(self, pre):
        df = make_df(4, 3)
        df["syn"] = ["Set", "True", "Not set", "False"]
        X = pre.transform(df)
        b = next(b for b in pre.core_blocks if b["name"] == "syn")
        assert X[:, b["start"]].tolist() == [1.0, 1.0, 0.0, 0.0]

    def test_text_and_code_encode_identically(self, pre):
        df = make_df(2, 3)
        df["message_type"] = ["Publish Message", "3"]
        X = pre.transform(df)
        b = next(b for b in pre.core_blocks if b["name"] == "message_type")
        blk = X[:, b["start"]:b["start"] + b["width"]]
        assert np.array_equal(blk[0], blk[1])
        assert b["categories"][int(np.argmax(blk[0]))] == "3"

    def test_unseen_to_other_and_na_to_none(self, pre):
        df = make_df(3, 3)
        df["protocol"] = ["TCP", "ICMPv9", None]
        X = pre.transform(df)
        b = next(b for b in pre.core_blocks if b["name"] == "protocol")
        cats = b["categories"]
        assert cats[np.argmax(X[1, b["start"]:b["start"] + b["width"]])] == "OTHER"
        assert cats[np.argmax(X[2, b["start"]:b["start"] + b["width"]])] == "NONE"

    def test_multi_value_uses_first_element(self, pre):
        df = make_df(2, 3)
        df["topic_length"] = ["40,41", "40"]
        X = pre.transform(df)
        b = next(b for b in pre.core_blocks if b["name"] == "topic_length")
        assert X[0, b["start"]] == X[1, b["start"]]


class TestFitRules:
    def test_constant_column_has_unit_std(self, pre, train_val):
        p = pre.params["will_message_length"]            # all-NA in the synthetic data -> filled with 0 -> constant
        assert p["constant"] and p["std"] == 1.0
        df = make_df(50, 4)
        df["keep_alive"] = "60"                           # constant but applicable everywhere
        q = Preprocessor(DECISIONS, PCFG).fit(df).params["keep_alive"]
        assert q["constant"] and q["std"] == 1.0
        assert np.isfinite(pre.transform(train_val[0])).all()

    def test_negative_time_clipped_and_counted(self, pre, train_val):
        assert pre.params["time_delta_from_previous_displayed_frame"]["n_negative_in_train"] == 1
        assert pre.audit(train_val[0])["negative_clipped"]["time_delta_from_previous_displayed_frame"] == 1

    def test_na_flag_range(self, pre):
        names = {b["name"] for b in pre.core_blocks if b["type"] == "na_flag"}
        assert "irtt_is_na" in names and "keep_alive_is_na" in names          # 40% / 90% NA: inside (0.5%, 99.5%)
        assert "user_name_length_is_na" not in names                           # ~0.05% NA: below the lower bound
        assert "will_topic_length_is_na" in names                              # 99.9% NA: ultra-sparse fix adds a flag
        assert pre.schema()["n_na_flags"] == len(names)

    def test_spec_rule_without_ultra_fix(self, train_val):
        spec = Preprocessor(DECISIONS, {**PCFG, "ultra_sparse_fix": False}).fit(train_val[0])
        names = {b["name"] for b in spec.core_blocks if b["type"] == "na_flag"}
        assert "will_topic_length_is_na" not in names                          # spec: no flag above 99.5% NA

    def test_ultra_sparse_keeps_distinct_values(self):
        """With 99.9% NA the spec rule clips every applicable row to +5 sigma; the fix keeps values apart."""
        df = make_df(4000, 5)
        df["will_message_length"] = [None] * 4000
        df.loc[[0, 1000, 2000, 3000], "will_message_length"] = ["2", "3164", "3164", "2"]
        fixed = Preprocessor(DECISIONS, PCFG).fit(df)
        spec = Preprocessor(DECISIONS, {**PCFG, "ultra_sparse_fix": False}).fit(df)
        app = df["will_message_length"].notna().to_numpy()
        zf = fixed.transform(df)[:, next(b["start"] for b in fixed.core_blocks if b["name"] == "will_message_length")]
        zs = spec.transform(df)[:, next(b["start"] for b in spec.core_blocks if b["name"] == "will_message_length")]
        assert len(np.unique(zf[app])) == 2 and np.all(zf[~app] == 0.0)
        assert len(np.unique(zs[app])) == 1                                    # spec rule merges them
        flag = next(b["start"] for b in fixed.core_blocks if b["name"] == "will_message_length_is_na")
        assert fixed.transform(df)[:, flag].tolist() == (~app).astype(float).tolist()
        inv = fixed.inverse_transform(fixed.transform(df))["will_message_length"]
        assert inv.isna().tolist() == (~app).tolist()
        assert np.allclose(inv[app].to_numpy(), [2, 3164, 3164, 2], rtol=1e-4)

    def test_sparse_column_stays_distinguishable(self, pre, train_val):
        X = pre.transform(train_val[0])
        b = next(b for b in pre.core_blocks if b["name"] == "will_topic_length_is_na")
        non_na = train_val[0]["will_topic_length"].notna().to_numpy()
        assert X[non_na, b["start"]].max() == 0.0 and X[~non_na, b["start"]].min() == 1.0

    def test_log1p_only_for_skewed_nonneg(self, pre):
        assert pre.params["calculated_window_size"]["log1p"] is True
        assert pre.params["frame_length_on_wire_2"]["log1p"] is False

    def test_diagnostic_columns_are_outside_core(self, pre, train_val):
        core_cols = {b["column"] for b in pre.core_blocks}
        assert "time_since_first_frame_in_this_tcp_stream" not in core_cols
        Xd = pre.transform_diagnostic(train_val[0])
        assert Xd.shape == (len(train_val[0]), pre.schema()["diagnostic"]["n_features"]) and Xd.shape[1] >= 1
        assert np.isfinite(Xd).all()

    def test_dropped_columns_never_used(self, pre):
        used = {b["column"] for b in pre.core_blocks}
        for c in ("source", "info", "no", "frame_length_on_wire_1", "n_mqtt_msgs"):
            assert c not in used


class TestRoundTrip:
    def test_inverse_recovers_inputs(self, pre, train_val):
        df = train_val[0]
        X = pre.transform(df)
        inv = pre.inverse_transform(X)
        # numeric: relative error < 1e-4 where nothing was clipped and the value is applicable
        for col in ("frame_length_on_wire_2", "calculated_window_size", "tcp_segment_len", "irtt"):
            p = pre.params[col]
            raw = pd.to_numeric(df[col], errors="coerce")
            z = X[:, next(b["start"] for b in pre.core_blocks if b["name"] == col)]
            ok = raw.notna().to_numpy() & (np.abs(z) < PCFG["clip_sigma"] - 1e-3)
            err = np.abs(inv[col].to_numpy()[ok] - raw.to_numpy()[ok]) / (1.0 + np.abs(raw.to_numpy()[ok]))
            assert err.max() < 1e-4, (col, err.max())
        # NA restored where a flag exists
        assert inv["irtt"].isna().to_numpy().tolist() == df["irtt"].isna().to_numpy().tolist()
        # categorical and binary
        assert inv["protocol"].tolist() == df["protocol"].tolist()   # all values are inside the top-K
        syn = df["syn"].map({"Set": 1.0, "True": 1.0, "Not set": 0.0, "False": 0.0})
        assert inv["syn"].tolist() == syn.tolist()


class TestNoLeakage:
    def test_transform_does_not_change_fitted_state(self, pre, train_val):
        before = json.dumps(pre.schema(), sort_keys=True)
        pre.transform(train_val[1])
        pre.transform(make_df(500, 9, shift=3.0))
        pre.audit(train_val[1])
        assert json.dumps(pre.schema(), sort_keys=True) == before

    def test_fit_depends_only_on_the_data_given(self, train_val):
        a = Preprocessor(DECISIONS, PCFG).fit(train_val[0])
        b = Preprocessor(DECISIONS, PCFG).fit(train_val[0])
        assert json.dumps(a.schema(), sort_keys=True) == json.dumps(b.schema(), sort_keys=True)
        c = Preprocessor(DECISIONS, PCFG).fit(pd.concat(train_val, ignore_index=True))   # control: leak would show here
        assert json.dumps(a.schema(), sort_keys=True) != json.dumps(c.schema(), sort_keys=True)


class TestPipeline:
    def _setup(self, tmp_path):
        cfg = load_config()
        cfg = copy.deepcopy(cfg)
        cfg["paths"]["work_dir"] = str(tmp_path / "work")
        cfg["label_mode"] = "6class"
        cfg["harmonize"]["decisions_path"] = str(ROOT / "configs" / "feature_decisions.yaml")
        d = tmp_path / "work" / "interim" / "6class"
        d.mkdir(parents=True)
        labels = ["NORMAL", "BCF", "DELAYED", "SYN", "INVALID", "WILL"]
        for i, s in enumerate(("train", "val", "test")):
            df = make_df(600, i)
            df["label"] = [labels[j % 6] for j in range(len(df))]
            df["class11"] = df["label"]
            df["group_id"] = [f"g{s}{j % 3}" for j in range(len(df))]
            df["stream_id"] = [f"g{s}:{j % 5}" for j in range(len(df))]
            df.to_parquet(d / f"{s}.parquet", index=False)
        return cfg

    def test_outputs_and_consistency(self, tmp_path):
        cfg = self._setup(tmp_path)
        schema = run_preprocess(cfg)
        out = processed_dir(cfg)
        for f in ("train.npz", "val.npz", "test.npz", "feature_schema.json", "label_map.json", "preprocessor.joblib"):
            assert (out / f).exists(), f
        z = np.load(out / "train.npz")
        assert z["X"].shape == (600, schema["n_features"]) and z["X"].dtype == np.float32
        assert set(z["y"].tolist()) == set(range(6))
        assert z["X_diag"].shape[0] == 600 and z["group_id"].shape == (600,)
        pre = joblib.load(out / "preprocessor.joblib")
        df = pd.read_parquet(tmp_path / "work" / "interim" / "6class" / "train.parquet")
        assert np.array_equal(pre.transform(df), z["X"])
        assert json.loads((out / "label_map.json").read_text())["NORMAL"] == 0
        assert schema["class_distribution"]["train"]["BCF"] == 100

    def test_label_map_order_follows_config(self):
        cfg = {"label_mode": "6class", "quota": {"NORMAL": [1, 1, 1], "BCF": [1, 1, 1]}}
        assert label_map(cfg) == {"NORMAL": 0, "BCF": 1}
