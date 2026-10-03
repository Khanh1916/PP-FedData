"""Tests for Phase 3 split + sampling, on a small synthetic dataset with the real 33-column layout."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from ppfeddata.data.harmonize import SHARED_COLUMNS_ORIG  # noqa: E402
from ppfeddata.data.split_sample import (  # noqa: E402
    QuotaError, allocate, interim_dir, run_sample, subclass_quotas,
)

ROWS = 400


def _write_csv(path, source, group, n=ROWS, seed=0):
    rng = np.random.default_rng(seed)
    d = {c: rng.integers(0, 9, n).astype(str) for c in SHARED_COLUMNS_ORIG}
    d["Protocol"] = rng.choice(["TCP", "MQTT", "STP"], n, p=[0.5, 0.4, 0.1])
    d["Stream index"] = (np.arange(n) // 40).astype(str)   # streams are contiguous, one per 40 rows
    d["Stream index"][[0, n - 1]] = "99"                    # one stream spanning the whole capture
    if source == "normal":
        d["Label"] = ["Normal"] * n
        d["Capture_ID"] = [f"Pcap Files/{group}.pcap"] * n
    pd.DataFrame(d).to_csv(path, index=False)


def _make_dataset(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    recs = []

    def add(class11, scenario, atype, group, source, fname):
        p = raw / fname
        _write_csv(p, source, group, seed=len(recs))
        recs.append(dict(path=str(p), filename=fname, scenario=scenario, attack_type=atype,
                         class6=scenario, class11=class11, group_id=group, rows=ROWS, cols=33,
                         bytes=p.stat().st_size, source=source, empty=False))

    for g in range(5):
        for part in range(2):  # Normal: capture split in two parts
            add("NORMAL", "NORMAL", "NONE", f"NormalData{g}", "normal", f"n{g}_{part}.csv")
    for g in range(4):
        add("BCF_DoS", "BCF", "DoS", f"BF1_DoS_{g}", "attack", f"bd{g}.csv")
    for g in range(3):
        add("BCF_DDoS", "BCF", "DDoS", f"BF1_DDoS_{g}", "attack", f"bdd{g}.csv")
    add("WILL_DoS", "WILL", "DoS", "WILL_DoS_1", "attack", "w1.csv")  # single group -> block split
    return pd.DataFrame(recs)


def _cfg(tmp_path, work="work", **split_over):
    cfg = {
        "label_mode": "11class",
        "paths": {"work_dir": str(tmp_path / work)},
        "split": {"test_groups_per_subclass": 1, "val_groups_per_subclass": 1,
                  "fallback_block_split": {"n_blocks": 10, "gap_rows": 5},
                  "split_seed": 0, "min_eval_rows": 5, "max_peak_gb": 4.0},
        "quota_11class": {"NORMAL": [40, 10, 10], "BCF_DoS": [20, 5, 5], "BCF_DDoS": [20, 5, 5],
                          "WILL_DoS": [10, 5, 5]},
        "train_sampling": {"max_rows_per_stream": None},
    }
    cfg["split"].update(split_over)
    return cfg


FILTERS = {"protocol": ["TCP", "MQTT"]}


def _run(tmp_path, work="work", cfg_over=None, **split_over):
    files = _make_dataset(tmp_path) if not (tmp_path / "raw").exists() else pd.read_csv(tmp_path / "files.csv")
    files.to_csv(tmp_path / "files.csv", index=False)
    cfg = _cfg(tmp_path, work, **split_over)
    if cfg_over:
        cfg.update(cfg_over)
    return cfg, run_sample(cfg, files=files, filters=FILTERS, use_cache=False)


def _load(cfg):
    d = interim_dir(cfg)
    return {s: pd.read_parquet(d / f"{s}.parquet") for s in ("train", "val", "test")}


class TestAllocate:
    def test_even(self):
        assert allocate(10, [100, 100, 100]) == [4, 3, 3]

    def test_water_filling(self):
        assert allocate(10, [2, 100, 100]) == [2, 4, 4]

    def test_capacity_shortfall(self):
        assert allocate(10, [2, 3]) == [2, 3]


class TestQuotas:
    def test_6class_split_even_and_conserved(self):
        cfg = {"label_mode": "6class", "quota": {"NORMAL": [10, 4, 6], "BCF": [7, 3, 5]}}
        q = subclass_quotas(cfg)
        assert q["NORMAL"] == (10, 4, 6)
        assert q["BCF_DoS"] == (4, 2, 3) and q["BCF_DDoS"] == (3, 1, 2)
        assert sum(x[0] for k, x in q.items() if k.startswith("BCF")) == 7

    def test_11class_uses_table(self):
        cfg = {"label_mode": "11class", "quota_11class": {"A": [1, 2, 3]}}
        assert subclass_quotas(cfg) == {"A": (1, 2, 3)}


class TestRunSample:
    def test_counts_groups_and_filters(self, tmp_path):
        cfg, m = _run(tmp_path)
        dfs = _load(cfg)
        for sub, q in cfg["quota_11class"].items():
            for split, n in zip(("train", "val", "test"), q):
                assert (dfs[split]["class11"] == sub).sum() == n, (sub, split)
        all_df = pd.concat(dfs.values())
        assert (all_df.groupby("group_id")["split"].nunique() == 1).all()
        assert all_df["protocol"].isin(["TCP", "MQTT"]).all()
        assert not all_df.duplicated(["source_file", "row_idx"]).any()
        assert set(all_df["data_source"]) <= {"normal", "attack"}
        assert all_df["source"].str.fullmatch(r"\d").all()   # raw `source` column is kept untouched
        assert "capture_id" not in all_df.columns   # raw Capture_ID/Label are replaced by our own metadata
        assert set(all_df["label"]) == set(cfg["quota_11class"])

    def test_train_spread_evenly_over_groups(self, tmp_path):
        cfg, _ = _run(tmp_path)
        tr = _load(cfg)["train"]
        per = tr[tr.class11 == "NORMAL"].groupby("group_id").size()
        assert len(per) == 3 and per.max() - per.min() <= 1   # 5 captures - 1 val - 1 test = 3 train groups

    def test_block_split_gap_and_units(self, tmp_path):
        cfg, m = _run(tmp_path)
        units = m["subclasses"]["WILL_DoS"]
        assert units["split_kind"] == "block"
        all_units = [u for v in units["units"].values() for u in v]
        assert all("#blk" in u["unit_id"] for u in all_units)
        ranges = sorted(u["row_range"] for u in all_units)
        size = ROWS // 10
        assert ranges[0][0] == 0 and ranges[0][1] == size - 5   # gap_rows=5 dropped at the end of each block
        df = pd.concat(_load(cfg).values())
        w = df[df.class11 == "WILL_DoS"]
        for _, r in w.iterrows():
            blk = int(r["group_id"].split("#blk")[1])
            hi = ROWS if blk == 9 else (blk + 1) * size - 5
            assert blk * size <= int(r["row_idx"]) < hi
        assert w.groupby("group_id")["split"].nunique().max() == 1

    def test_cross_block_streams_dropped(self, tmp_path):
        cfg, m = _run(tmp_path)
        assert m["subclasses"]["WILL_DoS"]["cross_block"]["cross_block_streams"] >= 1
        assert m["streams_in_multiple_splits"] == {}
        df = pd.concat(_load(cfg).values())
        assert not (df["stream_index"] == "99").any() or df[df.stream_index == "99"].class11.isin(
            ["NORMAL", "BCF_DoS", "BCF_DDoS"]).all()

    def test_cross_block_option_off_keeps_streams(self, tmp_path):
        _, m = _run(tmp_path, drop_cross_block_streams=False)
        assert m["subclasses"]["WILL_DoS"]["cross_block"]["rows_removed"] == 0

    def test_deterministic_same_seed_same_hash(self, tmp_path):
        _, m1 = _run(tmp_path, "w1")
        _, m2 = _run(tmp_path, "w2")
        assert m1["parquet_sha256"] == m2["parquet_sha256"]

    def test_different_seed_changes_assignment(self, tmp_path):
        _, m1 = _run(tmp_path, "w1")
        _, m2 = _run(tmp_path, "w2", split_seed=1)
        assert m1["parquet_sha256"] != m2["parquet_sha256"]

    def test_quota_reduced_with_warning(self, tmp_path):
        over = {"quota_11class": {"NORMAL": [40, 10, 10], "BCF_DoS": [20, 5, 5], "BCF_DDoS": [20, 5, 5],
                                  "WILL_DoS": [10, 5, 500]}}
        cfg, m = _run(tmp_path, cfg_over=over, min_eval_rows=1)
        assert any("WILL_DoS/test: quota reduced" in w for w in m["warnings"])
        assert m["quota_actual"]["WILL_DoS"]["test"] < 500

    def test_eval_class_too_small_raises(self, tmp_path):
        files = _make_dataset(tmp_path)
        cfg = _cfg(tmp_path, min_eval_rows=1000)
        with pytest.raises(QuotaError):
            run_sample(cfg, files=files, filters=FILTERS, use_cache=False)

    def test_max_rows_per_stream(self, tmp_path):
        cfg, _ = _run(tmp_path, cfg_over={"train_sampling": {"max_rows_per_stream": 2}})
        tr = _load(cfg)["train"]
        assert tr.groupby(["group_id", "stream_id"]).size().max() <= 2

    def test_capture_id_mismatch_raises(self, tmp_path):
        files = _make_dataset(tmp_path)
        bad = files[files.source == "normal"].iloc[0]
        d = pd.read_csv(bad["path"])
        d["Capture_ID"] = "Pcap Files/Other.pcap"
        d.to_csv(bad["path"], index=False)
        with pytest.raises(RuntimeError, match="Capture_ID"):
            run_sample(_cfg(tmp_path), files=files, filters=FILTERS, use_cache=False)

    def test_row_count_mismatch_raises(self, tmp_path):
        files = _make_dataset(tmp_path)
        files.loc[0, "rows"] = ROWS + 1
        with pytest.raises(RuntimeError, match="inventory says"):
            run_sample(_cfg(tmp_path), files=files, filters=FILTERS, use_cache=False)

    def test_manifest_written(self, tmp_path):
        cfg, m = _run(tmp_path)
        mf = json.loads((interim_dir(cfg) / "split_manifest.json").read_text())
        assert mf["split_seed"] == 0 and set(mf["parquet_sha256"]) == {"train", "val", "test"}
        assert mf["peak_rss_gb"] is not None


class TestSharedManifest:
    def test_copy_is_identical_and_has_no_local_paths(self, tmp_path):
        files = _make_dataset(tmp_path)
        cfg = _cfg(tmp_path)
        cfg["paths"]["shared_manifest_dir"] = str(tmp_path / "shared")
        run_sample(cfg, files=files, filters=FILTERS, use_cache=False)
        src = (interim_dir(cfg) / "split_manifest.json").read_text(encoding="utf-8")
        dst = (tmp_path / "shared" / "split_manifest_11class.json").read_text(encoding="utf-8")
        assert src == dst
        assert str(tmp_path) not in dst          # no absolute local paths leak into the shared file
        assert "stream_index" not in dst         # no data rows

    def test_disabled_when_not_configured(self, tmp_path):
        files = _make_dataset(tmp_path)
        run_sample(_cfg(tmp_path), files=files, filters=FILTERS, use_cache=False)
        assert not (tmp_path / "shared").exists()


class TestRowFilterSafety:
    def test_missing_decisions_file_raises(self, tmp_path):
        from ppfeddata.data.split_sample import load_row_filters
        with pytest.raises(FileNotFoundError):
            load_row_filters({"harmonize": {"decisions_path": str(tmp_path / "nope.yaml")}})

    def test_reads_all_sources_filters(self, tmp_path):
        from ppfeddata.data.split_sample import load_row_filters
        p = tmp_path / "d.yaml"
        p.write_text("_row_filters:\n  all_sources:\n    protocol: [TCP, MQTT]\n", encoding="utf-8")
        assert load_row_filters({"harmonize": {"decisions_path": str(p)}}) == {"protocol": ["TCP", "MQTT"]}
