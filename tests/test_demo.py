"""Phase 13 tests: the limitations list, the logic behind the demo (sampling from a trained CVAE, results readers), the demo pages, the generated blocks of the README and the
reproducibility bundle. The sampling tests use a tiny CVAE trained on the synthetic table of test_preprocess; `test_real_*` compare with the real artifacts when they exist."""
import io
import json
import re
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ppfeddata import demo_lib as dl  # noqa: E402
from ppfeddata import limitations as lim  # noqa: E402
from ppfeddata import package as pk  # noqa: E402
from ppfeddata import readme_gen as rg  # noqa: E402
from ppfeddata.cli import build_parser  # noqa: E402
from ppfeddata.data.preprocess import Preprocessor  # noqa: E402
from ppfeddata.models.cvae import build_layout  # noqa: E402
from ppfeddata.models.generate import gen_stats_from_cfg, generate  # noqa: E402
from ppfeddata.models.train import train_cvae  # noqa: E402
from ppfeddata.utils import load_config  # noqa: E402
from tests.test_preprocess import DECISIONS, PCFG, make_df  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HP = {"latent_dim": 4, "hidden": [32, 16], "beta": 0.5, "beta_warmup_epochs": 2, "lr": 3e-3, "batch_size": 128, "epochs": 3, "class_balanced_sampler": False}
CLASSES = ["NORMAL", "BCF", "SYN"]


@pytest.fixture(autouse=True)
def tuned_trial_21(monkeypatch):
    import ppfeddata.fl.m1 as m1
    monkeypatch.setattr(m1, "tuned_setup", lambda cfg: ("M1d-t21", {}, 1.0))


# --------------------------------------------------------------------------------------------------
# A tiny trained world: the real Preprocessor on a synthetic table, a CVAE trained for 3 epochs, artifacts in the layout of the experiments
# --------------------------------------------------------------------------------------------------
class World:
    pass


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("demo")
    df = make_df(1500, seed=0)
    pre = Preprocessor(DECISIONS, PCFG).fit(df)
    X = pre.transform(df)
    schema = pre.schema()
    schema["label_map"] = {c: i for i, c in enumerate(CLASSES)}
    y = np.random.default_rng(0).integers(0, 3, len(X)).astype(np.int64)
    model, hist = train_cvae(X[:1200], y[:1200], X[1200:], y[1200:], 3, build_layout(schema), HP, 0, patience=None)
    art = tmp / "artifacts"

    def save_fl(name, seed=0):
        d = art / f"{name}_{seed}"
        d.mkdir(parents=True)
        torch.save(model.state_dict(), d / "final_state.pt")
        (d / "spec.json").write_text(json.dumps({"hp": HP}), encoding="utf-8")

    save_fl("B3")
    save_fl("M1d-t21-eps5")
    d = art / "B2_0"
    d.mkdir(parents=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config(), "hp": HP, "history": hist}, d / "model.pt")
    w = World()
    w.cfg = {"label_mode": "6class", "seeds": [0, 1], "compute": {"artifacts_dir": str(art), "runs_csv": str(tmp / "results" / "runs.csv")},
             "generate": {"target_per_class": 40, "residual_noise": True, "snap_support": False, "snap_max_unique": 256}, "dp": {"delta": 1e-5},
             "fl": {"num_clients": 5}, "paths": {"work_dir": str(tmp / "data"), "shared_manifest_dir": str(tmp / "manifests")}}
    w.assets = dl.Assets({"train": {"X": X, "y": y}}, schema, pre)
    w.model, w.schema, w.pre, w.X, w.y, w.tmp = model, schema, pre, X, y, tmp
    return w


def gen(world, label):
    return next(g for g in dl.generators(world.cfg) if g.label == label)


# --------------------------------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------------------------------
def test_generators_follow_the_spec_matrix(world):
    gs = {g.label: g for g in dl.generators(world.cfg)}
    assert set(gs) == {"B2", "B3", "M2", "M1-eps1", "M1-eps5", "M1-eps10", "M3", "MG-eps1", "MG-eps5", "MG-eps10"}
    assert gs["MG-eps5"].kind == "mg" and gs["MG-eps5"].dp and gs["MG-eps5"].secagg and gs["MG-eps5"].run_name == "MGs-eps5"
    assert "Skellam" in gs["MG-eps5"].protections
    assert [gs[k].dp for k in ("B2", "B3", "M2", "M1-eps5", "M3")] == [False, False, False, True, True]
    assert [gs[k].secagg for k in ("B2", "B3", "M2", "M1-eps5", "M3")] == [False, False, True, False, True]
    assert gs["B2"].federated is False and gs["B3"].federated is True
    assert gs["M1-eps5"].run_name == "M1d-t21-eps5" and gs["M3"].run_name == "M3d-t21-eps5" and gs["M1-eps5"].eps_target == 5.0 and gs["B3"].eps_target is None
    assert "client-side DP-SGD" in gs["M3"].protections and "secure aggregation" in gs["M3"].protections


def test_available_generators_are_the_ones_with_a_model_file(world):
    avail = {g.label: s for g, s in dl.available_generators(world.cfg)}
    assert avail == {"B2": [0], "B3": [0], "M1-eps5": [0]}                 # seed 1 and the other configurations have no artifacts
    assert dl.model_file(world.cfg, gen(world, "B2"), 0).name == "model.pt" and dl.model_file(world.cfg, gen(world, "B3"), 0).name == "final_state.pt"


@pytest.mark.parametrize("label", ["B2", "B3", "M1-eps5"])
def test_load_model_reads_both_checkpoint_formats(world, label):
    m = dl.load_model(world.cfg, gen(world, label), 0, world.schema)
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), world.model.state_dict().values())) and not m.training
    with pytest.raises(FileNotFoundError):
        dl.load_model(world.cfg, gen(world, label), 1, world.schema)


# --------------------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------------------
def test_sample_is_deterministic_and_is_the_generate_of_the_evaluation(world):
    g = gen(world, "B3")
    a = dl.sample(world.cfg, g, 0, None, 30, 7, "residual", assets=world.assets)
    b = dl.sample(world.cfg, g, 0, None, 30, 7, "residual", assets=world.assets)
    c = dl.sample(world.cfg, g, 0, None, 30, 8, "residual", assets=world.assets)
    stats = gen_stats_from_cfg(world.model, world.X, world.y, world.schema, world.cfg["generate"])
    X, y = generate(world.model, world.schema, [30, 30, 30], 7, stats=stats)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.X, X) and np.array_equal(a.y, y) and not np.array_equal(a.X, c.X)
    assert a.variant == "residual" and len(a.table) == 90 and a.classes == CLASSES


def test_sample_reproduces_the_synthetic_set_the_evaluation_saved(world):
    """The claim of the demo: same model, seed and counts give the rows of artifacts/<run>/synthetic.npz."""
    k, target = 3, int(world.cfg["generate"]["target_per_class"])
    for label, variant in (("B3", "plain"), ("B3", "residual"), ("M1-eps5", "plain")):
        g = gen(world, label)
        stats = gen_stats_from_cfg(world.model, world.X, world.y, world.schema, world.cfg["generate"]) if variant == "residual" else None
        Xs, ys = generate(world.model, world.schema, [target] * k, 0, stats=stats)           # what evaluate_generator does
        p = dl.synthetic_path(world.cfg, g, 0, variant)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, X=Xs, y=ys.astype(np.int16))
        s = dl.sample(world.cfg, g, 0, None, target, 0, variant, assets=world.assets)
        assert np.array_equal(s.X, np.load(p)["X"]) and np.array_equal(s.y, np.load(p)["y"])
    assert dl.synthetic_path(world.cfg, gen(world, "B3"), 0, "plain").parent.name == "B3-plain_0"
    assert dl.synthetic_path(world.cfg, gen(world, "B2"), 0, "plain").parent.name == "B2_0"           # B2 has no separate plain run


def test_dp_generator_cannot_use_the_residual_noise_and_plain_uses_only_the_weights(world):
    with pytest.raises(ValueError, match="epsilon"):
        dl.sample(world.cfg, gen(world, "M1-eps5"), 0, None, 5, 0, "residual", assets=world.assets)
    blank = dl.Assets({"train": {"X": np.zeros_like(world.X), "y": world.y}}, world.schema, world.pre)         # the train split must not matter
    a = dl.sample(world.cfg, gen(world, "M1-eps5"), 0, None, 20, 1, "plain", assets=world.assets)
    b = dl.sample(world.cfg, gen(world, "M1-eps5"), 0, None, 20, 1, "plain", assets=blank)
    assert np.array_equal(a.X, b.X)
    r = dl.sample(world.cfg, gen(world, "B3"), 0, None, 20, 1, "residual", assets=world.assets)
    assert not np.array_equal(r.X, dl.sample(world.cfg, gen(world, "B3"), 0, None, 20, 1, "plain", assets=world.assets).X)


def test_residual_variant_falls_back_to_plain_when_the_config_switches_it_off(world):
    cfg = {**world.cfg, "generate": {**world.cfg["generate"], "residual_noise": False}}
    s = dl.sample(cfg, gen(world, "B3"), 0, None, 10, 0, "residual", assets=world.assets)
    assert s.variant == "plain" and s.notes and np.array_equal(s.X, dl.sample(cfg, gen(world, "B3"), 0, None, 10, 0, "plain", assets=world.assets).X)


def test_sample_validates_its_inputs(world):
    g = gen(world, "B3")
    for kw, msg in (({"classes": ["NOPE"]}, "unknown classes"), ({"n_per_class": 0}, "between 1"), ({"n_per_class": dl.MAX_PER_CLASS + 1}, "between 1"), ({"variant": "x"}, "variant")):
        args = {"classes": None, "n_per_class": 5, "variant": "plain", **kw}
        with pytest.raises(ValueError, match=msg):
            dl.sample(world.cfg, g, 0, args["classes"], args["n_per_class"], 0, args["variant"], assets=world.assets)


def test_one_class_gives_only_that_class(world):
    s = dl.sample(world.cfg, gen(world, "B3"), 0, ["BCF"], 25, 0, "plain", assets=world.assets)
    assert len(s.table) == 25 and set(s.table["label"]) == {"BCF"} and set(s.y) == {1}


def test_table_is_in_raw_units_and_empty_where_a_feature_does_not_apply(world):
    s = dl.sample(world.cfg, gen(world, "B3"), 0, None, 200, 3, "residual", assets=world.assets)
    t, pre = s.table, world.pre
    assert list(t.columns) == ["label"] + [b["column"] for b in pre.core_blocks if b["type"] != "na_flag"]
    for b in pre.core_blocks:
        c, p = b["column"], pre.params[b["column"]]
        if b["type"] == "numeric":
            v = t[c].dropna().astype(float)
            assert v.between(p["raw_min"], p["raw_max"]).all()
            if p["is_integer"]:
                assert str(t[c].dtype) == "Int64" and (v == np.round(v)).all()
        elif b["type"] == "binary":
            assert set(t[c].dropna().unique()) <= {0, 1}
        elif b["type"] == "categorical":
            assert set(t[c].dropna().unique()) <= set(p["categories"]) - {"NONE"}
        elif b["type"] == "na_flag":
            continue
    flag = {b["column"]: b for b in pre.core_blocks if b["type"] == "na_flag"}
    for col, b in flag.items():                                                   # a cell is empty exactly where the `_is_na` flag of the encoded row is 1
        assert np.array_equal(t[col].isna().to_numpy(), s.X[:, b["start"]] >= 0.5)


def test_csv_downloads(world):
    s = dl.sample(world.cfg, gen(world, "B3"), 0, ["SYN"], 12, 0, "plain", assets=world.assets)
    raw, enc = pd.read_csv(io.BytesIO(s.csv())), pd.read_csv(io.BytesIO(s.csv(encoded=True)))
    assert len(raw) == len(enc) == 12 and set(raw["label"]) == set(enc["label"]) == {"SYN"}
    assert enc.shape[1] == world.schema["n_features"] + 1 and np.allclose(enc.drop(columns="label").to_numpy(), s.X, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------------------------------
# real artifacts (skipped when the data / models are not on this machine)
# --------------------------------------------------------------------------------------------------
HAVE_REAL = (ROOT / "artifacts" / "B3_0" / "final_state.pt").exists() and (ROOT / "data" / "processed" / "6class" / "preprocessor.joblib").exists()


@pytest.mark.skipif(not HAVE_REAL, reason="needs the trained models and data/processed")
def test_real_models_reproduce_every_saved_synthetic_set(monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = load_config()
    assets, n = dl.load_assets(cfg), 0
    k, target = len(assets.classes), int(cfg["generate"]["target_per_class"])
    for g, seeds in dl.available_generators(cfg):
        model = dl.load_model(cfg, g, seeds[0], assets.schema)
        for variant in ("plain", "residual"):
            p = dl.synthetic_path(cfg, g, seeds[0], variant)
            if (variant == "residual" and g.dp) or (variant == "plain" and g.kind == "b2") or not p.exists():
                continue
            s = dl.sample(cfg, g, seeds[0], None, target, seeds[0], variant, assets=assets, model=model)
            ref = np.load(p)
            assert np.array_equal(s.X, ref["X"]) and np.array_equal(s.y.astype(np.int16), ref["y"]), (g.label, variant)
            n += 1
    assert n >= 8 and k == 6


# --------------------------------------------------------------------------------------------------
# results readers
# --------------------------------------------------------------------------------------------------
def test_split_markdown_and_image_parts():
    md = "# T\n\nintro\n\n## 1. A\n\ntext a\n\n### 1.1 sub\n\nsub text\n\n## 2. B\n\n```\n# not a heading\n```\n\nafter\n\n![fig](../figures/x.png)\n\ntail\n"
    s2, s3 = dl.split_markdown(md, 2), dl.split_markdown(md, 3)
    assert list(s2) == ["1. A", "2. B"] and "sub text" in s2["1. A"] and "### 1.1 sub" in s2["1. A"] and "# not a heading" in s2["2. B"]
    assert list(s3) == ["1.1 sub"] and s3["1.1 sub"] == "sub text"                  # a level-3 section ends at the next heading of level 3 or higher
    parts = dl.md_parts(s2["2. B"], Path("/base/reports"))
    assert [p[0] for p in parts] == ["md", "img", "md"] and parts[1][2] == "fig" and parts[1][1].name == "x.png" and parts[1][1].parent.name == "figures"


def test_quality_picks_the_plain_variant_when_the_ledger_has_it():
    rows = []
    for cfgname, f1 in (("B3-TSTR-rf", 0.40), ("B3-TAug-rf", 0.45), ("B3-plain-TSTR-rf", 0.38), ("B3-plain-TAug-rf", 0.44), ("B0-rf", 0.447), ("B2-TSTR-rf", 0.41), ("B2-TAug-rf", 0.448)):
        rows.append({"config": cfgname, "macro_f1_mean": f1, "macro_f1_std": 0.01, "c2st_auc_mean_mean": 0.999, "mia_auc_mean_mean": 0.49, "dup_rate_mean": 1e-4, "dcr_ratio_mean_mean": 0.9,
                     "dp_eps_max_mean": np.nan})
    summ = pd.DataFrame(rows)
    g3 = dl.Generator("B3", "t", "B3", "b3", False, False, True, None)
    g2 = dl.Generator("B2", "t", "B2", "b2", False, False, False, None)
    assert dl.quality(summ, g3, "plain")["config"] == "B3-plain" and dl.quality(summ, g3, "plain")["tstr_rf"] == (0.38, 0.01)
    assert dl.quality(summ, g3, "residual")["config"] == "B3" and dl.quality(summ, g3, "residual")["tstr_rf"][0] == 0.40
    assert dl.quality(summ, g2, "plain")["config"] == "B2" and dl.quality(summ, g2, "plain")["eps_max"] is None            # B2 has no plain twin and no DP
    assert dl.quality(None, g3, "plain") is None and dl.quality(summ.iloc[:0], g3, "plain") is None


def R6_fixture():
    cand = lambda lab, taug, tstr, ok: {"label": lab, "taug_f1": taug, "tstr_f1": tstr, "mia_auc": 0.49, "eps_max": None, "time_ratio": 1.0, "ok_mia": True, "ok_eps": True,  # noqa: E731
                                        "ok_overhead": ok, "eligible": ok}
    return {"R6": {"candidates": [cand("B3", 0.4478, 0.396, True), cand("M2", 0.4478, 0.398, True), cand("M1-eps1", 0.4477, 0.227, False)], "rule": {"mia_auc_max": 0.55},
                   "literal": "B3", "tied": ["B3", "M2"], "recommended": "M2", "tie_break_used": True, "dp_alternative": {"recommended": "M1-eps1"}}}


def test_recommendation_table_and_picks():
    rec = dl.recommendation(R6_fixture())
    assert list(rec["table"]["configuration"]) == ["B3", "M2", "M1-eps1"] and rec["recommended"] == "M2" and rec["literal"] == "B3" and rec["tie_break_used"] and rec["dp_alternative"] == "M1-eps1"
    assert list(rec["table"]["eligible"]) == [True, True, False]
    assert dl.recommendation(None) is None and dl.recommendation({"R6": {}}) is None


def test_premise_is_unsupported_only_when_no_generator_helps_and_the_cvae_always_loses():
    from ppfeddata.interpret_report import premise

    def R(effect, r2):
        return {"R1": {"rows": [{"label": "B2", "reference": False, "rf": {"macro_f1": {"effect": effect, "delta": 0.001}}},
                                {"label": "B1b", "reference": True, "rf": {"macro_f1": {"effect": "better", "delta": 0.05}}}]},
                "R2": {"rows": [{"macro_f1": {"effect": e}} for e in r2]}}
    assert premise(R("none", ["worse", "worse"]))["unsupported"] is True
    assert premise(R("better", ["worse", "worse"]))["unsupported"] is False and premise(R("none", ["worse", "none"]))["unsupported"] is False
    assert premise({"R1": {"rows": []}, "R2": {"rows": []}}) is None and dl.premise(None) is None


def fake_sources(tmp, rows_removed=(12, 99)):
    man = {"rows": {"train": 10, "val": 4, "test": 6}, "label_counts": {"B": {"train": 3, "val": 2, "test": 3}, "A": {"train": 7, "val": 2, "test": 3}},
           "quota_actual": {"A_DoS": {"train": 7, "val": 2, "test": 3}}, "subclasses": {
               "A_DoS": {"split_kind": "block", "n_units": 10, "cross_block": {"cross_block_streams": 3, "rows_removed": rows_removed[0], "eligible_before": 1000}},
               "A_DDoS": {"split_kind": "block", "n_units": 10, "cross_block": {"cross_block_streams": 3, "rows_removed": rows_removed[1], "eligible_before": 1000}},
               "NORMAL": {"split_kind": "group", "n_units": 49}}}
    sch = {"n_features": 3, "label_map": {"A": 0, "B": 1}, "fit": {"split": "train", "n_rows": 2000}, "settings": {"clip_sigma": 5.0, "numeric_scale": {"time_delta_from_previous_displayed_frame": 1000.0},
           "multi_policy": "first_only"}, "audit": {"train": {"clipped_sigma": {"time_delta_from_previous_displayed_frame": 4}}},
           "blocks": [{"name": "x", "type": "numeric", "column": "x", "start": 0, "width": 1}, {"name": "f", "type": "binary", "column": "f", "start": 1, "width": 1},
                      {"name": "k", "type": "categorical", "column": "k", "start": 2, "width": 1}]}
    d = Path(tmp) / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    (d / "split_manifest_6class.json").write_text(json.dumps(man), encoding="utf-8")
    (d / "feature_schema_6class.json").write_text(json.dumps(sch), encoding="utf-8")
    return {"label_mode": "6class", "paths": {"work_dir": str(Path(tmp) / "nodata"), "shared_manifest_dir": str(d)}, "thresholds": {"mia_auc_max": 0.55, "eps_max_recommend": 5.0}}


def test_data_overview_from_the_committed_manifest_and_schema(tmp_path):
    cfg = fake_sources(tmp_path)
    cfg["compute"] = {"runs_csv": str(tmp_path / "results" / "runs.csv")}
    ov = dl.data_overview(cfg)
    assert list(ov["classes"].index) == ["A", "B"] and ov["classes"].loc["A", "train"] == 7 and ov["rows"]["test"] == 6
    sub = ov["subclasses"].set_index("sub-class")
    assert sub.loc["A_DoS", "rows dropped (streams across blocks)"] == pytest.approx(0.012) and pd.isna(sub.loc["NORMAL", "rows dropped (streams across blocks)"])
    assert ov["features"]["blocks"] == {"numeric": ["x"], "binary": ["f"], "categorical": ["k"]} and ov["features"]["n_features"] == 3
    assert dl.imbalance_ratio(ov["classes"]) == ("A", "B", pytest.approx(7 / 3))
    none = dl.data_overview({"label_mode": "6class", "paths": {"shared_manifest_dir": str(tmp_path / "empty"), "work_dir": str(tmp_path / "empty")}})
    assert none["manifest"] is False and none["classes"] is None and dl.imbalance_ratio(None) is None


# --------------------------------------------------------------------------------------------------
# limitations
# --------------------------------------------------------------------------------------------------
def test_limitations_cover_every_item_of_the_spec_list(tmp_path):
    items = lim.limitations(fake_sources(tmp_path))
    low = [(i["topic"] + " " + i["text"]).lower() for i in items]
    for what, phrase in lim.SPEC_LIST.items():
        assert any(phrase.lower() in t for t in low), what
    assert [i["id"] for i in items] == [f"L{n:02d}" for n in range(1, len(items) + 1)] and all(i["source"] for i in items)
    assert not re.search(r"\bNone\b|\bnan\b|[{}]", " ".join(i["text"] for i in items))


def test_limitation_numbers_are_read_from_the_manifest_and_the_schema(tmp_path):
    text = {i["topic"]: i["text"] for i in lim.limitations(fake_sources(tmp_path))}
    assert "2 of the 3 sub-classes" in text["Block splits inside one capture"] and "1.2 %" in text["Block splits inside one capture"] and "9.9 %" in text["Block splits inside one capture"]
    t = text["Tail of the time-gap feature"]
    assert "4 of 2,000 train rows (0.20 %)" in t and "x1000" in t and "5 sigma" in t
    assert "(2,000 rows)" in text["Normalisation uses central statistics"]
    other = {i["topic"]: i["text"] for i in lim.limitations(fake_sources(tmp_path / "o", rows_removed=(50, 70)))}
    assert "5.0 %" in other["Block splits inside one capture"] and "7.0 %" in other["Block splits inside one capture"]


def test_limitations_without_any_source_file_keep_every_item_and_show_no_number(tmp_path):
    cfg = {"label_mode": "6class", "paths": {"work_dir": str(tmp_path / "x"), "shared_manifest_dir": str(tmp_path / "y")}}
    full, bare = lim.limitations(fake_sources(tmp_path / "f")), lim.limitations(cfg)
    assert [i["topic"] for i in bare] == [i["topic"] for i in full]
    t = {i["topic"]: i["text"] for i in bare}
    assert "shares are not shown" in t["Block splits inside one capture"] and "small share" in t["Tail of the time-gap feature"] and "few capture groups" in t["Test split from few capture groups"]


def test_limitations_use_the_interpretation_and_the_ledger_when_given(tmp_path):
    cfg = fake_sources(tmp_path)
    R = {"meta": {"n_groups": 11, "n_streams": 20343}, "why": {"b0": {"rf": 0.447, "mlp": 0.328}, "c2st": {"B2": 0.9996, "M1-eps1": 0.99998}},
         "R4": {"positive_control": {"available": True, "overfit_cvae_detected": False, "overfit_cvae_auc": 0.519}}}
    t = {i["topic"]: i["text"] for i in lim.limitations(cfg, R, pd.DataFrame({"config": ["B0-rf"]}))}
    assert "11 capture groups and 20,343 TCP streams" in t["Test split from few capture groups"] and "was not run" in t["Test split from few capture groups"]
    assert "0.447 (RF) and 0.328 (MLP)" in t["Attack classes are hard to tell apart packet by packet"]
    assert "0.9996-1.0000" in t["Weak fidelity and privacy diagnostics"] and "0.519" in t["Weak fidelity and privacy diagnostics"] and "not met" in t["Weak fidelity and privacy diagnostics"]
    ran = {i["topic"]: i["text"] for i in lim.limitations(cfg, R, pd.DataFrame({"config": ["B0-rf", "A4-seed1-TAug-rf"]}))}
    assert "A4 (other split seeds) was run" in ran["Test split from few capture groups"]
    R["R4"]["positive_control"]["overfit_cvae_detected"] = True
    assert "not met" not in {i["topic"]: i["text"] for i in lim.limitations(cfg, R)}["Weak fidelity and privacy diagnostics"]


def test_packet_level_limitation_gives_the_size_of_the_correlated_units_when_measured(tmp_path):
    cfg = fake_sources(tmp_path)
    topic = "Packet-level data, record-level epsilon"
    base = {i["topic"]: i["text"] for i in lim.limitations(cfg)}[topic]
    assert "of one TCP stream and of one capture are correlated" in base and "strongly" not in base and "In the train split" not in base           # no measurement: no number
    unit = {"classes": {"NORMAL": {"rows_per_stream_mean": 1.2, "rows_per_group_max": 1277}, "DELAYED": {"rows_per_stream_mean": 1.0, "rows_per_group_max": 188},
                        "WILL": {"rows_per_stream_mean": 1.0, "rows_per_group_max": 63}}}
    R = {"meta": {"rare_classes": ["DELAYED", "WILL"]}, "guide": {"privacy_unit": unit}}
    t = {i["topic"]: i["text"] for i in lim.limitations(cfg, R)}[topic]
    assert "a stream gives 1.0-1.2 rows on average" in t and "up to 63-188 rows of a rare class" in t and "the capture, not the stream (section 9.10)" in t      # NORMAL is not a rare class


def test_render_md_has_one_bullet_per_item(tmp_path):
    items = lim.limitations(fake_sources(tmp_path))
    md = lim.render_md(items)
    assert len(md) == len(items) and all(m.startswith("- **") for m in md) and "*(" in md[0] and "*(" not in lim.render_md(items, with_source=False)[0]


# --------------------------------------------------------------------------------------------------
# README blocks
# --------------------------------------------------------------------------------------------------
READMES = ["README.md", "README.vi.md"]


def readme_text(name="README.md"):
    return (ROOT / name).read_text(encoding="utf-8")


def test_replace_block_keeps_the_rest_and_is_idempotent():
    b, e = rg.markers("x")
    text = f"before\n{b}\nold\n{e}\nafter\n"
    once = rg.replace_block(text, "x", "new line")
    assert once == f"before\n{b}\nnew line\n{e}\nafter\n" and rg.replace_block(once, "x", "new line") == once
    with pytest.raises(ValueError, match="no `<!-- BEGIN GENERATED: y"):
        rg.replace_block(text, "y", "z")
    with pytest.raises(ValueError):
        rg.replace_block(f"{e}\n{b}\n", "x", "z")                         # END before BEGIN


@pytest.mark.parametrize("name", READMES)
def test_readme_carries_every_generated_block_in_order(name):
    t = readme_text(name)
    pos = [t.index(rg.markers(n)[0]) for n in rg.BLOCKS]
    ends = [t.index(rg.markers(n)[1]) for n in rg.BLOCKS]
    assert pos == sorted(pos) and all(b < e for b, e in zip(pos, ends))


@pytest.mark.parametrize("name", READMES)
def test_readme_results_and_limitations_are_what_the_generators_write_from_the_committed_json(name, monkeypatch):
    p = ROOT / "results" / "interpretation.json"
    if not p.exists():
        pytest.skip("no results/interpretation.json")
    monkeypatch.chdir(ROOT)
    cfg, R = load_config(), json.loads(p.read_text(encoding="utf-8"))
    t = readme_text(name)

    def block(key):
        b, e = rg.markers(key)
        return t[t.index(b) + len(b):t.index(e)].strip("\n")
    assert block("results") == rg.results_block(R, cfg).strip("\n"), f"{name} results block is stale: run `ppfeddata aggregate`"
    assert block("limitations") == rg.limitations_block(lim.limitations(cfg, R, None)).strip("\n"), f"{name} limitations block is stale: run `ppfeddata aggregate`"
    assert "**Why a CVAE at all?**" in block("results") and "Not tested" in block("results") and "**Which of M1, M2, M3 for which requirement?**" in block("results")


def test_results_block_reports_the_numbers_of_the_json(tmp_path):
    p = ROOT / "results" / "interpretation.json"
    if not p.exists():
        pytest.skip("no results/interpretation.json")
    cfg, R = load_config(), json.loads(p.read_text(encoding="utf-8"))
    b = rg.results_block(R, cfg)
    r6 = R["R6"]
    assert f"**{r6['recommended']}**" in b and f"{R['why']['b0']['rf']:.3f}" in b
    for c in r6["candidates"]:
        assert re.search(rf"\| {re.escape(c['label'])} \| {c['taug_f1']:.4f} \| {c['tstr_f1']:.4f} \|", b)
    assert f"{R['meta']['n_runs']} runs" in b


def test_times_block_from_a_trial_session(tmp_path):
    art = tmp_path / "art" / "_trial"
    art.mkdir(parents=True)
    (art / "experiment_status_6class.json").write_text(json.dumps({"seeds": [0], "items": [
        {"id": "B0", "group": "matrix", "status": "done", "seconds": 30.0}, {"id": "B3", "group": "matrix", "status": "done", "seconds": 90.0},
        {"id": "A1-a0.1", "group": "extension", "status": "done", "seconds": 600.0}]}), encoding="utf-8")
    rep = tmp_path / "res" / "reports"
    rep.mkdir(parents=True)
    (rep / "compute_budget.md").write_text("# x\n\nAuto. Hardware/software: Python 3.13, CPU, 10 threads. Commit `abc`.\n", encoding="utf-8")
    cfg = {"label_mode": "6class", "compute": {"artifacts_dir": str(tmp_path / "art"), "runs_csv": str(tmp_path / "res" / "runs.csv")}}
    t = rg.times_block(cfg)
    assert "**2.0 min**" in t and "on Python 3.13, CPU, 10 threads" in t and "| B3 | done | 1.5 |" in t and "A1-a0.1" not in t              # the extension is not part of the matrix
    assert "No trial session found" in rg.times_block({**cfg, "compute": {**cfg["compute"], "artifacts_dir": str(tmp_path / "nowhere")}})


def test_update_readme_rewrites_the_blocks_and_only_them(tmp_path):
    cfg = {**fake_sources(tmp_path), "compute": {"artifacts_dir": str(tmp_path / "art"), "runs_csv": str(tmp_path / "res" / "runs.csv")}}
    text = "# T\nintro\n" + "".join(f"\n{rg.markers(n)[0]}\nold {n}\n{rg.markers(n)[1]}\n" for n in rg.BLOCKS) + "\noutro\n"
    p = tmp_path / "README.md"
    p.write_text(text, encoding="utf-8")
    done = rg.update_readme(p, cfg, None)                                    # no interpretation: the results block is left as it is
    new = p.read_text(encoding="utf-8")
    assert done == ["libraries", "times", "limitations"] and "old results" in new and "old libraries" not in new and "old times" not in new and "old limitations" not in new
    assert new.startswith("# T\nintro\n") and new.endswith("\noutro\n") and "10 limitations" not in new and "limitations (the list of section 10" in new


@pytest.mark.parametrize("name", READMES)
def test_every_command_of_the_readme_parses_with_the_real_cli(name):
    cmds = re.findall(r"python -m ppfeddata\.cli ([^`\n]+)", readme_text(name))
    assert len(cmds) >= 25
    parser = build_parser()
    for c in cmds:
        argv = shlex.split(re.sub(r"<[^>]*>", "1", c))
        if argv != ["--help"]:                                                 # --help prints the usage and exits with status 0 by design
            parser.parse_args(argv)                                            # raises SystemExit on an unknown command or option


@pytest.mark.parametrize("name", READMES)
def test_readme_links_point_to_files_that_exist(name):
    for target in re.findall(r"\]\(([^)#\s]+)\)", readme_text(name)):
        if not target.startswith("http"):
            assert (ROOT / target).exists(), target


def test_report_links_use_forward_slashes_and_point_to_committed_files():
    """GitHub does not read ..\\figures\\x.png (a Windows path written by Path on Windows): the image is not shown."""
    for md in sorted((ROOT / "results" / "reports").glob("*.md")):
        for target in re.findall(r"\]\(([^)#\s]+)\)", md.read_text(encoding="utf-8")):
            if target.startswith("http"):
                continue
            assert "\\" not in target, f"{md.name}: {target}"
            assert (md.parent / target).exists(), f"{md.name}: {target}"


@pytest.mark.parametrize("name", READMES)
def test_readme_has_a_full_command_for_every_cli_command(name):
    named = {c.split()[0] for c in re.findall(r"python -m ppfeddata\.cli ([^`\n]+)", readme_text(name))}
    every = set(build_parser()._subparsers._group_actions[0].choices)
    assert every <= named, sorted(every - named)


def test_the_two_readmes_say_the_same_thing_in_two_languages():
    en, vi = readme_text("README.md"), readme_text("README.vi.md")
    assert "[Tiếng Việt](README.vi.md)" in en.split("\n\n")[1] and "[English](README.md)" in vi.split("\n\n")[1]                 # each points to the other at the top
    cmd = lambda s: [re.sub(r"<[^>]*>", "<x>", c) for c in re.findall(r"python -m ppfeddata\.cli ([^`\n]+)", s)]               # noqa: E731  (the words inside <...> are translated)
    assert cmd(en) == cmd(vi), "the commands (and their order) differ between the two READMEs"
    assert len(re.findall(r"^## ", en, flags=re.M)) == len(re.findall(r"^## ", vi, flags=re.M))
    assert re.findall(r"BEGIN GENERATED: (\w+)", en) == re.findall(r"BEGIN GENERATED: (\w+)", vi) == list(rg.BLOCKS)
    paths = lambda s: re.findall(r"`((?:configs|results|src|demo)/[^`]+)`", s.split("<!-- BEGIN GENERATED: results -->")[0])   # noqa: E731
    assert paths(en) and paths(en) == paths(vi), "the files the two READMEs name before the results block differ"


def test_update_readmes_writes_every_readme_that_exists(tmp_path):
    cfg = {**fake_sources(tmp_path), "compute": {"artifacts_dir": str(tmp_path / "art"), "runs_csv": str(tmp_path / "res" / "runs.csv")}}
    text = "# T\n" + "".join(f"\n{rg.markers(n)[0]}\nold {n}\n{rg.markers(n)[1]}\n" for n in rg.BLOCKS)
    a, b, missing = tmp_path / "README.md", tmp_path / "README.vi.md", tmp_path / "README.xx.md"
    a.write_text(text, encoding="utf-8")
    b.write_text(text, encoding="utf-8")
    done = rg.update_readmes(cfg, None, None, paths=(a, b, missing))
    assert done == [f"{n}: {k}" for n in ("README.md", "README.vi.md") for k in ("libraries", "times", "limitations")]
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8") and not missing.exists()


# --------------------------------------------------------------------------------------------------
# reproducibility bundle
# --------------------------------------------------------------------------------------------------
FREEZE = ["torch==2.14.1", "-e git+https://github.com/someone/PP-FedData.git@abc#egg=ppfeddata", "Flask==3.0", "numpy==2.3.2", "ppfeddata @ file:///C:/Users/x/PP-FedData"]


def test_sanitize_freeze_hides_the_local_install_in_place_and_keeps_pips_order():
    out = pk.sanitize_freeze(FREEZE)
    assert out == ["torch==2.14.1", pk.EDITABLE_NOTE, "Flask==3.0", "numpy==2.3.2"]            # one note where the first editable line was, the second one dropped
    assert not any("file:" in x or "git+" in x for x in out) and pk.sanitize_freeze(["a==1"]) == ["a==1"]


def test_manifest_paths_are_relative_to_the_repository(tmp_path):
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    f = root / "results" / "a.csv"
    f.write_text("x", encoding="utf-8")
    assert pk._rel(f, root) == "results/a.csv" and pk._rel(tmp_path / "outside" / "b.csv", root).endswith("outside/b.csv")


def make_repo(tmp):
    root = Path(tmp) / "repo"
    (root / "configs" / "exp").mkdir(parents=True)
    (root / "configs" / "default.yaml").write_text("seeds: [0]\n", encoding="utf-8")
    (root / "configs" / "local.yaml").write_text("paths: {raw_root: 'C:/private/raw'}\n", encoding="utf-8")
    (root / "configs" / "exp" / "B0.yaml").write_text("id: B0\n", encoding="utf-8")
    (root / "requirements-lock.txt").write_text("old\n", encoding="utf-8")
    cfg = fake_sources(tmp)
    res = Path(tmp) / "results"
    (res / "reports").mkdir(parents=True)
    (res / "runs.csv").write_text("a\n1\n", encoding="utf-8")
    (res / "reports" / "final_report.md").write_text("# r\n", encoding="utf-8")
    cfg.update(compute={"runs_csv": str(res / "runs.csv"), "artifacts_dir": str(Path(tmp) / "art")}, seeds=[0, 1, 2])
    return root, cfg, res


def test_package_writes_the_bundle(tmp_path):
    root, cfg, res = make_repo(tmp_path)
    r = pk.package(cfg, root=root, freeze=FREEZE)
    out = res / "repro"
    assert (out / "pip_freeze.txt").read_text(encoding="utf-8") == (root / "requirements-lock.txt").read_text(encoding="utf-8") and r["lock_updated"]
    assert (out / "configs" / "default.yaml").exists() and (out / "configs" / "exp" / "B0.yaml").exists() and not (out / "configs" / "local.yaml").exists()
    env = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    assert env["label_mode"] == "6class" and env["seeds"] == [0, 1, 2] and env["libraries"]["numpy"] and env["python"]
    man = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    ledger = next(k for k in man["files"] if k.endswith("runs.csv"))
    assert man["files"][ledger]["sha256"] == pk.sha256(res / "runs.csv") and man["files"][ledger]["bytes"] == len((res / "runs.csv").read_bytes())
    assert any(k.endswith("split_manifest_6class.json") for k in man["files"]) and any(k.endswith("feature_schema_6class.json") for k in man["files"])
    assert any(k.endswith("summary.csv") for k in man["missing"]) and any(k.endswith("interpretation.json") for k in man["missing"])           # reported, not invented
    assert not any("private" in json.dumps(v) for v in man["files"].values())


def test_package_copies_the_feature_schema_next_to_the_split_manifest(tmp_path):
    root, cfg, res = make_repo(tmp_path)
    (tmp_path / "manifests" / "feature_schema_6class.json").unlink()
    proc = tmp_path / "nodata" / "processed" / "6class"
    proc.mkdir(parents=True)
    (proc / "feature_schema.json").write_text('{"n_features": 3}', encoding="utf-8")
    pk.package(cfg, root=root, freeze=FREEZE)
    assert json.loads((tmp_path / "manifests" / "feature_schema_6class.json").read_text(encoding="utf-8")) == {"n_features": 3}


def test_package_can_leave_the_lock_file_alone_and_reports_a_missing_schema(tmp_path):
    root, cfg, res = make_repo(tmp_path)
    (tmp_path / "manifests" / "feature_schema_6class.json").unlink()
    r = pk.package(cfg, root=root, freeze=FREEZE, update_lock=False)
    assert (root / "requirements-lock.txt").read_text(encoding="utf-8") == "old\n" and not r["lock_updated"]
    assert any("feature_schema.json" in m for m in r["missing"])


def test_cli_knows_the_new_commands():
    p = build_parser()
    assert p.parse_args(["demo", "--port", "8700", "--headless"]).port == 8700
    a = p.parse_args(["package", "--no-update-lock", "--out-dir", "x"])
    assert a.command == "package" and a.no_update_lock and a.out_dir == "x"


# --------------------------------------------------------------------------------------------------
# the demo pages (Streamlit's own test runner)
# --------------------------------------------------------------------------------------------------
PAGES = ["1. Dữ liệu", "2. Kết quả", "3. Đánh đổi và khuyến nghị", "4. Tạo mẫu", "5. Mô hình đe dọa và hạn chế"]


def app():
    st_test = pytest.importorskip("streamlit.testing.v1")
    return st_test.AppTest.from_file(str(ROOT / "demo" / "app.py"), default_timeout=180)


@pytest.mark.parametrize("page", PAGES)
def test_every_demo_page_runs_without_an_exception(page, monkeypatch):
    monkeypatch.chdir(ROOT)
    at = app()
    at.run()
    assert not at.exception
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error
    assert any(page.split(". ")[1].split()[0] in h.value for h in at.header) or page.startswith("1")        # the page's own header is shown


def test_demo_sidebar_states_the_main_conclusion_from_the_json(monkeypatch):
    p = ROOT / "results" / "interpretation.json"
    if not p.exists():
        pytest.skip("no results/interpretation.json")
    monkeypatch.chdir(ROOT)
    at = app()
    at.run()
    R = json.loads(p.read_text(encoding="utf-8"))
    from ppfeddata.interpret_report import premise
    pm = premise(R)
    side = " ".join(m.value for m in at.sidebar.markdown)
    assert ("không cải thiện IDS" in side) == bool(pm and pm["unsupported"])


def test_demo_page_3_shows_which_configuration_for_which_requirement(monkeypatch):
    rep = ROOT / "results" / "reports" / "final_report.md"
    if not rep.exists() or "### 9.10 " not in rep.read_text(encoding="utf-8"):
        pytest.skip("the committed report has no section 9.10")
    monkeypatch.chdir(ROOT)
    at = app()
    at.run()
    at.sidebar.radio(key="page").set_value(PAGES[2]).run()
    assert not at.exception and not at.error
    assert any("Cấu hình nào cho yêu cầu nào" in s.value for s in at.subheader)
    text = " ".join(m.value for m in at.markdown)
    assert "Requirement to configuration" in text and "What in the IoT / MQTT data changes the choice" in text
    assert not any(e.label.startswith("9.10") for e in at.expander)                                       # shown once, under its own subheader


@pytest.mark.skipif(not HAVE_REAL, reason="needs the trained models and data/processed")
def test_demo_generates_samples_from_the_real_models(monkeypatch):
    monkeypatch.chdir(ROOT)
    at = app()
    at.run()
    at.sidebar.radio(key="page").set_value(PAGES[3]).run()
    assert not at.exception
    next(b for b in at.button if b.label == "Sinh mẫu").click().run()
    assert not at.exception and not at.error
    s = at.session_state["samples"]
    assert len(s.table) == 6 * 100 and s.generator.label and set(s.table["label"]) == {"NORMAL", "BCF", "DELAYED", "SYN", "INVALID", "WILL"}


# --------------------------------------------------------------------------------------------------
# installation files (the README tells a reader to install them: they must name what the code needs)
# --------------------------------------------------------------------------------------------------
IMPORT_TO_DISTRIBUTION = {"sklearn": "scikit-learn", "imblearn": "imbalanced-learn", "yaml": "pyyaml"}
OPTIONAL = {"polars"}                                           # listed by the spec as optional ("to read faster"); no module imports it


def requirement_names(path):
    names = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names[re.split(r"[=<>~! @]", line)[0].lower().replace("_", "-")] = line
    return names


def test_requirements_name_every_third_party_package_the_code_imports():
    import ast
    import sys
    std, found = set(sys.stdlib_module_names), set()
    for root in ("src", "demo"):
        for p in (ROOT / root).rglob("*.py"):
            for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(n, ast.Import):
                    found |= {a.name.split(".")[0] for a in n.names}
                elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                    found.add(n.module.split(".")[0])
    needed = {IMPORT_TO_DISTRIBUTION.get(m, m).lower() for m in found - std - {"ppfeddata"}}
    have = set(requirement_names(ROOT / "requirements.txt"))
    assert needed <= have, sorted(needed - have)
    assert "ray" in have, "Flower's simulation backend imports ray; flwr[simulation] does not install it on Windows with Python 3.13 (SPEC_DEVIATIONS 8.1)"


def test_the_lock_file_pins_every_requirement_to_one_version():
    req, lock = requirement_names(ROOT / "requirements.txt"), requirement_names(ROOT / "requirements-lock.txt")
    for name in set(req) - OPTIONAL:
        assert name in lock and "==" in lock[name], name
    assert not any(v.startswith("-e") or "file:" in v or "git+" in v for v in lock.values())


def test_demo_on_a_clone_without_results_data_or_artifacts_shows_hints_and_no_error(tmp_path, monkeypatch):
    """A fresh clone has no data/, artifacts/ or summary.csv: every page must say what to run, none may raise."""
    import sys

    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["compute"] = {**cfg["compute"], "artifacts_dir": str(tmp_path / "art"), "runs_csv": str(tmp_path / "res" / "runs.csv")}
    cfg["paths"] = {**cfg["paths"], "work_dir": str(tmp_path / "data"), "shared_manifest_dir": str(tmp_path / "manifests")}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", ["app.py", "--config", str(path)])
    at = app()
    at.run()
    assert not at.exception
    hints = {0: "split_manifest", 1: "summary.csv", 2: "interpretation.json", 3: "artifacts/"}
    for i, page in enumerate(PAGES):
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception, (page, [str(e.value) for e in at.exception])
        assert not at.error, (page, [e.value for e in at.error])
        if i in hints:
            assert any(hints[i] in w.value for w in at.warning), (page, [w.value for w in at.warning])
    assert any("limitations" in str(m.value) or "L01" in str(m.value) for m in at.markdown)               # page 5 still lists the limitations without any source file
