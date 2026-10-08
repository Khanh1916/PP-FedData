"""Follow-up round P1: group-level DP (stream / capture units) and the honest-client threshold of FedDP-Marginal."""
import numpy as np
import pandas as pd
import pytest

from ppfeddata import group_dp as gd
from ppfeddata.models import marginal as mg


def test_units_are_held_whole_by_one_client_and_capped():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 40)
    streams = rng.integers(0, 8, len(groups))
    y = groups % 3
    for unit in gd.UNITS:
        u = gd.unit_ids(groups, streams, unit)
        parts = gd.unit_partition(y, u, 4, 0.5, seed=1, min_size=50)
        owner = np.empty(len(y), int)
        for k, p in enumerate(parts):
            owner[p] = k
        assert sorted(np.concatenate(parts).tolist()) == list(range(len(y)))                       # every row once
        assert (pd.Series(owner).groupby(u).nunique() == 1).all()                                   # a unit is never split
        c = gd.cap(parts[0], u, 3, seed=2)
        assert set(c) <= set(parts[0]) and pd.Series(u[c]).value_counts().max() <= 3
    assert gd.unit_ids(groups, streams, "capture").max() == 29


def test_sensitivity_scales_with_the_rows_per_unit():
    sizes, split = [37, 666, 37], (0.3, 0.1, 0.6)
    g1, g4 = mg.noise_scales(5.0, 1e-5, 37, split, "gaussian", 1), mg.noise_scales(5.0, 1e-5, 37, split, "gaussian", 4)
    assert all(g4[s] == pytest.approx(4 * g1[s]) for s in g1)                                      # std scales with m
    s1, s4 = mg.noise_scales(5.0, 1e-5, 37, split, "skellam", 1), mg.noise_scales(5.0, 1e-5, 37, split, "skellam", 4)
    assert mg.eps_at(s4, 37, 1e-5, "skellam", rows=4)["eps"] == pytest.approx(5.0, rel=1e-4)
    assert mg.eps_at(s1, 37, 1e-5, "skellam", rows=4)["eps"] > 5.0                                  # the record-level noise does not cover 4 rows
    assert all(s4[s] > 10 * s1[s] for s in s1)                                                       # variance about m^2 times
    assert sizes


def test_honest_threshold_holds_the_target_with_t_clients():
    sc = mg.noise_scales(5.0, 1e-5, 37, (0.3, 0.1, 0.6), "skellam")
    info = mg.privacy_info(sc, 37, 1e-5, 1 / 3, "skellam", K=5)
    curve = mg.eps_honest_curve(sc, 37, 1e-5, 5, "skellam", share=1 / 3)
    assert info["eps"] == pytest.approx(5.0, rel=1e-4) and curve[3] == pytest.approx(5.0, rel=1e-4)
    assert info["eps_all_honest"] < 5.0 < info["eps_one_honest"] and curve[1] > curve[2] > curve[3] > curve[5]
    base = mg.privacy_info(sc, 37, 1e-5, 1 / 5, "skellam", K=5)                                      # t = K: as before
    assert base["eps_all_honest"] == pytest.approx(base["eps"]) and base["eps_one_honest"] > info["eps_one_honest"]


def test_fit_with_rows_and_threshold_reports_them():
    from tests.test_models import make_schema, synthetic_data
    s = {**make_schema(), "label_map": {"A": 0, "B": 1, "C": 2}}
    X, y = synthetic_data(1500)
    parts = [np.arange(0, 500), np.arange(500, 1000), np.arange(1000, 1500)]
    m = mg.fit(X, y, parts, s, 5.0, 1e-5, mechanism="skellam", rows=2, honest_t=2)
    assert m.info["unit_rows"] == 2 and m.info["honest_t"] == 2 and m.info["noise_share"] == 0.5
    assert m.info["eps"] == pytest.approx(5.0, rel=1e-4) and m.info["eps_all_honest"] < 5.0


def test_limitation_text_from_the_p1_results():
    from ppfeddata import limitations as lim
    row = lambda c, u, e, f, **kw: {"config": c, "unit": u, "eps": e, "tstr_f1": f, **kw}        # noqa: E731
    pu = {"units": [row("MGs-eps5", "packet", 5.0, 0.41), row("MGs-strm2-eps5", "stream", 5.0, 0.37), row("MGs-cap25-eps5", "capture", 5.0, 0.20, rows_kept=3675)],
          "thresholds": [{**row("MGs-eps5", None, 5.0, 0.41), "t": 5}, {**row("MGs-t3-eps5", None, 5.0, 0.41), "t": 3}, {**row("MGs-t2-eps5", None, 5.0, 0.39), "t": 2}]}
    unit, thr = lim._group_dp_text(pu)
    assert "costs 0.04-0.04" in unit and "0.370 vs 0.410 at epsilon 5" in unit and "3,675 rows kept" in unit
    assert "t = 3 changed TSTR macro-F1 by +0.000 to +0.000" in thr and "t = 2 changed TSTR macro-F1 by -0.020" in thr and thr.endswith("across epsilon 5.")
    assert lim._group_dp_text({}) == ("", "")


def test_limitations_read_the_p1_results(tmp_path):
    import json

    from ppfeddata import limitations as lim
    from tests.test_demo import fake_sources
    row = lambda c, u, e, f, **kw: {"config": c, "unit": u, "eps": e, "tstr_f1": f, **kw}        # noqa: E731
    pu = {"units": [row("MGs-eps5", "packet", 5.0, 0.41), row("MGs-strm2-eps5", "stream", 5.0, 0.37), row("MGs-cap25-eps5", "capture", 5.0, 0.20, rows_kept=3675)],
          "thresholds": [{**row("MGs-eps5", None, 5.0, 0.41), "t": 5}, {**row("MGs-t3-eps5", None, 5.0, 0.41), "t": 3}]}
    cfg = fake_sources(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "privacy_units.json").write_text(json.dumps(pu), encoding="utf-8")
    t = {i["topic"]: (i["text"], i["source"]) for i in lim.limitations({**cfg, "compute": {"runs_csv": str(tmp_path / "results" / "runs.csv")}})}
    honest, unit = t["Distributed DP assumes honest clients"], t["Packet-level data, record-level epsilon"]
    assert honest[0].endswith("across epsilon 5.") and "P1.2" in honest[1] and "mia_auc_max" not in honest[0]
    assert "3,675 rows kept" in unit[0] and "privacy_units.md" in unit[1]
    bare = {i["topic"]: i["text"] for i in lim.limitations(cfg)}
    assert "threshold t" not in bare["Distributed DP assumes honest clients"] and "whole unit" not in bare["Packet-level data, record-level epsilon"]


def test_unit_rows_in_the_scorecard_and_the_unit_requirement():
    import json

    from ppfeddata import recommend as rc
    from ppfeddata import scorecard as sc
    from tests.test_scorecard import RARE, _gen, _summary
    rows = _gen("MGs-strm1-eps5", 0.40, 0.40, eps=5.0) + _gen("MGs-cap50-eps10", 0.35, 0.35, eps=10.0) + _gen("MGs-t3-eps5", 0.39, 0.39, eps=5.0)
    card = sc._from_json(json.loads(json.dumps(sc._jsonable(sc.build(pd.concat([_summary(), pd.DataFrame(rows)], ignore_index=True), rare=RARE)))))
    by = {r["label"]: r for r in card["rows"]}
    assert by["MGs-strm1-eps5"]["unit"] == "stream" and by["MGs-cap50-eps10"]["unit"] == "capture" and by["MGs-t3-eps5"]["honest_t"] == 3
    assert by["M1-eps5"]["unit"] == "packet" and not isinstance(by["B3"]["unit"], str)            # no DP: no unit (None, NaN after JSON)
    r = rc.recommend(card, rc.Requirement("x", False, True, 10.0, unit="capture"))
    assert {x["label"] for x in r["meets"]} == {"MGs-cap50-eps10"}
    assert "privacy-units" in rc.command("MGs-strm2-eps5", 5.0) and "privacy-units" in rc.command("MGr-t3-eps1", 1.0)
    assert "tune-marginal" in rc.command("MGs-eps5", 5.0)
