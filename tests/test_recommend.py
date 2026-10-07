"""Optimisation O4: `recommend` filters the scorecard by a deployment requirement and picks from the front."""
import json
import math

import pandas as pd

from ppfeddata import recommend as rc
from ppfeddata import scorecard as sc
from tests.test_scorecard import RARE, _gen, _summary


def _card():
    extra = _gen("MGd-eps5", 0.38, 0.38, eps=5.0, bytes_=3e5) + _gen("MGl-eps5", 0.33, 0.33, eps=5.0, bytes_=3e5)
    for r in extra:
        r.update(rounds_mean=3.0, dp_eps_one_honest_mean=12.4 if r["config"].startswith("MGd") else 5.0)
    s = pd.concat([_summary(), pd.DataFrame(extra)], ignore_index=True)
    for c, n in (("B3", 30.0), ("M2", 30.0), ("M1d-t21", 30.0), ("M3d-t21", 30.0)):
        s.loc[s["config"].str.startswith(c), "rounds_mean"] = n
    return sc._from_json(json.loads(json.dumps(sc._jsonable(sc.build(s, rare=RARE)))))       # through JSON, as `run` reads it


def test_trusted_lab_takes_the_best_detection_without_dp():
    r = rc.recommend(_card(), rc.Requirement("lab"))
    assert r["best"]["label"] in ("B3", "M2") and set(r["ties_with_best"]) <= {"B3", "M2"}


def test_untrusted_server_excludes_plain_fedavg_and_keeps_local_dp():
    r = rc.recommend(_card(), rc.Requirement("x", trust_server=False, max_eps=5.0))
    labels = {x["label"] for x in r["meets"]}
    assert "B3" not in labels and "M2" not in labels                        # M2 has no DP, epsilon 5 required
    assert {"MGd-eps5", "MGl-eps5", "M1-eps5"} <= labels
    assert r["best"]["label"] == "MGd-eps5" and r["command"] == "python -m ppfeddata.cli tune-marginal --final --eps 5"
    assert any("server sees" in w for x in rc.recommend(_card(), rc.Requirement("y", trust_server=False))["rejected"] for w in x["rejected_because"])


def test_colluding_clients_use_the_one_honest_client_epsilon():
    r = rc.recommend(_card(), rc.Requirement("x", trust_server=False, trust_clients=False, max_eps=5.0))
    rej = {x["label"]: x["rejected_because"] for x in r["rejected"]}
    assert "MGd-eps5" in rej and "12.4" in rej["MGd-eps5"][0]
    assert r["best"]["label"] == "MGl-eps5"                                  # local DP holds whatever the other clients do


def test_bandwidth_and_rare_recall_limits():
    card = _card()
    r = rc.recommend(card, rc.Requirement("x", max_mb_total=10.0))
    assert all(x["mb_total"] <= 10.0 for x in r["meets"] if not math.isnan(x["mb_total"]))
    assert {x["label"] for x in r["meets"]} >= {"MGd-eps5"} and "B3" not in {x["label"] for x in r["meets"]}     # 4 MB x 30 rounds
    r2 = rc.recommend(card, rc.Requirement("x", min_rare_recall=0.2))
    assert all(x["rare_recall"] >= 0.2 for x in r2["meets"])


def test_files_are_written(tmp_path):
    (tmp_path / "scorecard.json").write_text(json.dumps(sc._jsonable(_card())), encoding="utf-8")
    out = rc.run({"compute": {"runs_csv": str(tmp_path / "runs.csv")}}, out_dir=tmp_path)
    assert len(out) == len(rc.SCENARIOS)
    md = (tmp_path / "reports" / "recommend.md").read_text(encoding="utf-8")
    assert "## consortium" in md and "Recommended:" in md
    html = (tmp_path / "reports" / "recommend.html").read_text(encoding="utf-8")
    assert "__ROWS__" not in html and "MGd-eps5" in html and json.loads((tmp_path / "recommend.json").read_text(encoding="utf-8"))
