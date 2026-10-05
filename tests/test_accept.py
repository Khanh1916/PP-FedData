"""Phase 13 tests: the acceptance checks (Definition of Done) decide PASS / FAIL / PARTIAL from the evidence they are given, and a missing input is never a pass."""
from pathlib import Path

import numpy as np
import pandas as pd

from ppfeddata import acceptance as ac
from ppfeddata import limitations as lim
from tests.test_demo import fake_sources

ROOT = Path(__file__).resolve().parents[1]
QUOTA = {"A": [7, 2, 3], "B": [3, 2, 3]}
MAN = {"rows": {"train": 10, "val": 4, "test": 6}, "label_counts": {"A": {"train": 7, "val": 2, "test": 3}, "B": {"train": 3, "val": 2, "test": 3}},
       "streams_in_multiple_splits": {}, "subclasses": {"A_DoS": {"units": {"train": [{"unit_id": "f1"}], "val": [{"unit_id": "f2"}], "test": [{"unit_id": "f3"}]}}}}


def test_phase_tests_need_a_test_file_per_phase(tmp_path):
    ok = ac.check_phase_tests(ROOT / "tests")
    assert ok.status == "PASS" and "14 phases" in ok.evidence and "pass / fail result" in ok.evidence
    bad = ac.check_phase_tests(tmp_path)
    assert bad.status == "FAIL" and "test_demo.py" in bad.evidence


def test_counts_must_equal_the_quotas():
    assert ac.check_counts(MAN, QUOTA).status == "PASS"
    wrong = {**MAN, "label_counts": {**MAN["label_counts"], "B": {"train": 3, "val": 2, "test": 4}}}
    r = ac.check_counts(wrong, QUOTA)
    assert r.status == "FAIL" and "B: manifest [3, 2, 4] vs quota [3, 2, 3]" in r.evidence
    assert ac.check_counts({**MAN, "rows": {"train": 10, "val": 4, "test": 7}}, QUOTA).status == "FAIL"
    assert ac.check_counts(None, QUOTA).status == "FAIL"
    assert ac.check_counts({"rows": MAN["rows"]}, QUOTA).status == "FAIL"                               # no label_counts: not a pass


def test_group_split_rules():
    ids = {"train": np.array(["a", "b"]), "val": np.array(["c"]), "test": np.array(["d"])}
    assert ac.check_group_split(MAN, ids, True, []).status == "PASS"
    assert ac.check_group_split(MAN, {**ids, "val": np.array(["c", "a"])}, True, []).status == "FAIL"                   # a group in train and val
    assert ac.check_group_split(MAN, ids, None, []).status == "PARTIAL"                                                   # the test hash could not be checked
    assert ac.check_group_split(MAN, ids, False, []).status == "FAIL" and "DOES NOT match" in ac.check_group_split(MAN, ids, False, []).evidence
    assert ac.check_group_split(MAN, ids, True, ["tune.py"]).status == "FAIL"                                              # the test split is read while tuning
    assert ac.check_group_split({**MAN, "streams_in_multiple_splits": {"s1": ["train", "val"]}}, ids, True, []).status == "FAIL"
    assert ac.check_group_split(MAN, None, True, []).status == "PASS"                                                      # falls back to the units of the manifest
    shared = {**MAN, "subclasses": {"A_DoS": {"units": {"train": [{"unit_id": "f1"}], "val": [{"unit_id": "f1"}], "test": []}}}}
    assert ac.check_group_split(shared, None, True, []).status == "FAIL"
    assert ac.check_group_split(None, None, None, []).status == "FAIL"


LEAK = "## C1 presence\n## C2 NA only\n## C3 single feature\n## C4 split gap\n## C5 DoS vs DDoS\n"
CFG = {"harmonize": {"g1_log": {"approved_on": "2026-10-03", "decisions": ["x"]}, "g2_log": {"approved_on": "2026-10-03", "decisions": ["y"]}}}


def test_leakage_report_needs_all_checks_and_both_decision_logs():
    assert ac.check_leakage_report(LEAK, CFG).status == "PASS"
    assert ac.check_leakage_report(LEAK.replace("C4", "C9"), CFG).status == "FAIL"
    assert ac.check_leakage_report(None, CFG).status == "FAIL"
    no_g2 = {"harmonize": {"g1_log": CFG["harmonize"]["g1_log"]}}
    r = ac.check_leakage_report(LEAK, no_g2)
    assert r.status == "FAIL" and "G2: no" in r.evidence
    assert ac.check_leakage_report(LEAK, {"g1_log": CFG["harmonize"]["g1_log"], "g2_log": CFG["harmonize"]["g2_log"]}).status == "PASS"       # found at any depth
    assert ac.check_leakage_report(LEAK, {"harmonize": {"g1_log": {"approved_on": "d"}, "g2_log": CFG["harmonize"]["g2_log"]}}).status == "FAIL"  # no decisions recorded


def test_seeds_must_all_be_done():
    done = {"B0": {0: True, 1: True, 2: True}, "B3": {0: True, 1: True, 2: False}}
    r = ac.check_seeds(done, [0, 1, 2])
    assert r.status == "FAIL" and "B3 seed 2" in r.evidence
    assert ac.check_seeds({"B0": {0: True, 1: True, 2: True}}, [0, 1, 2]).status == "PASS"
    assert ac.check_seeds({"B0": {0: True}}, [0, 1]).status == "FAIL"                                  # a seed that was never looked at is missing


def test_epsilon_must_reach_the_target_and_not_exceed_it():
    df = pd.DataFrame({"dp_target_eps": [1.0, 5.0, 5.0, np.nan], "dp_eps_max": [0.997, 4.999, 5.0, np.nan]})
    r = ac.check_epsilon(df, {"status": "not triggered"})
    assert r.status == "PASS" and "3 DP rows" in r.evidence and "not triggered" in r.evidence
    assert ac.check_epsilon(pd.DataFrame({"dp_target_eps": [5.0], "dp_eps_max": [5.3]}), None).status == "FAIL"        # above the target by more than 2 %
    assert ac.check_epsilon(pd.DataFrame({"dp_target_eps": [5.0], "dp_eps_max": [2.0]}), None).status == "FAIL"        # far below it: not "matching"
    assert ac.check_epsilon(None, None).status == "FAIL" and ac.check_epsilon(pd.DataFrame({"x": [1]}), None).status == "FAIL"


def test_secagg_check_t_sa1():
    assert ac.check_secagg({"tsa1": {"agg_err_max": 3e-5, "error_bound": 4e-5}}).status == "PASS"
    assert ac.check_secagg({"tsa1": {"agg_err_max": 5e-5, "error_bound": 4e-5}}).status == "FAIL"
    assert ac.check_secagg(None).status == "FAIL" and ac.check_secagg({}).status == "FAIL"


def test_positive_control_fails_when_the_attack_misses_an_overfitted_cvae():
    pc = {"available": True, "overfit_cvae_auc": 0.519, "overfit_cvae_detected": False, "copier_auc": {"0.0": 0.977}, "copier_detected_at_zero_noise": True, "threshold": 0.55}
    r = ac.check_positive_control({"R4": {"positive_control": pc}})
    assert r.status == "FAIL" and "NOT MET for the CVAE" in r.evidence and "0.519" in r.evidence and "0.977" in r.evidence
    assert ac.check_positive_control({"R4": {"positive_control": {**pc, "overfit_cvae_detected": True, "overfit_cvae_auc": 0.7}}}).status == "PASS"
    assert ac.check_positive_control({"R4": {"positive_control": {"available": False}}}).status == "FAIL" and ac.check_positive_control(None).status == "FAIL"


def test_report_must_be_auto_generated_and_reproducible():
    text = "# r\n\nAuto-generated by `ppfeddata aggregate` from the ledger\n"
    assert ac.check_report(text, text, 0.0).status == "PASS"
    r = ac.check_report(text, text + "x", 0.0)
    assert r.status == "FAIL" and "DIFFERENT" in r.evidence
    assert ac.check_report(text, None, 0.0).status == "PARTIAL" and "--no-regenerate" in ac.check_report(text, None, None).evidence
    assert ac.check_report("# r\n", "# r\n", 0.0).status == "FAIL" and ac.check_report(None, None, None).status == "FAIL"


def test_compute_budget_needs_the_measured_table():
    ok = "## Measured on one client-sized dataset\n\n| run | rows |\n|---|---|\n| dp_9000 | 9000 |\n\n**DP-SGD costs 8.99x the plain epoch**\n"
    r = ac.check_compute_budget(ok)
    assert r.status == "PASS" and "8.99x" in r.evidence
    assert ac.check_compute_budget("# nothing measured\n").status == "FAIL" and ac.check_compute_budget(None).status == "FAIL"


def test_limitations_must_be_in_the_list_the_report_and_the_readme(tmp_path):
    items = lim.limitations(fake_sources(tmp_path))
    everything = "\n".join(i["topic"] for i in items)
    assert ac.check_limitations(items, everything, everything, lim.SPEC_LIST).status == "PASS"
    r = ac.check_limitations(items, everything, everything.replace("One dataset", ""), lim.SPEC_LIST)
    assert r.status == "FAIL" and f"{len(items) - 1} in the README" in r.evidence
    assert ac.check_limitations(items, "", "", lim.SPEC_LIST).status == "FAIL"                           # listed, but not in the report or the README
    short = [i for i in items if i["topic"] != "Protocol filter"]
    r = ac.check_limitations(short, everything, everything, lim.SPEC_LIST)
    assert r.status == "FAIL" and "missing: ['protocol filter']" in r.evidence


def test_readme_check_lists_what_is_missing():
    full = "## Cài đặt\n## Dữ liệu\n## Chạy từng Phase\n## Tái lập\n## Cấu trúc thư mục\n## Thời gian chạy tham khảo\n## Hạn chế\nrequirements-lock.txt\n"
    r = ac.check_readme(full, {"a", "b"}, {"a", "b"})
    assert r.status == "PARTIAL" and "clean machine" in r.evidence                                  # structure is complete; 'enough' needs a human
    r = ac.check_readme(full, {"a"}, {"a", "b"})
    assert r.status == "FAIL" and "command b" in r.evidence
    assert ac.check_readme("# x\n", set(), set()).status == "FAIL" and ac.check_readme(None, set(), set()).status == "FAIL"


def test_render_counts_the_statuses_and_keeps_the_table_well_formed():
    items = [ac.Item("D1", "c | with a pipe", "PASS", "e | x"), ac.Item("D2", "c2", "FAIL", "bad"), ac.Item("D3", "c3", "PARTIAL", "half")]
    md = ac.render(items, {"label_mode": "6class", "seeds": [0]})
    assert "**1 PASS, 1 PARTIAL, 0 MANUAL, 1 FAIL.**" in md
    rows = [ln for ln in md.splitlines() if ln.startswith("| D")]
    assert len(rows) == 3 and all(ln.count("|") == 5 for ln in rows) and "e / x" in md and "c / with a pipe" in md


def test_accept_on_the_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    from ppfeddata.utils import load_config
    path, items = ac.write_report(load_config(), regenerate=False, out=tmp_path / "dod.md")
    by = {i.id: i for i in items}
    assert path.exists() and len(items) == 12
    assert by["D1"].status == "PASS" and by["D2"].status == "PASS" and by["D4"].status == "PASS" and by["D9"].status == "PASS"
    assert by["D7"].status == "PARTIAL"                                                                  # not regenerated: the row says so
    assert by["D10"].status in ("PARTIAL", "FAIL") and by["D6c"].status in ("PASS", "FAIL")
    assert "Definition of Done" in path.read_text(encoding="utf-8")
