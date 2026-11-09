"""Section 9.13: the protocol measurement, the binary scoring of saved predictions and the rules of the published table."""
import numpy as np
import yaml

from ppfeddata import compare_published as cp


def _data(n=(60, 20, 30)):
    out, start = {}, 0
    for sp, m in zip(cp.SPLITS, n):
        y = np.tile([0, 1, 2], m)
        idx = np.arange(start, start + len(y), dtype=float)
        out[sp] = {"X": idx[:, None], "y": y, "ids": {"ip": idx + 1.0}}
        start += len(y)
    return out


def test_ip_to_int_is_stable_and_tolerates_a_missing_address():
    assert cp.ip_to_int("192.168.90.100") == 192 * 256**3 + 168 * 256**2 + 90 * 256 + 100
    assert cp.ip_to_int("nan") == 0.0


def test_redraw_keeps_the_class_sizes_and_never_repeats_a_row():
    data = _data()
    d = cp.redraw(data, seed=0)
    for c in range(3):
        assert (d["train"]["y"] == c).sum() == (data["train"]["y"] == c).sum() and (d["test"]["y"] == c).sum() == (data["test"]["y"] == c).sum()
    tr, te = d["train"]["X"][:, 0], d["test"]["X"][:, 0]
    assert not set(tr) & set(te) and len(set(tr)) == len(tr)
    assert np.array_equal(d["train"]["ids"]["ip"], tr + 1.0)                      # the identifiers follow their rows
    assert set(te) != set(data["test"]["X"][:, 0])                                # the rows are really redrawn
    assert np.array_equal(cp.redraw(data, seed=0)["test"]["X"], d["test"]["X"])   # and deterministically


def test_closest_f1_counts_any_attack_label_as_detected_and_ignores_other_classes():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 2, 2, 0, 0, 0])                # class 1: one packet flagged (as class 2), one missed; one normal flagged
    assert cp.closest_f1(y, p, [1]) == 2 * 1 / (2 * 1 + 1 + 1)
    assert cp.binary_f1(y > 0, p > 0) == 2 * 1 / (2 * 1 + 1 + 3)


def _out():
    row = lambda s, i, m: {"split": s, "identifiers": i, "macro_f1": m, "macro_f1_std": 0.001, "binary_f1": m + 0.1, "binary_f1_std": 0.001, "runs": []}  # noqa: E731
    return {"seeds": [0, 1, 2], "n_test": 30000, "n_pooled": 130500, "n_features": 59, "max_diff_vs_ledger": 0.0, "subclasses": 11, "block_subclasses": 8,
            "study": {"B0": {"macro_f1": 0.45, "binary_f1": 0.79}, "SMOTE": {"macro_f1": 0.5, "binary_f1": 0.82}, "MG": {"label": "MGs-eps10", "classifier": "mlp", "macro_f1": 0.43, "binary_f1": 0.84}},
            "protocol": [row(*c, m) for c, m in zip(cp.CONDITIONS, (0.45, 0.84, 0.46, 0.5, 0.94, 0.93))],
            "closest": [{"name": "Normal / BCF", "study": "Zhang et al. 2026", "f1": 1.0, "real": 0.888, "synthetic": 0.857}]}


def _pub():
    return yaml.safe_load(cp.PUBLISHED.read_text(encoding="utf-8"))


def test_the_published_table_holds_only_studies_read_at_the_source():
    pub = _pub()
    assert [p["key"] for p in pub["included"]] == ["alatram2023", "zhang2026", "parra2026"]
    for p in pub["included"]:
        assert all(p[k] for k in ("role", "citation", "url", "read", "task", "input", "split", "result")) and p["read"].startswith("full text")
    assert all(p["why"] for p in pub["uses_dataset_not_read"])
    assert all(t["study"] in {p["name"] for p in pub["included"]} for t in pub["closest_task"])


def test_the_section_explains_the_selection_and_quotes_no_second_hand_score():
    pub = _pub()
    text = "\n".join(cp.section(_out(), pub))
    assert "**Why only these three.**" in text and "not a judgement of their quality" in text
    for p in pub["included"] + pub["uses_dataset_not_read"]:
        assert p["name"] in text
    for second_hand in ("98.82", "91.23", "96.95", "| 93"):
        assert second_hand not in text


def test_the_section_does_not_claim_that_published_scores_are_inflated():
    text = "\n".join(cp.section(_out(), _pub()))
    assert "**What this does not show:** that any published score is inflated" in text
    assert "do not say which ones the classifiers received" in text
    assert "macro-F1 moves by +0.010" in text                                     # the numbers come from the JSON


def test_no_section_before_the_command_ran(tmp_path):
    assert cp.section_for({"compute": {"runs_csv": str(tmp_path / "runs.csv")}}) == []
