"""Section 9.14: the two ways of redrawing the classes that have several captures, and the text rendered from the JSON."""
import numpy as np

from ppfeddata import split_check as sc


def _captures(n_caps=6, per_cap=50, rows_per_stream=2):
    group = np.repeat([f"cap{i}" for i in range(n_caps)], per_cap)
    time = np.tile(np.arange(per_cap, dtype=float), n_caps)
    stream = np.array([f"{g}:{int(t) // rows_per_stream}" for g, t in zip(group, time)])
    return group, time, stream


def test_group_draw_never_shares_a_capture_between_train_and_test():
    group, _, _ = _captures()
    tr, te = sc.group_draw(group, n_train=150, n_test=60, rng=np.random.default_rng(0))
    assert len(te) == 60 and len(tr) == 150
    assert not set(group[tr]) & set(group[te])
    assert np.array_equal(sc.group_draw(group, 150, 60, np.random.default_rng(0))[1], te)     # deterministic


def test_block_candidates_take_one_contiguous_stretch_of_every_capture_and_drop_straddling_streams():
    group, time, stream = _captures(rows_per_stream=3)
    tr_pool, te_pool = sc.block_candidates(group, time, stream, n_test=60, rng=np.random.default_rng(0))
    for c in np.unique(group):
        t_tr, t_te = time[tr_pool][group[tr_pool] == c], time[te_pool][group[te_pool] == c]
        assert len(t_te) == 10 and t_te.max() - t_te.min() == 9                               # same share of every capture, one contiguous stretch
        assert not ((t_tr > t_te.min()) & (t_tr < t_te.max())).any()                          # no training row inside the stretch
    assert not set(stream[tr_pool]) & set(stream[te_pool])                                    # streams cut by the boundary are dropped
    assert len(tr_pool) < len(group) - len(te_pool)
    starts = {time[te_pool][group[te_pool] == c].min() for c in np.unique(group)}
    assert len(starts) > 1                                                                    # the stretch is not always at the same place
    cand = sc.block_candidates(group, time, stream, 60, np.random.default_rng(0))
    assert all(np.array_equal(a, b) for a, b in zip(cand, (tr_pool, te_pool)))               # deterministic for a given generator
    tr, te = sc.block_draw(cand, n_train=100, n_test=60, rng=np.random.default_rng(1))
    assert set(group[tr]) == set(group[te]) == set(group)                                     # every capture is on both sides


def _out(f_group, f_block, std=0.004):
    classes = ["NORMAL", "BCF", "DELAYED"]
    arm = lambda f: {"macro_f1": float(np.mean(list(f.values()))), "macro_f1_std": 0.001, "f1": f, "f1_std": {c: std for c in classes},  # noqa: E731
                     "recall": {c: 0.9 for c in classes},
                     "runs": [{"seed": 0, "macro_f1": 0.4, "f1": f, "recall": {c: 0.9 for c in classes},
                               "counts": {"NORMAL": {"train": 900, "test": 50, "test_captures": 3, "shared_captures": 0}}}]}
    return {"seeds": [0, 1, 2], "group_subclasses": ["NORMAL"], "block_subclasses": ["DELAYED_DoS", "DELAYED_DDoS"],
            "n_captures": {"NORMAL": 49, "DELAYED_DoS": 10, "DELAYED_DDoS": 10}, "group_classes": ["NORMAL", "BCF"], "classes": classes,
            "quota": {"NORMAL": {"train": 1000, "val": 10, "test": 50}}, "max_diff_vs_ledger": 0.0,
            "arms": {"study": arm(f_group), "group": arm(f_group), "block": arm(f_block)}}


def test_the_section_reads_a_small_difference_as_no_measurable_inflation():
    text = "\n".join(sc.section(_out({"NORMAL": 0.52, "BCF": 0.56, "DELAYED": 0.36}, {"NORMAL": 0.519, "BCF": 0.565, "DELAYED": 0.36})))
    assert "### 9.14" in text and "does not inflate these scores measurably" in text
    assert "(NORMAL 900 of 1,000)" in text                                                    # the smaller training size is stated
    assert "it is not a correction for DELAYED" in text and "The row of the study is exactly the ledger's B0." in text


def test_the_section_reports_a_large_difference_as_such():
    text = "\n".join(sc.section(_out({"NORMAL": 0.52, "BCF": 0.56, "DELAYED": 0.36}, {"NORMAL": 0.60, "BCF": 0.56, "DELAYED": 0.36})))
    assert "does not score like testing on unseen captures" in text and "does not inflate" not in text


def test_no_section_before_the_command_ran(tmp_path):
    assert sc.section_for({"compute": {"runs_csv": str(tmp_path / "runs.csv")}}) == []
