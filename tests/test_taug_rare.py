"""Optimisation round: TAugR (real data + synthetic rows for the rare classes only, ratio chosen on validation)."""
import json

import numpy as np
import pandas as pd

from ppfeddata.eval import taug_rare as tr
from ppfeddata.eval.runs import RunLedger


def _xy(n_per_class, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.full(n, c) for c, n in enumerate(n_per_class)]).astype(np.int64)
    X = (rng.normal(size=(len(y), 4)) + y[:, None] + shift).astype(np.float32)
    return X, y


def test_only_rare_classes_get_synthetic_rows_in_proportion_to_their_real_count():
    Xr, yr = _xy([100, 50, 10, 6], 0)
    Xs, ys = _xy([40, 40, 40, 3], 1)
    X, y, used = tr.build_rare_aug(Xr, yr, Xs, ys, rare=[2, 3], ratio=2.0, seed=0)
    assert used == {2: 20, 3: 3}                                   # class 3: only 3 synthetic rows in the pool
    assert np.array_equal(np.bincount(y), [100, 50, 30, 9])
    assert np.array_equal(X[:len(Xr)], Xr)                        # all real rows kept, first


def test_ratio_is_chosen_on_seed_0_and_used_for_every_seed(tmp_path):
    cfg = {"label_mode": "6class", "seeds": [0, 1], "compute": {"artifacts_dir": str(tmp_path), "runs_csv": str(tmp_path / "runs.csv")},
           "eval": {"rf": {"n_estimators": 10}, "mlp": {"hidden": [8], "max_iter": 20, "early_stopping": False}}}
    schema = {"label_map": {"A": 0, "B": 1, "C": 2}}
    data = {s: dict(zip(("X", "y"), _xy(n, i))) for i, (s, n) in enumerate((("train", [200, 30, 20]), ("val", [50, 20, 20]), ("test", [50, 20, 20])))}
    for seed in (0, 1):
        d = tmp_path / f"G_{seed}"
        d.mkdir()
        Xs, ys = _xy([100, 100, 100], 10 + seed)
        np.savez_compressed(d / "synthetic.npz", X=Xs, y=ys.astype(np.int16))
    ledger = RunLedger(tmp_path / "runs.csv")
    chosen = tr.evaluate_prefix(cfg, "G", data, schema, ledger, rare=[1, 2], seeds=[0, 1], classifiers=("rf",))
    df = ledger.frame()
    assert list(df["config"]) == ["G-TAugR-rf", "G-TAugR-rf"] and set(df["seed"]) == {0, 1}
    assert chosen["rf"] in tr.RATIOS and set(df["taugr_ratio"]) == {chosen["rf"]}
    curve = json.loads(df["taugr_val_curve"].iloc[0])
    assert set(map(float, curve)) == set(tr.RATIOS) and curve[str(chosen["rf"])] == max(curve.values())
    assert (df["protocol"] == "TAugR").all() and (df["n_synthetic"] > 0).all()
    assert tr.evaluate_prefix(cfg, "G", data, schema, ledger, rare=[1, 2], seeds=[0, 1], classifiers=("rf",)) == {}   # resume: nothing to do


def test_scorecard_reports_the_taugr_gain_against_b0():
    from ppfeddata import aggregate as ag
    from ppfeddata import scorecard as sc
    from tests.test_scorecard import RARE, _summary
    assert ag.parse_config("M1o-t7-eps5-plain-TAugR-rf")["protocol"] == "TAugR"
    s = _summary()
    extra = [{"config": f"B3-TAugR-{c}", "n_seeds": 3, "val_macro_f1_mean": v, "macro_f1_mean": f, "macro_f1_std": 0.002, "taugr_ratio_mean": 0.5}
             for c, v, f in (("rf", 0.50, 0.47), ("mlp", 0.40, 0.99))]
    by = {r["label"]: r for r in sc.build(pd.concat([s, pd.DataFrame(extra)], ignore_index=True), rare=RARE)["rows"]}
    b3 = by["B3"]
    assert b3["taugr_classifier"] == "rf" and abs(b3["taugr_gain"] - (0.47 - 0.45)) < 1e-9 and b3["taugr_ratio"] == 0.5
    assert np.isnan(by["M2"]["taugr_gain"])
