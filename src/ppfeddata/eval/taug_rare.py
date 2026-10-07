"""Optimisation round, TAug for the rare classes (`TAugR`).

TAug (Phase 6) adds synthetic rows to every class up to `generate.target_per_class` (20 000): the rare classes (1 000-3 000 real rows) then get
up to 19 000 synthetic rows each, so the classifier mostly learns the generator's version of them, and the gain over B0 was ~0 everywhere (O0).
`TAugR` keeps all real rows and adds synthetic rows to the RARE classes only (train rows <= 5 000, as in the reports), `ratio` x the real count
of each class. The ratio is chosen on the VALIDATION macro-F1 of seed 0 (one choice per generator and classifier, from `RATIOS`) and then used
for every seed, so the seeds stay independent repetitions of one fixed recipe. Ratio 0 (= B0) is not a candidate: the protocol measures what
the synthetic rows add. Limitation: like the classifier choice of the scorecard, the choice uses the real non-private validation split.

It reads the synthetic set already saved by `evaluate_generator` (`artifacts/<prefix>_<seed>/synthetic.npz`), so no generator is trained
again. One ledger row per (prefix, classifier, seed): config `<prefix>-TAugR-<clf>`, with the chosen ratio and the validation curve.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ppfeddata.eval.baselines import _flat_metrics, load_data
from ppfeddata.eval.overhead import Timer
from ppfeddata.eval.runs import RunLedger, artifacts_dir, run_id, runs_csv_path, save_predictions
from ppfeddata.eval.utility import compute_metrics, full_proba, make_classifier
from ppfeddata.utils import config_hash, git_commit

logger = logging.getLogger("ppfeddata.eval.taug_rare")

RATIOS = (0.25, 0.5, 1.0, 2.0)
CLASSIFIERS = ("rf", "mlp")


def taugr_name(prefix: str, clf: str) -> str:
    return f"{prefix}-TAugR-{clf}"


def build_rare_aug(X_real: np.ndarray, y_real: np.ndarray, X_syn: np.ndarray, y_syn: np.ndarray, rare: list[int], ratio: float,
                   seed: int) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """All real rows + round(ratio x real count) synthetic rows of every rare class (fewer when the synthetic pool is smaller)."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y_real, minlength=max(rare) + 1)
    parts_X, parts_y, used = [X_real], [y_real], {}
    for c in rare:
        pool = np.flatnonzero(y_syn == c)
        want = int(round(ratio * counts[c]))
        pick = rng.choice(pool, size=min(want, len(pool)), replace=False)
        parts_X.append(X_syn[pick])
        parts_y.append(y_syn[pick])
        used[int(c)] = int(len(pick))
    return np.vstack(parts_X), np.concatenate(parts_y), used


def _fit(cfg: dict[str, Any], clf_name: str, X: np.ndarray, y: np.ndarray, seed: int):
    clf = make_classifier(clf_name, cfg, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    return clf


def _val_f1(clf, data, k: int, classes: list[str]) -> float:
    X, y = data["val"]["X"], data["val"]["y"]
    return float(compute_metrics(y, clf.predict(X), full_proba(clf, X, k), classes)["macro_f1"])


def load_synthetic(cfg: dict[str, Any], prefix: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    p = artifacts_dir(cfg) / run_id(prefix, seed, cfg["label_mode"]) / "synthetic.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p}: run the generator of {prefix!r} (seed {seed}) first")
    z = np.load(p)
    return z["X"], z["y"].astype(np.int64)


def choose_ratio(cfg: dict[str, Any], prefix: str, clf_name: str, data, schema, rare: list[int], ratios=RATIOS,
                 seed: int = 0) -> tuple[float, dict[float, float]]:
    """Ratio with the best validation macro-F1 on `seed` (ties: the smaller ratio)."""
    k = len(schema["label_map"])
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    Xs, ys = load_synthetic(cfg, prefix, seed)
    curve = {}
    for r in ratios:
        Xa, ya, _ = build_rare_aug(data["train"]["X"], data["train"]["y"], Xs, ys, rare, r, seed)
        curve[float(r)] = _val_f1(_fit(cfg, clf_name, Xa, ya, seed), data, k, classes)
        logger.info("%s %s ratio %g: val macro-F1 %.4f", prefix, clf_name, r, curve[float(r)])
    best = max(curve, key=lambda r: (curve[r], -r))
    return best, curve


def evaluate_prefix(cfg: dict[str, Any], prefix: str, data, schema, ledger: RunLedger, rare: list[int], seeds: list[int],
                    classifiers=CLASSIFIERS, resume: bool = True, extra: dict[str, Any] | None = None) -> dict[str, float]:
    """Choose the ratio on seed 0 per classifier, then one ledger row per seed. Returns {classifier: ratio}."""
    mode, k = cfg["label_mode"], len(schema["label_map"])
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    chosen = {}
    for clf_name in classifiers:
        name = taugr_name(prefix, clf_name)
        if resume and all(ledger.done(run_id(name, s, mode)) for s in seeds):
            logger.info("skip %s (all seeds done)", name)
            continue
        ratio, curve = choose_ratio(cfg, prefix, clf_name, data, schema, rare, seed=int(cfg["seeds"][0]))
        chosen[clf_name] = ratio
        for seed in seeds:
            rid = run_id(name, seed, mode)
            if resume and ledger.done(rid):
                continue
            Xs, ys = load_synthetic(cfg, prefix, seed)
            Xa, ya, used = build_rare_aug(data["train"]["X"], data["train"]["y"], Xs, ys, rare, ratio, seed)
            with Timer() as t_fit:
                clf = _fit(cfg, clf_name, Xa, ya, seed)
            res = {}
            for split in ("test", "val"):
                X, y = data[split]["X"], data[split]["y"]
                pred, proba = clf.predict(X), full_proba(clf, X, k)
                res[split] = compute_metrics(y, pred, proba, classes)
                save_predictions(cfg, rid, split, pred, proba if split == "test" else None,
                                 meta={"run_id": rid, "config": name, "seed": seed, "n_train": int(len(ya))} if split == "test" else None)
            row = {"run_id": rid, "config": name, "classifier": clf_name, "protocol": "TAugR", "seed": seed, "label_mode": mode,
                   "config_hash": config_hash(cfg), "git_commit": git_commit(), "n_train": int(len(ya)), "n_synthetic": int(sum(used.values())),
                   "fit_time_s": t_fit.seconds, "n_iter": int(getattr(clf, "n_iter_", 0)) if clf_name == "mlp" else 0,
                   "taugr_ratio": float(ratio), "taugr_val_curve": json.dumps(curve), **(extra or {}),
                   **_flat_metrics(res["test"]), **_flat_metrics(res["val"], "val_")}
            ledger.append(row)
            logger.info("%s: ratio %g, test macro-F1 %.4f (val %.4f)", rid, ratio, row["macro_f1"], row["val_macro_f1"])
    return chosen


def scorecard_prefixes(cfg: dict[str, Any]) -> list[str]:
    """Generator prefixes of the scorecard rows (the valid variant of each protected entry), as found in the ledger."""
    import pandas as pd

    from ppfeddata import scorecard as sc

    summ = pd.read_csv(Path(cfg["compute"]["runs_csv"]).parent / "summary.csv")
    out = []
    for r in sc.candidate_rows(summ):
        if r["valid"]:
            p = r["tstr_config"].rsplit("-TSTR-", 1)[0]
            if p not in out:
                out.append(p)
    return out


def run(cfg: dict[str, Any], prefixes: list[str] | None = None, seeds: list[int] | None = None, resume: bool = True) -> dict[str, dict[str, float]]:
    from ppfeddata.tune_dp_full import rare_labels

    data, schema = load_data(cfg)
    rare = rare_labels(cfg, schema)
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    ledger = RunLedger(runs_csv_path(cfg))
    prefixes = prefixes or scorecard_prefixes(cfg)
    logger.info("TAugR: rare labels %s, prefixes %s", rare, prefixes)
    return {p: evaluate_prefix(cfg, p, data, schema, ledger, rare, seeds, resume=resume) for p in prefixes}
