"""Optimisation O3(a): membership inference against FedDP-Marginal with access to the released model, with positive controls.

Same design as `mia_model.py` (the CVAE attack), so the two can be compared: the train pool is split per seed into halves H1 / H2; model A is
fitted on H1 and the reference B on H2 with the same recipe (5 clients drawn at random inside the half, the same epsilon and settings); a
record of H1 is a member of A. Scores: `plain` = log P(record | class) under A (members fit better), `calibrated` = log P_A - log P_B (removes
how typical the record is). AUC inside each class, both directions averaged; TPR at 1 % and 0.1 % FPR.

What is released here is the model itself (noisy count tables), so this is the strongest adversary of the threat model: it sees everything
the server publishes. Positive controls (no noise, eps = 1e7): fitted on 500 rows with 32 coarse bins (cells small enough that one record
shows), and on a full half with the setting used. A useful attack must detect the first; if it does not, an AUC near 0.5 for the DP
configurations proves nothing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ppfeddata.mia_model import _mean_rows, attack, halves
from ppfeddata.models import marginal as mg
from ppfeddata.models import marginal_bn as bn

logger = logging.getLogger("ppfeddata.mia_marginal")


def _split(idx: np.ndarray, K: int, seed: int) -> list[np.ndarray]:
    return [np.sort(p) for p in np.array_split(np.random.default_rng(20_000 + seed).permutation(idx), K)]


def fit_score(X, y, idx_fit, schema, eps, delta, setting: dict[str, Any], K: int, seed: int) -> np.ndarray:
    """Fit on the rows `idx_fit` (split over K clients, Skellam noise split over them) and return log P(record | class) for every row of X."""
    opts = dict(setting.get("options") or {})
    kw = {"bins": int(opts.pop("bins", setting["bins"])), "split": opts.pop("split", setting["split"])}
    parts = _split(idx_fit, K, seed)
    if setting.get("options") is None:                       # the tree of O2 (with the post-processing of O3.3 for MGr; threshold t of P1.2)
        m = mg.fit(X, y, parts, schema, eps, delta, seed=seed, mechanism="skellam", honest_t=setting.get("honest_t"), **kw)
        if setting.get("refine"):
            m = mg.refine(m, **setting["refine"])
        return bn.loglik_tree(m, X, y, alpha=m.info.get("alpha", 1.0))
    m = bn.fit(X, y, parts, schema, eps, delta, seed=seed, mechanism="skellam", **kw, **opts)
    return bn.loglik(m, X, y)


def run_one(X, y, schema, classes, rare, eps, delta, setting, K, seed, n: int | None = None) -> dict[str, Any]:
    h1, h2 = halves(len(y), seed)
    if n is not None:                                         # small worlds for the control: n members, n references
        rng = np.random.default_rng(30_000 + seed)
        h1, h2 = np.sort(rng.choice(h1, n, replace=False)), np.sort(rng.choice(h2, n, replace=False))
    keep = np.concatenate([h1, h2])
    La = -fit_score(X, y, h1, schema, eps, delta, setting, K, seed)[keep]
    Lb = -fit_score(X, y, h2, schema, eps, delta, setting, K, seed + 1)[keep]
    in_a = np.zeros(len(keep), bool)
    in_a[: len(h1)] = True
    r = attack(La, Lb, in_a, y[keep], classes, rare)
    r.update({"seed": seed, "eps": eps, "n_per_model": int(len(h1))})
    return r


def default_settings(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The released configurations: MGs (tree, settings of configs/best_marginal.yaml) at every epsilon, and MGb where O3(b) chose options."""
    import yaml

    from ppfeddata.tune_marginal import BEST_PATH, BN_PATH, REFINE_PATH

    best = yaml.safe_load(BEST_PATH.read_text(encoding="utf-8"))["searches"]
    bnb = yaml.safe_load(BN_PATH.read_text(encoding="utf-8"))["searches"] if BN_PATH.exists() else {}
    rfb = yaml.safe_load(REFINE_PATH.read_text(encoding="utf-8"))["searches"] if REFINE_PATH.exists() else {}
    out = {}
    for e in cfg["dp"]["epsilons"]:
        b = best[f"eps{float(e):g}"]["best"]
        out[f"MGs-eps{float(e):g}"] = {"eps": float(e), "bins": int(b["bins"]), "split": list(b["split"]), "options": None}
        ref = (rfb.get(f"eps{float(e):g}") or {}).get("chosen")
        if ref is not None:
            out[f"MGr-eps{float(e):g}"] = {"eps": float(e), "bins": int(b["bins"]), "split": list(b["split"]), "options": None, "refine": dict(ref)}
        opts = (bnb.get(f"eps{float(e):g}") or {}).get("chosen_options")
        if opts is not None:
            out[f"MGb-eps{float(e):g}"] = {"eps": float(e), "bins": int(b["bins"]), "split": list(b["split"]), "options": dict(opts)}
    out.update(threshold_settings(cfg))
    return out


def threshold_settings(cfg: dict[str, Any], ts: tuple[int, ...] = (3, 2)) -> dict[str, dict[str, Any]]:
    """The framework's configurations with the honest-client threshold t (P1.2): every client adds 1/t of the noise, so the released model
    (all clients honest) carries K/t times the calibrated variance. Same bins / split / post-processing as `privacy_units.final`."""
    from ppfeddata.privacy_units import _setting, name

    out = {}
    for e in cfg["dp"]["epsilons"]:
        s, ref, v = _setting(float(e))
        for t in ts:
            out[name(v, float(e), t=t)] = {"eps": float(e), **s, "options": None, "honest_t": int(t), **({"refine": dict(ref)} if ref else {})}
    return out


def run_all(cfg: dict[str, Any], settings: dict[str, dict[str, Any]] | None = None, seeds: list[int] | None = None, out: str | Path | None = None) -> dict[str, Any]:
    """`settings`: {label: {"eps": e, "bins": b, "split": s, "options": None (tree) | dict (marginal_bn)}}."""
    from ppfeddata.data.preprocess import processed_dir
    from ppfeddata.interpret import to_jsonable
    from ppfeddata.utils import config_hash, git_commit

    settings = settings or default_settings(cfg)
    seeds = [int(s) for s in (seeds if seeds is not None else cfg["seeds"])]
    d = processed_dir(cfg)
    schema = json.loads((d / "feature_schema.json").read_text(encoding="utf-8"))
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    q = cfg.get("quota", {})
    rare = [c for c in classes if c in q and q[c][0] <= 5000] or classes[-4:]
    tr = np.load(d / "train.npz", allow_pickle=True)
    X, y = tr["X"], tr["y"].astype(np.int64)
    K, delta = int(cfg["fl"]["num_clients"]), float(cfg["dp"]["delta"])
    any_setting = next(iter(settings.values()))
    controls = {"nonoise_500_bins32": ({"bins": 32, "split": any_setting["split"], "options": None}, 500),
                "nonoise_half": ({k: any_setting[k] for k in ("bins", "split", "options")}, None)}
    res: dict[str, Any] = {"classes": classes, "rare": rare, "seeds": seeds, "attack": "log-likelihood under the released tables, calibrated by a reference model",
                           "config_hash": config_hash(cfg), "git_commit": git_commit(), "controls": {}, "configs": {}}
    for name, (setting, n) in controls.items():
        rows = [run_one(X, y, schema, classes, rare, 1e7, delta, setting, K, s, n) for s in seeds]
        res["controls"][name] = _mean_rows(rows)
        logger.info("control %s: calibrated AUC %.3f, plain %.3f", name, res["controls"][name]["calibrated"]["auc_mean"]["mean"],
                    res["controls"][name]["plain"]["auc_mean"]["mean"])
    for label, setting in settings.items():
        rows = [run_one(X, y, schema, classes, rare, float(setting["eps"]), delta, setting, K, s) for s in seeds]
        res["configs"][label] = _mean_rows(rows)
        logger.info("%s: calibrated AUC %.3f (rare %s), plain %.3f", label, res["configs"][label]["calibrated"]["auc_mean"]["mean"],
                    res["configs"][label]["calibrated"].get("auc_rare_mean", {}).get("mean"), res["configs"][label]["plain"]["auc_mean"]["mean"])
    p = Path(out) if out else Path(cfg["compute"]["runs_csv"]).parent / "mia_marginal.json"
    p.write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    return res
