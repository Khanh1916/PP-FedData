"""Follow-up round, Gate P2: what the IDS side can do with the synthetic data (item 5) and local augmentation at each client (item 6).

Item 5 (IDS trained on synthetic data only, TSTR). Two IDS-side options, the generators unchanged:
- `rows`: all the saved synthetic rows (20,000 per class) instead of `tune.syn_per_class` (5,000);
- `weights`: the class probabilities of the classifier are multiplied by a weight vector chosen on VALIDATION (coordinate search over
  `GRID`, maximising the mean validation macro-F1 of the 3 seeds), then the arg-max is taken. Validation and test are class-balanced like
  the synthetic data, so this is not a prior-shift correction: a classifier trained on synthetic rows labels too many real rows NORMAL or
  BCF (classes whose synthetic distribution is wide), and the weights move that decision boundary. The weights use the non-private
  validation split, like every other choice of the study; on traffic with another class mix they would have to be chosen again.
Reported for every generator (CVAE and FedDP-Marginal alike), RF and MLP, on the real test split; an IDS improvement, not a generator one.

Item 6 (local augmentation). Each client of the federation (the non-IID partition of the run seed) trains its own IDS on its own real rows,
then on its own rows plus synthetic rows of the federation's generator for every class it holds fewer than `FILL` rows of (filled up to
`FILL`), the generator trained by all the clients. A client then gets attack classes it rarely or never saw; this is the use of synthetic data
that can add information (the TAug of the main study added data generated from the same rows, so it could not). Reference: the IDS trained on
synthetic data only (`tune.syn_per_class` rows per class, one model per seed). RF, real test split, every client of 3 seeds.
Results: results/ids_followup.json, results/reports/ids_followup.md.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("ppfeddata.ids_followup")

GENERATORS = {"B3": "B3", "M2": "M2", "M1o-eps5": "M1o-t27-eps5-plain", "MG-eps1": "MGr-eps1", "MG-eps5": "MGs-eps5", "MG-eps10": "MGs-eps10",
              "MGl-eps5": "MGl-eps5"}
GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
FILL = 5000


def _synthetic(cfg: dict[str, Any], prefix: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from ppfeddata.eval.taug_rare import load_synthetic
    return load_synthetic(cfg, prefix, seed)


def _head(X: np.ndarray, y: np.ndarray, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.concatenate([np.flatnonzero(y == c)[:n] for c in range(k)])
    return X[idx], y[idx]


def _fit(cfg: dict[str, Any], clf: str, X: np.ndarray, y: np.ndarray, seed: int):
    from ppfeddata.eval.utility import make_classifier
    m = make_classifier(clf, cfg, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)
    return m


def _metrics(y: np.ndarray, pred: np.ndarray, classes: list[str], rare: list[int]) -> dict[str, float]:
    from ppfeddata.eval.utility import compute_metrics
    k = len(classes)
    m = compute_metrics(y, pred, np.eye(k, dtype=np.float32)[pred], classes)
    return {"macro_f1": float(m["macro_f1"]), "bin_f1": float(m["binary"]["f1"]),
            "rare_recall": float(np.mean([m["per_class"][classes[c]]["recall"] for c in rare]))}


def choose_weights(P_val: list[np.ndarray], y_val: np.ndarray, classes: list[str], rare: list[int], rounds: int = 2) -> np.ndarray:
    """Per-class probability weights maximising the mean validation macro-F1 over the seeds (coordinate search over GRID)."""
    k = P_val[0].shape[1]
    w = np.ones(k)
    score = lambda w_: np.mean([_metrics(y_val, (P * w_).argmax(1), classes, rare)["macro_f1"] for P in P_val])      # noqa: E731
    best = score(w)
    for _ in range(rounds):
        for c in range(k):
            for g in GRID:
                w2 = w.copy()
                w2[c] = g
                s = score(w2)
                if s > best + 1e-9:
                    best, w = s, w2
    return w


def item5(cfg: dict[str, Any], seeds: list[int] | None = None) -> dict[str, Any]:
    from ppfeddata.eval.baselines import load_data
    from ppfeddata.eval.utility import full_proba
    from ppfeddata.tune_dp_full import rare_labels

    data, schema = load_data(cfg)
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    k, rare, spc = len(classes), rare_labels(cfg, schema), int(cfg["tune"]["syn_per_class"])
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    out = {}
    for label, prefix in GENERATORS.items():
        for clf in ("rf", "mlp"):
            res: dict[str, Any] = {}
            for opt, n in (("base", spc), ("rows", None)):
                Pv, Pt = [], []
                for s in seeds:
                    Xs, ys = _synthetic(cfg, prefix, s)
                    if n is not None:
                        Xs, ys = _head(Xs, ys, n, k)
                    m = _fit(cfg, clf, Xs, ys, s)
                    Pv.append(full_proba(m, data["val"]["X"], k))
                    Pt.append(full_proba(m, data["test"]["X"], k))
                plain = [_metrics(data["test"]["y"], P.argmax(1), classes, rare) for P in Pt]
                w = choose_weights(Pv, data["val"]["y"], classes, rare)
                weighted = [_metrics(data["test"]["y"], (P * w).argmax(1), classes, rare) for P in Pt]
                res[opt] = {"plain": _agg(plain), "weighted": _agg(weighted), "weights": w.tolist()}
            out[f"{label}|{clf}"] = res
            logger.info("IDS %s %s: base %.3f, +weights %.3f, all rows %.3f, all rows + weights %.3f (macro-F1, test)", label, clf,
                        res["base"]["plain"]["macro_f1"][0], res["base"]["weighted"]["macro_f1"][0], res["rows"]["plain"]["macro_f1"][0],
                        res["rows"]["weighted"]["macro_f1"][0])
    return out


def _agg(rows: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    return {key: (float(np.mean([r[key] for r in rows])), float(np.std([r[key] for r in rows]))) for key in rows[0]}


def item6(cfg: dict[str, Any], seeds: list[int] | None = None) -> dict[str, Any]:
    from ppfeddata.eval.baselines import load_data
    from ppfeddata.tune_dp_full import rare_labels
    from ppfeddata.tune_marginal import _parts

    data, schema = load_data(cfg)
    classes = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    k, rare, spc = len(classes), rare_labels(cfg, schema), int(cfg["tune"]["syn_per_class"])
    X, y = data["train"]["X"], np.asarray(data["train"]["y"], dtype=np.int64)
    seeds = list(cfg["seeds"]) if seeds is None else seeds
    rows = []
    for s in seeds:
        parts = _parts(cfg, y, schema, s)
        syn = {label: _synthetic(cfg, prefix, s) for label, prefix in GENERATORS.items()}
        only = {label: _metrics(data["test"]["y"], _fit(cfg, "rf", *_head(Xs, ys, spc, k), s).predict(data["test"]["X"]), classes, rare)
                for label, (Xs, ys) in syn.items()}                      # synthetic data only: the same model for every client of the seed
        for cid, idx in enumerate(parts):
            Xc, yc = X[idx], y[idx]
            cnt = np.bincount(yc, minlength=k)
            r = {"seed": s, "client": cid, "n_rows": int(len(idx)), "class_counts": cnt.tolist()}
            r["local"] = _metrics(data["test"]["y"], _fit(cfg, "rf", Xc, yc, s).predict(data["test"]["X"]), classes, rare)
            for label, (Xs, ys) in syn.items():
                add = [np.flatnonzero(ys == c)[: max(0, FILL - int(cnt[c]))] for c in range(k)]
                add = np.concatenate(add)
                Xa, ya = np.vstack([Xc, Xs[add]]), np.concatenate([yc, ys[add]])
                r[label] = _metrics(data["test"]["y"], _fit(cfg, "rf", Xa, ya, s).predict(data["test"]["X"]), classes, rare)
                r[f"only:{label}"] = only[label]
            rows.append(r)
            logger.info("local aug seed %d client %d (%d rows, classes %s): local %.3f, + MG-eps5 %.3f, + B3 %.3f", s, cid, len(idx), cnt.tolist(),
                        r["local"]["macro_f1"], r["MG-eps5"]["macro_f1"], r["B3"]["macro_f1"])
    summary = {}
    for label in ["local"] + list(GENERATORS) + [f"only:{g}" for g in GENERATORS]:
        summary[label] = {key: (float(np.mean([r[label][key] for r in rows])), float(np.std([r[label][key] for r in rows]))) for key in ("macro_f1", "bin_f1", "rare_recall")}
        if label != "local":
            d = [r[label]["macro_f1"] - r["local"]["macro_f1"] for r in rows]
            summary[label]["gain_macro_f1"] = (float(np.mean(d)), float(np.std(d)))
            summary[label]["clients_better"] = int(sum(x > 0 for x in d))
    return {"rows": rows, "summary": summary, "n": len(rows)}


def run(cfg: dict[str, Any], items: tuple[str, ...] = ("5", "6")) -> dict[str, Any]:
    from ppfeddata.interpret import to_jsonable

    res = Path(cfg["compute"]["runs_csv"]).parent
    p = res / "ids_followup.json"
    out = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    for item in items:                                                   # saved after each item
        out[f"item{item}"] = {"5": item5, "6": item6}[item](cfg)
        p.write_text(json.dumps(to_jsonable(out), indent=1), encoding="utf-8")
        (res / "reports" / "ids_followup.md").write_text(render(out), encoding="utf-8")
    return out


def _f(v: Any, d: int = 3) -> str:
    try:
        return f"{float(v[0]):.{d}f} ± {float(v[1]):.{d}f}" if isinstance(v, (list, tuple)) else f"{float(v):.{d}f}"
    except (TypeError, ValueError, IndexError):
        return "n/a"


def render(out: dict[str, Any]) -> str:
    L = ["# IDS-side options and local augmentation (follow-up round, Gate P2)", "",
         "Generated by `ppfeddata ids-followup`; do not edit. Real test split, mean ± std over seeds (item 5) or over clients x seeds (item 6).", ""]
    if out.get("item5"):
        L += ["## Item 5: IDS trained on synthetic data only (TSTR), IDS-side options", "",
              "`base` = 5,000 synthetic rows per class (as in the study); `all rows` = 20,000 per class; `+ weights` = class-probability weights chosen on "
              "validation (mean of 3 seeds). Validation and test are class-balanced like the synthetic data, so the weights do not correct a class prior: "
              "they mostly lower NORMAL (and BCF for most FedDP-Marginal sets), which an IDS trained on synthetic rows predicts too often. An improvement of the IDS, not of the generator; "
              "on traffic with another class mix the weights must be chosen again.", "",
              "| generator | classifier | base macro-F1 | + weights | all rows | all rows + weights | rare recall: base → best | binary F1: base → best |",
              "|---|---|---|---|---|---|---|---|"]
        for key, r in out["item5"].items():
            g, c = key.split("|")
            cands = {"base+w": r["base"]["weighted"], "rows": r["rows"]["plain"], "rows+w": r["rows"]["weighted"]}
            best = max(cands.values(), key=lambda v: v["macro_f1"][0])
            L.append(f"| {g} | {c} | {_f(r['base']['plain']['macro_f1'])} | {_f(r['base']['weighted']['macro_f1'])} | {_f(r['rows']['plain']['macro_f1'])} | "
                     f"{_f(r['rows']['weighted']['macro_f1'])} | {_f(r['base']['plain']['rare_recall'][0])} → {_f(best['rare_recall'][0])} | "
                     f"{_f(r['base']['plain']['bin_f1'][0])} → {_f(best['bin_f1'][0])} |")
        L.append("")
    if out.get("item6"):
        s = out["item6"]["summary"]
        L += ["## Item 6: local augmentation at each client (RF)", "",
              f"Each client trains its own IDS on its own real rows (`local`), then with synthetic rows of the federation's generator for every class it holds "
              f"fewer than {FILL} rows of. `X only` = the IDS trained on {FILL:,} synthetic rows per class of X and nothing else (one model per seed, "
              f"the same for every client), for reference. {out['item6']['n']} client models (5 clients x 3 seeds).", "",
              "| training data of a client | macro-F1 | binary F1 | rare recall | gain over local (macro-F1) | clients better |", "|---|---|---|---|---|---|"]
        for label, v in s.items():
            name = label if label == "local" else (label[5:] + " only (same model for every client)" if label.startswith("only:") else "local + " + label)
            L.append(f"| {name} | {_f(v['macro_f1'])} | {_f(v['bin_f1'])} | {_f(v['rare_recall'])} | "
                     f"{_f(v.get('gain_macro_f1')) if label != 'local' else '-'} | {v.get('clients_better', '-') if label != 'local' else '-'} |")
        L.append("")
    return "\n".join(L) + "\n"
