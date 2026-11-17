"""Section 9.14 of the report: how much testing on a block of the same capture moves the score.

8 of the 11 sub-classes come from a single capture, so their validation and test rows are blocks of the capture their training rows come from, usually with training blocks before and after
(block split); only NORMAL, BCF_DoS and BCF_DDoS have several captures and are split by whole capture (group split). This measurement asks what the
block split does to a score, on the only sub-classes where both splits are possible.

For the group-split sub-classes, the sampled rows of the three splits are pooled and drawn again in two ways, with the class sizes of the study:
- `group`: whole captures go to the test split until it is full, the training rows come from the other captures (as in the study);
- `block`: inside every capture (rows in time order), a contiguous stretch at a random position forms the test candidates and the rows before and after it the training candidates; TCP streams
  that would straddle the boundary are dropped from the training side (as in the study's block split).
The other sub-classes keep the rows of the study. The IDS is the one of B0 (RF, the study's settings, 3 seeds). The difference block - group in the
F1 of NORMAL and BCF is the optimism of testing on the same capture, for these classes.

Nothing here trains a generator or writes to the ledger, and no number of the study changes. Writes results/split_check.json and
results/reports/split_check.md; `section` renders 9.14 from the JSON.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("ppfeddata.split_check")

SPLITS = ("train", "val", "test")
ARMS = ("study", "group", "block")
ARM_TEXT = {"study": "split of the study", "group": "group-split classes redrawn by whole capture",
            "block": "group-split classes redrawn by block of the same capture"}


def _load(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from ppfeddata.data.preprocess import processed_dir
    from ppfeddata.data.split_sample import interim_dir

    out = {}
    for sp in SPLITS:
        z = np.load(processed_dir(cfg) / f"{sp}.npz", allow_pickle=True)
        d = pd.read_parquet(interim_dir(cfg) / f"{sp}.parquet", columns=["group_id", "class11", "epoch_time", "stream_id"])
        if not (d["group_id"].astype(str).to_numpy() == z["group_id"]).all():
            raise ValueError(f"{sp}: the rows of the parquet file and of the npz file differ")
        out[sp] = {"X": z["X"].astype(np.float64), "y": z["y"].astype(int), "group": d["group_id"].astype(str).to_numpy(),
                   "sub": d["class11"].astype(str).to_numpy(), "stream": d["stream_id"].astype(str).to_numpy(),
                   "time": pd.to_numeric(d["epoch_time"], errors="coerce").fillna(0.0).to_numpy(float)}
    return out


def _pool(data: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([data[s][k] for s in SPLITS]) for k in data["train"]}


def group_draw(group: np.ndarray, n_train: int, n_test: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Indices (into `group`) of n_test test rows from whole captures and up to n_train training rows from the other captures."""
    caps = rng.permutation(np.unique(group))
    test_caps, size = [], 0
    for c in caps:
        if size >= n_test:
            break
        test_caps.append(c)
        size += int((group == c).sum())
    is_test = np.isin(group, test_caps)
    te = rng.choice(np.flatnonzero(is_test), size=min(n_test, int(is_test.sum())), replace=False)
    pool = np.flatnonzero(~is_test)
    tr = rng.choice(pool, size=min(n_train, len(pool)), replace=False)
    return np.sort(tr), np.sort(te)


def block_candidates(group: np.ndarray, time: np.ndarray, stream: np.ndarray, n_test: int,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Training and test candidates of the block split, as in the study: inside every capture (rows in time order), one contiguous stretch at a
    random position holds the test candidates (the same share of each capture), and the rows before and after it are training candidates, minus
    every TCP stream that reaches the test side."""
    frac = n_test / len(group)
    test_cand, train_cand = [], []
    for c in np.unique(group):
        idx = np.flatnonzero(group == c)
        idx = idx[np.argsort(time[idx], kind="stable")]
        k = int(np.ceil(frac * len(idx)))
        start = int(rng.integers(0, len(idx) - k + 1))
        te_c = idx[start:start + k]
        tr_c = np.concatenate([idx[:start], idx[start + k:]])
        train_cand.append(tr_c[~np.isin(stream[tr_c], stream[te_c])])
        test_cand.append(te_c)
    return np.concatenate(train_cand), np.concatenate(test_cand)


def block_draw(candidates: tuple[np.ndarray, np.ndarray], n_train: int, n_test: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Indices of n_test test rows and up to n_train training rows drawn from the block candidates."""
    tr_pool, te_pool = candidates
    te = rng.choice(te_pool, size=min(n_test, len(te_pool)), replace=False)
    tr = rng.choice(tr_pool, size=min(n_train, len(tr_pool)), replace=False)
    return np.sort(tr), np.sort(te)


def resplit(data: dict[str, dict[str, Any]], subs: list[str], quota: dict[str, dict[str, int]], arm: str, seed: int) -> dict[str, Any]:
    """The study's train/test rows, with the rows of the sub-classes `subs` drawn again by `arm` ("group" or "block")."""
    rng = np.random.default_rng(seed)
    pool = _pool(data)
    keep_tr = ~np.isin(data["train"]["sub"], subs)
    keep_te = ~np.isin(data["test"]["sub"], subs)
    X_tr, y_tr, X_te, y_te, sub_te = [data["train"]["X"][keep_tr]], [data["train"]["y"][keep_tr]], [data["test"]["X"][keep_te]], [data["test"]["y"][keep_te]], [data["test"]["sub"][keep_te]]
    counts = {}
    for i, s in enumerate(subs):
        rows = np.flatnonzero(pool["sub"] == s)
        n_te = int(quota[s]["test"])
        # the block candidates depend only on (seed, sub-class), so both arms can use the same number of training rows:
        # the quota, or fewer if the block split leaves fewer candidates
        cand = block_candidates(pool["group"][rows], pool["time"][rows], pool["stream"][rows], n_te, np.random.default_rng([seed, i]))
        n_tr = min(int(quota[s]["train"]), len(cand[0]))
        if arm == "group":
            tr, te = group_draw(pool["group"][rows], n_tr, n_te, rng)
        else:
            tr, te = block_draw(cand, n_tr, n_te, rng)
        tr, te = rows[tr], rows[te]
        if set(pool["stream"][tr]) & set(pool["stream"][te]):
            raise AssertionError(f"{arm}/{s}: a TCP stream is in both splits")
        counts[s] = {"train": int(len(tr)), "test": int(len(te)), "test_captures": int(len(set(pool["group"][te]))),
                     "shared_captures": int(len(set(pool["group"][tr]) & set(pool["group"][te])))}
        X_tr.append(pool["X"][tr]); y_tr.append(pool["y"][tr])
        X_te.append(pool["X"][te]); y_te.append(pool["y"][te]); sub_te.append(pool["sub"][te])
    return {"train": {"X": np.concatenate(X_tr), "y": np.concatenate(y_tr)}, "test": {"X": np.concatenate(X_te), "y": np.concatenate(y_te)},
            "counts": counts}


def per_class(y: np.ndarray, p: np.ndarray, names: dict[int, str]) -> dict[str, Any]:
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    labels = sorted(names)
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {"macro_f1": float(f1_score(y, p, average="macro", labels=labels, zero_division=0)),
            "f1": {names[c]: float(v) for c, v in zip(labels, f1)}, "recall": {names[c]: float(v) for c, v in zip(labels, rec)},
            "precision": {names[c]: float(v) for c, v in zip(labels, prec)}}


def measure(cfg: dict[str, Any], data: dict[str, dict[str, Any]], subs: list[str], quota: dict[str, dict[str, int]], names: dict[int, str]) -> dict[str, Any]:
    from ppfeddata.eval.utility import make_classifier

    out = {}
    for arm in ARMS:
        runs = []
        for seed in [int(s) for s in cfg["seeds"]]:
            if arm == "study":
                d = {"train": data["train"], "test": data["test"], "counts": {}}
            else:
                d = resplit(data, subs, quota, arm, seed)
            rf = make_classifier("rf", cfg, seed).fit(d["train"]["X"], d["train"]["y"])
            m = per_class(d["test"]["y"], rf.predict(d["test"]["X"]), names)
            m.update(seed=seed, counts=d["counts"], n_train=int(len(d["train"]["y"])), n_test=int(len(d["test"]["y"])))
            runs.append(m)
            logger.info("%s / seed %d: macro-F1 %.4f, F1 %s", arm, seed, m["macro_f1"], {k: round(v, 3) for k, v in m["f1"].items()})
        mean = lambda key, c: float(np.mean([r[key][c] for r in runs]))  # noqa: E731
        std = lambda key, c: float(np.std([r[key][c] for r in runs]))  # noqa: E731
        out[arm] = {"runs": runs, "macro_f1": float(np.mean([r["macro_f1"] for r in runs])), "macro_f1_std": float(np.std([r["macro_f1"] for r in runs])),
                    "f1": {c: mean("f1", c) for c in names.values()}, "f1_std": {c: std("f1", c) for c in names.values()},
                    "recall": {c: mean("recall", c) for c in names.values()}}
    return out


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    from ppfeddata.data.preprocess import processed_dir
    from ppfeddata.interpret import to_jsonable

    res = Path(cfg["compute"]["runs_csv"]).parent
    manifest = json.loads((res / "manifests" / f"split_manifest_{cfg['label_mode']}.json").read_text(encoding="utf-8"))
    subs = [s for s, v in manifest["subclasses"].items() if v["split_kind"] == "group"]
    block_subs = [s for s, v in manifest["subclasses"].items() if v["split_kind"] == "block"]
    label_map = json.loads((processed_dir(cfg) / "label_map.json").read_text(encoding="utf-8"))
    names = {int(v): k for k, v in label_map.items()}
    data = _load(cfg)
    arms = measure(cfg, data, subs, manifest["quota_actual"], names)
    df = pd.read_csv(Path(cfg["compute"]["runs_csv"]), low_memory=False)
    b0 = df[(df["config"] == "B0-rf") & (df["label_mode"] == cfg["label_mode"])].drop_duplicates("seed").set_index("seed")["macro_f1"]
    group_classes = sorted({data["train"]["y"][data["train"]["sub"] == s][0] for s in subs})
    out = {"seeds": [int(s) for s in cfg["seeds"]], "group_subclasses": subs, "block_subclasses": block_subs,
           "n_captures": {s: int(manifest["subclasses"][s]["n_units"]) for s in manifest["subclasses"]},
           "group_classes": [names[int(c)] for c in group_classes], "classes": [names[c] for c in sorted(names)],
           "quota": {s: {k: int(v) for k, v in manifest["quota_actual"][s].items()} for s in subs},
           "max_diff_vs_ledger": max(abs(float(b0[r["seed"]]) - r["macro_f1"]) for r in arms["study"]["runs"]),
           "arms": arms}
    out = to_jsonable(out)
    (res / "split_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    write_report(cfg, out)
    return out


def section(out: dict[str, Any]) -> list[str]:
    """The lines of section 9.14."""
    A, gc = out["arms"], out["group_classes"]
    blk = ", ".join(sorted({s.rsplit("_", 1)[0] for s in out["block_subclasses"]}))
    d = out["max_diff_vs_ledger"]
    same = "The row of the study is exactly the ledger's B0." if d < 1e-9 else f"The row of the study differs from the ledger's B0 by at most {d:.1e} per seed."
    caps = "; ".join(f"{s} {out['n_captures'][s]}" for s in out["group_subclasses"])
    n_tr = {s: [r["counts"][s]["train"] for r in A["block"]["runs"]] for s in out["group_subclasses"]}
    rng_txt = lambda v: f"{min(v):,}" if min(v) == max(v) else f"{min(v):,} to {max(v):,}"  # noqa: E731
    short = [f"{s} {rng_txt(v)} of {out['quota'][s]['train']:,}" for s, v in n_tr.items() if min(v) < out["quota"][s]["train"]]
    sizes = (f" Both redrawn splits use the same number of training rows per sub-class: the quota, or fewer where the block split drops the TCP streams that straddle its boundary ({'; '.join(short)})."
             if short else " Both redrawn splits use the training quota of the study.")
    L = ["### 9.14 How much testing on the same capture moves the score", "",
         f"Only {len(out['group_subclasses'])} of {len(out['n_captures'])} sub-classes have several captures and are split by whole capture ({caps} captures). The other {len(out['block_subclasses'])}, which make up the classes {blk}, come from one capture each, so their test rows are a block of the capture their training rows come from, usually with training blocks before and after it. "
         f"This section measures what such a block split does to a score, on the classes where both splits are possible ({', '.join(gc)}): their sampled rows are pooled and drawn again by whole capture or by block of the same capture (one contiguous stretch at a random position of every capture, with training rows before and after it and the straddling TCP streams dropped, as in the study's block split), with the class sizes of the study, while the other classes keep the rows of the study. "
         f"IDS of B0 (RF, {len(out['seeds'])} seeds).{sizes} No number above changes (`split-check`, SPEC_DEVIATIONS R.12; details in `results/reports/split_check.md`). {same}", "",
         "| Split | Macro-F1 | " + " | ".join(f"F1 {c}" for c in out["classes"]) + " |", "|---|---|" + "---|" * len(out["classes"])]
    for arm in ARMS:
        a = A[arm]
        L.append(f"| {ARM_TEXT[arm]} | {a['macro_f1']:.3f} ± {a['macro_f1_std']:.3f} | " + " | ".join(f"{a['f1'][c]:.3f}" for c in out["classes"]) + " |")
    L.append("")
    for c in gc:
        g, b = A["group"], A["block"]
        L.append(f"- **{c}:** F1 {g['f1'][c]:.3f} by whole capture against {b['f1'][c]:.3f} by block of the same capture ({b['f1'][c] - g['f1'][c]:+.3f}); recall {g['recall'][c]:.3f} against {b['recall'][c]:.3f}.")
    dm = A["block"]["macro_f1"] - A["group"]["macro_f1"]
    gap = max(abs(A["block"]["f1"][c] - A["group"]["f1"][c]) for c in gc)
    spread = max(A[arm]["f1_std"][c] for arm in ("group", "block") for c in gc)
    verdict = (f"- **Reading:** the largest difference in class F1 is {gap:.3f}, within twice the largest seed-to-seed standard deviation of the two splits ({2 * spread:.3f}). For {', '.join(gc)}, testing on a block of the same capture scores about as testing on unseen captures, once the identifier columns are removed; the block split does not inflate these scores measurably."
               if gap <= 2 * spread else
               f"- **Reading:** the largest difference in class F1 is {gap:.3f}, above twice the largest seed-to-seed standard deviation of the two splits ({2 * spread:.3f}). For {', '.join(gc)}, testing on a block of the same capture does not score like testing on unseen captures.")
    L += [verdict,f"- **Macro-F1:** {A['group']['macro_f1']:.3f} against {A['block']['macro_f1']:.3f} ({dm:+.3f}); the classes {blk} are the same rows in both arms, so the difference comes from {', '.join(gc)} and from how they are confused with the rest.",
          f"- **What it means for the study:** the test scores of {blk} are measured on other traffic of the capture the model was trained on. The difference above estimates, for the classes where it can be measured, how much such a test differs from a test on unseen captures; it is not a correction for {blk}, whose single capture leaves no unseen capture to test on. "
          "All configurations of the study share the same split, so comparisons between them are affected alike; the absolute level of a score is what may not carry over to new captures.", ""]
    return L


def section_for(cfg: dict[str, Any]) -> list[str]:
    p = Path(cfg["compute"]["runs_csv"]).parent / "split_check.json"
    return section(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else []


def write_report(cfg: dict[str, Any], out: dict[str, Any]) -> None:
    L = ["# How much testing on the same capture moves the score", "", "Written by `split-check`; do not edit.", ""]
    L += [ln.replace("### 9.14 ", "## ") for ln in section(out)]
    L += ["## Rows drawn per seed", "", "| Split | Seed | Sub-class | Train rows | Test rows | Test captures | Captures in both splits |", "|---|---|---|---|---|---|---|"]
    L += [f"| {arm} | {r['seed']} | {s} | {c['train']} | {c['test']} | {c['test_captures']} | {c['shared_captures']} |"
          for arm in ("group", "block") for r in out["arms"][arm]["runs"] for s, c in r["counts"].items()]
    L += ["", "## Scores per seed", "", "| Split | Seed | Macro-F1 | " + " | ".join(f"F1 {c}" for c in out["classes"]) + " | " + " | ".join(f"Recall {c}" for c in out["group_classes"]) + " |",
          "|---|---|---|" + "---|" * (len(out["classes"]) + len(out["group_classes"]))]
    L += [f"| {arm} | {r['seed']} | {r['macro_f1']:.4f} | " + " | ".join(f"{r['f1'][c]:.4f}" for c in out["classes"]) + " | " + " | ".join(f"{r['recall'][c]:.4f}" for c in out["group_classes"]) + " |"
          for arm in ARMS for r in out["arms"][arm]["runs"]]
    res = Path(cfg["compute"]["runs_csv"]).parent
    (res / "reports" / "split_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
