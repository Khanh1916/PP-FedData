"""Phase 13: the logic behind demo/app.py, kept out of the Streamlit script so that it can be tested without Streamlit.

- `generators` / `sample`: draw synthetic rows from a trained CVAE of the spec matrix (B2, B3, M1, M2, M3) for one class or all classes, and return them in raw units.
  The path is the one of the evaluation (`models.generate.generate`): the same model, seed and counts give the same rows as `artifacts/<run>/synthetic.npz`.
  A DP configuration is sampled with the plain decoder only: the per-class residual noise is estimated on the pooled train split and is not covered by epsilon
  (SPEC_DEVIATIONS 7.1, 8.8), so releasing rows made with it would void the guarantee the configuration is named after.
- results readers: `summary.csv`, `interpretation.json`, the sections of `final_report.md`, the data overview from the split manifest and the feature schema.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppfeddata import limitations
from ppfeddata.eval.runs import artifacts_dir, run_id

MAX_PER_CLASS = 50_000
VARIANTS = {"plain": "plain decoder", "residual": "plain decoder + per-class residual noise"}
ALL_CLASSES = "ALL"


# --------------------------------------------------------------------------------------------------
# Generators of the spec matrix
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Generator:
    label: str                      # configuration id of the matrix: B2, B3, M1-eps5, M2, M3
    title: str
    run_name: str                   # prefix of the artifact folder: B2, B3, M1d-t21-eps5, M2, M3d-t21-eps5
    kind: str                       # b2 | b3 | m1 | m2 | m3 | mg (FedDP-Marginal, spec v1.5)
    dp: bool
    secagg: bool
    federated: bool
    eps_target: float | None

    @property
    def protections(self) -> str:
        parts = ["federated" if self.federated else "centralised"]
        if self.kind == "mg":
            return ", ".join(parts + [f"DP count tables (Skellam noise split over the clients, target epsilon {self.eps_target:g})", "secure aggregation (SecAgg+)"])
        if self.dp:
            parts.append(f"client-side DP-SGD (target epsilon {self.eps_target:g})")
        if self.secagg:
            parts.append("secure aggregation (SecAgg+)")
        return ", ".join(parts)


def generators(cfg: dict[str, Any]) -> list[Generator]:
    """The CVAE configurations of the spec matrix (group `matrix`): the ones that have a generator model."""
    from ppfeddata.run_experiment import fl_run_name, load_matrix

    out = []
    for e in load_matrix():
        if e.group != "matrix" or e.kind not in ("b2", "b3", "m1", "m2", "m3"):
            continue
        out.append(Generator(e.id, e.title, "B2" if e.kind == "b2" else str(fl_run_name(cfg, e)), e.kind, dp=e.kind in ("m1", "m3"), secagg=e.kind in ("m2", "m3"),
                             federated=e.kind != "b2", eps_target=float(e.params["eps"]) if "eps" in e.params else None))
    # FedDP-Marginal (spec v1.5): the framework's generator; the Flower SecAgg+ runs (MGr = with the post-processing of O3.3 where it was chosen)
    for e in (1, 5, 10):
        name = f"MGr-eps{e}" if e == 1 and (artifacts_dir(cfg) / run_id(f"MGr-eps{e}", int(cfg["seeds"][0]), cfg["label_mode"])).exists() else f"MGs-eps{e}"
        out.append(Generator(f"MG-eps{e}", f"FedDP-Marginal, epsilon {e}", name, "mg", dp=True, secagg=True, federated=True, eps_target=float(e)))
    return out


def run_dir(cfg: dict[str, Any], g: Generator, seed: int) -> Path:
    return artifacts_dir(cfg) / run_id(g.run_name, int(seed), cfg["label_mode"])


def model_file(cfg: dict[str, Any], g: Generator, seed: int) -> Path:
    return run_dir(cfg, g, seed) / {"b2": "model.pt", "mg": "mg_model.pkl"}.get(g.kind, "final_state.pt")


def available_seeds(cfg: dict[str, Any], g: Generator) -> list[int]:
    return [int(s) for s in cfg["seeds"] if model_file(cfg, g, s).exists()]


def available_generators(cfg: dict[str, Any]) -> list[tuple[Generator, list[int]]]:
    return [(g, s) for g in generators(cfg) if (s := available_seeds(cfg, g))]


def synthetic_path(cfg: dict[str, Any], g: Generator, seed: int, variant: str) -> Path:
    """The synthetic set the evaluation saved for this model: `<run>[-plain]_<seed>/synthetic.npz` (B2 has no separate plain run)."""
    name = g.run_name + ("-plain" if variant == "plain" and g.kind not in ("b2", "mg") else "")
    return artifacts_dir(cfg) / run_id(name, int(seed), cfg["label_mode"]) / "synthetic.npz"


# --------------------------------------------------------------------------------------------------
# Assets and sampling
# --------------------------------------------------------------------------------------------------
@dataclass
class Assets:
    data: dict[str, dict[str, np.ndarray]]
    schema: dict[str, Any]
    pre: Any                                  # the fitted Preprocessor

    @property
    def classes(self) -> list[str]:
        return [c for c, _ in sorted(self.schema["label_map"].items(), key=lambda kv: kv[1])]


def load_assets(cfg: dict[str, Any]) -> Assets:
    import joblib

    from ppfeddata.data.preprocess import processed_dir
    from ppfeddata.eval.baselines import load_data

    data, schema = load_data(cfg)
    return Assets(data, schema, joblib.load(processed_dir(cfg) / "preprocessor.joblib"))


def load_model(cfg: dict[str, Any], g: Generator, seed: int, schema: dict[str, Any]):
    """The trained CVAE of a run: `model.pt` (centralised, with its own config) or `final_state.pt` + `spec.json` (federated)."""
    import torch

    from ppfeddata.fl import core
    from ppfeddata.models.cvae import CVAE, build_layout

    f = model_file(cfg, g, seed)
    if not f.exists():
        raise FileNotFoundError(f"no trained model for {g.label} seed {seed}: {f}")
    if g.kind == "mg":                      # the released (noisy) tables; MGr adds the post-processing chosen on validation (no privacy cost)
        import pickle

        import yaml

        from ppfeddata.models import marginal as mg
        from ppfeddata.tune_marginal import REFINE_PATH

        with open(f, "rb") as fh:
            m = pickle.load(fh)
        if g.run_name.startswith("MGr") and REFINE_PATH.exists():
            ref = (yaml.safe_load(REFINE_PATH.read_text(encoding="utf-8"))["searches"].get(f"eps{g.eps_target:g}") or {}).get("chosen")
            m = mg.refine(m, **ref) if ref else m
        return m
    layout, k = build_layout(schema), len(schema["label_map"])
    if g.kind == "b2":
        ck = torch.load(f, map_location="cpu", weights_only=True)
        c = ck["config"]
        model, state = CVAE(layout, c["n_classes"], c["latent_dim"], tuple(c["hidden"]), c["layernorm"]), ck["state_dict"]
    else:
        spec = json.loads((f.parent / "spec.json").read_text(encoding="utf-8"))
        model, state = core.make_model(layout, k, spec["hp"]), torch.load(f, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def readable_table(X: np.ndarray, y: np.ndarray, pre: Any, class_names: list[str]) -> pd.DataFrame:
    """Encoded rows -> raw units, one column per original feature, plus `label`. A cell is empty where the feature does not apply (its `_is_na` flag is 1 or the category is NONE).
    Integer-valued and binary columns are whole numbers (the decoder returns them with float error); numeric ones are clipped to the range seen in train."""
    df = pre.inverse_transform(np.asarray(X))
    for b in pre.core_blocks:
        c, p = b["column"], pre.params[b["column"]]
        if b["type"] == "numeric":
            df[c] = df[c].clip(p["raw_min"], p["raw_max"]) + 0.0               # back inside the train range; `+ 0.0` turns -0.0 into 0.0
            if p.get("is_integer"):
                df[c] = df[c].round().astype("Int64")
        elif b["type"] == "binary":
            df[c] = df[c].astype("Int64")
    df.insert(0, "label", [class_names[int(i)] for i in y])
    return df


@dataclass
class Samples:
    X: np.ndarray                              # encoded rows (n, D)
    y: np.ndarray
    table: pd.DataFrame                        # raw units + label
    generator: Generator
    train_seed: int
    sample_seed: int
    variant: str                               # the variant actually used
    n_per_class: int
    classes: list[str]
    notes: list[str]

    def csv(self, encoded: bool = False) -> bytes:
        if encoded:
            cols = [f"x{i}" for i in range(self.X.shape[1])]
            df = pd.DataFrame(self.X, columns=cols)
            df.insert(0, "label", self.table["label"].to_numpy())
            return df.to_csv(index=False, float_format="%.8g").encode("utf-8")
        return self.table.to_csv(index=False, float_format="%.10g").encode("utf-8")


def sample(cfg: dict[str, Any], g: Generator, train_seed: int, classes: list[str] | None, n_per_class: int, sample_seed: int = 0, variant: str = "plain",
           *, assets: Assets | None = None, model=None) -> Samples:
    """`n_per_class` synthetic rows for each class in `classes` (None = every class), deterministic given (model, classes, n, sample_seed, variant)."""
    from ppfeddata.models.generate import gen_stats_from_cfg, generate

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    if variant == "residual" and g.dp:
        raise ValueError(f"{g.label} uses DP: the residual noise is estimated on the pooled train split, outside epsilon, so only the plain decoder can be sampled")
    n = int(n_per_class)
    if not 1 <= n <= MAX_PER_CLASS:
        raise ValueError(f"n_per_class must be between 1 and {MAX_PER_CLASS}, got {n}")
    assets = assets or load_assets(cfg)
    names = assets.classes
    sel = list(names) if not classes else list(classes)
    unknown = [c for c in sel if c not in names]
    if unknown:
        raise ValueError(f"unknown classes {unknown}; the classes are {names}")
    model = model if model is not None else load_model(cfg, g, train_seed, assets.schema)
    notes: list[str] = []
    stats = None
    if variant == "residual":
        stats = gen_stats_from_cfg(model, assets.data["train"]["X"], assets.data["train"]["y"], assets.schema, cfg["generate"])
        if stats is None:
            variant = "plain"
            notes.append("generate.residual_noise is off in the config: the plain decoder was sampled")
    if g.kind == "mg":
        from ppfeddata.models import marginal as mg
        X, y = mg.sample(model, assets.schema, [n if c in sel else 0 for c in names], int(sample_seed))
    else:
        X, y = generate(model, assets.schema, {names.index(c): n for c in sel}, int(sample_seed), stats=stats)
    return Samples(X, y, readable_table(X, y, assets.pre, names), g, int(train_seed), int(sample_seed), variant, n, sel, notes)


# --------------------------------------------------------------------------------------------------
# What the experiments measured for a generator
# --------------------------------------------------------------------------------------------------
def _cell(summ: pd.DataFrame, config: str, col: str) -> float | None:
    r = summ.loc[summ["config"] == config, col] if col in summ.columns else []
    return float(r.iloc[0]) if len(r) and pd.notna(r.iloc[0]) else None


def quality(summ: pd.DataFrame | None, g: Generator, variant: str) -> dict[str, Any] | None:
    """Mean and std over seeds of the measured numbers of `g` in the variant that is sampled (config names of the ledger)."""
    if summ is None or not len(summ):
        return None
    have = set(summ["config"])
    prefix = g.run_name + ("-plain" if variant == "plain" and f"{g.run_name}-plain-TSTR-rf" in have else "")
    if f"{prefix}-TSTR-rf" not in have:
        return None

    def ms(config: str, metric: str = "macro_f1"):
        m, s = _cell(summ, config, f"{metric}_mean"), _cell(summ, config, f"{metric}_std")
        return None if m is None else (m, s)

    out: dict[str, Any] = {"config": prefix, "tstr_rf": ms(f"{prefix}-TSTR-rf"), "taug_rf": ms(f"{prefix}-TAug-rf"), "tstr_mlp": ms(f"{prefix}-TSTR-mlp"),
                           "taug_mlp": ms(f"{prefix}-TAug-mlp"), "real_only_rf": ms("B0-rf"), "real_only_mlp": ms("B0-mlp")}
    for k, col in (("c2st_auc", "c2st_auc_mean"), ("mia_auc", "mia_auc_mean"), ("dup_rate", "dup_rate"), ("dcr_ratio", "dcr_ratio_mean"), ("eps_max", "dp_eps_max")):
        out[k] = _cell(summ, f"{prefix}-TSTR-rf", f"{col}_mean")
    return out


# --------------------------------------------------------------------------------------------------
# Results, report, interpretation
# --------------------------------------------------------------------------------------------------
def results_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["compute"]["runs_csv"]).parent


def load_summary(cfg: dict[str, Any]) -> pd.DataFrame | None:
    p = results_dir(cfg) / "summary.csv"
    return pd.read_csv(p) if p.exists() else None


def load_interpretation(cfg: dict[str, Any]) -> dict[str, Any] | None:
    p = results_dir(cfg) / "interpretation.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def report_path(cfg: dict[str, Any]) -> Path:
    return results_dir(cfg) / "reports" / "final_report.md"


def read_report(cfg: dict[str, Any]) -> str | None:
    p = report_path(cfg)
    return p.read_text(encoding="utf-8") if p.exists() else None


def split_markdown(md: str, level: int = 2) -> dict[str, str]:
    """Heading text -> body (up to the next heading of the same or a higher level; deeper headings stay in the body). Lines in code fences are not headings."""
    out: dict[str, list[str]] = {}
    key, fence = None, False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        m = None if fence else re.match(r"^(#{1,6}) +(.*\S)\s*$", line)
        if m and len(m.group(1)) <= level:
            key = m.group(2) if len(m.group(1)) == level else None
            if key is not None:
                out[key] = []
            continue
        if key is not None:
            out[key].append(line)
    return {k: "\n".join(v).strip("\n") for k, v in out.items()}


_IMG = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")


def md_parts(md: str, base: Path) -> list[tuple[str, Any, str]]:
    """Split markdown into [("md", text, ""), ("img", path, alt)]: Streamlit cannot show an image link to a local file."""
    parts, pos = [], 0
    for m in _IMG.finditer(md):
        if md[pos:m.start()].strip():
            parts.append(("md", md[pos:m.start()], ""))
        parts.append(("img", (Path(base) / m.group(2)).resolve(), m.group(1)))
        pos = m.end()
    if md[pos:].strip():
        parts.append(("md", md[pos:], ""))
    return parts


def figure(cfg: dict[str, Any], name: str) -> Path | None:
    p = results_dir(cfg) / "figures" / name
    return p if p.exists() else None


def recommendation(R: dict[str, Any] | None) -> dict[str, Any] | None:
    """The R6 table and the pick(s) of the rule, from interpretation.json."""
    if not R or not (R.get("R6") or {}).get("candidates"):
        return None
    r6 = R["R6"]
    rows = [{"configuration": c["label"], "TAug RF macro-F1": c["taug_f1"], "TSTR RF macro-F1": c["tstr_f1"], "MIA AUC": c["mia_auc"], "epsilon (max)": c["eps_max"],
             "time per round vs B3": c["time_ratio"], "MIA ok": c["ok_mia"], "epsilon ok": c["ok_eps"], "overhead ok": c["ok_overhead"], "eligible": c["eligible"]} for c in r6["candidates"]]
    return {"table": pd.DataFrame(rows), "recommended": r6.get("recommended"), "literal": r6.get("literal"), "tied": r6.get("tied") or [], "tie_break_used": bool(r6.get("tie_break_used")),
            "dp_alternative": (r6.get("dp_alternative") or {}).get("recommended"), "rule": r6.get("rule") or {}}


def premise(R: dict[str, Any] | None) -> dict[str, Any] | None:
    """Do R1 and R2 support the CVAE as a way to improve the IDS? (None without interpretation)"""
    from ppfeddata.interpret_report import premise as _premise

    return _premise(R) if R else None


def why_cvae_note(R: dict[str, Any] | None, cfg: dict[str, Any]) -> str | None:
    """The 'Why a CVAE at all?' paragraph of section 9.0 (generated from the same numbers as the report)."""
    if not R:
        return None
    from ppfeddata.interpret_report import answers

    return next((a for a in answers(R, cfg) if a.startswith("**Why a CVAE at all?**")), None)


# --------------------------------------------------------------------------------------------------
# Data overview
# --------------------------------------------------------------------------------------------------
def data_overview(cfg: dict[str, Any]) -> dict[str, Any]:
    """Class counts per split and sub-class table from the split manifest; the feature blocks from the feature schema. A missing file leaves its entry None."""
    man, sch = limitations.load_manifest(cfg), limitations.load_schema(cfg)
    out: dict[str, Any] = {"manifest": man is not None, "schema": sch is not None, "classes": None, "subclasses": None, "features": None, "rows": None}
    if man:
        lc = man.get("label_counts") or {}
        order = list(sch["label_map"]) if sch else list(lc)
        t = pd.DataFrame({c: lc[c] for c in order if c in lc}).T[["train", "val", "test"]]
        t.index.name = "class"
        out["classes"] = t
        out["rows"] = man.get("rows")
        sub = []
        for name, v in (man.get("subclasses") or {}).items():
            q = (man.get("quota_actual") or {}).get(name, {})
            cb = v.get("cross_block") or {}
            sub.append({"sub-class": name, "split": v.get("split_kind"), "units": v.get("n_units"), "train rows": q.get("train"), "val rows": q.get("val"), "test rows": q.get("test"),
                        "rows dropped (streams across blocks)": (cb["rows_removed"] / cb["eligible_before"]) if cb.get("eligible_before") else None})
        out["subclasses"] = pd.DataFrame(sub)
    if sch:
        by: dict[str, list[str]] = {}
        for b in sch["blocks"]:
            by.setdefault(b["type"], []).append(b["name"])
        out["features"] = {"n_features": sch["n_features"], "blocks": by}
    return out


def imbalance_ratio(classes: pd.DataFrame | None) -> tuple[str, str, float] | None:
    """(largest class, smallest class, ratio) of the train split."""
    if classes is None or not len(classes):
        return None
    tr = classes["train"]
    return str(tr.idxmax()), str(tr.idxmin()), float(tr.max() / tr.min())
