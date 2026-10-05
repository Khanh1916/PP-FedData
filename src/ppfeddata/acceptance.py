"""Phase 13: the Definition of Done of the spec, checked against the files of the repository (`ppfeddata accept` -> results/reports/dod_checklist.md).

Each item of the spec's final checklist becomes one row with a status and the evidence it was decided on:
PASS (met, with the numbers), FAIL (not met), PARTIAL (met except for something the row names), MANUAL (cannot be decided from files: say how to check it).
A check never turns a missing file into a pass: a missing input is a FAIL (or MANUAL when only a human can decide).
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppfeddata import limitations as lim

ROOT = Path(__file__).resolve().parents[2]
PHASE_TESTS = {0: ["test_seed.py", "test_config.py"], 1: ["test_inventory.py"], 2: ["test_harmonize.py"], 3: ["test_split_sample.py"], 4: ["test_preprocess.py"], 5: ["test_leakage.py"],
               6: ["test_eval.py"], 7: ["test_models.py"], 8: ["test_fl.py"], 9: ["test_dp.py"], 10: ["test_secagg.py"], 11: ["test_aggregate.py", "test_experiment.py"],
               12: ["test_interpret.py"], 13: ["test_demo.py", "test_accept.py"]}
EPS_TOLERANCE = 1.02                      # SPEC_DEVIATIONS 9.3: every client reaches epsilon_i <= 1.02 x the target


@dataclass
class Item:
    id: str
    criterion: str
    status: str                           # PASS | FAIL | PARTIAL | MANUAL
    evidence: str


def check_phase_tests(tests_dir: Path) -> Item:
    files = [f for fs in PHASE_TESTS.values() for f in fs]
    miss = [f for f in files if not (tests_dir / f).exists()]
    n = sum(len(re.findall(r"^\s*def test_", (tests_dir / f).read_text(encoding="utf-8"), flags=re.M)) for f in files if f not in miss)
    return Item("D1", "Every Phase has tests (`pytest -q` is green: run it, see below)", "FAIL" if miss else "PASS",
                (f"missing test files: {miss}" if miss else f"{len(PHASE_TESTS)} phases, {len(files)} test files, {n} test functions; "
                 "the pass / fail result of `python -m pytest -q` is not decided here: run it"))


def check_counts(manifest: dict[str, Any] | None, quota: dict[str, list[int]]) -> Item:
    crit = "Total rows and rows of every class match `expected_counts` (the quotas of the config)"
    if not manifest:
        return Item("D2", crit, "FAIL", "split manifest not found")
    lc, bad = manifest.get("label_counts") or {}, []
    for cls, want in quota.items():
        got = [int((lc.get(cls) or {}).get(s, -1)) for s in ("train", "val", "test")]
        if got != [int(x) for x in want]:
            bad.append(f"{cls}: manifest {got} vs quota {want}")
    rows = manifest.get("rows") or {}
    total_want = [sum(int(q[i]) for q in quota.values()) for i in range(3)]
    total_got = [int(rows.get(s, -1)) for s in ("train", "val", "test")]
    if total_got != total_want:
        bad.append(f"totals: manifest {total_got} vs quota {total_want}")
    return Item("D2", crit, "FAIL" if bad else "PASS", "; ".join(bad) if bad else f"{len(quota)} classes match; rows train/val/test = {total_got[0]:,} / {total_got[1]:,} / {total_got[2]:,}")


def check_group_split(manifest: dict[str, Any] | None, group_ids: dict[str, np.ndarray] | None, test_sha_ok: bool | None, tuning_mentions_test: list[str]) -> Item:
    crit = "No group appears in more than one split; the real test split is locked and not used for tuning"
    notes, status = [], "PASS"
    if group_ids:
        sets = {s: set(map(str, g)) for s, g in group_ids.items()}
        names = sorted(sets)
        shared = {f"{a}/{b}": len(sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:] if sets[a] & sets[b]}
        notes.append("group ids of the processed splits: " + (", ".join(f"{k} share {v}" for k, v in shared.items()) if shared else f"disjoint ({', '.join(f'{s} {len(sets[s])}' for s in names)})"))
        status = "FAIL" if shared else status
    elif manifest:
        units = {s: set() for s in ("train", "val", "test")}
        for v in (manifest.get("subclasses") or {}).values():
            for s, us in (v.get("units") or {}).items():
                units.setdefault(s, set()).update(u["unit_id"] for u in us)
        shared = {f"{a}/{b}": len(units[a] & units[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test")) if units[a] & units[b]}
        notes.append("units of the split manifest: " + (f"shared {shared}" if shared else "disjoint"))
        status = "FAIL" if shared else status
    else:
        return Item("D3", crit, "FAIL", "neither the processed data nor the split manifest was found")
    if manifest is not None:
        multi = manifest.get("streams_in_multiple_splits")
        notes.append(f"TCP streams in more than one split: {len(multi) if multi is not None else 'not recorded'}")
        status = "FAIL" if multi else status
    notes.append("test.parquet matches the hash of the manifest" if test_sha_ok else ("test.parquet DOES NOT match the hash of the manifest" if test_sha_ok is False else "test.parquet not on this machine: hash not checked"))
    if test_sha_ok is False:
        status = "FAIL"
    elif test_sha_ok is None and status == "PASS":
        status = "PARTIAL"
    notes.append("tune.py / tune_dp.py never read the test split" if not tuning_mentions_test else f"the test split is mentioned in {tuning_mentions_test}")
    if tuning_mentions_test:
        status = "FAIL"
    return Item("D3", crit, status, "; ".join(notes))


def _find(d: Any, key: str) -> Any:
    """First value of `key` anywhere in a nested dict (the decision logs sit under `harmonize` in the config)."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def check_leakage_report(text: str | None, cfg: dict[str, Any]) -> Item:
    crit = "`leakage_report.md` has C1-C5; the decisions of G1 and G2 are saved"
    if text is None:
        return Item("D4", crit, "FAIL", "leakage_report.md not found")
    have = [c for c in ("C1", "C2", "C3", "C4", "C5") if re.search(rf"\b{c}\b", text)]
    logs = {k: _find(cfg, k) for k in ("g1_log", "g2_log")}
    saved = {k: bool(v and v.get("approved_on") and v.get("decisions")) for k, v in logs.items()}
    ok = len(have) == 5 and all(saved.values())
    return Item("D4", crit, "PASS" if ok else "FAIL", f"checks found: {', '.join(have) or 'none'}; G1 decisions saved in the config: {'yes' if saved['g1_log'] else 'no'}"
                + (f" ({logs['g1_log']['approved_on']})" if saved["g1_log"] else "") + f"; G2: {'yes' if saved['g2_log'] else 'no'}" + (f" ({logs['g2_log']['approved_on']})" if saved["g2_log"] else ""))


def check_seeds(done: dict[str, dict[int, bool]], seeds: list[int]) -> Item:
    crit = "B0, B1, B2, B3, M1 (x3), M2, M3 all have every seed in `runs.csv`"
    miss = [f"{e} seed {s}" for e, d in done.items() for s in seeds if not d.get(s, False)]
    return Item("D5", crit, "FAIL" if miss else "PASS", f"missing: {miss}" if miss else f"{len(done)} configurations x seeds {seeds}: every expected ledger row exists")


def check_epsilon(df: pd.DataFrame | None, f5: dict[str, Any] | None) -> Item:
    crit = "The epsilon reached by DP is reported and matches the target"
    if df is None or "dp_eps_max" not in df or df["dp_eps_max"].notna().sum() == 0:
        return Item("D6a", crit, "FAIL", "no DP rows with `dp_eps_max` in the ledger")
    d = df[df["dp_eps_max"].notna() & df["dp_target_eps"].notna()]
    r = d["dp_eps_max"] / d["dp_target_eps"]
    ok = bool((r <= EPS_TOLERANCE).all() and (r >= 0.9).all())
    extra = f"; red flag F5 (reported epsilon far from the independent recomputation with our own RDP accountant): {f5['status']}" if f5 else ""
    return Item("D6a", crit, "PASS" if ok else "FAIL", f"{len(d)} DP rows, targets {sorted(set(map(float, d['dp_target_eps'])))}: achieved/target from {r.min():.4f} to {r.max():.4f} (bound {EPS_TOLERANCE}){extra}")


def check_secagg(checks: dict[str, Any] | None) -> Item:
    crit = "T-SA1 passes (secure aggregation equals plain FedAvg within the quantisation bound)"
    if not checks or "tsa1" not in checks:
        return Item("D6b", crit, "FAIL", "secagg_checks_<mode>.json not found")
    t = checks["tsa1"]
    ok = float(t["agg_err_max"]) <= float(t["error_bound"])
    return Item("D6b", crit, "PASS" if ok else "FAIL", f"largest aggregation error {float(t['agg_err_max']):.2e} against the bound {float(t['error_bound']):.2e}")


def check_positive_control(R: dict[str, Any] | None) -> Item:
    crit = "The membership-inference attack has a positive control (it detects a generator known to leak)"
    pc = ((R or {}).get("R4") or {}).get("positive_control") or {}
    if not pc.get("available"):
        return Item("D6c", crit, "FAIL", "no positive control on file")
    cop = pc.get("copier_detected_at_zero_noise")
    cop_auc = (pc.get("copier_auc") or {}).get("0.0")
    if pc.get("overfit_cvae_detected"):
        return Item("D6c", crit, "PASS", f"over-fitted CVAE detected (AUC {pc['overfit_cvae_auc']:.3f}, threshold {pc['threshold']})")
    return Item("D6c", crit, "FAIL", f"NOT MET for the CVAE: the attack did not detect an over-fitted CVAE (AUC {pc['overfit_cvae_auc']:.3f}, threshold {pc['threshold']}); "
                f"it does detect a verbatim copier (AUC {cop_auc:.3f}{'' if cop else ', below the threshold'}) and loses it as noise grows. An AUC near 0.5 therefore does not show privacy "
                "(section 9.4, SPEC_DEVIATIONS 7.7)")


def check_report(text: str | None, regenerated: str | None, integrity: float | None) -> Item:
    crit = "`final_report.md` is generated automatically; every number has a source in `results/`"
    if text is None:
        return Item("D7", crit, "FAIL", "final_report.md not found")
    auto = "Auto-generated by `ppfeddata aggregate`" in text
    ev = [f"header says auto-generated: {'yes' if auto else 'no'}"]
    if integrity is not None:
        ev.append(f"macro-F1 recomputed from the saved predictions equals the ledger (largest difference {integrity:.1e})")
    if regenerated is None:
        ev.append("regeneration not run here (`--no-regenerate`)")
        return Item("D7", crit, "PARTIAL" if auto else "FAIL", "; ".join(ev))
    same = regenerated == text
    ev.append("re-running `aggregate` into a scratch folder gives the same report, byte for byte" if same else "re-running `aggregate` gives a DIFFERENT report: the file is not what the code produces from the ledger")
    return Item("D7", crit, "PASS" if (auto and same) else "FAIL", "; ".join(ev))


def check_compute_budget(text: str | None) -> Item:
    crit = "`compute_budget.md` has measured numbers and records every reduction (if any)"
    if text is None:
        return Item("D8", crit, "FAIL", "compute_budget.md not found")
    measured = "Measured on one client-sized dataset" in text and bool(re.search(r"\|\s*dp_\d+\s*\|", text))
    ratio = re.search(r"DP-SGD costs ([0-9.]+)x the plain epoch", text)
    return Item("D8", crit, "PASS" if measured else "FAIL", ("measured epoch times with and without DP-SGD" + (f" (DP-SGD costs {ratio.group(1)}x a plain epoch)" if ratio else "")
                                                             + "; no reduction of the matrix was needed (SPEC_DEVIATIONS 7.9)") if measured else "no measured table found")


def check_limitations(items: list[dict[str, str]], report: str | None, readme: str | None, spec_phrases: dict[str, str]) -> Item:
    crit = "The list of limitations is complete (every item of the spec's list)"
    low = [(i["topic"] + " " + i["text"]).lower() for i in items]
    miss = [k for k, ph in spec_phrases.items() if not any(ph.lower() in t for t in low)]
    in_report = sum(i["topic"] in (report or "") for i in items)
    in_readme = sum(i["topic"] in (readme or "") for i in items)
    ev = f"{len(spec_phrases) - len(miss)} of {len(spec_phrases)} items of the spec's list are in `limitations.py` ({len(items)} items in all); {in_report} of {len(items)} appear in the final report (section 10), {in_readme} in the README"
    ok = not miss and in_report == len(items) and in_readme == len(items)
    return Item("D9", crit, "PASS" if ok else "FAIL", ev + (f"; missing: {miss}" if miss else ""))


def check_readme(text: str | None, commands: set[str], cli_commands: set[str]) -> Item:
    crit = "The README is enough for someone else to re-run everything from scratch"
    if text is None:
        return Item("D10", crit, "FAIL", "README.md not found")
    need = ("## Cài đặt", "## Dữ liệu", "## Chạy từng Phase", "## Tái lập", "## Cấu trúc thư mục", "## Thời gian chạy tham khảo", "## Hạn chế", "requirements-lock.txt")
    miss = [n for n in need if n not in text] + [f"command {c}" for c in sorted(cli_commands - commands)]
    return Item("D10", crit, "FAIL" if miss else "PARTIAL",
                (f"missing: {miss}" if miss else f"has install, data, every phase command ({len(cli_commands)} CLI commands), reproduction steps, layout, run times and limitations; "
                 "not tried on a clean machine (that is the only real test of 'enough')"))


# --------------------------------------------------------------------------------------------------
def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _read(p: Path) -> str | None:
    return p.read_text(encoding="utf-8") if p.exists() else None


def run_all(cfg: dict[str, Any], regenerate: bool = True, root: Path = ROOT) -> list[Item]:
    """Gather the evidence from the repository and decide every row."""
    from ppfeddata.data.preprocess import processed_dir
    from ppfeddata.eval.runs import RunLedger, run_id, runs_csv_path
    from ppfeddata.run_experiment import ledger_names, load_matrix

    mode, seeds = cfg["label_mode"], [int(s) for s in cfg["seeds"]]
    man, results = lim.load_manifest(cfg), Path(cfg["compute"]["runs_csv"]).parent
    R = json.loads((results / "interpretation.json").read_text(encoding="utf-8")) if (results / "interpretation.json").exists() else None
    quota = cfg["quota_11class"] if mode == "11class" else cfg["quota"]

    gids = None
    pd_ = processed_dir(cfg)
    if all((pd_ / f"{s}.npz").exists() for s in ("train", "val", "test")):
        gids = {s: np.load(pd_ / f"{s}.npz")["group_id"] for s in ("train", "val", "test")}
    interim = Path(cfg["paths"]["work_dir"]) / "interim" / mode / "test.parquet"
    sha_ok = (_sha(interim) == (man or {}).get("parquet_sha256", {}).get("test")) if interim.exists() and man else None
    tun = [f for f in ("tune.py", "tune_dp.py") if re.search(r"""['"]test['"]""", (_read(root / "src" / "ppfeddata" / f) or ""))]

    ledger = RunLedger(runs_csv_path(cfg))
    df = ledger.frame()
    done = {e.id: {s: all(ledger.done(run_id(n, s, mode)) for n in ledger_names(cfg, e)) for s in seeds} for e in load_matrix() if e.group == "matrix" and e.runnable}
    dp = df[df["config"].astype(str).str.contains("-plain-") & df["dp_eps_max"].notna()] if len(df) and "dp_eps_max" in df else None
    f5 = next((f for f in (R or {}).get("flags", []) if f.get("id") == "F5"), None)
    sec_json = Path(cfg["compute"]["artifacts_dir"]) / f"secagg_checks_{mode}.json"
    secagg = json.loads(sec_json.read_text(encoding="utf-8")) if sec_json.exists() else None

    report_text = _read(results / "reports" / "final_report.md")
    regenerated = None
    if regenerate and report_text is not None:
        from ppfeddata import aggregate as ag
        with tempfile.TemporaryDirectory() as td:
            ag.aggregate(cfg, out_dir=td)
            regenerated = (Path(td) / "reports" / "final_report.md").read_text(encoding="utf-8")
    items = lim.limitations(cfg, R, pd.read_csv(results / "summary.csv") if (results / "summary.csv").exists() else None)
    readme = _read(root / "README.md")
    from ppfeddata.cli import build_parser
    cli = set(build_parser()._subparsers._group_actions[0].choices)
    named = {c.split()[0] for c in re.findall(r"python -m ppfeddata\.cli ([^`\n]+)", readme or "")}
    leak = Path(cfg.get("leakage", {}).get("report_path", results / "reports" / "leakage_report.md"))

    return [check_phase_tests(root / "tests"), check_counts(man, quota), check_group_split(man, gids, sha_ok, tun), check_leakage_report(_read(leak), cfg),
            check_seeds(done, seeds), check_epsilon(dp, f5), check_secagg(secagg), check_positive_control(R),
            check_report(report_text, regenerated, (R or {}).get("meta", {}).get("integrity_max_abs_diff_vs_ledger")),
            check_compute_budget(_read(results / "reports" / "compute_budget.md")), check_limitations(items, report_text, readme, lim.SPEC_LIST), check_readme(readme, named, cli)]


def render(items: list[Item], cfg: dict[str, Any]) -> str:
    n = {s: sum(i.status == s for i in items) for s in ("PASS", "PARTIAL", "MANUAL", "FAIL")}
    L = ["# Definition of Done (spec, final checklist)", "",
         f"Generated by `ppfeddata accept` for label mode {cfg['label_mode']}, seeds {cfg['seeds']}. Each row is decided from the files of the repository; the evidence is what was found. "
         f"**{n['PASS']} PASS, {n['PARTIAL']} PARTIAL, {n['MANUAL']} MANUAL, {n['FAIL']} FAIL.** PARTIAL = met except for what the row says; FAIL = not met (a missing file is a FAIL, never a pass).", "",
         "| # | criterion | status | evidence |", "|---|---|---|---|"]
    L += [f"| {i.id} | {i.criterion.replace('|', '/')} | **{i.status}** | {i.evidence.replace('|', '/')} |" for i in items]
    L += ["", "`python -m pytest -q` is not run by this command (it takes about ten minutes): run it and read its summary line; D1 only checks that every phase has test files."]
    return "\n".join(L) + "\n"


def write_report(cfg: dict[str, Any], regenerate: bool = True, out: str | Path | None = None, root: Path = ROOT) -> tuple[Path, list[Item]]:
    items = run_all(cfg, regenerate, root)
    p = Path(out) if out else Path(cfg["compute"]["runs_csv"]).parent / "reports" / "dod_checklist.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(items, cfg), encoding="utf-8")
    return p, items
