"""Phase 13: the generated blocks of README.md.

The README keeps what does not change by hand (install, commands, layout) and takes everything that depends on a result from `results/interpretation.json`, the run ledger and
the trial session: the answers R1-R6 with their numbers, the reasons for using a CVAE although R1 and R2 are negative, the reference run times and the list of limitations.
`ppfeddata aggregate` rewrites the text between `<!-- BEGIN GENERATED: name -->` and `<!-- END GENERATED: name -->`, so the README cannot drift from the report.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ppfeddata import limitations as lim
from ppfeddata.interpret import PRIMARY

README = Path(__file__).resolve().parents[2] / "README.md"
README_VI = Path(__file__).resolve().parents[2] / "README.vi.md"
READMES = (README, README_VI)
# The results block is written in the README's language (one template per language, every number put in by the code); the other blocks
# and the full answers inside the results block are the report's English text in both READMEs.
BLOCKS = ("results", "libraries", "times", "limitations")


def markers(name: str) -> tuple[str, str]:
    return f"<!-- BEGIN GENERATED: {name} -->", f"<!-- END GENERATED: {name} -->"


def lang_of(path: str | Path) -> str:
    """'vi' for README.vi.md, else 'en'."""
    return "vi" if Path(path).name.endswith(".vi.md") else "en"


def _fmt(x: Any, nd: int = 3, dash: str = "-") -> str:
    return dash if x is None or (isinstance(x, float) and pd.isna(x)) else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def results_facts(R: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """The numbers of the short results block, from interpretation.json and the result files the report reads; a missing source leaves its key out."""
    from ppfeddata.interpret_report import _counts, _fed_dist, _pivot_facts, _q1_facts

    f: dict[str, Any] = {"meta": R["meta"]}
    W = R.get("why") or {}
    f["b0"] = (W.get("b0") or {}).get(PRIMARY)
    ref = {r["id"]: r for r in W.get("rows", [])}
    f["pooled"] = {lab: ref[k]["macro_f1"].get(PRIMARY) for k, lab in (("B1a:TRTR", "B1a"), ("B1b:TRTR", "B1b")) if k in ref}
    pv = _pivot_facts(cfg)
    if pv and f["b0"]:
        f["main"] = [(e, v["mg"]["tstr_f1"], v["cvae"]["tstr_f1"]) for e, v in pv["eps"].items()]
        if pv.get("mg_nodp") and pv.get("b3"):
            f["main_nodp"] = (pv["mg_nodp"]["tstr_f1"], pv["b3"]["tstr_f1"])
        mg = [v["mg"] for v in pv["eps"].values() if v["mg"].get("bytes_per_round")]
        cv = [v["cvae"] for v in pv["eps"].values() if v["cvae"].get("bytes_per_round") and v["cvae"].get("rounds")] + ([pv["b3"]] if (pv.get("b3") or {}).get("rounds") else [])
        if mg and cv:
            f["cost"] = {"mg_mb": (min(r["bytes_per_round"] for r in mg) / 1e6, max(r["bytes_per_round"] for r in mg) / 1e6),
                         "cvae_rounds": (min(r["rounds"] for r in cv), max(r["rounds"] for r in cv)),
                         "cvae_mb": (min(r["bytes_per_round"] * r["rounds"] for r in cv) / 1e6, max(r["bytes_per_round"] * r["rounds"] for r in cv) / 1e6)}
    gens = [r for r in R["R1"]["rows"] if not r["reference"]]
    refs = {r["label"]: r for r in R["R1"]["rows"] if r["reference"]}
    r1 = {}
    for clf in ("rf", "mlp"):
        cs = [r[clf]["macro_f1"] for r in gens if clf in r]
        if cs:
            r1[clf] = {"better": _counts(cs)["better"], "n": len(cs), "best": max(c["delta"] for c in cs),
                       "smote": (refs.get("B1b") or {}).get(clf, {}).get("macro_f1", {}).get("delta")}
    f["R1"] = r1
    if R["R2"]["rows"]:
        cs = [r["macro_f1"] for r in R["R2"]["rows"]]
        f["R2"] = {"worse": _counts(cs)["worse"], "n": len(cs), "lo": min(c["delta"] for c in cs), "hi": max(c["delta"] for c in cs)}
    f["R3"] = next((r["relative_loss"] for r in R["R3"]["rows"] if (r["protocol"], r["classifier"]) == ("TSTR", PRIMARY)), None)
    tstr = [c["macro_f1"]["delta"] for c in R["R4"].get("cost") or [] if c["protocol"] == "TSTR"]
    f["R4_cvae"] = min(tstr) if tstr else None
    q1 = _q1_facts(cfg)
    if q1:
        by = {}
        for d in q1["dp_cost"]:
            by.setdefault(float(d["eps"]), []).append(d["delta"])
        f["R4_mg"] = by
        f["mg_mia"] = (q1["mia"]["nodp"], min(q1["mia"]["dp"].values()), max(q1["mia"]["dp"].values())) if q1.get("mia", {}).get("dp") else None
        f["mg_secagg"] = q1.get("secagg_cost")
    pr = R["R5"]["pairs"]
    if pr:
        f["R5"] = {"worse": _counts([p["macro_f1"] for p in pr])["worse"], "n": len(pr), "overhead": next((o for o in R["R5"]["overhead"] if o["pair"].startswith("M2")), None)}
    r6 = R["R6"]
    if r6.get("candidates"):
        f["R6"] = {"lo": min(r["taug_f1"] for r in r6["candidates"]), "hi": max(r["taug_f1"] for r in r6["candidates"]), "recommended": r6.get("recommended"),
                   "tie_break": r6.get("tie_break_used")}
    D = _fed_dist(R)
    if D:
        cl = D["classifiers"]
        labs = [lab for lab in D["comparisons"]]
        fw = [D["references"][D["comparisons"][lab]["framework"]["against"]]["macro_f1_mean"] for lab in labs if "framework" in D["comparisons"][lab]]
        f["direct"] = {"eps": [cl[lab]["dp"]["target_eps"] for lab in labs], "ids": [cl[lab]["macro_f1_mean"] for lab in labs], "mg": fw,
                       "verdict": [D["comparisons"][lab].get("framework", {}).get("macro_f1", {}).get("effect", "none") for lab in labs],
                       "rounds": max(cl[lab]["dp"]["rounds"] for lab in labs), "nonoise": (cl.get("FedMLPddp-nonoise") or {}).get("macro_f1_mean"),
                       "mg_nodp": (D["references"].get((D.get("nonoise_comparison") or {}).get("against")) or {}).get("macro_f1_mean"),
                       "smote": (((R.get("fed_classifier") or {}).get("classifiers") or {}).get("FedMLPsm") or {}).get("macro_f1_mean")}
    worlds = [w for w in ((R.get("sensitivity") or {}).get("worlds") or [])]
    if len(worlds) > 1:
        from ppfeddata.sensitivity import verdicts
        v = verdicts({"worlds": worlds})
        f["robust"] = (sum(x["same_everywhere"] for x in v), len(v), len(worlds) - 1)
    f["flags"] = [x["id"] for x in R.get("flags") or [] if str(x.get("status", "")).startswith("TRIGGERED")], len(R.get("flags") or [])
    return f


TEXT = {
    "en": {
        "source": "*Generated by `ppfeddata aggregate` from `results/interpretation.json`: {runs} runs, seeds {seeds}, {rows} real test rows from {groups} capture groups. "
                  "Scores are macro-F1 on the real test split (0 to 1, higher is better), mean over the seeds.*",
        "main": "### Main result", "main_lead": "An IDS trained **only on synthetic data** (TSTR), the raw packets never leaving the gateways:",
        "main_head": "| ε | FedDP-Marginal (the framework) | best CVAE with DP | share of the real-data reference {b0} (MG · CVAE) |",
        "nodp": "no DP", "cost": "FedDP-Marginal: 3 rounds, {a}-{b} MB per round. CVAE: {r0:.0f}-{r1:.0f} rounds, {m0:.0f}-{m1:.0f} MB in total.",
        "short": "### Short answers", "short_head": "| question | answer | report |",
        "q": ["**R1** Do synthetic rows added to real data help the IDS?", "**R2** Is the CVAE better than simple class balancing?", "**R3** What does non-IID federated training cost?",
              "**R4** What does DP cost, and what does it bring?", "**R5** What does secure aggregation cost?", "**R6** Which configuration?"],
        "r1": "Hardly. {parts}.", "r1_part": "{clf}: {b} of {n} generators help, by at most {best} (SMOTE {smote})",
        "r2": "No. Worse than class weights and SMOTE in {w} of {n} comparisons ({lo} to {hi}).",
        "r3": "{p} % of TSTR macro-F1 ({clf}), centralised B2 against federated B3.",
        "r4": "CVAE: TSTR down by up to {cv}, and the membership attacks see nothing even without DP.",
        "r4_mg": " FedDP-Marginal: {e1} at ε 1, {e5} at ε 5-10; the attack detects its tables without DP (AUC {nodp}) and less with DP ({lo}-{hi}).",
        "r5": "No loss in utility ({w} of {n} comparisons worse). CVAE: {t}x time, {b}x bytes per parameter.",
        "r5_mg": " FedDP-Marginal: {t}x time, {b}x bytes.",
        "r6": "Not decided by utility: every candidate's TAug macro-F1 lies in {lo}-{hi}; the spec's rule picks {rec}{tb}. Choose by deployment requirement with `recommend`.",
        "r6_tb": " by a tie-break", "sec": ["9.1", "9.2", "9.3", "9.4, 9.12", "9.5, 9.12", "9.6, 9.10b"],
        "read": "### How to read it",
        "pooled": "**When the real data can be pooled**, synthetic data do not help: use class weights or SMOTE ({pp} against {b0} for real data only).",
        "split": "**When the raw data must stay at the gateways**, FedDP-Marginal trains an IDS at {lo}-{hi} % of the real-data reference under DP, far above the CVAE.",
        "direct": "**Against training the IDS itself by FL** (no generator, MLP against MLP): level without DP ({nn} against {mn}; federated SMOTE {sm}); "
                  "with distributed DP {pairs}, with {rounds} rounds instead of 3.",
        "pair": "ε {e}: {ids} against {mg} ({verdict})", "verdict": {"none": "level", "worse": "behind", "better": "ahead"},
        "robust": "**Robustness:** {s} of {n} verdicts hold in all {w} sensitivity worlds (other test groups, rare-class quotas, a per-stream cap).",
        "flags": "**Red flags:** {k} of {n} raised ({ids}), investigated in section 9.7.", "noflags": "**Red flags:** none of {n} raised.",
        "full": "Full answers (section 9.0 of the report)", "cand": "Candidate configurations of the recommendation rule (R6)",
        "cand_lead": "Macro-F1 of the {clf} classifier, mean over seeds:",
        "cand_head": "| configuration | TAug macro-F1 | TSTR macro-F1 | MIA AUC | epsilon (max) | time per round vs B3 | eligible |", "yes": "yes", "no": "no",
        "refs": "For reference (trained and tested on real data): {parts}.", "ref_names": {"B0": "real data only", "B1a": "class weights", "B1b": "SMOTE"},
    },
    "vi": {
        "source": "*Sinh bởi `ppfeddata aggregate` từ `results/interpretation.json`: {runs} lần chạy, seed {seeds}, {rows} gói test thật từ {groups} nhóm capture. "
                  "Điểm là macro-F1 trên tập test thật (0 đến 1, cao hơn là tốt hơn), trung bình các seed.*",
        "main": "### Kết quả chính", "main_lead": "IDS học **chỉ trên dữ liệu sinh** (TSTR), gói tin thô không rời gateway:",
        "main_head": "| ε | FedDP-Marginal (khung) | CVAE có DP tốt nhất | so với mốc dữ liệu thật {b0} (MG · CVAE) |",
        "nodp": "không DP", "cost": "FedDP-Marginal: 3 vòng, {a}-{b} MB mỗi vòng. CVAE: {r0:.0f}-{r1:.0f} vòng, tổng {m0:.0f}-{m1:.0f} MB.",
        "short": "### Trả lời ngắn", "short_head": "| câu hỏi | trả lời | báo cáo |",
        "q": ["**R1** Thêm dữ liệu sinh vào dữ liệu thật có giúp IDS?", "**R2** CVAE có hơn cách cân bằng lớp đơn giản?", "**R3** Học liên kết non-IID tốn bao nhiêu?",
              "**R4** DP tốn gì, đem lại gì?", "**R5** SecAgg tốn gì?", "**R6** Chọn cấu hình nào?"],
        "r1": "Hầu như không. {parts}.", "r1_part": "{clf}: {b}/{n} bộ sinh giúp được, tối đa {best} (SMOTE {smote})",
        "r2": "Không. Thua class weights và SMOTE ở {w}/{n} phép so ({lo} đến {hi}).",
        "r3": "{p} % TSTR macro-F1 ({clf}), B2 tập trung so với B3 liên kết.",
        "r4": "CVAE: TSTR giảm tới {cv}, và tấn công suy luận thành viên không phát hiện gì kể cả khi không DP.",
        "r4_mg": " FedDP-Marginal: {e1} ở ε 1, {e5} ở ε 5-10; tấn công phát hiện được bảng công bố không DP (AUC {nodp}) và kém hơn khi có DP ({lo}-{hi}).",
        "r5": "Không mất utility ({w}/{n} phép so kém hơn). CVAE: {t}× thời gian, {b}× byte mỗi tham số.",
        "r5_mg": " FedDP-Marginal: {t}× thời gian, {b}× byte.",
        "r6": "Utility không quyết định được: TAug macro-F1 của mọi ứng viên nằm trong {lo}-{hi}; luật của đề cương chọn {rec}{tb}. Chọn theo yêu cầu triển khai bằng `recommend`.",
        "r6_tb": " nhờ phá hoà", "sec": ["9.1", "9.2", "9.3", "9.4, 9.12", "9.5, 9.12", "9.6, 9.10b"],
        "read": "### Cách đọc",
        "pooled": "**Khi gộp được dữ liệu thật**, dữ liệu sinh không giúp gì: dùng class weights hoặc SMOTE ({pp} so với {b0} khi chỉ dùng dữ liệu thật).",
        "split": "**Khi gói thô phải ở lại gateway**, FedDP-Marginal huấn luyện được IDS đạt {lo}-{hi} % mốc dữ liệu thật dưới DP, vượt xa CVAE.",
        "direct": "**So với huấn luyện thẳng IDS bằng FL** (không bộ sinh, MLP với MLP): không DP thì ngang nhau ({nn} so với {mn}; SMOTE liên kết {sm}); "
                  "với DP phân tán {pairs}, và cần {rounds} vòng thay vì 3.",
        "pair": "ε {e}: {ids} so với {mg} ({verdict})", "verdict": {"none": "ngang", "worse": "kém hơn", "better": "hơn"},
        "robust": "**Độ bền:** {s}/{n} kết luận giữ nguyên ở cả {w} thế giới độ nhạy (nhóm test khác, hạn ngạch lớp hiếm khác, giới hạn gói mỗi luồng).",
        "flags": "**Cờ đỏ:** bật {k}/{n} ({ids}), đã điều tra ở mục 9.7.", "noflags": "**Cờ đỏ:** không cờ nào bật trong {n}.",
        "full": "Câu trả lời đầy đủ (mục 9.0 của báo cáo, tiếng Anh)", "cand": "Các cấu hình ứng viên của luật khuyến nghị (R6)",
        "cand_lead": "Macro-F1 của bộ phân loại {clf}, trung bình các seed:",
        "cand_head": "| cấu hình | TAug macro-F1 | TSTR macro-F1 | MIA AUC | ε (lớn nhất) | thời gian mỗi vòng so với B3 | đủ điều kiện |", "yes": "có", "no": "không",
        "refs": "Để tham chiếu (học và chấm trên dữ liệu thật): {parts}.", "ref_names": {"B0": "chỉ dữ liệu thật", "B1a": "class weights", "B1b": "SMOTE"},
    },
}


def _n(x: float, nd: int, lang: str, sign: bool = False) -> str:
    """A decimal in the README's language: a true minus sign, and a decimal comma in Vietnamese."""
    s = (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace("-", "−")
    return s.replace(".", ",") if lang == "vi" else s


def _int(x: int, lang: str) -> str:
    """A whole number with the language's thousands separator (30,000 / 30.000)."""
    s = f"{int(x):,}"
    return s.replace(",", ".") if lang == "vi" else s


def results_block(R: dict[str, Any], cfg: dict[str, Any], lang: str = "en") -> str:
    """The short results of the README in its language (main table, short answers R1-R6, how to read them), then the report's full answers
    (section 9.0) and the R6 candidate table, folded. Every number comes from interpretation.json and the result files the report reads."""
    from ppfeddata.interpret_report import answers

    T, f = TEXT[lang], results_facts(R, cfg)
    n = lambda x, nd=3, sign=False: _n(x, nd, lang, sign)                       # noqa: E731
    m = f["meta"]
    seeds = ", ".join(map(str, m["seeds"]))
    L = [T["source"].format(runs=m["n_runs"], seeds=seeds, rows=_int(m["n_test_rows"], lang), groups=m["n_groups"]), ""]
    b0 = f.get("b0")
    if f.get("main"):
        L += [T["main"], "", T["main_lead"], "", T["main_head"].format(b0=n(b0)), "|---|---|---|---|"]
        for e, mg, cv in f["main"]:
            L.append(f"| {e:g} | **{n(mg)}** | {n(cv)} | {mg / b0 * 100:.0f} % · {cv / b0 * 100:.0f} % |")
        if f.get("main_nodp"):
            mg, cv = f["main_nodp"]
            L.append(f"| {T['nodp']} | **{n(mg)}** | {n(cv)} (B3) | {mg / b0 * 100:.0f} % · {cv / b0 * 100:.0f} % |")
        if f.get("cost"):
            c = f["cost"]
            L += ["", T["cost"].format(a=n(c["mg_mb"][0], 2), b=n(c["mg_mb"][1], 2), r0=c["cvae_rounds"][0], r1=c["cvae_rounds"][1], m0=c["cvae_mb"][0], m1=c["cvae_mb"][1])]
        L.append("")
    ans: list[str] = []
    if f["R1"]:
        ans.append(T["r1"].format(parts=". ".join(T["r1_part"].format(clf=clf.upper(), b=v["better"], n=v["n"], best=n(v["best"], 3, True), smote=n(v["smote"], 3, True) if v["smote"] is not None else "-")
                                                for clf, v in f["R1"].items())))
    if f.get("R2"):
        ans.append(T["r2"].format(w=f["R2"]["worse"], n=f["R2"]["n"], lo=n(f["R2"]["lo"], 3, True), hi=n(f["R2"]["hi"], 3, True)))
    if f.get("R3") is not None:
        ans.append(T["r3"].format(p=n(f["R3"] * 100, 1), clf=PRIMARY.upper()))
    if f.get("R4_cvae") is not None:
        mg = ""
        if f.get("R4_mg") and f.get("mg_mia"):
            e1 = f["R4_mg"].get(1.0, [])
            e510 = f["R4_mg"].get(5.0, []) + f["R4_mg"].get(10.0, [])
            if e1 and e510:
                mg = T["r4_mg"].format(e1=f"{n(min(e1), 3, True)} {'đến' if lang == 'vi' else 'to'} {n(max(e1), 3, True)}",
                                       e5=f"{n(min(e510), 3, True)} {'đến' if lang == 'vi' else 'to'} {n(max(e510), 3, True)}",
                                       nodp=n(f["mg_mia"][0]), lo=n(f["mg_mia"][1]), hi=n(f["mg_mia"][2]))
        ans.append(T["r4"].format(cv=n(abs(f["R4_cvae"]))) + mg)
    if f.get("R5") and f["R5"]["overhead"]:
        o, s = f["R5"]["overhead"], f.get("mg_secagg")
        ans.append(T["r5"].format(w=f["R5"]["worse"], n=f["R5"]["n"], t=n(o["time_ratio"], 2), b=n(o["bytes_per_param_ratio"], 2))
                   + (T["r5_mg"].format(t=n(s["time_ratio"], 2), b=n(s["bytes_ratio"], 2)) if s else ""))
    if f.get("R6"):
        r = f["R6"]
        ans.append(T["r6"].format(lo=n(r["lo"], 4), hi=n(r["hi"], 4), rec=r["recommended"], tb=T["r6_tb"] if r["tie_break"] else ""))
    if len(ans) == 6:
        L += [T["short"], "", T["short_head"], "|---|---|---|"] + [f"| {q} | {a} | {s} |" for q, a, s in zip(T["q"], ans, T["sec"])] + [""]
    read = []
    if b0 and f.get("pooled"):
        read.append(T["pooled"].format(pp=" / ".join(f"{k} {n(v)}" for k, v in f["pooled"].items() if v is not None), b0=n(b0)))
    if f.get("main"):
        sh = [mg / b0 * 100 for _, mg, _ in f["main"]]
        read.append(T["split"].format(lo=f"{min(sh):.0f}", hi=f"{max(sh):.0f}"))
    d = f.get("direct")
    if d and d["nonoise"] is not None and d["mg_nodp"] is not None and len(d["mg"]) == len(d["ids"]):
        pairs = "; ".join(T["pair"].format(ids=n(i), mg=n(g), e=f"{e:g}", verdict=T["verdict"].get(v, v)) for e, i, g, v in zip(d["eps"], d["ids"], d["mg"], d["verdict"]))
        read.append(T["direct"].format(nn=n(d["nonoise"]), mn=n(d["mg_nodp"]), sm=n(d["smote"]) if d["smote"] is not None else "-", pairs=pairs,
                                       rounds=_int(d["rounds"], lang)))
    if f.get("robust"):
        s, k, w = f["robust"]
        read.append(T["robust"].format(s=s, n=k, w=w))
    trig, nf = f["flags"]
    if nf:
        read.append(T["flags"].format(k=len(trig), n=nf, ids=", ".join(trig)) if trig else T["noflags"].format(n=nf))
    if read:
        L += [T["read"], ""] + [f"- {x}" for x in read] + [""]
    L += [f"<details><summary>{T['full']}</summary>", ""] + [f"- {a}" for a in answers(R, cfg)] + ["", "</details>"]
    rows = R.get("R6", {}).get("candidates") or []
    if rows:
        L += ["", f"<details><summary>{T['cand']}</summary>", "", T["cand_lead"].format(clf=PRIMARY.upper()), "", T["cand_head"], "|---|---|---|---|---|---|---|"]
        fm = (lambda x, nd: _fmt(x, nd).replace(".", ",")) if lang == "vi" else _fmt   # noqa: E731
        for c in rows:
            L.append(f"| {c['label']} | {fm(c['taug_f1'], 4)} | {fm(c['tstr_f1'], 4)} | {fm(c['mia_auc'], 3)} | {fm(c['eps_max'], 3)} | {fm(c['time_ratio'], 2)} | "
                     f"{T['yes'] if c['eligible'] else T['no']} |")
        W = R.get("why") or {}
        ref = {r["id"]: r for r in W.get("rows", [])}
        parts = [f"{lab} {T['ref_names'][lab]}: " + ", ".join(f"{clf.upper()} {n(v)}" for clf, v in ref[k]["macro_f1"].items()) for k, lab in
                 (("B0:TRTR", "B0"), ("B1a:TRTR", "B1a"), ("B1b:TRTR", "B1b")) if k in ref]
        if parts:
            L += ["", T["refs"].format(parts="; ".join(parts))]
        L += ["", "</details>"]
    return "\n".join(L)


def hardware(cfg: dict[str, Any]) -> str | None:
    """The machine of the benchmark, as written in compute_budget.md."""
    p = Path(cfg["compute"]["runs_csv"]).parent / "reports" / "compute_budget.md"
    if not p.exists():
        return None
    m = re.search(r"Hardware/software: (.*?)\.\s+Commit", p.read_text(encoding="utf-8"), flags=re.S)
    return " ".join(m.group(1).split()) if m else None


def trial_session(cfg: dict[str, Any]) -> dict[str, Any] | None:
    from ppfeddata.run_experiment import status_path, workspace_cfg

    p = status_path(workspace_cfg(cfg, "trial"))
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def times_block(cfg: dict[str, Any]) -> str:
    """Wall time of the one-seed trial of the whole matrix (Gate G4), per configuration, from its session file."""
    s = trial_session(cfg)
    if not s or not s.get("items"):
        return "No trial session found (`artifacts/_trial`); see `results/reports/compute_budget.md` for the measured cost of one epoch and the extrapolation."
    items = [i for i in s["items"] if i.get("seconds") is not None and i.get("group") == "matrix"]
    total = sum(i["seconds"] for i in items) / 60
    hw = hardware(cfg)
    L = [f"Wall time of the whole matrix for **one seed**, run from scratch with one command (`python -m ppfeddata.cli run --stage trial`, seed {s['seeds']}), "
         f"{'on ' + hw if hw else 'on the machine of the experiments'}: **{total:.1f} min**. The evaluation of every generator (fidelity, privacy, TSTR and TAug with RF and MLP) is included. "
         "Source: `results/reports/g4_trial.md`; a Colab CPU session will differ.", "", "| configuration | status | minutes |", "|---|---|---|"]
    L += [f"| {i['id']} | {i['status']} | {i['seconds'] / 60:.1f} |" for i in items]
    return "\n".join(L)


def libraries_block() -> str:
    """Versions of the libraries the study depends on, as installed when the block is written (the exact lock is requirements-lock.txt)."""
    import platform

    from ppfeddata.package import lib_versions

    rows = [f"| {k} | {v} |" for k, v in lib_versions().items() if v]
    head = (f"Python {platform.python_version()} on {platform.system()} {platform.release()}. Versions installed when `ppfeddata aggregate` last wrote this block; "
            "the exact lock is `requirements-lock.txt` (a copy is in `results/repro/pip_freeze.txt`).")
    return "\n".join([head, "", "| library | version |", "|---|---|"] + rows)


def limitations_block(items: list[dict[str, str]]) -> str:
    return "\n".join([f"<details><summary>{len(items)} limitations (the list of section 10 of the final report and of page 5 of the demo)</summary>", ""]
                     + lim.render_md(items) + ["", "</details>"])


def replace_block(text: str, name: str, body: str) -> str:
    b, e = markers(name)
    i, j = text.find(b), text.find(e)
    if i < 0 or j < i:
        raise ValueError(f"README has no `{b}` ... `{e}` pair")
    return text[:i + len(b)] + "\n" + body.strip("\n") + "\n" + text[j:]


def update_readme(path: str | Path, cfg: dict[str, Any], R: dict[str, Any] | None, summ: pd.DataFrame | None = None) -> list[str]:
    """Rewrite the generated blocks of the README; returns the names written. A block whose source is missing is left as it is."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    done = []
    bodies = {"results": results_block(R, cfg, lang_of(p)) if R else None, "libraries": libraries_block(), "times": times_block(cfg),
              "limitations": limitations_block(lim.limitations(cfg, R, summ))}
    for name in BLOCKS:
        if bodies[name] is not None:
            text = replace_block(text, name, bodies[name])
            done.append(name)
    p.write_text(text, encoding="utf-8")
    return done


def update_readmes(cfg: dict[str, Any], R: dict[str, Any] | None, summ: pd.DataFrame | None = None, paths: tuple[Path, ...] = READMES) -> list[str]:
    """`update_readme` for every README that exists (English and Vietnamese); returns `"<file>: <block>"` for each block written."""
    return [f"{p.name}: {name}" for p in paths if Path(p).exists() for name in update_readme(p, cfg, R, summ)]
