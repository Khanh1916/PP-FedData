"""Phase 12: Markdown of section 9 of final_report.md, written from the dict returned by `ppfeddata.interpret.interpret`.

Every sentence that states a result is built from the computed verdicts and numbers, so it cannot disagree with the tables; the fixed text only
explains the rules and the threat model.
"""
from __future__ import annotations

from typing import Any

from ppfeddata.interpret import CLFS, MIA_CHANCE_BAND, PRIMARY, PROTOS

WORD = {"better": "better", "worse": "worse", "none": "no effect"}
SHORT = {"better": "better", "worse": "worse", "none": "="}


def _table(rows: list[dict[str, Any]]) -> str:
    from ppfeddata.aggregate import _table as t
    return t(rows)


def mark(c: dict[str, Any]) -> str:
    return " †" if c["streams"]["effect"] != c["effect"] else ""


def verdict(c: dict[str, Any]) -> str:
    return WORD[c["effect"]] + mark(c)


def short(c: dict[str, Any]) -> str:
    return SHORT[c["effect"]] + mark(c)


def dci(c: dict[str, Any], nd: int = 4) -> str:
    return f"{c['delta']:+.{nd}f} [{c['lo']:+.{nd}f}, {c['hi']:+.{nd}f}]"


def _counts(cs: list[dict[str, Any]]) -> dict[str, int]:
    return {k: sum(c["effect"] == k for c in cs) for k in ("better", "none", "worse")}


def _range(cs: list[dict[str, Any]], nd: int = 4) -> str:
    d = [c["delta"] for c in cs]
    return f"{min(d):+.{nd}f} to {max(d):+.{nd}f}"


def _pn(c: dict[str, Any]) -> str:
    return f"{c['protocol']}-{c['classifier'].upper()}"


# --------------------------------------------------------------------------------------------------
# Answers at a glance
# --------------------------------------------------------------------------------------------------
def answers(R: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    out = []
    th = cfg["thresholds"]
    rare = "/".join(R["meta"]["rare_classes"])

    # R1
    gens = [r for r in R["R1"]["rows"] if not r["reference"]]
    refs = {r["label"]: r for r in R["R1"]["rows"] if r["reference"]}
    if gens:
        parts, leads = [], []
        for clf in CLFS:
            cs = [r[clf]["macro_f1"] for r in gens if clf in r]
            if not cs:
                continue
            k = _counts(cs)
            smote = next((r[clf]["macro_f1"]["delta"] for lab, r in refs.items() if lab == "B1b" and clf in r), None)
            leads.append(f"{clf.upper()}: " + ("no generator passes the rule" if k["better"] == 0 else f"{k['better']} of {len(cs)} generators pass, the best by {max(c['delta'] for c in cs):+.3f}")
                         + (f" (SMOTE {smote:+.3f})" if smote is not None else ""))
            parts.append(f"{clf.upper()}: {k['better']} of {len(cs)} generators better, {k['none']} no effect, {k['worse']} worse (delta {_range(cs)})")
        lead = "; ".join(leads)
        rec = [c["delta"] for r in gens if "rf" in r for c in list(r["rf"]["per_class"].values()) + [r["rf"]["rare_mean"]]]
        sm = [f"{lab} {r[clf]['macro_f1']['delta']:+.3f} ({clf.upper()})" for lab, r in refs.items() for clf in CLFS if clf in r]
        sm_rare = [r["rf"]["rare_mean"]["delta"] for r in refs.values() if "rf" in r]
        out.append(f"**R1 - Does synthetic data improve the IDS?** Short answer: {lead}. Test macro-F1, real + synthetic minus real only: " + "; ".join(parts)
                   + (f". Recall of the rare classes ({rare}, RF) moves by at most {max(abs(x) for x in rec):.3f} for any generator" if rec else "")
                   + (f", against {min(sm_rare):+.3f} to {max(sm_rare):+.3f} (mean of the four) for the SMOTE / class-weight references" if sm_rare else "")
                   + (f". Reference macro-F1 deltas: {', '.join(sm)}." if sm else "."))

    # R2
    r2 = R["R2"]["rows"]
    if r2:
        cs = [r["macro_f1"] for r in r2]
        k = _counts(cs)
        out.append(f"**R2 - Is the CVAE better than the simple methods?** {'No' if k['better'] == 0 else 'Not in general'}. TAug with the CVAE (B2, B3) against class weights (B1a, RF) and SMOTE (B1b): "
                   f"{k['worse']} of {len(cs)} comparisons worse, {k['none']} no effect, {k['better']} better (macro-F1 delta {_range(cs)}; rare-class recall delta {_range([r['rare_mean'] for r in r2], 3)}).")

    # R3
    r3 = R["R3"]["rows"]
    if r3:
        txt = ", ".join(f"{_pn(r)} {r['macro_f1']['delta']:+.4f}" for r in r3)
        out.append(f"**R3 - What does non-IID FL cost?** B2 minus B3 in test macro-F1: {txt}. Relative loss in TSTR-RF: {next((r['relative_loss'] for r in r3 if (r['protocol'], r['classifier']) == ('TSTR', 'rf')), float('nan')) * 100:.1f} %. "
                   "Reported only; the spec sets no pass / fail threshold.")

    # R4
    r4 = R["R4"]
    if r4.get("cost"):
        tstr = [c["macro_f1"] for c in r4["cost"] if c["protocol"] == "TSTR"]
        taug = [c["macro_f1"] for c in r4["cost"] if c["protocol"] == "TAug"]
        k1, k2 = _counts(tstr), _counts(taug)
        mia, pc = r4.get("mia", {}), r4.get("positive_control", {})
        curve = r4["curves"].get(f"TSTR-{PRIMARY}")
        s = (f"**R4 - What does DP cost, and what does it bring?** Cost, plain decoder against the epsilon = infinity point: TSTR macro-F1 delta {_range(tstr)} ({k1['worse']} of {len(tstr)} comparisons worse), "
             f"TAug delta {_range(taug)} ({k2['none']} of {len(taug)} no effect). ")
        if curve:
            s += (f"Over epsilon = 1, 5, 10 the TSTR-{PRIMARY.upper()} curve is {'monotone within noise' if curve['monotone_within_noise'] else 'NOT monotone within noise'} and "
                  f"{'flat' if curve['flat'] else 'not flat'} (range {curve['range']:.3f} against seed std up to {curve['max_std']:.3f}). ")
        if mia:
            s += (f"Benefit: no empirical benefit is measurable. MIA AUC is {min(mia['values']):.3f}-{max(mia['values']):.3f} at every epsilon including infinity "
                  f"(spread {mia['spread']:.3f}), it {'does' if mia.get('approaches_chance') else 'does not'} approach 0.5 as epsilon shrinks, and the attack "
                  + (f"did not detect an over-fitted CVAE (AUC {pc['overfit_cvae_auc']:.3f}), so the positive control of the spec is not met for the CVAE" if pc.get("available") and not pc["overfit_cvae_detected"]
                     else "passed its positive control" if pc.get("available") else "has no positive control on file")
                  + ". What DP brings is the formal guarantee (record level, see 9.8).")
        out.append(s)

    # R5
    pr = R["R5"]["pairs"]
    if pr:
        cs = [p["macro_f1"] for p in pr]
        k = _counts(cs)
        ov = "; ".join(f"{o['pair']}: {o['time_ratio']:.2f}x time per round, {o['bytes_per_param_ratio']:.2f}x bytes per parameter per round" for o in R["R5"]["overhead"])
        out.append(f"**R5 - What does SecAgg cost?** Utility (M2 - B3 and M3 - M1-eps5, {len(cs)} comparisons): {k['none']} no effect, {k['better']} better, {k['worse']} worse "
                   f"(largest |delta| {max(abs(c['delta']) for c in cs):.4f}); {'the differences are inside the seed noise' if k['better'] + k['worse'] == 0 else 'some differences pass the rule'}. Overhead: {ov}.")

    # R6
    r6 = R["R6"]
    if r6.get("candidates"):
        rows = r6["candidates"]
        s = (f"**R6 - Recommended configuration.** Rule: MIA AUC <= {th['mia_auc_max']}, epsilon <= {th['eps_max_recommend']:g} when DP is used, time per round <= {th['overhead_ratio_max']:g}x plain FL (B3), "
             f"then the highest TAug macro-F1 ({PRIMARY.upper()}). ")
        if r6["recommended"]:
            s += (f"Eligible: {', '.join(r['label'] for r in rows if r['eligible'])}. The literal rule picks {r6['literal']}; tied within noise: {', '.join(r6['tied'])}; "
                  + (f"the tie is broken in favour of the stronger protection: **{r6['recommended']}**. " if r6["tie_break_used"] else f"**{r6['recommended']}**. "))
        else:
            s += "No configuration satisfies all three conditions; see the candidate table and the Pareto figure for the trade-off. "
        dpo = [r for r in rows if r["kind"] in ("dp", "dpsa")]
        if dpo:
            s += ("DP configurations fail the overhead filter ("
                  + ", ".join(f"{r['label']} {r['time_ratio']:.2f}x" for r in dpo if r["time_ratio"] and not r["ok_overhead"]) + ")" if any(not r["ok_overhead"] for r in dpo) else "DP configurations pass the filters")
            s += f", not utility or MIA: every candidate's TAug-{PRIMARY.upper()} macro-F1 lies in {min(r['taug_f1'] for r in rows):.4f}-{max(r['taug_f1'] for r in rows):.4f}. "
        alt = r6.get("dp_alternative")
        if alt:
            s += f"If a formal DP guarantee is required, the DP option the rule would pick without the overhead filter is {alt['recommended']} (tied: {', '.join(alt['tied'])})."
        out.append(s)

    # red flags
    W = R.get("why")
    if W and R["R1"]["rows"] and R["R2"]["rows"]:
        pm = _premise(R)
        by = {x["id"]: x for x in W["rows"]}
        tstr = ", ".join(f"{by[k]['macro_f1'][PRIMARY]:.3f} ({k.split(':')[0]})" for k in ("B3:TSTR", "M2:TSTR") if k in by and PRIMARY in by[k]["macro_f1"])
        b0 = W["b0"].get(PRIMARY)
        head = ("**Why a CVAE at all?** " + ("R1 and R2 do not support the CVAE as a way to improve the IDS" if pm["unsupported"] else "R1 and R2 support the CVAE only in part as a way to improve the IDS")
                + f" (worse than class weights and SMOTE in {pm['r2_worse']} of {pm['r2_n']} comparisons; generators better than real data only: "
                + ", ".join(f"{clf.upper()} {v['better']} of {v['n']}, at most {v['best']:+.3f}" + (f" against {pm['smote'][clf]:+.3f} for SMOTE" if pm["smote"].get(clf) is not None else "") for clf, v in pm["per_clf"].items()) + "). "
                "The CVAE is the premise of the spec (a federated, label-conditional generator whose data balance the classes of the IDS), not the outcome of a comparison between generators, and this study is its test.")
        tail = (" What is left of the case is the setting where raw data cannot be pooled, which TAug, B0 and B1 all need: there the synthetic data alone give "
                f"{PRIMARY.upper()} {tstr} against {b0:.3f} for real data only, without pooling the raw data. Not tested, so the CVAE is not shown to be the best option even there: training the classifier itself by FL, "
                "federated class weights or SMOTE, other generators. See 9.9." if tstr and b0 else " See 9.9.")
        out.append(head + tail)
    bits = [f"{f['id']} ({f['title']}): {f['status']}" for f in R["flags"]]
    out.append("**Red flags** (spec Phase 12): " + "; ".join(bits) + ". Details in 9.7.")
    return out


# --------------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------------
def _how_judged(R: dict[str, Any]) -> list[str]:
    m = R["meta"]
    groups = f"{m['n_groups']} capture groups" if m.get("n_groups") else "a handful of capture groups"
    streams = f" ({m['n_streams']:,} streams)" if m.get("n_streams") else ""
    return ["**How a difference is judged.** Every comparison is a difference of test macro-F1 (or of recall) between two configurations, averaged over the "
            f"{len(m['seeds'])} seeds (delta = first minus second). A difference *has an effect* only if (i) its 95 % paired-bootstrap interval excludes 0 and (ii) |delta| is larger "
            "than the seed-to-seed std of the two configurations (the larger of the two, ddof = 0, as in every table). The bootstrap draws the "
            f"{m['n_test_rows']:,} real test rows with replacement, stratified by class, with the same {m['n_boot']} resamples for every run, so the differences are paired. "
            f"Packets of one TCP stream are correlated, which makes that interval too narrow; every verdict is therefore repeated with a bootstrap over whole streams{streams} "
            "and marked **†** when the verdict changes. Neither interval reflects which capture files form the test split "
            f"({groups}; sensitivity A4 was not run) or training randomness beyond the seeds that were run, so a verdict is a statement about this test split and these seeds.", ""]


def _r1(R: dict[str, Any]) -> list[str]:
    rows = []
    for r in R["R1"]["rows"]:
        row = {"configuration": r["label"] + (" (reference)" if r["reference"] else "")}
        for clf in CLFS:
            c = r.get(clf)
            row[f"{clf.upper()}: delta macro-F1 [95% CI]"] = dci(c["macro_f1"]) if c else "-"
            row[f"{clf.upper()}: seed std"] = f"{c['macro_f1']['sigma']:.4f}" if c else "-"
            row[f"{clf.upper()}: verdict"] = verdict(c["macro_f1"]) if c else "-"
        rows.append(row)
    L = ["### 9.1 R1 - Does synthetic data improve the IDS?", "",
         "Delta = test macro-F1 of real + synthetic training data (TAug) minus real data only (B0). B1a (class weights) and B1b (SMOTE) are non-generative references; "
         "DP rows use the plain decoder. **†** = the stream bootstrap gives another verdict.", "", _table(rows), ""]
    rare = R["meta"]["rare_classes"]
    for clf in CLFS:
        rr = []
        for r in R["R1"]["rows"]:
            c = r.get(clf)
            if not c:
                continue
            row = {"configuration": r["label"] + (" (reference)" if r["reference"] else "")}
            for name, cc in c["per_class"].items():
                row[name] = f"{cc['delta']:+.3f} {short(cc)}"
            row["mean of the four [95% CI]"] = f"{dci(c['rare_mean'], 3)} {short(c['rare_mean'])}"
            rr.append(row)
        L += [f"Recall of the rare classes ({', '.join(rare)}), {clf.upper()}: delta TAug - B0 and verdict ('=' = no effect):", "", _table(rr), ""]
    return L


def _r2(R: dict[str, Any]) -> list[str]:
    pointer = " R1 and R2 do not favour the CVAE as an augmentation method; section 9.9 explains why the study uses it and what is left of the case." if _premise(R)["unsupported"] else ""
    rows = [{"CVAE": r["generator"], "compared with": f"{r['reference']}-{r['classifier'].upper()}", "delta macro-F1 [95% CI]": dci(r["macro_f1"]), "seed std": f"{r['macro_f1']['sigma']:.4f}",
             "verdict": verdict(r["macro_f1"]), "delta rare-class recall [95% CI]": dci(r["rare_mean"], 3), "verdict (recall)": verdict(r["rare_mean"])} for r in R["R2"]["rows"]]
    return ["### 9.2 R2 - Is the CVAE better than the simple methods?", "",
            "Delta = TAug macro-F1 of the CVAE (B2 centralised, B3 federated) minus the simple method trained on the same real data (B1a: class weights, RF only; B1b: SMOTE to the same per-class target)." + pointer,
            "", _table(rows), ""]


def _r3(R: dict[str, Any]) -> list[str]:
    rows = [{"protocol / classifier": _pn(r), "B2": f"{r['macro_f1']['mean_a']:.4f}", "B3": f"{r['macro_f1']['mean_b']:.4f}", "B2 - B3 [95% CI]": dci(r["macro_f1"]),
             "relative loss of B3": f"{r['relative_loss'] * 100:+.1f} %", "seed std": f"{r['macro_f1']['sigma']:.4f}", "verdict (information only)": verdict(r["macro_f1"])} for r in R["R3"]["rows"]]
    return ["### 9.3 R3 - What does non-IID federated training cost?", "",
            "B2 (centralised CVAE) minus B3 (the same CVAE trained with FedAvg over 5 non-IID clients). Positive = the centralised generator is better. The spec asks to report the loss, with no pass / fail threshold; the verdict column applies the rule of this section for information.",
            "", _table(rows), ""]


def _r4(R: dict[str, Any]) -> list[str]:
    r4 = R["R4"]
    L = ["### 9.4 R4 - What does DP cost, and what does it bring?", ""]
    if not r4.get("cost"):
        return L + ["_(no DP configuration with predictions in the ledger)_", ""]
    by: dict[str, dict[str, Any]] = {}
    for c in r4["cost"]:
        by.setdefault(c["label"], {"configuration": c["label"]})[f"{c['protocol']}-{c['classifier'].upper()}"] = f"{dci(c['macro_f1'])} {verdict(c['macro_f1'])}"
    L += ["**Cost.** Delta = test macro-F1 of the DP generator (plain decoder, epsilon = worst client) minus B3-plain, the epsilon = infinity point with the same decoder.", "", _table(list(by.values())), ""]
    prow = []
    for p in r4["privacy"]:
        f = lambda x, nd=4: f"{x[0]:.{nd}f} ± {x[1]:.{nd}f}" if x else "-"       # noqa: E731
        prow.append({"epsilon": "infinity (B3-plain)" if p["eps"] == "inf" else p["eps"], "achieved (max over clients and seeds)": f"{p['eps_max']:.3f}" if p["eps_max"] is not None else "-",
                     "MIA AUC": f(p["mia_auc"]), "DCR ratio": f(p["dcr_ratio"]), "duplicate rate": f(p["dup_rate"], 5), "validation ELBO (FL final)": f(p["val_elbo"], 2)})
    L += ["**Empirical privacy and generator quality against epsilon** (mean ± std over seeds; the validation ELBO is each run's own, with its own architecture and beta, so compare "
          "the infinity row with the others only roughly):", "", _table(prow), ""]
    crow = []
    for name, cv in r4["curves"].items():
        crow.append({"curve": name, **{f"eps {k}": f"{v[0]:.3f} ± {v[1]:.3f}" for k, v in cv["values"].items()}, "monotone within noise": "yes" if cv["monotone_within_noise"] else "NO",
                     "range": f"{cv['range']:.3f}", "largest seed std": f"{cv['max_std']:.3f}", "flat (range <= std)": "yes" if cv["flat"] else "no"})
    L += ["**Shape of the utility-privacy curve** (a step to a larger epsilon that is worse by more than the larger seed std of the two points would break monotonicity):", "", _table(crow), ""]
    mia, pc = r4["mia"], r4["positive_control"]
    L += ["**Membership inference.** " + (f"The AUC is within 0.5 ± {mia['chance_band']} at every epsilon, infinity included (spread {mia['spread']:.4f})"
                                          if mia["near_chance"] else f"The AUC leaves the band 0.5 ± {mia['chance_band']} for some epsilon (spread {mia['spread']:.4f})")
          + (f"; the gap to 0.5 is {mia['gap_at_inf']:.4f} at epsilon = infinity and {mia['gap_at_smallest_eps']:.4f} at the smallest epsilon, against a seed std of {mia['seed_std']:.4f}, so the AUC "
             + ("does approach 0.5 as epsilon shrinks." if mia["approaches_chance"] else "does not approach 0.5 as epsilon shrinks (it is already there).") if "approaches_chance" in mia else "."), ""]
    if pc.get("available"):
        cop = ", ".join(f"noise {float(k):g}: {float(v):.3f}" for k, v in sorted(pc["copier_auc"].items(), key=lambda kv: float(kv[0])))      # keys are strings after a JSON round trip
        L += [f"Positive control (b2_cvae.md): on a CVAE over-fitted on 498 members the attack reaches AUC {pc['overfit_cvae_auc']:.3f} (threshold {pc['threshold']}), so it "
              + ("detects an over-fitted CVAE." if pc["overfit_cvae_detected"] else "**does not detect an over-fitted CVAE**; the spec's condition 'MIA AUC approaches 0.5 as epsilon shrinks, after passing the positive control' "
                 "cannot be established with this attack")
              + f". On a pure copier of the members it reaches ({cop}). An AUC near 0.5 therefore says only that no synthetic row is a near-copy of a training row.", ""]
    return L


def _r5(R: dict[str, Any]) -> list[str]:
    r5 = R["R5"]
    L = ["### 9.5 R5 - What does secure aggregation cost?", ""]
    rows = [{"pair": p["pair"], "protocol / classifier": f"{p['protocol']}-{p['classifier'].upper()}", "delta macro-F1 [95% CI]": dci(p["macro_f1"]), "seed std": f"{p['macro_f1']['sigma']:.4f}",
             "verdict": verdict(p["macro_f1"])} for p in r5["pairs"]]
    L += ["Utility: SecAgg aggregates the updates up to a quantisation error of the order 1e-5 per aggregation, so a difference in the table is not a systematic effect of the protocol; it comes from "
          "the sensitivity of the training to perturbations of that size (SPEC_DEVIATIONS 10.4, controls in m2_m3_secagg.md). The criterion is that the difference stays inside the seed noise ('no effect').", "",
          _table(rows), ""]
    orow = [{"pair": o["pair"], "median s/round": f"{o['s_round'][0]:.2f} vs {o['s_round'][1]:.2f}", "time ratio": f"{o['time_ratio']:.2f}",
             "bytes / parameter / round": f"{o['bytes_per_param'][0]:.1f} vs {o['bytes_per_param'][1]:.1f}", "bytes ratio": f"{o['bytes_per_param_ratio']:.2f}"} for o in r5["overhead"]]
    L += ["Overhead (single-machine simulation: compute and bytes are measured, network latency is not):", "", _table(orow), ""]
    return L


def _r6(R: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    r6, th = R["R6"], cfg["thresholds"]
    L = ["### 9.6 R6 - Recommended configuration", ""]
    if not r6.get("candidates"):
        return L + ["_(no candidate)_", ""]
    yn = lambda b: "yes" if b else "NO"            # noqa: E731
    rows = [{"configuration": r["label"], f"TAug-{PRIMARY.upper()}": f"{r['taug_f1']:.4f}", f"TSTR-{PRIMARY.upper()}": f"{r['tstr_f1']:.4f}", "MIA AUC": f"{r['mia_auc']:.3f}",
             "epsilon (max)": f"{r['eps_max']:.3f}" if r["eps_max"] is not None else "-", "time / round vs B3": f"{r['time_ratio']:.2f}" if r["time_ratio"] is not None else "n/a (not federated)",
             f"MIA <= {th['mia_auc_max']}": yn(r["ok_mia"]), f"eps <= {th['eps_max_recommend']:g}": yn(r["ok_eps"]), f"overhead <= {th['overhead_ratio_max']:g}x": yn(r["ok_overhead"]),
             "eligible": yn(r["eligible"]) if r["federated"] else "no (pooled data)"} for r in r6["candidates"]]
    L += [f"Rule (spec Phase 12; thresholds from the config): among the configurations with MIA AUC <= {th['mia_auc_max']}, epsilon (max over clients and seeds) <= {th['eps_max_recommend']:g} when DP is used and "
          f"time per round <= {th['overhead_ratio_max']:g}x plain FL (B3), choose the highest TAug macro-F1 ({PRIMARY.upper()}, mean over seeds). Two additions that the spec does not state (SPEC_DEVIATIONS 12.5): "
          "B2 trains on pooled data, so it is shown but cannot be eligible in a study of federated generation; configurations whose TAug macro-F1 is not worse than the best one by the rule of this section "
          "count as tied, and a tie is broken in favour of the stronger formal protection (finite epsilon, the smaller the better; then SecAgg).", "", _table(rows), ""]
    if r6["recommended"]:
        taug_of = {r["label"]: r["taug_f1"] for r in r6["candidates"]}
        margin = taug_of[r6["literal"]] - taug_of[r6["recommended"]]
        L += [f"**Result.** The literal rule picks **{r6['literal']}** (highest TAug-{PRIMARY.upper()} among the eligible). Tied within noise: {', '.join(r6['tied'])}. "
              + (f"After the tie-break the recommendation is **{r6['recommended']}**; the literal pick leads it by {margin:.1e} in TAug-{PRIMARY.upper()}, far inside the seed std."
                 if r6["tie_break_used"] else f"The recommendation is **{r6['recommended']}**."), ""]
    else:
        L += ["**Result.** No configuration satisfies all three conditions. The trade-off is in the table above and in the Pareto figure (section 6): utility, empirical privacy risk and overhead cannot be optimised together here.", ""]
    taug, cand = [r["taug_f1"] for r in r6["candidates"]], r6["candidates"]
    gens = [r for r in R["R1"]["rows"] if not r["reference"] and PRIMARY in r]
    nb = sum(r[PRIMARY]["macro_f1"]["effect"] == "better" for r in gens)
    f1ev = next((f["evidence"] for f in R["flags"] if f["id"] == "F1"), None) or {}
    b0 = f1ev.get("B0", {}).get(PRIMARY)
    removed = lambda key: ", ".join(r["label"] for r in cand if r["federated"] and not r[key]) or "nobody"       # noqa: E731
    L += [f"**How much the ranking says.** The TAug-{PRIMARY.upper()} macro-F1 of all candidates lies in {min(taug):.4f}-{max(taug):.4f}"
          + (f" (real data only, B0-{PRIMARY.upper()}: {b0:.4f})" if b0 is not None else "") + f"; R1 found {nb} of {len(gens)} generators better than real data only for the {PRIMARY.upper()}. "
          "The ranking therefore separates the candidates little, and the outcome is decided by the filters and the tie-break. "
          f"Removed by the overhead filter: {removed('ok_overhead')}; by the epsilon limit: {removed('ok_eps')}; by the MIA filter: {removed('ok_mia')}"
          + (" (the attack is at chance for every generator, so it filters nothing)." if not any(r["federated"] and not r["ok_mia"] for r in cand) else "."), ""]
    alt = r6.get("dp_alternative")
    if alt:
        row = next(r for r in r6["candidates"] if r["label"] == alt["recommended"])
        L += [f"**If a formal DP guarantee is required.** Dropping only the overhead filter, the rule would pick {alt['literal']} (tied within noise: {', '.join(alt['tied'])}); with the tie-break, **{alt['recommended']}**: "
              f"epsilon {row['eps_max']:.3f}, TAug-{PRIMARY.upper()} {row['taug_f1']:.4f}, but TSTR-{PRIMARY.upper()} {row['tstr_f1']:.4f} and {row['time_ratio']:.2f}x the time per round of B3. "
              "If the synthetic data are to be released instead of the real data (the TSTR scenario), the utility column to read is TSTR, not TAug.", ""]
    tr = r6.get("tstr_ranking")
    if tr and r6["recommended"]:
        L += [f"**Ranked by TSTR instead of TAug.** When the synthetic data replace the real data, the macro-F1 to rank by is TSTR-{PRIMARY.upper()}. The same rule then picks {tr['literal']} "
              f"(tied within noise: {', '.join(tr['tied'])}) and, after the tie-break, **{tr['recommended']}**: "
              + ("the same recommendation as above." if tr["recommended"] == r6["recommended"] else "a different recommendation from the one above."), ""]
    L += ["**What R6 does not say.** R6 ranks the CVAE pipelines against each other; it does not say that a CVAE pipeline is better than not using one. "
          + ("R1 and R2 say it is not when the real training data can be pooled; section 9.9 gives the setting in which the recommendation applies." if _premise(R)["unsupported"]
             else "See R1 and R2 for how a CVAE pipeline compares with the simple methods."), ""]
    return L


def _flags(R: dict[str, Any]) -> list[str]:
    L = ["### 9.7 Red flags", "", "The spec lists five situations that must be investigated and never reported as success:", ""]
    L += [_table([{"flag": f"{f['id']} {f['title']}", "triggered when": f["rule"] or "-", "status": f["status"]} for f in R["flags"]]), ""]
    F = {f["id"]: f for f in R["flags"]}

    f = F["F1"]
    ev = f["evidence"] or {}
    if ev:
        L += [f"**{f['id']}** B0 macro-F1: RF {ev['B0'].get('rf', float('nan')):.4f}, MLP {ev['B0'].get('mlp', float('nan')):.4f}; the highest macro-F1 of any configuration is {ev['highest_macro_f1_of_any_configuration']:.4f} (threshold {ev['threshold']}).", ""]

    f = F["F2"]
    ev2 = f["evidence"] or {"rows": [], "summary": {}}
    sm = ev2["summary"]
    hit = [r for r in ev2["rows"] if r["vs_b0"]["effect"] == "better"]
    L += [f"**{f['id']} TSTR above TRTR.** {len(hit)} of {len(ev2['rows'])} generator / classifier pairs have TSTR better than B0 by the rule of this section."]
    if hit:
        rows = []
        for r in hit:
            row = {"generator": r["label"], "classifier": r["classifier"].upper(), "TSTR - B0 [95% CI]": dci(r["vs_b0"])}
            for name, c in r.get("vs_balanced_real", {}).items():
                row[f"TSTR - {name}-{r['classifier'].upper()} [95% CI]"] = f"{dci(c)} {verdict(c)}"
            row["recall NORMAL (TSTR - B0)"] = f"{r['recall_normal']['delta']:+.3f}"
            row["recall rare classes (TSTR - B0)"] = f"{r['recall_rare']['delta']:+.3f}"
            rows.append(row)
        cols = list(dict.fromkeys(k for row in rows for k in row))
        parts = ["Investigation: B0 is trained on the imbalanced real train pool and TSTR on a class-balanced synthetic set, so the fair real counterpart of TSTR is the class-balanced real reference (B1b, SMOTE)."]
        parts.append("In every flagged pair TSTR stays below that reference (table)." if sm.get("below_balanced_in_all_hits") else "**In at least one flagged pair TSTR is not below that reference.**")
        parts.append("In every flagged pair the gain over B0 comes with lower NORMAL recall and higher rare-class recall, the signature of class balancing." if sm.get("recall_signature_in_all_hits")
                     else "The recall pattern does not show the signature of class balancing in every flagged pair.")
        if sm.get("n_rf_pairs"):
            parts.append(f"For the RF, TSTR is worse than B0 in {sm['n_rf_below_b0']} of {sm['n_rf_pairs']} pairs.")
        parts.append("The preprocessor and the generators are fitted on the train split, the validation split only stops training and selects hyper-parameters, the test split is never used for fitting or tuning "
                     "(SPEC_DEVIATIONS 6.4), and the test groups are disjoint from the train groups.")
        parts.append("Status: " + ("not above the class-balanced reference, so no sign of label leakage in this comparison." if "investigated" in f["status"] else "**open.**"))
        L += ["", _table([{k: row.get(k, "-") for k in cols} for row in rows]), "", " ".join(parts), ""]
    else:
        L += [""]
    for n in f["notes"]:
        L += [n, ""]

    f = F["F3"]
    ev = f["evidence"]
    L += [f"**{f['id']} Copying.** Largest duplicate rate {ev['max_dup_rate'][0]['value']:.5f} ({ev['max_dup_rate'][0]['config']}, seed {ev['max_dup_rate'][0]['seed']}; limit {ev['dup_rate_max']}); "
          f"smallest DCR ratio {ev['min_dcr_ratio'][0]['value']:.3f} ({ev['min_dcr_ratio'][0]['config']}, seed {ev['min_dcr_ratio'][0]['seed']}; limit {ev['dcr_ratio_min']}); "
          f"{ev['n_generators_checked']} generator / seed pairs checked.", ""]

    f = F["F4"]
    ev = f["evidence"]
    rows = [{"configuration": x["config"], "mean macro-F1": f"{x['mean']:.4f}", "std over seeds": f"{x['std']:.4f}", "in the spec matrix": "yes" if x["in_matrix"] else "no (reference or extension)"} for x in ev["largest"]]
    bd = ev["soft_breakdown"]
    soft = ""
    if ev["n_above_seed_std_max"]:
        meth = ", ".join(f"{n} {m}" for m, n in bd["by_method"].items())
        proto = ", ".join(f"{n} {p}" for p, n in bd["by_protocol"].items())
        soft = (f"{ev['n_above_seed_std_max']} configurations exceed {ev['seed_std_max']} (by method: {meth}; by protocol: {proto}). "
                + ("These generators are trained with DP noise (M1, M3) or on very skewed client data (A1, alpha 0.1), and FL training is sensitive to small perturbations (SPEC_DEVIATIONS 10.4). "
                   if set(bd["by_method"]) <= {"M1", "M3", "A1"} else "")
                + "A difference between such rows is not interpretable below the std; more seeds would narrow it (3 seeds is the size the spec asks for).")
    L += [f"**{f['id']} Spread between seeds.** Largest std of macro-F1 over seeds (limit {ev['redflag']}; softer note level {ev['seed_std_max']}):", "", _table(rows), "",
          f"{ev['n_above_redflag']} of {ev['n_configurations']} configurations exceed {ev['redflag']} ({ev['n_above_redflag_in_matrix']} of them in the spec matrix); the largest std inside the matrix is "
          f"{ev['max_std_in_matrix']:.4f}. " + soft, ""]

    f = F["F5"]
    ev = f["evidence"] or {}
    if ev.get("runs"):
        by: dict[Any, list[dict[str, Any]]] = {}
        for r in ev["runs"]:
            by.setdefault(r["target"], []).append(r)
        rows = [{"target epsilon": f"{t:g}", "DP runs": len(v), "reported (min - max)": f"{min(x['reported'] for x in v):.4f} - {max(x['reported'] for x in v):.4f}",
                 "independent (min - max)": f"{min(x['independent'] for x in v):.4f} - {max(x['independent'] for x in v):.4f}", "largest relative difference (absolute)": f"{max(abs(x['rel_diff']) for x in v) * 100:.2f} %"}
                for t, v in sorted(by.items())]
        L += [f"**{f['id']} Epsilon recomputed independently.** Own implementation of the Renyi-DP accountant of the Sampled Gaussian mechanism (integer orders 2-256, the conversion Opacus uses), from each run's "
              f"client sizes, noise multipliers, batch size and logged step counters; the tests check it against Opacus' own functions on the same orders (relative difference below 1e-9). {ev['n_runs']} DP runs: the largest relative "
              f"difference to the reported epsilon is {ev['max_abs_rel_diff'] * 100:.2f} % (limit {ev['rtol'] * 100:.0f} %); the independent value is the same or slightly larger because Opacus also searches fractional orders.",
              "", _table(rows), ""]
        for n in f["notes"]:
            L += [n, ""]
    else:
        L += [f"**{f['id']}** {f['status']}.", ""]
    return L


def _protection() -> list[str]:
    return ["### 9.8 What secure aggregation and differential privacy each protect", "",
            "- **SecAgg** (M2, M3) hides each client's model update from the aggregation server: an honest-but-curious server sees only the sum of the updates of the clients that took part. "
            "It does not limit what the aggregated model, or the synthetic data generated from it, reveals about a training record, and it gives no epsilon (SecAgg+ is not credited with privacy amplification here). "
            "It does not defend against a malicious client.",
            "- **DP-SGD** (M1, M3) bounds what the trained weights, and therefore the synthetic data generated from them, reveal about one record of one client: epsilon at the level of a record (a packet), "
            "worst case over the clients, delta 1e-5. Packets of one TCP stream are strongly correlated, so an entire attack session is protected much less than epsilon suggests (group privacy); the class label "
            "that conditions the generator is not protected; the hyper-parameters were tuned on non-private validation data; the residual-noise variants of the decoder use statistics of the pooled train data and "
            "are outside epsilon (headline numbers use the plain decoder).",
            "- **M3** combines both: the server cannot see an individual (noisy) update, and the aggregate carries the DP guarantee of the clients' noise.",
            "- The empirical checks (duplicate rate, DCR ratio, membership inference) are weak: they detect near-copies only (section 4), so they cannot replace the formal guarantee, and a C2ST close to 1 shows that "
            "the synthetic rows are easy to tell from real ones (SPEC_DEVIATIONS 7.1, 7.6).", ""]


def _premise(R: dict[str, Any]) -> dict[str, Any]:
    """Do R1 and R2 support the CVAE as a way to improve the IDS, the premise of the spec?"""
    gens = [r for r in R["R1"]["rows"] if not r["reference"] and PRIMARY in r]
    nb = sum(r[PRIMARY]["macro_f1"]["effect"] == "better" for r in gens)
    r2 = R["R2"]["rows"]
    nw = sum(r["macro_f1"]["effect"] == "worse" for r in r2)
    per = {}
    for clf in CLFS:
        cs = [r[clf]["macro_f1"] for r in R["R1"]["rows"] if not r["reference"] and clf in r]
        if cs:
            per[clf] = {"better": _counts(cs)["better"], "n": len(cs), "best": max(c["delta"] for c in cs)}
    smote = {clf: next((r[clf]["macro_f1"]["delta"] for r in R["R1"]["rows"] if r["label"] == "B1b" and clf in r), None) for clf in CLFS}
    return {"r1_better": nb, "r1_n": len(gens), "r2_worse": nw, "r2_better": sum(r["macro_f1"]["effect"] == "better" for r in r2), "r2_n": len(r2), "per_clf": per, "smote": smote,
            "unsupported": bool(gens and r2 and nb == 0 and nw == len(r2))}


def premise(R: dict[str, Any]) -> dict[str, Any] | None:
    """Public view of `_premise` for the README and the demo; None when R1 or R2 has no rows."""
    return _premise(R) if R.get("R1", {}).get("rows") and R.get("R2", {}).get("rows") else None


def _r1_digest(R: dict[str, Any]) -> str:
    gens = [r for r in R["R1"]["rows"] if not r["reference"]]
    refs = {r["label"]: r for r in R["R1"]["rows"] if r["reference"]}
    parts = []
    for clf in CLFS:
        cs = [r[clf]["macro_f1"] for r in gens if clf in r]
        if cs:
            parts.append(f"{clf.upper()}: {_counts(cs)['better']} of {len(cs)} generators better than real data only (delta {_range(cs)})")
    sm = [f"{lab} {r[clf]['macro_f1']['delta']:+.3f} ({clf.upper()})" for lab, r in refs.items() for clf in CLFS if clf in r]
    return "; ".join(parts) + (f"; the simple references change macro-F1 by {', '.join(sm)}" if sm else "")


def _why_cvae(R: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    W = R.get("why")
    L = ["### 9.9 Why a CVAE, given R1 and R2", ""]
    if not W or not R["R1"]["rows"] or not R["R2"]["rows"]:
        return L + ["_(not available)_", ""]
    pm = _premise(R)
    by = {x["id"]: x for x in W["rows"]}
    mf = lambda key, clf=PRIMARY: by[key]["macro_f1"].get(clf) if key in by else None             # noqa: E731
    b0 = mf("B0:TRTR")

    L += ["**Where the choice comes from.** The CVAE is the premise of the study, not the outcome of a comparison between generators: the spec (title and 'Idea') asks whether a label-conditional CVAE trained by "
          "federated learning on non-IID clients, protected by DP-SGD, secure aggregation or both, can produce data that balance the classes for the IDS classifiers (RF and MLP). R1 and R2 are the test of "
          "that premise, and the spec asks for results that run against the expectation to be reported as they are.", ""]

    lead = "R1 and R2 do not support the premise." if pm["unsupported"] else "R1 and R2 support the premise only in part."
    L += [f"**What R1 and R2 say about that premise.** {lead} {_r1_digest(R)}. TAug with the CVAE is worse than class weights and SMOTE in {pm['r2_worse']} of {pm['r2_n']} comparisons "
          f"(macro-F1 delta {_range([r['macro_f1'] for r in R['R2']['rows']])}). "
          + ("When the real training data can be pooled and the goal is only a better IDS, class weights or SMOTE are the better choice: they use the same real data and need no generator. "
             "This is a negative result for the premise on this data, and it is reported as such." if pm["unsupported"] else "The simple methods are not uniformly better; see the tables of 9.1 and 9.2."), ""]

    rows = [{"setting": x["setting"], "option": x["option"], "needs pooled real data": "yes" if x["needs_pooled_real_data"] else "no",
             **{f"{clf.upper()} macro-F1": (f"{x['macro_f1'][clf]:.3f}" if clf in x["macro_f1"] else "-") for clf in CLFS}} for x in W["rows"]]
    L += ["**What the choice can still rest on.** TAug, B0 and B1 all train on the pooled real training data, so none of them is available when raw data cannot leave the clients, the setting of the threat model. "
          "There the federated CVAE is the only option of this study that produces data without pooling the raw records, and the question becomes what the synthetic data alone are worth (TSTR) and what protection can be "
          "layered on them. The table puts both settings side by side (test macro-F1, mean over seeds).", "", _table(rows), ""]

    ret, dpl = W["retention_primary"], W.get("dp_tstr_relative_loss")
    t3, t2 = mf("B3:TSTR"), mf("M2:TSTR")
    sent = []
    if t3 is not None and b0:
        sent.append(f"With the synthetic data alone the {PRIMARY.upper()} reaches {t3:.3f} (B3)" + (f" and {t2:.3f} (M2)" if t2 is not None else "") + f" against {b0:.3f} with real data only ("
                    + f"{ret['B3'] * 100:+.0f} %" + (f" and {ret['M2'] * 100:+.0f} %" if "M2" in ret else "") + ")")
    m3, mb0, mb1 = mf("B3:TSTR", "mlp"), mf("B0:TRTR", "mlp"), mf("B1b:TRTR", "mlp")
    if m3 is not None and mb0 is not None and m3 > mb0:
        sent.append(f"the MLP is above real data only ({m3:.3f} against {mb0:.3f}), a gap that flag F2 traces to B0 being trained on imbalanced data" + (f" (it stays below SMOTE, {mb1:.3f})" if mb1 is not None and m3 < mb1 else ""))
    if dpl:
        sent.append(f"DP lowers the synthetic-only macro-F1 by {abs(dpl['max']) * 100:.0f}-{abs(dpl['min']) * 100:.0f} % against epsilon = infinity at every epsilon tested (R4)")
    if sent:
        L += ["; ".join(sent) + ".", ""]

    ts = W.get("train_seconds", {})
    fit = ["it is label-conditional, so classes can be generated on demand", "its parameters form one fixed-size vector, so FedAvg and secure aggregation apply unchanged",
           "it has no BatchNorm, so the per-sample gradients of DP-SGD work in Opacus (spec 7.1; a test checks it)"]
    ok = lambda k: ts.get(k) is not None and ts[k] == ts[k]                                         # noqa: E731
    if ok("B3") and ok("B2"):
        fit.append(f"it trains on a CPU: {ts['B3']:.0f} s for the 30 federated rounds of B3 per seed ({ts['B2']:.0f} s centralised)" + (f", {ts['M1-eps5']:.0f} s with DP" if ok("M1-eps5") else ""))
    L += ["**Why this generator and not another.** The spec does not compare generators, so this is not a result. What can be said is which properties make the CVAE fit the design: " + "; ".join(fit)
          + ". That a GAN or another generator would be heavier or worse was not measured (the optional WGAN-GP baseline of the spec was not run, SPEC_DEVIATIONS 6.7).", ""]

    pc, mia = R["R4"].get("positive_control", {}), W.get("mia", {})
    near = bool(mia) and all(abs(v - 0.5) <= MIA_CHANCE_BAND for v in mia.values() if v == v)
    c2 = W["c2st"].get("B3")
    c2all = [v for v in W["c2st"].values() if v == v]
    c2txt = (f"close to 1 for every generator, from {min(c2all):.4f} to {max(c2all):.4f}" if c2all and min(c2all) > cfg["thresholds"]["c2st_auc_max"] else "see section 5")
    L += ["**What this does not show.**", "",
          (f"- The synthetic rows are easy to tell from real ones (C2ST AUC {c2:.4f} for B3; {c2txt}; section 5), so fidelity is poor and the premise was tested with a generator of this quality; "
           "a generator with better fidelity might behave differently (not tested).") if c2 is not None else "- Fidelity: see section 5.",
          "- Privacy: the membership-inference check is " + ("at chance for every generator, the non-private B3 included" + (f" ({W['mia']['B3']:.3f})" if W['mia'].get('B3') == W['mia'].get('B3') else "") if near else "not at chance for every generator")
          + (", and it did not pass its positive control on an over-fitted CVAE" if pc.get("available") and not pc["overfit_cvae_detected"] else "")
          + ", so the study cannot show that the synthetic data leak less than the real data. Only epsilon (M1, M3) gives a guarantee"
          + (f", and it costs {abs(dpl['max']) * 100:.0f}-{abs(dpl['min']) * 100:.0f} % of the synthetic-only macro-F1 (R4)." if dpl else "."),
          "- Not tested: federated training of the IDS classifier itself (for example FedAvg on the MLP), the obvious alternative when raw data cannot be pooled and no synthetic data have to be shared; federated "
          "versions of class weighting or SMOTE (B1 ran on pooled real data, so R2 compares with centralised simple methods); other generators (the optional WGAN-GP, TVAE or CTGAN-type models, diffusion); "
          "other ways of using the synthetic data than topping every class up to `target_per_class`.", ""]

    r6, bits = R["R6"], []
    b1 = [v for v in (mf("B1a:TRTR"), mf("B1b:TRTR")) if v is not None]
    if pm["unsupported"] and b0 is not None:
        bits.append(f"(1) the real data can be pooled and the goal is a better IDS: class weights or SMOTE, no generator ({PRIMARY.upper()} " + " / ".join(f"{v:.3f}" for v in b1) + f" against {b0:.3f} for real data only);")
    if r6.get("recommended"):
        alt = r6.get("dp_alternative")
        bits.append(f"({len(bits) + 1}) the raw data cannot leave the clients and data must be generated or shared: a federated CVAE, **{r6['recommended']}** by the rule of 9.6"
                    + (f", or {alt['recommended']} if a formal guarantee is required" if alt else "") + ", accepting a synthetic-only macro-F1 below that of real data.")
    L += ["**How to read R6 with this in mind.** R6 ranks the CVAE pipelines against each other; it does not say that a CVAE pipeline is better than not using one. Read R1 to R6 together: " + " ".join(bits), ""]
    return L


def render(R: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    m = R["meta"]
    L = ["## 9. Interpretation (Phase 12)", "",
         f"Generated by `ppfeddata aggregate` (module `interpret.py`) from the run ledger and the test-set predictions saved by every run ({m['n_runs']} runs, seeds {m['seeds']}, "
         f"{m['n_test_rows']:,} real test rows). The rules are those of the spec (Phase 12); the choices it leaves open are recorded in SPEC_DEVIATIONS 12.1-12.10. Every number is recomputed from those files "
         f"(the macro-F1 recomputed from the predictions equals the ledger's to {m['integrity_max_abs_diff_vs_ledger']:.0e}); `results/interpretation.json` holds the same numbers in machine-readable form.", ""]
    L += _how_judged(R)
    L += ["### 9.0 Answers at a glance", ""] + [f"- {a}" for a in answers(R, cfg)] + [""]
    L += _r1(R) + _r2(R) + _r3(R) + _r4(R) + _r5(R) + _r6(R, cfg) + _flags(R) + _protection() + _why_cvae(R, cfg)
    return L
