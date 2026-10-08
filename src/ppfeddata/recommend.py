"""Optimisation O4: which configuration for which deployment requirement (`ppfeddata recommend`).

Input: a requirement of an IoT / MQTT deployment. Output: the valid configurations of `results/scorecard.json` that meet it, their Pareto
front under the requirement, and the best one by the chosen priority, with every number read from the scorecard (nothing typed by hand).

Requirement (`Requirement`):
- `trust_server`: False = the aggregation server is honest-but-curious and must not see a client's update or table in the clear. Allowed:
  secure aggregation (M2, M3, M3f, MGd, ...) or local DP (each client's release is DP by itself: M1, MGl).
- `trust_clients`: False = clients may collude with the server. Distributed DP (noise split over the clients: M3f, MGd) then only guarantees
  the epsilon with a single honest client (`eps_one_honest`), which is used as its epsilon.
- `max_eps`: record-level epsilon budget (None = no DP required).
- `max_mb_round`, `max_mb_total`, `max_rounds`: bandwidth of the gateways / broker (MB per round, MB over the whole run, number of rounds).
- `priority`: `macro` (TSTR macro-F1, all 6 classes), `binary` (attack vs normal F1) or `rare` (mean recall of the rare attack classes);
  `min_rare_recall`: a floor on the rare-class recall.

Utility is TSTR (an IDS trained on synthetic data only): the requirement the data-generation framework answers when raw data cannot be
pooled. Configurations within the seed std of each other are ties (the scorecard's rule).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ppfeddata import scorecard as sc

PRIORITY = {"macro": "tstr_f1", "binary": "bin_f1", "rare": "rare_recall"}


@dataclass
class Requirement:
    name: str = "custom"
    trust_server: bool = True
    trust_clients: bool = True
    max_eps: float | None = None
    max_mb_round: float | None = None
    max_mb_total: float | None = None
    max_rounds: int | None = None
    priority: str = "macro"
    min_rare_recall: float | None = None
    note: str = ""
    honest_clients: int | None = None          # O4: epsilon must hold when only h clients add their share of the distributed noise
    max_mia: float | None = None               # O4: calibrated model-access MIA AUC must stay below (configurations not attacked are excluded)


# example requirements of IoT / MQTT deployments (section 9.10 of the report and the demo page)
SCENARIOS = [
    Requirement("lab", True, True, None, note="one operator owns the server and every gateway (in-house IDS lab): no DP needed, best detection"),
    Requirement("cloud-aggregator", False, True, 5.0, note="gateways of one company, aggregation in a third-party cloud: server not trusted, epsilon <= 5"),
    Requirement("consortium", False, False, 10.0, note="brokers of different companies that may collude with the aggregator: epsilon must hold with one honest client"),
    Requirement("strict-privacy", False, True, 1.0, note="sensitive deployments (e.g. healthcare IoT): epsilon <= 1"),
    Requirement("low-bandwidth", False, True, 10.0, max_mb_round=2.0, max_mb_total=100.0, note="gateways on NB-IoT / LoRa / metered 4G backhaul"),
    Requirement("rare-attacks", False, True, 5.0, priority="rare", note="the rare MQTT attacks (DELAYED, SYN, INVALID, WILL) matter most"),
    Requirement("honest-majority", False, True, 10.0, honest_clients=3, max_mia=0.55,
                note="brokers of 5 operators, at most 2 may collude with the aggregator; the released model must resist the membership attack (AUC < 0.55)"),
]


def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def effective(row: dict[str, Any], req: Requirement) -> dict[str, Any]:
    """The row with the epsilon that holds under the trust assumption and the MB totals."""
    r = dict(row)
    eps = _num(r.get("eps"))
    if r.get("dp_mode") == "distributed":
        h = 1 if not req.trust_clients else req.honest_clients
        if h is not None:
            curve = {int(k): _num(v) for k, v in (r.get("eps_honest") or {}).items()}
            # measured curve when there is one; otherwise only h = 1 is known (the one-honest epsilon, conservative for h > 1)
            eps = curve.get(int(h), _num(r.get("eps_one_honest", eps)))
    r["eps_eff"] = eps if r.get("dp") else math.inf
    r["mb_round"] = _num(r.get("bytes_per_round")) / 1e6
    r["mb_total"] = _num(r.get("bytes_total")) / 1e6
    return r


def reasons(r: dict[str, Any], req: Requirement) -> list[str]:
    """Why a row does not meet the requirement (empty = it does). Missing cost numbers do not exclude a row (reported as n/a)."""
    out = []
    if not req.trust_server and not (r.get("secagg") or r.get("dp_mode") == "local"):
        out.append("the server sees the client updates in the clear")
    if req.max_eps is not None and not (r["eps_eff"] <= req.max_eps * (1 + sc.EPS_TOL)):
        out.append(f"epsilon {r['eps_eff']:.3g} > {req.max_eps:g}" if math.isfinite(r["eps_eff"]) else "no DP")
    for key, lim, unit in (("mb_round", req.max_mb_round, "MB/round"), ("mb_total", req.max_mb_total, "MB in total"), ("rounds", req.max_rounds, "rounds")):
        v = _num(r.get(key))
        if lim is not None and not math.isnan(v) and v > lim:
            out.append(f"{v:.3g} {unit} > {lim:g}")
    if req.max_mia is not None and not (_num(r.get("mia_auc")) < req.max_mia):
        out.append("membership attack not measured" if math.isnan(_num(r.get("mia_auc"))) else f"MIA AUC {_num(r.get('mia_auc')):.3f} >= {req.max_mia:g}")
    if req.min_rare_recall is not None and not (_num(r.get("rare_recall")) >= req.min_rare_recall):
        out.append(f"rare-class recall {_num(r.get('rare_recall')):.3f} < {req.min_rare_recall:g}")
    return out


AXES_UTILITY = ("tstr_f1", "bin_f1", "rare_recall")


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """The axes a deployment chooses on (the same rule as the demo page): TSTR macro-F1, binary F1, rare recall (ties within the seed std),
    epsilon under the requirement (2 %), SecAgg, MB per round and MB in total (5 %)."""
    c = [sc._cmp_util(a, b, m) for m in AXES_UTILITY]
    c += [sc._cmp_low(a["eps_eff"], b["eps_eff"], sc.EPS_TOL), int(bool(a.get("secagg"))) - int(bool(b.get("secagg"))),
          sc._cmp_low(a["mb_round"], b["mb_round"], sc.COST_TOL), sc._cmp_low(a["mb_total"], b["mb_total"], sc.COST_TOL)]
    return all(v >= 0 for v in c) and any(v > 0 for v in c)


def command(label: str, eps: float | None) -> str:
    """Command that (re)produces a configuration of the scorecard."""
    base = label.split(" ")[0]
    e = f" --eps {eps:g}" if eps is not None and math.isfinite(eps) else ""
    if base.startswith(("M1o", "M3o")) and not base.endswith(("-c16", "-c32")):
        return f"python -m ppfeddata.cli tune-dp-full --final{e}"
    if base.startswith("M3f"):
        return f"python -m ppfeddata.cli tune-fedsgd --final{e}"
    if base.startswith("MG"):
        return f"python -m ppfeddata.cli tune-marginal --final{e}"
    if base.startswith("M2-c"):
        return f"python -m ppfeddata.cli secagg-bits --levels {base[3:]}"
    if base.startswith("M3o") and base.endswith(("-c16", "-c32")):
        return f"python -m ppfeddata.cli secagg-bits --levels {base[-3:]} --m3-eps {eps:g}"
    if base.startswith("M1-"):
        return f"python -m ppfeddata.cli m1 --tuned{e}"
    if base.startswith("M3-"):
        return f"python -m ppfeddata.cli m3 --eps {eps:g}"
    return {"B3": "python -m ppfeddata.cli b3", "M2": "python -m ppfeddata.cli m2"}.get(base, "")


def recommend(scorecard: dict[str, Any], req: Requirement) -> dict[str, Any]:
    rows = [effective(r, req) for r in scorecard["rows"] if r.get("valid") and not r.get("reference")]
    ok, rejected = [], []
    for r in rows:
        why = reasons(r, req)
        (rejected if why else ok).append({**r, "rejected_because": why} if why else r)
    key = PRIORITY[req.priority]
    cand = list(ok)
    for r in cand:
        r["dominated_by"] = [o["label"] for o in cand if o is not r and dominates(o, r)]
        r["pareto"] = not r["dominated_by"]
    front = sorted([r for r in cand if r["pareto"]], key=lambda r: -_num(r.get(key)))
    best = front[0] if front else None
    ties = [r["label"] for r in front[1:] if best is not None and sc._cmp_util(best, r, key) == 0]
    return {"requirement": asdict(req), "priority_metric": key, "best": best, "ties_with_best": ties, "front": front,
            "meets": sorted(cand, key=lambda r: -_num(r.get(key))), "rejected": rejected,
            "command": command(best["label"], best.get("eps_target")) if best else None}


# --------------------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------------------
def _f(x: Any, d: int = 3) -> str:
    v = _num(x)
    return "n/a" if math.isnan(v) else ("∞" if math.isinf(v) else f"{v:.{d}f}")


def render(results: list[dict[str, Any]]) -> str:
    L = ["# Which configuration for which deployment requirement (optimisation round O4)", "",
         "Generated by `ppfeddata recommend` from `results/scorecard.json`; do not edit. Utility = TSTR on the real test split (IDS trained on "
         "synthetic data only), mean over seeds; ties = within the seed std. ε = the record-level epsilon that holds under the requirement's trust "
         "assumption (distributed DP with possibly colluding clients: ε with a single honest client).", ""]
    for res in results:
        q, b = res["requirement"], res["best"]
        L += [f"## {q['name']}", "", f"{q['note']}" if q.get("note") else "", "",
              f"Requirement: server trusted **{'yes' if q['trust_server'] else 'no'}**, clients trusted **{'yes' if q['trust_clients'] else 'no'}**, "
              f"ε ≤ {q['max_eps'] if q['max_eps'] is not None else '∞'}, MB/round ≤ {q['max_mb_round'] or '∞'}, MB total ≤ {q['max_mb_total'] or '∞'}, "
              f"rounds ≤ {q['max_rounds'] or '∞'}, priority **{q['priority']}**" + (f", rare recall ≥ {q['min_rare_recall']}" if q.get("min_rare_recall") else "")
              + (f", at least {q['honest_clients']} honest clients" if q.get("honest_clients") else "") + (f", MIA AUC < {q['max_mia']}" if q.get("max_mia") else "") + ".", ""]
        if b is None:
            L += ["No valid configuration meets this requirement.", ""]
            continue
        L += [f"**Recommended: {b['label']}**" + (f" (ties within the seed std: {', '.join(res['ties_with_best'])})" if res["ties_with_best"] else "") +
              f". Command: `{res['command']}`.", "",
              "| configuration (front) | TSTR macro-F1 | binary F1 | rare recall | ε | SecAgg | DP | MB / round | rounds | MB total |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in res["front"]:
            L.append(f"| {r['label']} | {_f(r.get('tstr_f1'))} | {_f(r.get('bin_f1'))} | {_f(r.get('rare_recall'))} | {_f(r['eps_eff'], 2)} | "
                     f"{'yes' if r.get('secagg') else 'no'} | {r['dp_mode'] if isinstance(r.get('dp_mode'), str) else 'no'} | {_f(r['mb_round'], 2)} | {_f(r.get('rounds'), 0)} | {_f(r['mb_total'], 1)} |")
        if res["rejected"]:
            L += ["", "Excluded: " + "; ".join(f"{r['label']} ({', '.join(r['rejected_because'])})" for r in res["rejected"]) + "."]
        L.append("")
    return "\n".join(L) + "\n"


def _jsonable(x: Any) -> Any:
    return sc._jsonable(x)


def render_html(scorecard: dict[str, Any]) -> str:
    """A static page (no server): the valid rows of the scorecard and filters for the requirement; the same rules as `reasons` in JS."""
    rows = [effective(r, Requirement()) for r in scorecard["rows"] if r.get("valid") and not r.get("reference")]
    keep = ("label", "method", "dp", "dp_mode", "secagg", "eps", "eps_one_honest", "eps_honest", "mia_auc", "tstr_f1", "tstr_f1_std", "bin_f1", "bin_f1_std", "rare_recall",
            "rare_recall_std", "mb_round", "mb_total", "rounds")
    data = json.dumps(_jsonable([{k: r.get(k) for k in keep} for r in rows]), ensure_ascii=False)
    scen = json.dumps([asdict(s) for s in SCENARIOS], ensure_ascii=False)
    return HTML.replace("__ROWS__", data).replace("__SCENARIOS__", scen).replace("__EPS_TOL__", str(sc.EPS_TOL))


SLIM_KEYS = ("label", "method", "dp_mode", "secagg", "eps_eff", "tstr_f1", "tstr_f1_std", "bin_f1", "rare_recall", "mb_round", "rounds", "mb_total")


def slim(res: dict[str, Any]) -> dict[str, Any]:
    """What `recommend.json` keeps: the requirement, the best row and the front with the deciding numbers, the labels that meet it and
    why the others were excluded."""
    row = lambda r: {k: r.get(k) for k in SLIM_KEYS}      # noqa: E731
    return {"requirement": res["requirement"], "priority_metric": res["priority_metric"], "best": row(res["best"]) if res["best"] else None,
            "ties_with_best": res["ties_with_best"], "command": res["command"], "front": [row(r) for r in res["front"]],
            "meets": [r["label"] for r in res["meets"]], "rejected": {r["label"]: r["rejected_because"] for r in res["rejected"]}}


def run(cfg: dict[str, Any], req: Requirement | None = None, out_dir: str | Path | None = None) -> list[dict[str, Any]]:
    res_dir = Path(out_dir) if out_dir else Path(cfg["compute"]["runs_csv"]).parent
    card = sc._from_json(json.loads((res_dir / "scorecard.json").read_text(encoding="utf-8")))
    results = [recommend(card, r) for r in ([req] if req else SCENARIOS)]
    (res_dir / "reports").mkdir(parents=True, exist_ok=True)
    if req is None:
        (res_dir / "recommend.json").write_text(json.dumps(_jsonable([slim(r) for r in results]), indent=1, ensure_ascii=False), encoding="utf-8")
        (res_dir / "reports" / "recommend.md").write_text(render(results), encoding="utf-8")
        (res_dir / "reports" / "recommend.html").write_text(render_html(card), encoding="utf-8")
    return results


HTML = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PP-FedData recommend</title>
<style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--muted:#6b6b66;--line:#deded8;--acc:#1f6f8b;--ok:#2f7d4f;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--muted:#a3a39d;--line:#34342f;--acc:#6fb6d2;--ok:#7cc79a;--card:#1f1f1d}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:1.4rem;margin:.2rem 0 .8rem}
.f{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px 16px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
label{display:flex;flex-direction:column;font-size:.85rem;color:var(--muted)}
select,input{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:4px 6px;margin-top:2px}
.best{margin:14px 0;padding:12px;border-left:4px solid var(--ok);background:var(--card)}
.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
tr.out{color:var(--muted)} tr.front td:first-child{font-weight:600;color:var(--acc)}
.why{font-size:.8rem}
</style></head><body><main>
<h1>Cấu hình nào cho yêu cầu triển khai nào</h1>
<p style="color:var(--muted)">Số đo đọc từ <code>results/scorecard.json</code> (TSTR trên tập test thật, trung bình theo seed). ε là ε còn đúng dưới giả định tin cậy đã chọn.</p>
<div class="f">
<label>Kịch bản mẫu<select id="scn"><option value="">(tự chọn)</option></select></label>
<label>Tin server?<select id="ts"><option value="1">có</option><option value="0">không</option></select></label>
<label>Số client trung thực tối thiểu<select id="tc"><option value="">tất cả (tin các client)</option><option>4</option><option>3</option><option>2</option><option value="1">1 (các client khác có thể thông đồng)</option></select></label>
<label>MIA AUC tối đa<input id="mia" type="number" min="0.5" max="1" step="0.01" placeholder="không giới hạn"></label>
<label>ε tối đa<select id="eps"><option value="">không cần DP</option><option>1</option><option>5</option><option>10</option></select></label>
<label>MB / vòng tối đa<input id="mbr" type="number" min="0" step="0.1" placeholder="∞"></label>
<label>MB tổng tối đa<input id="mbt" type="number" min="0" step="1" placeholder="∞"></label>
<label>Số vòng tối đa<input id="rnd" type="number" min="1" step="1" placeholder="∞"></label>
<label>Ưu tiên<select id="pri"><option value="tstr_f1">macro-F1 (6 lớp)</option><option value="bin_f1">F1 nhị phân</option><option value="rare_recall">recall lớp hiếm</option></select></label>
<label>Recall lớp hiếm tối thiểu<input id="rr" type="number" min="0" max="1" step="0.01" placeholder="0"></label>
</div>
<div class="best" id="best"></div>
<div class="wrap"><table><thead><tr><th>cấu hình</th><th>macro-F1</th><th>F1 nhị phân</th><th>recall lớp hiếm</th><th>ε</th><th>SecAgg</th><th>DP</th><th>MB/vòng</th><th>vòng</th><th>MB tổng</th><th>lý do loại</th></tr></thead><tbody id="tb"></tbody></table></div>
</main>
<script>
const ROWS=__ROWS__, SC=__SCENARIOS__, TOL=__EPS_TOL__;
const $=id=>document.getElementById(id), num=v=>(v===null||v===undefined||v==="")?NaN:(v==="inf"?Infinity:+v);
const fmt=(v,d=3)=>{v=num(v);return isNaN(v)?"n/a":(v===Infinity?"∞":v.toFixed(d))};
SC.forEach((s,i)=>{const o=document.createElement("option");o.value=i;o.textContent=s.name+" - "+s.note;$("scn").appendChild(o)});
$("scn").onchange=()=>{const s=SC[$("scn").value];if(!s){draw();return}
 $("ts").value=s.trust_server?"1":"0";$("tc").value=s.trust_clients?(s.honest_clients?String(s.honest_clients):""):"1";$("mia").value=s.max_mia??"";$("eps").value=s.max_eps??"";$("mbr").value=s.max_mb_round??"";
 $("mbt").value=s.max_mb_total??"";$("rnd").value=s.max_rounds??"";$("pri").value={macro:"tstr_f1",binary:"bin_f1",rare:"rare_recall"}[s.priority];$("rr").value=s.min_rare_recall??"";draw()};
document.querySelectorAll("select,input").forEach(e=>{if(e.id!=="scn")e.addEventListener("input",()=>{$("scn").value="";draw()})});
function effEps(r,h){if(!r.dp)return Infinity;if(r.dp_mode!=="distributed"||!h)return num(r.eps);const c=r.eps_honest||{};return (c[h]!==undefined&&c[h]!==null)?num(c[h]):num(r.eps_one_honest)}
function cmp(a,b,m){const x=num(a[m]),y=num(b[m]);if(isNaN(x)||isNaN(y))return 0;const t=Math.max(num(a[m+"_std"])||0,num(b[m+"_std"])||0);return x>y+t?1:(y>x+t?-1:0)}
function low(x,y,rel){if(isNaN(x)||isNaN(y)||(x===Infinity&&y===Infinity))return 0;if(x===Infinity||y===Infinity)return x===Infinity?-1:1;return x<y*(1-rel)?1:(y<x*(1-rel)?-1:0)}
function dom(a,b){const c=["tstr_f1","bin_f1","rare_recall"].map(m=>cmp(a,b,m));c.push(low(a.e,b.e,TOL),(a.secagg?1:0)-(b.secagg?1:0),low(num(a.mb_round),num(b.mb_round),.05),low(num(a.mb_total),num(b.mb_total),.05));return c.every(v=>v>=0)&&c.some(v=>v>0)}
function draw(){const ts=$("ts").value==="1",tc=$("tc").value,MI=num($("mia").value),E=num($("eps").value),MR=num($("mbr").value),MT=num($("mbt").value),RN=num($("rnd").value),RR=num($("rr").value),P=$("pri").value;
 const rs=ROWS.map(r=>{const e=effEps(r,tc),w=[];
  if(!ts&&!(r.secagg||r.dp_mode==="local"))w.push("server thấy cập nhật");
  if(!isNaN(E)&&!(e<=E*(1+TOL)))w.push(e===Infinity?"không có DP":"ε "+fmt(e,2)+" > "+E);
  if(!isNaN(MR)&&num(r.mb_round)>MR)w.push(fmt(r.mb_round,2)+" MB/vòng");
  if(!isNaN(MT)&&num(r.mb_total)>MT)w.push(fmt(r.mb_total,1)+" MB tổng");
  if(!isNaN(RN)&&num(r.rounds)>RN)w.push(fmt(r.rounds,0)+" vòng");
  if(!isNaN(RR)&&!(num(r.rare_recall)>=RR))w.push("recall hiếm "+fmt(r.rare_recall));
  if(!isNaN(MI)&&!(num(r.mia_auc)<MI))w.push(isNaN(num(r.mia_auc))?"chưa đo MIA":"MIA "+fmt(r.mia_auc));
  return {...r,e,w}});
 const ok=rs.filter(r=>!r.w.length);ok.forEach(r=>r.front=!ok.some(o=>o!==r&&dom(o,r)));
 rs.sort((a,b)=>(a.w.length?1:0)-(b.w.length?1:0)||(b.front?1:0)-(a.front?1:0)||num(b[P])-num(a[P]));
 const fr=ok.filter(r=>r.front).sort((a,b)=>num(b[P])-num(a[P])),best=fr[0];
 $("best").innerHTML=best?`<b>Khuyến nghị: ${best.label}</b> · ${({tstr_f1:"macro-F1",bin_f1:"F1 nhị phân",rare_recall:"recall lớp hiếm"})[P]} ${fmt(best[P])} · ε ${fmt(best.e,2)}`
  +(fr.slice(1).filter(r=>cmp(best,r,P)===0).length?` · ngang nhau trong độ lệch seed: ${fr.slice(1).filter(r=>cmp(best,r,P)===0).map(r=>r.label).join(", ")}`:""):"Không có cấu hình hợp lệ nào thoả yêu cầu.";
 $("tb").innerHTML=rs.map(r=>`<tr class="${r.w.length?"out":(r.front?"front":"")}"><td>${r.label}</td><td>${fmt(r.tstr_f1)}</td><td>${fmt(r.bin_f1)}</td><td>${fmt(r.rare_recall)}</td><td>${fmt(r.e,2)}</td><td>${r.secagg?"có":"không"}</td><td>${r.dp_mode||"không"}</td><td>${fmt(r.mb_round,2)}</td><td>${fmt(r.rounds,0)}</td><td>${fmt(r.mb_total,1)}</td><td class="why">${r.w.join("; ")}</td></tr>`).join("")}
draw();
</script></body></html>
"""
