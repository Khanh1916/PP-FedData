"""PP-FedData demo (Streamlit): data overview, results, trade-offs and recommendation, sample generation, threat model and limitations.

Run from the repository root:   python -m ppfeddata.cli demo        (or: streamlit run demo/app.py -- --config configs/default.yaml)

Everything shown is read from `results/` and `artifacts/`, which the experiments wrote; the only computation here is sampling from the trained CVAEs
(`ppfeddata.demo_lib.sample`, the code path of the evaluation). The logic lives in `src/ppfeddata/demo_lib.py` and `limitations.py`, where it is tested.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                                                  # the paths of the config are relative to the repository root
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                             # noqa: E402
import streamlit as st                                          # noqa: E402

from ppfeddata import demo_lib as dl                            # noqa: E402
from ppfeddata import limitations                               # noqa: E402
from ppfeddata.utils import load_config                         # noqa: E402

PAGES = ["1. Dữ liệu", "2. Kết quả", "3. Đánh đổi và khuyến nghị", "4. Tạo mẫu", "5. Mô hình đe dọa và hạn chế"]
VARIANT_LABELS = {"residual": "Bộ giải mã + nhiễu dư theo lớp (biến thể của B2, B3, M2 trong báo cáo)", "plain": "Bộ giải mã thuần"}
FIG_TABS = [("Macro-F1 theo cấu hình", "f1_by_config.png", "Macro-F1 trên tập test thật (thanh lỗi = độ lệch chuẩn giữa các seed)."),
            ("Recall lớp hiếm", "recall_rare_classes.png", "Recall của các lớp hiếm (DELAYED, SYN, INVALID, WILL), bộ phân loại RF."),
            ("Utility - privacy", "utility_privacy.png", "Macro-F1 và MIA AUC theo ε (ε = ∞ là B3 không DP)."),
            ("Độ trung thực", "fidelity_vs_eps.png", "Wasserstein và C2ST của dữ liệu sinh theo ε."),
            ("Chi phí", "overhead.png", "Thời gian mỗi vòng FL và số byte mỗi vòng."),
            ("Pareto", "pareto.png", "Macro-F1 so với MIA AUC; kích thước điểm theo overhead.")]


# --------------------------------------------------------------------------------------------------
# Cached loaders (keyed on the file modification time so a new `ppfeddata aggregate` shows up without a restart)
# --------------------------------------------------------------------------------------------------
def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--config", default=None)
    return ap.parse_known_args(sys.argv[1:])[0]


@st.cache_resource(show_spinner=False)
def get_cfg(path: str | None) -> dict:
    return load_config(path)


def _stamp(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def get_summary(path: str | None, stamp: float):
    return dl.load_summary(get_cfg(path))


@st.cache_data(show_spinner=False)
def get_interpretation(path: str | None, stamp: float):
    return dl.load_interpretation(get_cfg(path))


@st.cache_data(show_spinner=False)
def get_report(path: str | None, stamp: float):
    return dl.read_report(get_cfg(path))


@st.cache_resource(show_spinner=False)
def get_assets(path: str | None):
    return dl.load_assets(get_cfg(path))


@st.cache_resource(show_spinner=False)
def get_model(path: str | None, label: str, seed: int):
    cfg = get_cfg(path)
    g = next(x for x in dl.generators(cfg) if x.label == label)
    return dl.load_model(cfg, g, seed, get_assets(path).schema)


def show_md(md: str, base: Path) -> None:
    """Markdown with the image links of the report turned into st.image (the report links the figures relatively)."""
    for kind, val, alt in dl.md_parts(md, base):
        if kind == "md":
            st.markdown(val)
        elif Path(val).exists():
            st.image(str(val), caption=alt or None, width="stretch")


def protections_vi(cfg: dict, g: dl.Generator) -> str:
    fl, dp = cfg.get("fl", {}), cfg.get("dp", {})
    parts = [f"huấn luyện liên kết (FedAvg, {fl.get('num_clients', '?')} client non-IID)" if g.federated else "huấn luyện tập trung (không FL)"]
    if g.dp:
        parts.append(f"DP-SGD ở client (ε mục tiêu {g.eps_target:g}, δ = {dp.get('delta', '?'):g})")
    if g.secagg:
        parts.append("tổng hợp an toàn SecAgg+")
    return ", ".join(parts)


def fmt_ms(x) -> str:
    return "-" if not x else (f"{x[0]:.3f} ± {x[1]:.3f}" if x[1] is not None else f"{x[0]:.3f}")


# --------------------------------------------------------------------------------------------------
# Page 1: data
# --------------------------------------------------------------------------------------------------
def page_data(cfg: dict) -> None:
    st.header("Tổng quan dữ liệu")
    ov = dl.data_overview(cfg)
    if not ov["manifest"]:
        st.warning("Không thấy `results/manifests/split_manifest_<mode>.json`. Chạy `python -m ppfeddata.cli sample` (Phase 3) để tạo.")
        return
    st.markdown(
        "Mỗi dòng là **một gói tin mạng** (packet) của hệ thống IoT dùng giao thức MQTT, với các đặc trưng đã trích xuất. "
        f"Nhãn ở chế độ **{cfg['label_mode']}**: lưu lượng bình thường (NORMAL) và các kiểu tấn công DoS/DDoS (BCF - Basic Connect Flooding, DELAYED, SYN, INVALID, WILL). "
        "Dữ liệu đã được lấy mẫu theo hạn ngạch từng lớp và chia theo **nhóm capture**: không nhóm nào nằm ở hai tập (xem `leakage_report.md`).")
    rows = ov["rows"] or {}
    c = st.columns(4)
    for col, split in zip(c[:3], ("train", "val", "test")):
        col.metric(f"Số dòng {split}", f"{rows.get(split, 0):,}")
    if ov["features"]:
        c[3].metric("Số đặc trưng sau mã hoá", ov["features"]["n_features"])
    cls = ov["classes"]
    st.subheader("Phân bố lớp")
    st.dataframe(cls, width="stretch")
    st.bar_chart(cls["train"], y_label="số dòng train", x_label="lớp", sort=False)
    ratio = dl.imbalance_ratio(cls)
    notes = []
    if ratio:
        notes.append(f"Ở tập train lớp lớn nhất ({ratio[0]}) gấp **{ratio[2]:.0f} lần** lớp nhỏ nhất ({ratio[1]}): đây là mất cân bằng mà dữ liệu sinh được dùng để bù.")
    if cls["val"].nunique() == 1 and cls["test"].nunique() == 1:
        notes.append(f"Val và test được lấy đều {int(cls['val'].iloc[0]):,} và {int(cls['test'].iloc[0]):,} dòng mỗi lớp nên không phản ánh tỉ lệ thật của lưu lượng.")
    for n in notes:
        st.markdown("- " + n)
    with st.expander("Các lớp con (DoS / DDoS) và cách chia train/val/test"):
        sub = ov["subclasses"].copy()
        sub["rows dropped (streams across blocks)"] = sub["rows dropped (streams across blocks)"].map(lambda v: "-" if pd.isna(v) else f"{100 * v:.1f} %")
        st.dataframe(sub, hide_index=True, width="stretch")
        st.caption("`group`: chia theo file capture. `block`: lớp con chỉ có một file capture, chia thành các khối liên tiếp; các TCP stream vắt qua nhiều khối bị bỏ (cột cuối).")
    if ov["features"]:
        with st.expander("Các đặc trưng"):
            names = {"numeric": "số", "binary": "nhị phân", "na_flag": "cờ trống (_is_na)", "categorical": "phân loại (one-hot)"}
            for t, cols in ov["features"]["blocks"].items():
                st.markdown(f"**{names.get(t, t)}** ({len(cols)}): " + ", ".join(f"`{x}`" for x in cols))
    f1, f2 = dl.figure(cfg, "partition_alpha0.5_seed0.png"), dl.figure(cfg, "presence_heatmap.png")
    if f1:
        st.image(str(f1), caption=f"Chia train cho các client theo Dirichlet (hình `{f1.name}`): mỗi client giữ một tỉ lệ lớp rất khác nhau (non-IID).", width="stretch")
    if f2:
        st.image(str(f2), caption="Tỉ lệ ô không trống của từng cột theo lớp con (kiểm toán Phase 2).", width="stretch")


# --------------------------------------------------------------------------------------------------
# Page 2: results
# --------------------------------------------------------------------------------------------------
def page_results(cfg: dict, path: str | None) -> None:
    st.header("Kết quả thí nghiệm")
    summ = get_summary(path, _stamp(dl.results_dir(cfg) / "summary.csv"))
    if summ is None:
        st.warning("Chưa có `results/summary.csv`. Chạy `python -m ppfeddata.cli aggregate`.")
        return
    seeds = sorted({s for v in summ["seeds"].astype(str) for s in v.split(",")})
    st.caption(f"{len(summ)} cấu hình trong sổ chạy, seed {', '.join(seeds)}. Macro-F1 đo trên **tập test thật**: trung bình ± độ lệch chuẩn giữa các seed (`results/summary.csv`).")
    c1, c2, c3 = st.columns(3)
    only = c1.toggle("Chỉ các cấu hình của ma trận spec", value=True)
    clf = c2.radio("Bộ phân loại", ["Tất cả", "rf", "mlp"], horizontal=True)
    proto = c3.radio("Giao thức đánh giá", ["Tất cả", "TRTR", "TSTR", "TAug"], horizontal=True,
                     help="TRTR: huấn luyện và test trên dữ liệu thật (B0, B1). TSTR: huấn luyện chỉ bằng dữ liệu sinh. TAug: dữ liệu thật + dữ liệu sinh.")
    d = summ
    if only:
        d = d[d["in_matrix"]]
    if clf != "Tất cả":
        d = d[d["classifier"] == clf]
    if proto != "Tất cả":
        d = d[d["protocol"] == proto]
    cols = {"config": "cấu hình", "protocol": "giao thức", "classifier": "bộ phân loại", "n_seeds": "số seed", "macro_f1_mean": "macro-F1", "macro_f1_std": "± std",
            "balanced_acc_mean": "balanced acc", "c2st_auc_mean_mean": "C2ST AUC", "mia_auc_mean_mean": "MIA AUC", "dp_eps_max_mean": "ε đạt (max)", "round_s_median_mean": "giây/vòng FL"}
    show = d[[k for k in cols if k in d.columns]].rename(columns=cols)
    st.dataframe(show.style.format(na_rep="-", precision=4), hide_index=True, width="stretch")
    st.download_button("Tải summary.csv", summ.to_csv(index=False).encode("utf-8"), file_name="summary.csv", mime="text/csv")
    st.subheader("Hình")
    tabs = st.tabs([t for t, _, _ in FIG_TABS])
    for tab, (_, fname, cap) in zip(tabs, FIG_TABS):
        with tab:
            p = dl.figure(cfg, fname)
            if p:
                st.image(str(p), caption=cap, width="stretch")
            else:
                st.info(f"Chưa có `results/figures/{fname}`.")
    md = get_report(path, _stamp(dl.report_path(cfg)))
    if md:
        with st.expander("Báo cáo cuối (sinh tự động, tiếng Anh): các bảng 2a-8c"):
            secs = dl.split_markdown(md, 2)
            for k, v in secs.items():
                if k[:2] in ("1.", "2a", "2b", "3.", "4.", "5.", "6.", "7.", "8.", "8b", "8c"):
                    st.markdown(f"#### {k}")
                    show_md(v, dl.report_path(cfg).parent)


# --------------------------------------------------------------------------------------------------
# Page 3: trade-offs and recommendation
# --------------------------------------------------------------------------------------------------
def page_tradeoff(cfg: dict, path: str | None) -> None:
    st.header("Đánh đổi utility - privacy - overhead và khuyến nghị")
    R = get_interpretation(path, _stamp(dl.results_dir(cfg) / "interpretation.json"))
    if R is None:
        st.warning("Chưa có `results/interpretation.json`. Chạy `python -m ppfeddata.cli aggregate` (Phase 12).")
        return
    note = dl.why_cvae_note(R, cfg)
    if note:
        st.warning("Trước khi đọc khuyến nghị:\n\n" + note)
    md = get_report(path, _stamp(dl.report_path(cfg)))
    secs = dl.split_markdown(md, 3) if md else {}
    base = dl.report_path(cfg).parent
    rec = dl.recommendation(R)
    if rec:
        st.subheader("Cấu hình khuyến nghị theo quy tắc R6")
        rule = rec["rule"]
        st.markdown(f"Quy tắc (ngưỡng ở `thresholds` trong config): MIA AUC ≤ {rule.get('mia_auc_max')}, ε ≤ {rule.get('eps_max')} nếu có DP, thời gian mỗi vòng ≤ "
                    f"{rule.get('overhead_ratio_max')} × FL thường (B3); trong các cấu hình thoả, chọn macro-F1 TAug (RF) cao nhất.")
        if rec["recommended"]:
            extra = f" Quy tắc theo đúng chữ chọn **{rec['literal']}**; các cấu hình hoà trong nhiễu: {', '.join(rec['tied'])}; hoà được phá theo mức bảo vệ." if rec["tie_break_used"] else ""
            st.success(f"Khuyến nghị: **{rec['recommended']}**." + extra)
        else:
            st.info("Không cấu hình nào thoả cả ba điều kiện; xem bảng và hình Pareto.")
        if rec["dp_alternative"]:
            st.markdown(f"Nếu cần bảo đảm DP hình thức, lựa chọn DP mà quy tắc sẽ chọn nếu bỏ điều kiện overhead là **{rec['dp_alternative']}**.")
        st.dataframe(rec["table"], hide_index=True, width="stretch")
        st.caption("Khuyến nghị chỉ xếp hạng các cấu hình CVAE với nhau; nó không nói rằng dùng CVAE tốt hơn không dùng (xem cảnh báo ở đầu trang và mục 9.6, 9.9 của báo cáo).")
    guide = next((k for k in secs if k.split()[0] == "9.10"), None)
    if guide:
        st.subheader("Cấu hình nào cho yêu cầu nào (M1, M2, M3)")
        st.caption("Utility không phân biệt được các cấu hình (R6), nên lựa chọn dựa trên cái mà mỗi cấu hình bảo vệ và chi phí đo được, kèm các đặc thù của dữ liệu IoT/MQTT. "
                   "Nội dung sinh từ mục 9.10 của `final_report.md` (tiếng Anh, như mọi đoạn trích từ báo cáo).")
        show_md(secs[guide], base)
    c = st.columns(2)
    for col, name, cap in ((c[0], "utility_privacy.png", "Utility - privacy theo ε."), (c[1], "overhead.png", "Overhead của từng cấu hình.")):
        p = dl.figure(cfg, name)
        if p:
            col.image(str(p), caption=cap, width="stretch")
    p = dl.figure(cfg, "pareto.png")
    if p:
        st.image(str(p), caption="Pareto: macro-F1 so với MIA AUC, kích thước điểm theo overhead.", width="stretch")
    for key in [k for k in secs if k.split()[0] in ("9.0", "9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7", "9.8", "9.9")]:
        with st.expander(key, expanded=key.startswith("9.9")):
            show_md(secs[key], base)


# --------------------------------------------------------------------------------------------------
# Page 4: sample generation
# --------------------------------------------------------------------------------------------------
def page_generate(cfg: dict, path: str | None) -> None:
    st.header("Tạo mẫu từ CVAE đã huấn luyện")
    avail = dl.available_generators(cfg)
    if not avail:
        st.warning("Không thấy mô hình đã huấn luyện trong `artifacts/` (hoặc thiếu dữ liệu đã xử lý trong `data/processed/`). Chạy các Phase 7-10 trước (xem README).")
        return
    try:
        assets = get_assets(path)
    except Exception as e:                                                     # missing processed data / preprocessor
        st.error(f"Không nạp được dữ liệu đã xử lý: {e}")
        return
    st.markdown("Chọn cấu hình, lớp và số lượng; bảng là các **gói tin tổng hợp** do bộ giải mã CVAE sinh ra, đổi về đơn vị gốc (ô trống = đặc trưng không áp dụng). "
                "Cùng cấu hình, seed, lớp và số lượng luôn cho cùng một bảng.")
    by = {g.label: (g, seeds) for g, seeds in avail}
    c1, c2 = st.columns(2)
    label = c1.selectbox("Cấu hình", list(by), format_func=lambda k: f"{k} - {by[k][0].title}")
    g, seeds = by[label]
    train_seed = c2.selectbox("Seed huấn luyện của mô hình", seeds)
    st.caption(f"Cách huấn luyện và bảo vệ: {protections_vi(cfg, g)}.")
    c3, c4, c5 = st.columns(3)
    cls_opt = c3.selectbox("Lớp", ["Tất cả các lớp"] + assets.classes)
    n = c4.number_input("Số mẫu mỗi lớp", min_value=1, max_value=dl.MAX_PER_CLASS, value=100, step=50)
    sample_seed = c5.number_input("Seed lấy mẫu", min_value=0, value=0, step=1)
    if g.dp:
        variant = "plain"
        st.info("Cấu hình có DP chỉ sinh bằng **bộ giải mã thuần**: nhiễu dư theo lớp được ước lượng từ toàn bộ train nên nằm ngoài ε. Mẫu là hậu xử lý của mô hình đã huấn luyện với DP.")
    else:
        variant = st.radio("Cách sinh", ["residual", "plain"], horizontal=True, format_func=lambda v: VARIANT_LABELS[v],
                           help="Nhiễu dư: thêm nhiễu Gauss theo độ lệch chuẩn phần dư của từng lớp (ước lượng trên train) vào giá trị trung bình của bộ giải mã; đây là biến thể dùng cho TAug/TSTR của B2, B3, M2.")
    form = st.radio("Dạng bảng tải về", ["Đơn vị gốc", f"Đặc trưng đã mã hoá ({assets.schema['n_features']} cột)"], horizontal=True)
    if st.button("Sinh mẫu", type="primary"):
        try:
            with st.spinner("Đang nạp mô hình và sinh mẫu..."):
                st.session_state["samples"] = dl.sample(cfg, g, int(train_seed), None if cls_opt == "Tất cả các lớp" else [cls_opt], int(n), int(sample_seed), variant,
                                                         assets=assets, model=get_model(path, g.label, int(train_seed)))
        except Exception as e:
            st.session_state.pop("samples", None)
            st.error(f"Không sinh được mẫu: {e}")
    s = st.session_state.get("samples")
    if s is not None:
        st.subheader(f"{s.generator.label}, seed huấn luyện {s.train_seed}, seed lấy mẫu {s.sample_seed}: {len(s.table):,} dòng ({', '.join(s.classes)})")
        for note in s.notes:
            st.caption(note)
        st.dataframe(s.table.head(500), hide_index=True, width="stretch")
        if len(s.table) > 500:
            st.caption(f"Hiển thị 500 dòng đầu; tệp CSV có đủ {len(s.table):,} dòng.")
        enc = form.startswith("Đặc trưng")
        st.download_button("Tải CSV", lambda: s.csv(encoded=enc), file_name=f"{s.generator.label}_seed{s.train_seed}_{'-'.join(s.classes) if len(s.classes) < 6 else 'all'}_n{s.n_per_class}.csv", mime="text/csv")
    st.subheader("Chất lượng đã đo của cấu hình này")
    q = dl.quality(get_summary(path, _stamp(dl.results_dir(cfg) / "summary.csv")), g, "plain" if g.dp else variant)
    if q is None:
        st.info("Chưa có số đo của cấu hình này trong `results/summary.csv`.")
    else:
        m = st.columns(4)
        m[0].metric("TSTR macro-F1 (RF)", fmt_ms(q["tstr_rf"]), help="Huấn luyện bộ phân loại chỉ bằng dữ liệu sinh, test trên dữ liệu thật.")
        m[1].metric("Chỉ dữ liệu thật (B0, RF)", fmt_ms(q["real_only_rf"]))
        m[2].metric("C2ST AUC", "-" if q["c2st_auc"] is None else f"{q['c2st_auc']:.4f}", help="Gần 1: bộ phân loại phân biệt dữ liệu sinh với dữ liệu thật gần như hoàn hảo.")
        m[3].metric("MIA AUC", "-" if q["mia_auc"] is None else f"{q['mia_auc']:.3f}", help="Tấn công suy luận thành viên; gần 0,5 là ngẫu nhiên. Tấn công này yếu và chưa qua đối chứng dương với CVAE.")
        st.caption(f"Số đo của `{q['config']}` (trung bình ± std qua các seed). " + (f"ε đạt (max theo client, δ = {cfg['dp']['delta']:g}): {q['eps_max']:.3f}." if q["eps_max"] else "Không có DP: không có bảo đảm ε."))
    if not g.dp:
        st.warning("Cấu hình này không có bảo đảm quyền riêng tư hình thức. Mẫu sinh có thể mang thông tin về dữ liệu train; các phép đo thực nghiệm (MIA, tỉ lệ trùng) chỉ là bằng chứng yếu.")
    st.caption("Mẫu chỉ để minh hoạ. So TSTR với \"chỉ dữ liệu thật\" ở trên để biết dữ liệu sinh thay được dữ liệu thật đến đâu; kết luận về IDS ở trang 3.")


# --------------------------------------------------------------------------------------------------
# Page 5: threat model and limitations
# --------------------------------------------------------------------------------------------------
def page_threat(cfg: dict, path: str | None) -> None:
    st.header("Mô hình đe dọa và hạn chế")
    sa = cfg.get("secagg", {})
    st.markdown(
        "**Mô hình đe dọa (spec).** Máy chủ FL là *honest-but-curious*: chạy đúng giao thức nhưng cố suy luận từ những gì nó nhận được. "
        "Các client trung thực và **không thông đồng** với nhau hay với máy chủ. Mô phỏng chạy trên một máy, không có mạng thật.")
    st.markdown("| Cơ chế | Bảo vệ khỏi | Không bảo vệ |\n|---|---|---|\n"
                "| DP-SGD ở client | Rò rỉ thông tin về từng **bản ghi** huấn luyện qua mô hình toàn cục và dữ liệu sinh từ nó (hậu xử lý). ε ở mức bản ghi | Nhãn lớp; cả một phiên tấn công gồm nhiều packet tương quan; thống kê chuẩn hoá tính tập trung; việc tune siêu tham số |\n"
                f"| SecAgg (SecAgg+ của Flower; {sa.get('num_shares', '?')} chia sẻ, ngưỡng khôi phục {sa.get('reconstruction_threshold', '?')}) | Máy chủ thấy **cập nhật của từng client**: chỉ nhận tổng đã che | Mô hình tổng hợp cuối cùng và dữ liệu sinh từ nó vẫn có thể rò rỉ; không đem lại bảo đảm ε |\n"
                "| DP + SecAgg | Cả hai: máy chủ không thấy cập nhật từng client, và mô hình/dữ liệu sinh có bảo đảm ε | Các giới hạn của DP ở trên |")
    st.caption("Theo spec: SecAgg bảo vệ cập nhật của từng client khỏi máy chủ, còn DP bảo vệ khỏi rò rỉ từ mô hình và dữ liệu sinh.")
    R = get_interpretation(path, _stamp(dl.results_dir(cfg) / "interpretation.json"))
    md = get_report(path, _stamp(dl.report_path(cfg)))
    if md:
        secs = dl.split_markdown(md, 3)
        k = next((x for x in secs if x.startswith("9.8")), None)
        if k:
            with st.expander(k + " (báo cáo, tiếng Anh)"):
                st.markdown(secs[k])
    st.subheader("Những điều dưới đây áp dụng cho mọi con số")
    summ = get_summary(path, _stamp(dl.results_dir(cfg) / "summary.csv"))
    items = limitations.limitations(cfg, R, summ)
    st.caption(f"{len(items)} mục, sinh từ `limitations.py` (cùng danh sách với mục 10 của báo cáo và README; tiếng Anh). Con số lấy từ manifest, schema, config và interpretation.json.")
    for it in items:
        st.markdown(f"**{it['id']} {it['topic']}.** {it['text']}  \n*Nguồn: {it['source']}*")
    note = dl.why_cvae_note(R, cfg)
    if note:
        st.subheader("Vì sao vẫn dùng CVAE?")
        st.markdown(note + "\n\nChi tiết: mục 9.9 của báo cáo (trang 3).")


# --------------------------------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="PP-FedData demo", page_icon=":material/shield:", layout="wide")
    path = _args().config
    cfg = get_cfg(path)
    st.title("PP-FedData")
    st.caption("Sinh dữ liệu bảo toàn quyền riêng tư trong học liên kết để hỗ trợ phát hiện tấn công MQTT DoS/DDoS trên IoT. Demo chạy cục bộ; số liệu đọc từ `results/` và `artifacts/`.")
    page = st.sidebar.radio("Trang", PAGES, key="page")
    R = get_interpretation(path, _stamp(dl.results_dir(cfg) / "interpretation.json"))
    pm = dl.premise(R)
    if pm and pm["unsupported"]:
        st.sidebar.markdown("**Kết luận chính (trang 3):** theo R1 và R2, dữ liệu sinh không cải thiện IDS khi gộp được dữ liệu thật. Lý do vẫn dùng CVAE và phạm vi còn lại của nó được nêu ở trang 3.")
    elif pm:
        st.sidebar.markdown("**Kết luận chính:** xem R1, R2 và mục 9.9 ở trang 3 để biết dữ liệu sinh có cải thiện IDS hay không và vì sao dùng CVAE.")
    {PAGES[0]: lambda: page_data(cfg), PAGES[1]: lambda: page_results(cfg, path), PAGES[2]: lambda: page_tradeoff(cfg, path),
     PAGES[3]: lambda: page_generate(cfg, path), PAGES[4]: lambda: page_threat(cfg, path)}[page]()


main()
