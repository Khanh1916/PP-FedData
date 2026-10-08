# `demo/` — Streamlit demo

```bash
python -m ppfeddata.cli demo
```

`app.py` is a Streamlit app (interface in Vietnamese) that presents the study to someone who will not read the code. It only **reads**
what the experiments wrote in `results/` and `artifacts/`; the one computation it does is sampling new rows from a trained generator
through the same code path as the evaluation. Its logic lives in `src/ppfeddata/demo_lib.py` and `src/ppfeddata/limitations.py`, where it
is tested (`tests/test_demo.py`); `app.py` only lays it out.

| Page | Shows | Reads |
|---|---|---|
| 1. Dữ liệu (data) | classes, splits, class imbalance, sources, client partition | `results/manifests/`, `results/figures/` |
| 2. Kết quả (results) | summary table, the figures and sections of the final report | `results/summary.csv`, `results/figures/`, `results/reports/final_report.md` |
| 3. Đánh đổi và khuyến nghị (trade-offs, recommendation) | utility - privacy - cost trade-off, the recommended configuration, a filter by deployment requirement with the rules of `recommend` | `results/interpretation.json`, `results/scorecard.json`, `results/reports/final_report.md` |
| 4. Tạo mẫu (generate) | sample rows from FedDP-Marginal (MG-eps1/5/10) or a CVAE: choose the configuration, the class and the number of rows; download CSV | trained generators in `artifacts/`, `results/summary.csv` |
| 5. Mô hình đe dọa và hạn chế (threat model, limitations) | what SecAgg and DP protect, why FedDP-Marginal, the limitations | `results/interpretation.json`, `results/reports/final_report.md`, `limitations.py` |

Pages 2 and 4 need the experiment outputs (`results/summary.csv`, `artifacts/`), which are not committed: run the experiments first or
copy them from a machine that did. No telemetry is sent (`.streamlit/config.toml`).
