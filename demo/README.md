# `demo/` — Streamlit demo

```bash
python -m ppfeddata.cli demo
```

`app.py` is a Streamlit app (interface in English, like the report it quotes; SPEC_DEVIATIONS R.16) that presents the study to someone who will not read the code. FedDP-Marginal, the framework's generator, comes first on every page;
the CVAE routes of the spec are shown as the baseline (SPEC_DEVIATIONS R.15). It only **reads**
what the experiments wrote in `results/` and `artifacts/`; the one computation it does is sampling new rows from a trained generator
through the same code path as the evaluation. Its logic lives in `src/ppfeddata/demo_lib.py` and `src/ppfeddata/limitations.py`, where it
is tested (`tests/test_demo.py`); `app.py` only lays it out.

| Page | Shows | Reads |
|---|---|---|
| 1. Data | classes, splits, class imbalance, sources, client partition | `results/manifests/`, `results/figures/` |
| 2. Results | summary table, the figures and sections of the final report | `results/summary.csv`, `results/figures/`, `results/reports/final_report.md` |
| 3. Trade-offs and recommendation | why FedDP-Marginal, the configuration for a deployment requirement with the rules of `recommend`, section 9.10b, the utility - privacy - cost figures; in collapsed sections the spec's rule R6, the guide of the CVAE route (9.10) and why the spec chose the CVAE | `results/interpretation.json`, `results/scorecard.json`, `results/reports/final_report.md` |
| 4. Generate samples | sample rows from FedDP-Marginal (MG-eps1/5/10, listed first) or a CVAE: choose the configuration, the class and the number of rows; download CSV | trained generators in `artifacts/`, `results/summary.csv` |
| 5. Threat model and limitations | what each mechanism of FedDP-Marginal and of the CVAE route protects, why FedDP-Marginal, the limitations | `results/interpretation.json`, `results/reports/final_report.md`, `limitations.py` |

Pages 2 and 4 need the experiment outputs (`results/summary.csv`, `artifacts/`), which are not committed: run the experiments first or
copy them from a machine that did. No telemetry is sent (`.streamlit/config.toml`).
