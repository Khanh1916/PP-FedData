# `notebooks/` — empty on purpose

The spec reserves this folder for notebooks that are **only for viewing** (for example an exploratory look at the data) and hold no logic.
None was needed: every analysis of the study is a command of `python -m ppfeddata.cli`, so it is tested (`tests/`), resumable and
recorded in the run ledger, and its output is a committed report. Where to look instead:

| To see | Open |
|---|---|
| the data: classes, splits, columns, leakage checks | `results/reports/g1_feature_review.md`, `results/reports/leakage_report.md` |
| every result with its interpretation | `results/reports/final_report.md` (start with section 9.0) |
| results interactively, and new synthetic rows | the Streamlit demo: `python -m ppfeddata.cli demo` ([demo/README.md](../demo/README.md)) |

If a notebook is added here, keep the rule of the spec: it reads `results/` and `artifacts/` and calls functions of `src/ppfeddata`;
any computation that a result depends on goes into the package, with a test, and not into the notebook. Clear the outputs before
committing (they can hold data rows; the repository is public and never holds data).
