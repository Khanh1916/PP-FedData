# Clean-install check of the README (2026-10-05)

Written by hand from the logs of that session (the only file of `results/reports/` that is not generated); `ppfeddata accept` (row D10) reads it.

- **Copy:** the files tracked by git plus untracked non-ignored files (`git ls-files -co --exclude-standard`), 2.8 MB, no `data/`, no `artifacts/`, no raw data. Taken before the follow-up changes of 2026-10-05 (fed-baseline, sensitivity).
- **Environment:** new venv with Python 3.13.5 on Windows 11.
- **Steps run, as the README says:** `python -m pip install -r requirements-lock.txt` (exit 0), `python -m pip install -e .` (exit 0), `python -m ppfeddata.cli --help`, `python -m pytest -q`, `python -m ppfeddata.cli run --stage trial --dry-run`.
- **Versions installed:** torch 2.14.1, flwr 1.39.0, opacus 1.6.0, ray 2.59.0, numpy 2.3.2, scikit-learn 1.9.1, streamlit 1.65.0 (the ones of the lock).
- **Result of `pytest -q`:** 346 passed, 9 skipped (the skipped tests need the processed data or the trained models), 7 warnings, 954.58 s, exit 0.
- **Result of the dry run:** the plan lists the 10 configurations of the matrix, all `todo` (about 65 min), no error.
- **Not checked:** Phases 1-5 from the raw data in the clean copy (the raw data were not copied), and running the matrix there. The sensitivity worlds of `ppfeddata sensitivity` did re-run Phase 3-4 from the raw data in the main checkout, reusing the scan cache.
