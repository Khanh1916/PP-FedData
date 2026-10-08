# Clean-install check of the README (2026-10-05, raw-data phases 2026-10-08)

Written by hand from the logs of those sessions (the only file of `results/reports/` that is not generated); `ppfeddata accept` (row D10) reads it.

## 2026-10-05: install and tests

- **Copy:** the files tracked by git plus untracked non-ignored files (`git ls-files -co --exclude-standard`), 2.8 MB, no `data/`, no `artifacts/`, no raw data. Taken before the follow-up changes of 2026-10-05 (fed-baseline, sensitivity).
- **Environment:** new venv with Python 3.13.5 on Windows 11.
- **Steps run, as the README says:** `python -m pip install -r requirements-lock.txt` (exit 0), `python -m pip install -e .` (exit 0), `python -m ppfeddata.cli --help`, `python -m pytest -q`, `python -m ppfeddata.cli run --stage trial --dry-run`.
- **Versions installed:** torch 2.14.1, flwr 1.39.0, opacus 1.6.0, ray 2.59.0, numpy 2.3.2, scikit-learn 1.9.1, streamlit 1.65.0 (the ones of the lock).
- **Result of `pytest -q`:** 346 passed, 9 skipped (the skipped tests need the processed data or the trained models), 7 warnings, 954.58 s, exit 0.
- **Result of the dry run:** the plan lists the 10 configurations of the matrix, all `todo` (about 65 min), no error.

## 2026-10-08: Phases 1-5 from the raw data in a clean copy (optimisation round O5)

- **Copy:** `git ls-files -co --exclude-standard` at commit 1a0c6c8, 214 files, 3.8 MB, no `data/`, no `artifacts/`, no `configs/local.yaml`; the raw data (17 GB, outside the repository) given through the environment variable `PPFEDDATA_RAW_ROOT`, as the README says.
- **Environment:** new venv, Python 3.13 on Windows 11; `python -m pip install -r requirements-lock.txt` (exit 0, 523 s), `python -m pip install -e .` (exit 0, 14 s), `python -m ppfeddata.cli --help` (exit 0).
- **Phases run, the README commands as written:** `inventory` (exit 0, 126 s), `harmonize` (exit 0, 2,547 s: the full scan of the raw CSV files, no cache), `sample` (exit 0, 414 s), `preprocess` (exit 0, 14 s), `check` (exit 0, 175 s). About 61 min for Phases 1-5 on this machine.
- **Result: Phases 1-5 from the raw data give processed data identical to the main checkout:** `train.npz` (88,500 x 60), `val.npz` (12,000 x 60) and `test.npz` (30,000 x 60) are array-for-array equal, `feature_schema.json` and `configs/feature_decisions.yaml` are equal. The split manifest (`results/manifests/split_manifest_6class.json`) has the same split and the same rows; only its provenance fields differ: `git_commit` / `git_dirty` (the copy is not a git checkout), `config_hash` (the raw-data path comes from the environment instead of `configs/local.yaml`) and the measured peak memory (0.686 against 0.663 GB).
- **Not re-run in the clean copy:** the experiments of Phases 6-14 (hours of machine time; their code is covered by the test suite and by the run ledger of the main checkout).
