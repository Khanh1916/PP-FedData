# How to use PP-FedData

**English** | [Tiếng Việt](USAGE.vi.md)

Step by step, by what you want to do. The [README](../README.md) is the reference: every command with its output (section "Running each phase"), the results and the glossary of the configuration names (B0 ... M3, MGs, MGr, MGd, MGl). Every time and number below comes from a file of the repository or from a run, named next to it. Run every command from the root of the repository.

## 1. Choose a path

| I want to... | Raw data needed | Time on the study's machine | Section |
|---|---|---|---|
| read the results and the conclusions | no | nothing to run | 3 |
| see the study in the browser (demo) | no; pages 2 and 4 need runs | install only | 4 |
| find the configuration for my deployment | no | a few seconds | 5 |
| generate synthetic packets with FedDP-Marginal | yes | data preparation about 1 h, then the runs | 6 |
| re-run the whole study | yes | the CVAE matrix alone 28.0 min per seed; searches and rounds on top | 7 |
| change K, α, ε, the quotas, the seeds | depends on the stage | | 8 |
| use my own data | my own data | | 9 |
| fix an error | | | 10 |

Sources of the times: section 5 is one run of this guide (2026-11-27); data preparation from `data/README.md`; the CVAE matrix from the README, section "Reference run times".

## 2. Set up (once)

```bash
git clone https://github.com/Khanh1916/PP-FedData.git
cd PP-FedData
python -m venv .venv                               # optional; Windows: .venv\Scripts\activate   Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-lock.txt     # the library versions of the runs (requirements.txt: the newest allowed)
python -m pip install -e .
python -m ppfeddata.cli --help                     # the 41 commands
python -m pytest -q                                # 10-15 minutes on a laptop CPU, needs no raw data
```

- Python 3.10 or newer; the runs used Python 3.13.5 on Windows 11, CPU only.
- Without a virtual environment, when the packages sit in the user site-packages, do not run Python with `-I`: isolated mode hides them and `import numpy` fails.
- PyTorch for CPU only: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- If you will commit: `git config core.hooksPath .githooks` turns on the check that refuses data and run outputs.

**Raw data (paths D and E).** Download DoS-DDoS-MQTT-IoT from Kaggle (`alaaalatram/dosddos-mqtt-iot`, about 12 GB of CSV) and tell the code where it is, in `configs/local.yaml` (never committed) or in the environment variable `PPFEDDATA_RAW_ROOT`:

```yaml
paths:
  raw_root: "D:/path/to/DoS-DDoS-MQTT-IoT_Dataset"
```

The release has the Normal traffic only as PCAP. `python -m ppfeddata.cli extract-normal` extracts it with the authors' export settings into `NormalData/CSV_Authors_Settings` under `raw_root` (needs TShark; about 30 min). Why: README, section "Where the data come from".

## 3. Path A: read the results

Nothing to install. Open on GitHub:

| File | What it answers |
|---|---|
| `results/reports/final_report.md` | everything: 9.0 short answers, 9.9b why FedDP-Marginal, 9.10b which configuration for which requirement, 10 limitations |
| `results/reports/recommend.md` | the recommended configuration for 11 example IoT deployments (lab, cloud-aggregator, consortium, ...) |
| `results/reports/recommend.html` | the same rules with filters you set; download the file and open it in a browser (GitHub shows its source) |
| `results/reports/scorecard.md` | every configuration on every axis: utility, ε, SecAgg, membership attack, cost |
| `results/figures/` | the figures of the report |
| `results/reports/data_provenance.md` | where the raw data come from and how that was checked |

## 4. Path B: the demo

```bash
python -m ppfeddata.cli demo                       # opens http://localhost:8501
python -m ppfeddata.cli demo --port 8700 --headless
```

| Page | On a fresh clone | Needs |
|---|---|---|
| 1. Data | works | `results/manifests/` (committed) |
| 2. Results | shows what to run | `results/summary.csv`, written by `aggregate` after the runs (not committed) |
| 3. Trade-offs and recommendation | works | `results/interpretation.json`, `results/scorecard.json`, the report (committed) |
| 4. Generate samples | shows what to run | trained generators in `artifacts/` and the processed data in `data/processed/` (path D or E) |
| 5. Threat model and limitations | works | committed files |

The demo sends no usage statistics (`.streamlit/config.toml`). Stop it with Ctrl+C.

## 5. Path C: the configuration for a deployment

`recommend` reads only `results/scorecard.json`, which is committed: no raw data, no run. Describe the deployment with options:

| Option | Meaning |
|---|---|
| `--untrusted-server` | the aggregation server must not see a gateway's update or tables in the clear (secure aggregation or local DP) |
| `--untrusted-clients` | gateways may collude with the server: distributed DP then counts with a single honest gateway |
| `--honest-clients 3` | distributed DP must hold when only 3 gateways add their noise share (measured curve) |
| `--max-eps 5` | record-level ε budget; without it no DP is required |
| `--max-mb-round 2`, `--max-mb-total 100`, `--max-rounds 3` | bandwidth of the gateways |
| `--priority rare` | what to maximise: `macro` (macro-F1 of the 6 classes, the default), `binary` (attack against normal), `rare` (recall of the rare attacks) |
| `--min-rare-recall 0.2` | a floor on the recall of the rare classes |
| `--max-mia 0.55` | the calibrated membership attack must stay below this AUC |

Example: the gateways of one company, aggregation in a third-party cloud, ε ≤ 5:

```bash
python -m ppfeddata.cli recommend --untrusted-server --max-eps 5
```

One run of this guide (2026-11-27) printed:

```text
custom: MGd-eps5 (tstr_f1 0.426, epsilon 5); front ['MGd-eps5', 'MGs-t3-eps5', 'MGs-eps2', 'MGr-eps2', 'MGs-t3-eps1', 'MGs-eps1', 'MGs-eps0.5']; command: python -m ppfeddata.cli tune-marginal --final --eps 5
```

Read it as: the best configuration (TSTR macro-F1 on the real test split, mean over seeds), the Pareto front under the requirement, and the command that reproduces it. MGd is the simulated form of the distributed noise (Gaussian, summed in one process); MGs and MGr send the same releases through Flower SecAgg+ (README glossary).
Without options, `recommend` runs the 11 example deployments and rewrites `results/recommend.json`, `results/reports/recommend.md` and `results/reports/recommend.html`; from the same scorecard they come out unchanged. Demo page 3 applies the same rules.

## 6. Path D: generate synthetic packets with FedDP-Marginal

The settings chosen on validation by the study are committed (`configs/best_marginal.yaml`, `configs/best_marginal_refine.yaml`, `configs/best_marginal_bn.yaml`), so no search is needed.

**Step 1. Prepare the data (Phases 1-4).** About one hour on the study's machine, most of it the full scan of `harmonize` (`data/README.md`).

```bash
python -m ppfeddata.cli extract-normal             # once: the Normal CSV from the PCAP (needs TShark, about 30 min)
python -m ppfeddata.cli inventory                  # data/inventory/: every raw file, its class and capture group
python -m ppfeddata.cli harmonize                  # common schema and format audit (Gate G1; the decisions are in configs/)
python -m ppfeddata.cli sample                     # split by capture group, class quotas -> data/interim/6class/
python -m ppfeddata.cli preprocess                 # encoded features -> data/processed/6class/
python -m ppfeddata.cli check                      # recommended: leakage checks C1-C5 (Gate G2) -> results/reports/leakage_report.md
```

`sample` also rewrites the committed `results/manifests/split_manifest_6class.json`. Look at its `parquet_sha256` entries in `git diff`: the same hashes mean your split is the study's (other entries, such as `git_commit`, change with every run). If `git status` shows another committed file changed, read the diff before going on.

**Step 2. Run FedDP-Marginal at one ε.**

```bash
python -m ppfeddata.cli tune-marginal --final --eps 5 --seeds 0
```

With the committed settings this trains, for seed 0, the variants the study kept at ε = 5: MGs-eps5 (Skellam noise split over the 5 gateways, three releases through Flower SecAgg+), MGr-eps5 (MGs plus the post-processing chosen on validation: the framework's configuration at ε = 5), MGd-eps5 and MGl-eps5 (distributed Gaussian noise and local DP, simulated in one process) and MGb-eps5 (the Bayesian-network option chosen at ε = 5).
Each is evaluated like every generator (TSTR, TAug, fidelity, privacy) and gets its rows in `results/runs.csv`. The three Flower releases took 20 s in total per run (final report, table of 9.6); the evaluation comes on top. Without `--seeds 0` the three seeds of the study run; `--eps 1 5 10` gives the three budgets.

Each run leaves a folder in `artifacts/`, for example `MGr-eps5_0`: `mg_model.pkl` (the released noisy tables), `synthetic.npz` (the synthetic set that was evaluated, encoded), `fidelity_privacy.json`, `mg_rounds.jsonl` (bytes and seconds of each release).

**Step 3. Get the rows in original units.** Either demo page 4: choose MG-eps5, a class and the rows per class, press "Generate samples", download the CSV (original units or encoded features). Or in Python, from the root of the repository:

```python
from ppfeddata import demo_lib as dl
from ppfeddata.utils import load_config

cfg = load_config(None)                                              # configs/default.yaml + configs/local.yaml
g = next(x for x in dl.generators(cfg) if x.label == "MG-eps5")      # the framework's configuration at epsilon 5
s = dl.sample(cfg, g, train_seed=0, classes=None, n_per_class=1000, sample_seed=0)
s.table.to_csv("mg_eps5_seed0.csv", index=False)                    # original units, one column per feature, plus `label`
```

One run of this code (2026-11-27) gave 6,000 rows, 1,000 per class, and 26 columns: `label` and 25 features. The rows are post-processing of the DP release, so drawing more spends no more privacy budget; the same model, seed, classes and number give the same rows. At most 50,000 rows per class per call.

**Step 4 (optional). Put the runs into a report.** `aggregate` turns the run ledger into the summary, the figures and the report, and rewrites the committed ones in `results/`. With only some runs, write them elsewhere:

```bash
python -m ppfeddata.cli aggregate --out-dir preview
```

With `--out-dir` the READMEs are left alone.

## 7. Path E: re-run the whole study

The order and the gates are those of the spec; the README table "Running each phase" lists every command with its output.

1. Set up with `requirements-lock.txt` and the raw data (section 2).
2. Data: `check-extraction` (optional provenance check, needs TShark, about 1.5 h), then section 6, step 1. Gates G1 (`harmonize`) and G2 (`check`) are approvals by a person: read `results/reports/g1_feature_review.md` and `results/reports/leakage_report.md` before going on.
3. Baselines (Phase 6, Gate G3): `baseline`, then read `results/reports/g3_baseline.md`.
4. The CVAE matrix (Phases 7-11). The committed `configs/best_cvae.yaml` and `configs/best_cvae_dp.yaml` let you skip `tune`, `tune-dp` and `verify-dp`. `run --stage trial --dry-run` prints the plan; `run --stage trial` runs the whole matrix for one seed in its own workspace (`artifacts/_trial`, 28.0 min on the study's machine); read `results/reports/g4_trial.md` (Gate G4), and only then `run --stage full --approve-g4` (three seeds).
5. `aggregate` (about 75 s): summary, figures, report, `results/interpretation.json`, the generated blocks of the READMEs.
6. The later rounds (optimisation O0-O4, follow-up P1-P2, Q1, R): the commands of the README table, in its order. Most write a report of their own; those whose `--help` says so need `aggregate` and `scorecard` after them. The committed `best_*.yaml` files let you skip the searches and run only the `--final` stages.
7. `scorecard`, `recommend` and `aggregate` again, then `accept` (the Definition of Done, D1-D10) and `package --no-update-lock` (the reproducibility bundle `results/repro/`).

Runs are keyed by their name and seed: a finished run is skipped when a command is repeated, and `--no-resume` forces it again where the command has that option.

## 8. Change settings

Every parameter is in `configs/default.yaml`; `configs/local.yaml` (not committed) is merged on top, and `PPFEDDATA_RAW_ROOT` overrides the raw path. To try other values without mixing them with the study's runs, copy `configs/default.yaml` to a new file, change what you need, point `compute.runs_csv` and `compute.artifacts_dir` (and `paths.work_dir` when the data preparation changes) to new places, and give the file with `--config`, before the command:

```bash
python -m ppfeddata.cli --config configs/my_run.yaml tune-marginal --final --eps 5 --seeds 0
```

Keep the ledger apart: a run whose name and seed are already in the ledger is taken as done and skipped, even if the settings changed.

| Key | Default | What it changes |
|---|---|---|
| `seeds` | `[0, 1, 2]` | the seeds of every run (partition, noise, sampling, classifier) |
| `paths.raw_root` | `./raw/DoS-DDoS-MQTT-IoT_Dataset` | where the raw CSV are (set it in `configs/local.yaml`) |
| `paths.work_dir` | `./data` | the processed data and the client partitions |
| `compute.runs_csv`, `compute.artifacts_dir` | `./results/runs.csv`, `./artifacts` | the run ledger; models, synthetic sets, predictions |
| `label_mode` | `6class` | 6 classes, or 11 with DoS and DDoS apart (data preparation and `robustness` only) |
| `split.split_seed` | `0` | which capture groups go to validation and test, which rows are sampled |
| `quota.<class>` | `NORMAL: [60000, 2000, 5000]` ... | rows per class in train, validation, test; the order of the classes is the order of the labels |
| `fl.num_clients` | `5` | the number of gateways K |
| `fl.dirichlet_alpha` | `0.5` | how unevenly the classes are spread over the gateways (smaller = more uneven) |
| `fl.rounds`, `fl.local_epochs` | `30`, `2` | CVAE route only; FedDP-Marginal always sends three releases |
| `dp.epsilons` | `[1, 5, 10]` | the ε values a command runs when `--eps` is left out |
| `dp.delta` | `1.0e-5` | δ of every DP guarantee |
| `secagg.num_shares`, `secagg.reconstruction_threshold` | `5`, `3` | secret sharing of SecAgg+ |
| `generate.target_per_class` | `20000` | rows per class of each synthetic set that is evaluated |
| `eval.rf.n_estimators`, `eval.bootstrap` | `200`, `1000` | size of the random forest, bootstrap resamples of the report |
| `thresholds.mia_auc_max`, `thresholds.eps_max_recommend` | `0.55`, `5.0` | limits of the spec's rule R6 |

**Another ε for FedDP-Marginal.** The committed searches cover ε = 0.5, 1, 2, 5, 10 and 20. For another value, search first, then run:

```bash
python -m ppfeddata.cli tune-marginal --eps 3                # grid of 4 bin counts x 3 budget splits, mean of 3 seeds on validation
python -m ppfeddata.cli tune-marginal --final --eps 3
```

The searches write to `configs/best_marginal.yaml` (and `configs/best_marginal_refine.yaml`, `configs/best_marginal_bn.yaml` with `--refine`, `--bn`) whatever `--config` says: a new search changes these committed files and keeps the entries of the other ε values. The final runs of a new ε give MGd, MGl and MGs; MGr and MGb only after `--refine` and `--bn` chose something at that ε. The demo lists ε = 1, 5 and 10 only.

**Other K or α.** The study ran FedDP-Marginal with α = 0.1 and 10 and with 10 and 20 gateways through `robustness`, with the settings chosen for 5 gateways and α = 0.5 (`results/reports/robustness.md`). For your own values, change `fl.num_clients` or `fl.dirichlet_alpha` in a copy of the config with its own ledger and run `tune-marginal --final`; the partitions go to `data/partitions/`, with α, the seed and K in their names.

## 9. Use your own data

| Stage | Tied to DoS-DDoS-MQTT-IoT? |
|---|---|
| `inventory` (Phase 1) | yes: it reads `<raw_root>/<scenario folder>/<attack type>/CSV Files/*.csv` and the Normal files `Pcap_Files__NormalData<n>_part<k>.csv` under `paths.normal_csv_dir`; folders, classes and expected row counts come from `configs/label_map.yaml`, and it logs "VERIFICATION FAILED" when the counts differ |
| `harmonize`, `sample`, `preprocess` (Phases 2-4) | yes: the columns of the TShark export (`SHARED_COLUMNS` in `data/harmonize.py`), the row filter TCP / MQTT and the decisions of Gate G1 (`configs/feature_decisions.yaml`, `harmonize.user_overrides` in `configs/default.yaml`) |
| generators, evaluation, reports (Phase 6 on) | no: they read only `data/processed/<mode>/` and the config |

1. **The same kind of data** (TShark CSV exports of MQTT traffic with these columns): use the same folder layout, edit `configs/label_map.yaml` (folders, classes, expected counts) and `quota` (one entry per class), then run Phases 1-5 and read the reports of Gates G1 and G2 before going on.
2. **Other columns or another domain**: either change the column handling of Phases 2-4 (`SHARED_COLUMNS` and the parsers in `data/harmonize.py` and `data/preprocess.py`), or write `data/processed/<mode>/` yourself in the format of `preprocess`:
   - `train.npz`, `val.npz`, `test.npz` with `X` (encoded features), `y` (class index), `X_diag`, `group_id`, `stream_id`, `class11`;
   - `feature_schema.json` with `blocks` (each with `name`, `type` among numeric / binary / na_flag / categorical, `column`, `start`, `width`), `n_features`, `label_map` and the record `fit` (split `train` and its row count: the commands refuse data whose preprocessor was not fitted on train alone);
   - `label_map.json`, and `preprocessor.joblib` for rows in original units.

   The study did not test this route. A new generator: [ARCHITECTURE.md](ARCHITECTURE.md), section 10.

## 10. Troubleshooting

| Symptom | What to do |
|---|---|
| `import numpy` fails although the packages are installed | Python was run with `-I`, which hides the user site-packages: run it without `-I`, or install in a virtual environment |
| a Flower run stops making progress with the CPU idle (Windows) | a Ray worker hangs: stop the command with Ctrl+C and start it again; finished runs are skipped and a Flower run continues from its last checkpoint (every 5 rounds) |
| a Flower run dies in the middle | a round that misses a client's reply stops at once and the run restarts from its checkpoint at most twice; if it still fails, run the command again |
| a long run was killed after being moved to the background | start long commands in their own terminal from the beginning; moving a running job to the background killed the Ray workers once |
| `ray` is missing on Windows with Python 3.13 | `flwr[simulation]` does not install it there; install `requirements.txt`, which lists `ray` |
| `check-extraction` or `extract-normal` cannot start | both need TShark: install Wireshark and put `tshark` on the PATH |
| `inventory` logs "VERIFICATION FAILED" | the raw files differ from the release, or `paths.normal_csv_dir` points to another extraction: compare with `results/reports/data_provenance.md` |
| files not found after moving the raw data | update `paths.raw_root` and run `inventory` again; `sample` then reads the raw files once more |
| `sample` stops: "classes below 1000 rows ... stop and ask the user" | a validation or test class is smaller than `split.min_eval_rows`: the quotas or the split must change, a decision for a person |
| `sample` stops on peak memory | the run used more than `split.max_peak_gb` (4.0 GB) |
| "feature_schema.json has no `fit` record" or "not fitted on the train split alone" | run `preprocess` again |
| `KeyError: 'eps3'` from `tune-marginal --final` | no setting is chosen at that ε yet: run `tune-marginal --eps 3` first |
| a command only logs "skip ..." | the run is already in `results/runs.csv`: use `--no-resume` where the command has it, or another `compute.runs_csv` |
| `run --stage full` refuses to start | it needs `--approve-g4`, given only after reading `results/reports/g4_trial.md` |
| demo page 2 or 4 shows a hint instead of data | they need `results/summary.csv` and `artifacts/` (section 4) |
| `accept` exits with code 1 | a row of `results/reports/dod_checklist.md` failed: its evidence column says what is missing |
| `git commit` is refused by the pre-commit hook | it refuses data, run outputs and files over 5 MB on purpose: leave them out of the commit |

The log of each Flower run is `fl.log` in its folder under `artifacts/`.
