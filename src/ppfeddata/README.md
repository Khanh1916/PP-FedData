# `src/ppfeddata/` — the package

Everything the CLI runs. Each sub-command of `python -m ppfeddata.cli` (`cli.py`) imports only the modules of its stage. How the pieces fit
together: [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md). The modules are listed below by layer, in pipeline order.

## Data preparation (Phases 1-4) — `data/`

| Module | Role |
|---|---|
| `data/inventory.py` | scan the raw CSV files, count rows, assign a capture group to every file (`inventory`) |
| `data/harmonize.py` | harmonise column names, presence matrix, format audit, feature decisions for Gate G1 (`harmonize`) |
| `data/split_sample.py` | group-aware train / val / test split and streaming sampling to the class quotas (`sample`) |
| `data/parse_multi.py` | parse multi-value cells such as "Publish Message,Publish Message" |
| `data/preprocess.py` | fit-on-train encoding, `feature_schema.json`, processed arrays (`preprocess`) |

## Checks (Phase 5) — `checks/`

| Module | Role |
|---|---|
| `checks/leakage.py` | leakage and label-reliability checks C1-C5 (`check`) |
| `checks/leakage_report.py` | Markdown report and figures of those checks |

## Evaluation (Phase 6 onward) — `eval/`

| Module | Role |
|---|---|
| `eval/baselines.py` | B0 (real only), B1a (class weights), B1b (SMOTE); data loading shared by every stage; the Markdown table helper |
| `eval/utility.py` | classifiers (RF, MLP), protocols TRTR / TSTR / TAug, utility metrics |
| `eval/fidelity.py` | distances between synthetic and real data, C2ST |
| `eval/privacy.py` | duplicate rate, distance to closest record, membership attack on the synthetic data |
| `eval/overhead.py` | time, bytes per round, peak RAM |
| `eval/stats.py` | stratified bootstrap intervals |
| `eval/compare.py` | paired bootstrap comparisons between runs |
| `eval/dp_check.py` | epsilon of DP-SGD recomputed without Opacus (red flag F5) |
| `eval/runs.py` | the run ledger `results/runs.csv` and per-run predictions, with resume |
| `eval/taug_rare.py` | TAugR: real data + synthetic rows for the rare classes only |

## Generators — `models/`

| Module | Role |
|---|---|
| `models/cvae.py` | conditional VAE over the encoded features (the baseline generator) |
| `models/train.py` | CVAE training loop (reused for the local FL epochs) |
| `models/generate.py` | sampling synthetic rows from a trained CVAE |
| `models/b2.py` | B2: the centralised CVAE, trained and evaluated |
| `models/benchmark.py` | compute benchmark (plain vs DP-SGD) and extrapolation |
| `models/marginal.py` | **FedDP-Marginal**: DP class-conditional marginals, Chow-Liu tree, Gaussian / Skellam noise and accounting, post-processing, sampling |
| `models/marginal_bn.py` | FedDP-Marginal generalised to a per-class Bayesian network (option tested in O3) |

## Federated learning — `fl/` and the partition

| Module | Role |
|---|---|
| `partition.py` | per-class Dirichlet split of the train pool over K clients |
| `group_dp.py` | stream / capture units, unit-aware partition and per-unit cap for group-level DP |
| `fl/core.py` | FedAvg itself, without Flower (unit-testable) |
| `fl/app.py` | Flower ServerApp + ClientApp around `fl/core.py` |
| `fl/run.py` | launch one Flower simulation (Ray) in its own process, with checkpoints |
| `fl/dp_utils.py` | client-side DP-SGD with Opacus and its accounting |
| `fl/dp_stats.py` | DP per-class residual statistics of the CVAE |
| `fl/secagg.py` | Flower SecAgg+ wiring for the CVAE (M2, M3), compact encodings |
| `fl/bandwidth.py` | M2 with uint32 / uint16 masked vectors |
| `fl/dpfedsgd.py` | DP-FedSGD with the noise split over the clients (M3f, simulated) |
| `fl/b3.py`, `fl/m1.py`, `fl/m2.py` | the CVAE configurations B3, M1, M2 / M3 and their reports |
| `fl/mg_app.py` | the three FedDP-Marginal releases through Flower's real SecAgg+ |

## Experiment control and tuning

| Module | Role |
|---|---|
| `run_experiment.py` | the experiment matrix of `configs/exp/*.yaml` (`run --stage trial / full`) |
| `tune.py` | Optuna search of the CVAE on non-private data (`tune`) |
| `tune_dp.py` | DP search of the CVAE on a one-client proxy (`tune-dp`, `verify-dp`) |
| `tune_dp_full.py` | DP search of the CVAE at full scale, one study per epsilon (`tune-dp-full`) |
| `tune_fedsgd.py` | search and final runs of DP-FedSGD (`tune-fedsgd`) |
| `tune_marginal.py` | settings of FedDP-Marginal chosen on validation, final runs, options and post-processing (`tune-marginal`) |
| `privacy_units.py` | group-level DP and honest-client threshold (`privacy-units`) |
| `ids_followup.py` | IDS-side options and local augmentation (`ids-followup`) |
| `robustness.py` | FedDP-Marginal under other federations and 11 classes (`robustness`) |
| `sensitivity.py` | other test groups (A4) and a per-stream cap (A5) (`sensitivity`) |
| `fed_classifier.py`, `fed_protected.py` | the IDS itself trained by FL, plain and protected (`fed-baseline`) |
| `mia_model.py`, `mia_marginal.py` | membership attacks with access to the released CVAE / FedDP-Marginal model (`mia`, `mia-marginal`) |

## Analysis and reporting

| Module | Role |
|---|---|
| `aggregate.py` | ledger -> `results/summary.csv`, six figures, `final_report.md`, README blocks (`aggregate`) |
| `interpret.py` | rules R1-R6, red flags, recommendation rule, from the ledger and the predictions |
| `interpret_report.py` | section 9 of the final report, built from `interpret.py` and the JSON results |
| `limitations.py` | the list of limitations shared by the report, the READMEs and the demo |
| `scorecard.py` | every configuration on every metric, Pareto front (`scorecard`) |
| `recommend.py` | configurations that meet a deployment requirement, example IoT scenarios (`recommend`) |
| `readme_gen.py` | the generated blocks of README.md and README.vi.md |

## Delivery and plumbing

| Module | Role |
|---|---|
| `cli.py` | the single entry point; one sub-command per stage |
| `utils.py` | config loading, seeding, hashing, logging |
| `demo_lib.py` | the logic of the Streamlit demo (`demo/app.py`), testable without Streamlit |
| `acceptance.py` | Definition of Done checked against the files (`accept`) |
| `package.py` | reproducibility bundle `results/repro/` (`package`) |
| `__init__.py`, `__main__.py` | package marker; `python -m ppfeddata` |
