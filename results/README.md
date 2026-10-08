# `results/` — what the study produced

Every file here is **written by a command**; do not edit them by hand (change the code and re-run the command). Start with
[`reports/final_report.md`](reports/final_report.md): section 9.0 summarises everything. How results flow from runs to reports:
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md), section 6.

Not committed (too large, rebuilt by the runs): `runs.csv` (the run ledger: one row per configuration, protocol, classifier and seed) and
`summary.csv` (mean and std per configuration, written by `aggregate`).

## Reports — `reports/`

| File | Command | Content |
|---|---|---|
| `final_report.md` | `aggregate` | **the main report**: tables, figures, interpretation R1-R6 and red flags (section 9), limitations (section 10) |
| `recommend.md`, `recommend.html` | `recommend` | which configuration for which deployment requirement; the HTML page has the same filters |
| `scorecard.md` | `scorecard` | every configuration on every metric and the Pareto front |
| `privacy_units.md` | `privacy-units` | group-level DP (stream, capture) and honest-client threshold of FedDP-Marginal |
| `ids_followup.md` | `ids-followup` | IDS-side options and local augmentation at each client |
| `robustness.md` | `robustness` | FedDP-Marginal under other federations and 11 classes |
| `sensitivity.md` | `sensitivity` | other test groups (A4) and a per-stream cap (A5) |
| `g1_feature_review.md` | `harmonize` | column formats and feature decisions (Gate G1) |
| `leakage_report.md` | `check` | leakage and label-reliability checks C1-C5 (Gate G2) |
| `g3_baseline.md` | `baseline` | baselines B0, B1a, B1b (Gate G3) |
| `b2_cvae.md` | `b2` | centralised CVAE |
| `compute_budget.md` | `benchmark` | measured cost of one epoch and extrapolation |
| `b3_fl.md` | `b3` | CVAE with non-IID FedAvg |
| `m1_dp.md`, `m1_dp_tuned.md` | `m1`, `m1 --tuned` | CVAE with client-side DP-SGD (Phase 7 and DP-specific hyper-parameters) |
| `m2_m3_secagg.md` | `secagg-report` | CVAE with SecAgg+ (M2) and DP + SecAgg+ (M3), checks T-SA1-3 |
| `g4_trial.md` | `run --stage trial` | one-seed trial of the whole matrix (Gate G4) and run times |
| `dod_checklist.md` | `accept` | the spec's Definition of Done, item by item |
| `clean_install.md` | written by hand | record of the re-run of Phases 1-5 on a clean copy (item D10) |

## Machine-readable results

| File | Command | Content |
|---|---|---|
| `interpretation.json` | `aggregate` | every number of section 9 of the report (rules, intervals, red flags) |
| `scorecard.json`, `scorecard_baseline.json` | `scorecard`, `scorecard --freeze-baseline` | scorecard rows and the frozen reference of the optimisation round |
| `recommend.json` | `recommend` | requirement, front and pick of every example scenario |
| `privacy_units.json` | `privacy-units` | rows of the P1 tables |
| `ids_followup.json` | `ids-followup` | rows of the P2 tables, per client |
| `robustness.json` | `robustness` | federations and 11-class results |
| `sensitivity.json` | `sensitivity` | A4 / A5 verdicts |
| `mia_model.json` | `mia` | membership attack on the released CVAE models, positive controls |
| `mia_marginal.json` | `mia-marginal` | membership attack on the released FedDP-Marginal tables, positive controls |
| `fed_classifier.json` | `fed-baseline` | the IDS trained directly by FL (plain and protected) |

## Figures — `figures/`

| File | Command | Shows |
|---|---|---|
| `f1_by_config.png`, `recall_rare_classes.png`, `utility_privacy.png`, `fidelity_vs_eps.png`, `overhead.png`, `pareto.png` | `aggregate` | the six figures of the final report |
| `presence_heatmap.png` | `harmonize` | which columns exist in which source |
| `c3_single_feature.png`, `c3b_leave_one_out.png`, `c4_split_gap.png` | `check` | leakage checks |
| `partition_alpha0.5_seed0.png`, `fl_convergence.png` | `b3` | class share per client, FL convergence |
| `utility_privacy_m1.png`, `utility_privacy_m1_tuned.png` | `m1`, `m1 --tuned` | utility against epsilon for M1 |
| `secagg_checks.png` | `secagg-report` | secure vs plain aggregation per round |

## Data descriptions and reproducibility

| Path | Command | Content |
|---|---|---|
| `manifests/split_manifest_6class.json`, `manifests/split_manifest_11class.json` | `sample` | capture groups of each split, counts, hashes, library versions (no data row) |
| `manifests/feature_schema_6class.json` | `package` | encoding of every feature (copy of the processed schema) |
| `repro/` | `package` | `pip_freeze.txt`, copy of `configs/`, `environment.json`, `MANIFEST.json` (SHA-256 of the results) |
