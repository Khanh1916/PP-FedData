# `artifacts/` — what the runs leave on disk (not committed)

Every file here is **written by a command**; only this README and `.gitkeep` are committed. The rest (trained models, synthetic data,
predictions, Optuna studies, logs) is large and derived from the dataset, so it stays on the machine that ran the experiments. The
folder is `compute.artifacts_dir` in `configs/default.yaml`; the processed data and partitions are in `data/` (`paths.work_dir`).

Who reads it: `aggregate` (round logs and test predictions for the report, the paired bootstrap), `taug-rare` (the saved synthetic
sets), `secagg-report` (the round times of M2 and M3) and the demo (trained generators and synthetic sets, see
[demo/README.md](../demo/README.md)). The numbers themselves are in `results/runs.csv` (one row per run, not committed); the committed
summaries are in [results/](../results/README.md).

## Run directories: `<config>_<seed>`

One directory per run, named by its run id (`eval/runs.py`, `run_id`): the configuration and the seed, e.g. `B3_0`, `MGs-eps5_2`;
11-class runs add the mode, e.g. `B0-rf_11class_0`. The run id is the key of the row in `results/runs.csv`. The configuration names
(B0 ... M3, MGs, MGr, MGd, MGl, the extensions A1-A5, the search trials O1) are explained in the
[glossary of the main README](../README.md#glossary).

| Kind of run | Examples | Files |
|---|---|---|
| CVAE trained centrally (B2) | `B2_0` | `model.pt`, `synthetic.npz` (the synthetic set that was evaluated), `fidelity_privacy.json` |
| CVAE trained by Flower (B3, M1, M2, M3 and their tuned variants, A1, O1 search trials) | `B3_0`, `M1d-t18-eps5_0`, `O1-eps5-t12_0` | `spec.json` (settings of the run), `fl.log`, `rounds.jsonl` (one line per round: training loss, seconds per round and per client, bytes, peak memory, optimiser steps per client; validation loss), `ckpt/` (the two newest checkpoints and `latest.json`, used to resume), `final_state.pt`; generators that were evaluated add `synthetic.npz` and `fidelity_privacy.json`; DP runs add `dp_stats.json` (the residual statistics); SecAgg+ runs add `secagg_rounds.jsonl` and `clip_check_c<k>.jsonl` (one per client) |
| FedDP-Marginal through Flower (MGs, MGr with SecAgg+; MGp without) | `MGs-eps5_0`, `MGp-nodp_0` | `spec.json`, `fl.log`, `mg_rounds.jsonl` (bytes and seconds of each release, check of the aggregate), `mg_model.pkl`, `client<k>_<stage>.npy` (each client's vector, written off-protocol only to check that the aggregate equals their exact sum), `synthetic.npz`, `fidelity_privacy.json` |
| FedDP-Marginal simulated in process (MGd, MGl, MGb) | `MGd-eps5_0` | `synthetic.npz`, `fidelity_privacy.json` |
| Classifier runs (TSTR, TAug, TAugR, B0, B1, the federated classifiers) | `B3-TSTR-rf_0`, `B0-rf_0`, `FedMLPcwT-mlp_0` | `meta.json`, `preds/test.npz` and `preds/val.npz` (predicted class and probabilities), which the paired bootstrap of the report reads |
| SecAgg+ checks T-SA1-3 (`secagg-check`) | `B3-chk_0`, `M2-chk_0`, `B3-noise1_0`, `chk-M2-chk-TSTR-rf_0` | the same seed trained plain and through SecAgg+, and plain with tiny noise as controls, with `states/` (the weights after every round); the `chk-` directories hold their evaluation |

## Files at the top level

| File | Command | Content |
|---|---|---|
| `experiment_status_6class.json` | `run` | the last session of the experiment matrix: what ran, what was skipped |
| `optuna_cvae_6class.db` | `tune` | Optuna study (SQLite) of the CVAE search without DP; `tune` resumes from it |
| `optuna_dp_eps5_6class.db` | `tune-dp` | DP search of the CVAE (one-client proxy) |
| `optuna_dpfull_eps1_6class.db`, `optuna_dpfull_eps5_6class.db`, `optuna_dpfull_eps10_6class.db` | `tune-dp-full` | DP search of the CVAE at full scale, one study per epsilon |
| `B2_positive_control_6class.json` | `b2` | the over-fitted CVAE used as the positive control of the membership attack |
| `b3_sanity_6class.json`, `m1_sanity_6class.json` | `b3`, `m1` | sanity checks of B3 and M1 |
| `compute_benchmark_6class.json` | `benchmark` | measured times, the raw data of `results/reports/compute_budget.md` |
| `secagg_checks_6class.json`, `secagg_checks_ledger_6class.csv` | `secagg-check` | checks T-SA1-3 (secure against plain aggregation per round) and their own ledger |
| `fed_protected_cache.json` | `fed-baseline --protected` | finished tuning and runs, so that an interrupted run resumes; keyed by the config hash |

## Sub-folders starting with `_`

| Folder | Written by | Content |
|---|---|---|
| `_trial/` | `run --stage trial` | the one-seed trial of the whole matrix (Gate G4), kept apart from the main runs |
| `_sens/<world>/` | `sensitivity` | the run directories of each sensitivity world (A2, A4, A5), same layout as above |
| `_mia/s<seed><A or B>/` | `mia` | the models of the membership attack with model access: A trained on one half of the train pool, B on the other (the reference) |
| `_extraction_check/` | `check-extraction` | the TShark re-extractions of the provenance check |
| `_data_check/`, `_round_r/` | by hand | logs of the provenance session and the scripts that chained round R (`r2.sh` ... `r5.sh`) with their logs |

## Resuming and running again

The commands resume: a run already in `results/runs.csv` is skipped, an interrupted Flower run continues from `ckpt/latest.json`, and
the Optuna studies continue from their `.db`. To run something again, use `--no-resume` where the command has it (`baseline`, `b2`,
`b3`, `m1`, `m2`, `m3`, `secagg-check`, `run`). Deleting this folder loses the models and predictions but not the committed results;
`aggregate` then needs the experiments to be run again.
