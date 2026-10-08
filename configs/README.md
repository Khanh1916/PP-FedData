# `configs/` — parameters and chosen settings

Every command reads `default.yaml` (or the file given with `--config`), deep-merges `local.yaml` next to it if it exists, and lets the
environment variable `PPFEDDATA_RAW_ROOT` override `paths.raw_root` (`utils.load_config`). How the configuration drives the pipeline:
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md), section 7.

| File | Written by | Role |
|---|---|---|
| `default.yaml` | hand (spec section 3) | every parameter: seeds, paths, label mode, split quotas, preprocessing, CVAE, FL (5 clients, 30 rounds, Dirichlet α 0.5), DP (ε 1 / 5 / 10, δ 1e-5), SecAgg, generation, evaluation, tuning, thresholds |
| `local.yaml` | you (not committed) | machine-specific paths, e.g. `paths.raw_root`; the environment variable `PPFEDDATA_RAW_ROOT` also works |
| `feature_decisions.yaml` | `harmonize`, approved at Gate G1 | role of every raw column (numeric, binary, categorical, dropped) |
| `label_map.yaml` | hand | mapping of the raw scenario directories to class codes, for the 6-class and 11-class modes |
| `best_cvae.yaml` | `tune` | CVAE hyper-parameters chosen without DP |
| `best_cvae_dp.yaml` | `verify-dp` | CVAE hyper-parameters chosen under DP-SGD (one-client proxy) |
| `best_cvae_dp_full.yaml` | `tune-dp-full` | DP search at full scale, one result per epsilon |
| `best_marginal.yaml` | `tune-marginal` | FedDP-Marginal: coarse bins and budget split per epsilon |
| `best_marginal_bn.yaml` | `tune-marginal --bn` | Bayesian-network options tried and the verdict |
| `best_marginal_refine.yaml` | `tune-marginal --refine` | post-processing chosen per epsilon (MGr) |
| `best_group_dp.yaml` | `privacy-units` | rows per unit m chosen per unit (stream, capture) and epsilon |
| `exp/*.yaml` | hand | the experiment matrix run by `run`: one file per configuration (B0 ... M3, extensions A1-A5) |

All `best_*.yaml` files are chosen on the **validation** split with the mean of 3 seeds where the spec requires it; they are committed so
that the tuning stages can be skipped. The reproducibility bundle keeps a copy in `results/repro/configs/`.
