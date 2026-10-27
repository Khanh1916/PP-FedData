# `data/` — the working data of the pipeline (not committed)

Every file here is **written by a command**; only this README and `.gitkeep` are committed. The folder is `paths.work_dir` in
`configs/default.yaml`. It holds the inventory of the raw dataset, the sampled rows, the encoded features and the client partitions;
everything in it can be deleted and rebuilt with the commands below.

**The raw dataset is not here.** It stays where it was downloaded, read only (rule 2 of the
[spec](../docs/PP-FedData_Implementation_Spec.md)), and `paths.raw_root` points to it: set it in `configs/local.yaml` (git-ignored)
or in the environment variable `PPFEDDATA_RAW_ROOT`. Keeping the raw data outside this folder means that clearing the working data
can never touch it. Where the raw data come from is in the main README, section [Data](../README.md#data).

## Sub-folders

`<mode>` is `6class` or `11class` (`label_mode`).

| Folder | Written by | Content |
|---|---|---|
| `inventory/` | `inventory` (Phase 1) | `files.csv` (every raw file: absolute path, scenario, attack type, class, capture group, rows, bytes), `class_counts.csv`, `group_counts.csv` |
| `inventory/` | `harmonize` (Phase 2) | `header_check.csv`, `presence_matrix.csv`, `format_audit.csv`, `token_audit.csv`, `duplicate_pairs.csv`, `time_feature_audit.csv`; they are also the cache of `harmonize` (`--recompute` computes them again) |
| `interim/<mode>/` | `sample` (Phase 3) | `train.parquet`, `val.parquet`, `test.parquet` (the sampled raw rows, harmonised columns kept as strings, with the label, capture group and stream ids), `split_manifest.json`; a copy without machine paths is committed as `results/manifests/split_manifest_<mode>.json` |
| `interim/_scan/` | `sample` | cache of pass A (the row filter and the stream ids of every raw file), keyed by the file path, its size and the row filter; `sample --no-cache` ignores it |
| `processed/<mode>/` | `preprocess` (Phase 4) | `train.npz`, `val.npz`, `test.npz` (encoded features `X`, labels `y`, group and stream ids), `feature_schema.json` (committed copy: `results/manifests/feature_schema_<mode>.json`), `label_map.json`, `preprocessor.joblib` |
| `partitions/` | the first federated command that needs them (`b3`, `m1`, `m2`, `m3`, the FedDP-Marginal runs, `fed-baseline`) | client row indices, Dirichlet split by label: `alpha<α>_seed<s>.json` (5 clients), `_11class` for the 11-class mode, `_k<K>` for other client counts (`robustness`); `units_<stream or capture>_m<m>_alpha<α>_seed<s>_k<K>_<mode>.json`, where every stream or capture is held by one client (`privacy-units`) |
| `_sens/<world>/` | `sensitivity` | the same layout (`inventory/`, `interim/`, `processed/`, `partitions/`, `manifests/`) for each sensitivity world: `A2-r10`, `A2-r200`, `A4-s1`, `A4-s2`, `A5-cap20` |
| `_mia/s<seed><A or B>/` | `mia` | `partitions/` and `processed/` of the two halves of the train pool: model A is trained on one half, the reference B on the other |

## Rebuilding and moving

- Deleting this folder loses nothing that cannot be rebuilt: `inventory`, `harmonize`, `sample`, `preprocess` (Phases 1-4, about an
  hour on the machine of the study, most of it the full scan of `harmonize`), then the commands that write the partitions and the
  sub-folders starting with `_`. Rebuilding with the same seeds and library versions gives the same files (the split manifest
  records the SHA-256 of every Parquet file).
- `files.csv` and the keys of `interim/_scan/` contain the absolute paths of the raw files. If the raw data move, update
  `paths.raw_root` and run `inventory` again; `sample` then misses the scan cache and reads the raw files once more.
- Never committed: `.gitignore` blocks everything under `data/` except `.gitkeep` and this README.
