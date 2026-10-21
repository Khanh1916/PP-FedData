# PP-FedData — Implementation specification (study protocol, v1.5)

**English** | [Tiếng Việt](PP-FedData_Implementation_Spec.vi.md)

> **Two files go together.** This file (the spec) holds the **rules**: what must be done, in which order, and how it is accepted; it holds no results. [`SPEC_DEVIATIONS.md`](SPEC_DEVIATIONS.md) holds the **evidence**: every place where the work differs from the spec or goes into more detail, with the reason, the measurements and the user's decision; every entry has a code (for example 3.5, O2.3, P1.2) that the report, the READMEs and the source code cite. Read the spec for the rules; look up SPEC_DEVIATIONS by code to know why and what was measured.
>
> This is the protocol of the study: whoever carries it out, by hand or with the help of an AI agent, follows the same rules. Every step runs with the `python -m ppfeddata.cli ...` commands listed in the README (section "Running each phase"); below, "the implementer" is the person or agent who runs those steps and "the user" is the person who approves at the gates. Read everything before writing code. Work **through the Phases in order**; every Phase has *Deliverables* and a *Gate* (acceptance condition). Do not move to the next Phase before the Gate is met. v1.2 folded in the changes learnt while doing Phases 0-5; **v1.3 folded in the changes of Phases 6-13** (paragraphs that start with "**v1.3:**"); **v1.4 added Phase 14, the optimisation round of M1, M2, M3**; **v1.5 moved the framework's main generator to FedDP-Marginal (Phase 14, O2-O5), the CVAE becoming the baseline**. The detailed evidence and measurements are in `SPEC_DEVIATIONS.md` (the evidence log, kept; this spec holds only rules, no results); the 🔶 items and the decisions G1-G4 the user settled are written in each Phase.

**Topic:** a privacy-preserving data generation framework in federated learning (FL) to strengthen the detection of MQTT DoS/DDoS attacks on IoT.
**Idea (v1.1-v1.4):** a label-conditional CVAE trained by FL on non-IID data; three protection mechanisms compared: **DP** (DP-SGD at the client), **SecAgg** (Flower), and **DP + SecAgg**; the generated data balance the classes for the IDS classifiers (Random Forest, MLP).
**Idea (v1.5):** the main generator is **FedDP-Marginal** (per-class marginal count tables with DP, aggregated over the federation, Chow-Liu tree); two protection variants: **distributed** (Skellam noise split over the clients, summed through Flower's SecAgg+) and **local** (every client adds the full noise). The CVAE (M1, M2, M3) stays as the baseline. Configurations are chosen by deployment requirement (`recommend`).
**Threat model:** honest-but-curious server; honest clients that do not collude. **v1.5:** with distributed DP, ε is also reported per number of honest clients h = 1…K.

---

## 0. Operating rules

1. Follow the Phases in order. There are **4 HUMAN GATES (🛑)**: stop, write a short report, wait for the user's confirmation.
2. **Do not modify or delete the raw data** (CSV, PCAP). Read only.
3. Do not hard-code paths, seeds or hyper-parameters: everything lives in `configs/*.yaml`.
4. Fixed seeds `{0,1,2}` for python/numpy/torch/sklearn. Every result is stored with its `seed`, `config_hash`, `git_commit` (when available).
5. **Never invent numbers.** Every number in a report must be read from a result file. Bad or unexpected results are reported as they are.
6. Every module has unit tests; `pytest -q` must be green before moving to a new Phase.
7. Before using the **Flower** and **Opacus** APIs, run `pip show <pkg>` and read the documentation of the installed version. Do not rely on memory. Pin the versions in `requirements.txt`.
8. The raw data are very large (39.0 million rows after round R, about 12 GB of CSV): **always read in chunks / streaming**, never load everything into RAM.
9. Windows paths contain spaces: use `pathlib.Path`, not string concatenation. With multiprocessing on Windows, put the main code under `if __name__ == "__main__":`.
10. After 2 unsuccessful fix attempts: write `BLOCKERS.md` (error, command, log, hypotheses) and stop; do not carry on by guessing.
11. **Label 🔶 = TO BE VERIFIED.** Attached to (a) thresholds / figures that are heuristics set by the author (settled by the user, kept under `thresholds` in the config), (b) library APIs or behaviour not yet verified on the installed version. For a 🔶 item the implementer must check / measure it in practice and write the result in the report, not assume it is right.
12. Every long step (FL training, Optuna, the experiment matrix) must **write checkpoints and be resumable** (see section 1.1).

---

## 1. Environment

There are two distinct running stages:

| Stage | Where | Work |
|---|---|---|
| **A. Data preparation** (Phases 1–5) | Personal Windows machine (the raw data are here) | Statistics, group split, sampling, preprocessing → small Parquet exports (tens of MB) |
| **B. Experiments** (Phases 6–13) | Colab CPU or the personal machine | CVAE, FL, DP, SecAgg, evaluation |

Libraries (pin the versions after installing): `python>=3.10`, `numpy`, `pandas`, `pyarrow`, `polars` (optional, for fast reading), `scikit-learn`, `imbalanced-learn`, `torch` (CPU), `flwr`, `opacus`, `optuna`, `scipy`, `matplotlib`, `psutil`, `pyyaml`, `joblib`, `pytest`, `streamlit` (demo).

### 1.1 Moving the data to Colab and surviving session loss
- Upload to Colab only the **processed data** (`processed/*.npz`, `feature_schema.json`, `split_manifest.json`, `partitions/`), which are small (estimated tens of MB 🔶). Do not upload raw data.
- How: copy to Google Drive and mount Drive in Colab. Point `compute.artifacts_dir` to Drive; **every** checkpoint, the Optuna SQLite files and `results/runs.csv` are written to Drive, never to Colab's temporary disk (lost when the session ends).
- Every Colab session starts with a single setup cell: install `requirements.txt`, mount Drive, `pip install -e .`, print the library versions.
- Every long process must be resumable: FL saves a checkpoint every `checkpoint_every_rounds` rounds; Optuna uses SQLite on Drive; `run_experiment.py` skips the runs already finished.

### 1.2 Compute budget (🔶 must be measured, not estimated)
- The spec does not commit to any running time. The implementer measures it at step **7.4** and writes `results/reports/compute_budget.md`.
- General note 🔶: DP-SGD (per-sample gradients) is usually several times slower than plain training; the FL simulation runs on one machine, so network latency is not measured.
- **v1.3:** Flower simulates with **Ray**; the clients run as parallel actors sharing the CPU, so the time of a round is not the sum of the clients' times (SPEC_DEVIATIONS 8.10); `ray` must be in `requirements.txt` (the `flwr[simulation]` extra does not install Ray on Windows with Python 3.13). Round 1 includes Ray's start-up, so take the median from round 2 on.

---

## 2. Repository layout

```
ppfeddata/
├── configs/
│   ├── default.yaml            # see section 3
│   ├── local.yaml              # per machine (raw-data path), NOT committed, deep-merged over default.yaml
│   ├── label_map.yaml          # mapping of folders/files → classes
│   ├── feature_decisions.yaml  # written in Phase 2 from the data + the user's decisions (G1, G2)
│   └── exp/                    # 1 file per experiment configuration (B0..M3)
├── src/ppfeddata/
│   ├── data/      inventory.py, harmonize.py, split_sample.py, parse_multi.py, preprocess.py, partition.py
│   ├── models/    cvae.py, generate.py
│   ├── fl/        app.py (ServerApp/ClientApp), run.py, core.py, b3.py, m1.py, m2.py, dp_utils.py, secagg.py   # v1.3: secagg_sim.py not needed
│   ├── eval/      utility.py, fidelity.py, privacy.py, overhead.py, stats.py, runs.py (run ledger), baselines.py, compare.py (paired bootstrap), dp_check.py (independent ε recomputation)
│   ├── checks/    leakage.py, leakage_report.py
│   ├── tune.py, tune_dp.py, run_experiment.py, aggregate.py, interpret.py, interpret_report.py,
│   │   limitations.py, demo_lib.py, readme_gen.py, package.py, acceptance.py, cli.py   # v1.3: Phases 11-13
├── tests/
├── notebooks/     00_eda.ipynb ... (view only, no logic)
├── data/          inventory/, interim/<mode>/, processed/<mode>/, partitions/   (NOT committed; <mode> = 6class | 11class)
├── artifacts/     {run_id}/ model.pt, synthetic.parquet, preds/
├── results/       runs.csv, summary.csv, figures/, reports/, manifests/   (manifests/split_manifest_<mode>.json is committed: group ids, counts, hashes, library versions only)
├── demo/          app.py
├── tools/         extract_normal_pcap.py (the script that extracted the Normal CSV from the PCAP, kept as it was run)   # v1.5, data provenance
├── docs/          PP-FedData_Implementation_Spec.md / .vi.md (this file), SPEC_DEVIATIONS.md / .vi.md (evidence log of the spec changes), ARCHITECTURE.md / .vi.md   # moved into docs/ on 2026-10-09; English and Vietnamese versions since 2026-10-21
├── requirements.txt, requirements-lock.txt, README.md, README.vi.md, BLOCKERS.md (if any)
```

Single CLI: `python -m ppfeddata.cli <command> --config ...` with command ∈ {`inventory`, `harmonize`, `sample`, `preprocess`, `check`, `baseline`, `tune`, `b2`, `benchmark`, `b3`, `m1`, `tune-dp`, `verify-dp`, `m2`, `m3`, `secagg-check`, `secagg-report`, `run`, `aggregate`, `demo`, `package`, `accept`, `fed-baseline`, `sensitivity`, `mia`, `scorecard`, `tune-dp-full`, `taug-rare`, `tune-fedsgd`, `secagg-bits`, `tune-marginal`, `recommend`, `mia-marginal`, `robustness`, `privacy-units`, `ids-followup`, `mg-nodp`, `check-extraction`, `extract-normal`} (**v1.3**; `scorecard`, `tune-dp-full`, `taug-rare`, `tune-fedsgd`, `secagg-bits`, `tune-marginal`, `recommend`, `mia-marginal` and `robustness` **v1.4**; `fed-baseline`, `sensitivity` and `mia` are the follow-ups after Phase 13: the classifier trained directly by FL, the extensions A4/A5, and membership inference with access to the released model).

---

## 3. Default configuration (`configs/default.yaml`)

```yaml
seeds: [0, 1, 2]
paths:
  raw_root: "./raw/DoS-DDoS-MQTT-IoT_Dataset"   # the real path goes in configs/local.yaml (not committed) or the PPFEDDATA_RAW_ROOT environment variable; quoted
  normal_csv_dir: "NormalData/CSV_Authors_Settings"   # round R (the authors' export settings); first extraction: NormalData/CSV_Extracted_Split
  work_dir: "./data"
  shared_manifest_dir: "./results/manifests"   # copy of split_manifest (no data rows) to commit
label_mode: "6class"            # "6class" | "11class"  (settled at G2). Binary Normal/Attack is only a side table derived from the multi-class predictions
split:
  test_groups_per_subclass: 1
  val_groups_per_subclass: 1
  fallback_block_split: {n_blocks: 10, gap_rows: 1000}
  split_seed: 0                  # decides WHICH groups/blocks go to val/test and which rows are sampled (A4 changes this value)
  min_groups_for_group_split: 3  # fewer groups -> block split
  min_eval_rows: 1000            # val/test of a class below this -> stop and ask the user
  max_peak_gb: 4.0               # Gate of Phase 3: peak memory
  drop_cross_block_streams: true # block split: drop the TCP streams that span several blocks (the gap cannot separate them)
quota:                          # [train, val, test] rows per class (6 classes)
  NORMAL:    [60000, 2000, 5000]
  BCF:       [20000, 2000, 5000]   # Basic Connect Flooding
  DELAYED:   [3000,  2000, 5000]
  SYN:       [3000,  2000, 5000]
  INVALID:   [1500,  2000, 5000]
  WILL:      [1000,  2000, 5000]
quota_11class:                  # [train, val, test]; used when label_mode = 11class
  NORMAL:        [60000, 2000, 5000]
  BCF_DoS:       [10000, 1000, 2500]
  BCF_DDoS:      [10000, 1000, 2500]
  DELAYED_DoS:   [1500,  1000, 2500]
  DELAYED_DDoS:  [1500,  1000, 2500]
  SYN_DoS:       [1500,  1000, 2500]
  SYN_DDoS:      [1500,  1000, 2500]
  INVALID_DoS:   [750,   1000, 2500]
  INVALID_DDoS:  [750,   1000, 2500]
  WILL_DoS:      [500,   1000, 2500]
  WILL_DDoS:     [500,   1000, 2500]
train_sampling:
  max_rows_per_stream: null     # null = off. An integer (e.g. 20) = cap on the packets per stream_id in the train pool (see Phase 3, Phase 9)
preprocess:
  multi_value: {message_type: first, numeric_multi: first}   # v1.2: the Normal export keeps only the first element (occurrence=f), so only the first element is comparable
  categorical_top_k: 10
  log1p_skew_threshold: 2.0
  clip_sigma: 5
  na_flag_range: [0.005, 0.995]  # add a <column>_is_na flag when the share of empty cells in train lies INSIDE this range
  ultra_sparse_fix: true         # empty >= upper bound (but not 100 %): statistics over the rows where it applies + a flag (false = the rule of v1.1)
  numeric_scale: {time_delta_from_previous_displayed_frame: 1000}   # multiplied before log1p (seconds -> ms): for a time column < 1 s, log1p is almost the identity, so 1.04 % of the rows were clipped at 5σ; after the scaling 0.08 %
cvae: {latent_dim: 16, hidden: [128, 64], beta: 0.5, beta_warmup_epochs: 10,
       lr: 1.0e-3, batch_size: 256, epochs: 50, class_balanced_sampler: false,
       patience: 5}                  # v1.3: early stopping on the validation loss, counted after the beta warm-up
fl: {num_clients: 5, rounds: 30, local_epochs: 2, dirichlet_alpha: 0.5, min_client_size: 500}
dp: {epsilons: [1, 5, 10], delta: 1.0e-5, max_grad_norm: 1.0}   # v1.3: the DP-specific max_grad_norm comes from configs/best_cvae_dp.yaml (Phase 9)
secagg: {num_shares: 5, reconstruction_threshold: 3, clipping_range: 16.0, max_weight: 100000}   # v1.3: 8 -> 16 (|w| observed up to 11.4); max_weight is the public upper bound of n_i (Flower's default 1000 clipped the FedAvg weights)
generate: {target_per_class: 20000, residual_noise: true, snap_support: false, snap_max_unique: 256}   # v1.3: per-class residual noise (Phase 7)
eval:
  rf: {n_estimators: 200}
  smote: {k_neighbors: 5}
  mlp: {hidden: [128, 64], max_iter: 100, early_stopping: true}
  bootstrap: 1000
tune: {n_trials: 30, syn_per_class: 5000, rf_trees: 100, epochs: 30}
thresholds:                     # 🔶 starting heuristics — settled by the user, not a scientific standard
  c2st_auc_max: 0.95            # the synthetic data must not be obviously fake
  dup_rate_max: 0.01
  dcr_ratio_min: 0.5
  mia_auc_max: 0.55
  overhead_ratio_max: 3.0
  seed_std_max: 0.02
  leakage_gap_flag: 0.05        # macro-F1 gap between a random split and a group (or stream) split
  single_feature_f1_flag: 0.9   # C3: a single column whose macro-F1 or best per-class F1 exceeds this -> inspect its meaning
  na_only_f1_flag: 0.5          # C2: an RF on the _is_na flags alone above this -> the model learns the extraction procedure
  dos_ddos_f1_flag: 0.6         # C5: DoS/DDoS F1 below this -> not separable (chance = 0.5)
  eps_max_recommend: 5.0        # v1.3, Phase 12 (R6): a DP configuration is recommended only if ε (max over clients and seeds) ≤ this
  seed_std_redflag: 0.05        # v1.3, Phase 12: red flag on the std of macro-F1 across seeds (seed_std_max = 0.02 is the soft note level)
  f1_near_one: 0.95             # v1.3, Phase 12: red flag when the macro-F1 of B0 ≥ this -> suspect leakage (back to Phase 5)
  eps_recompute_rtol: 0.10      # v1.3, Phase 12: red flag when the reported ε differs from the independent recomputation by more than this ratio
leakage: {rf_trees: 100, cv_folds: 5}
harmonize:                      # 🔶 heuristics of Phase 2; the user's decisions are in user_overrides / row_filters / g1_log / g2_log
  suspect_presence_diff: 0.95
  absent_presence_max: 0.001
  present_presence_min: 0.01
  time_ratio_suspect: 10.0
  token_min_share: 0.001
  token_min_count: 100
  # user_overrides, row_filters (protocol: [TCP, MQTT]), g1_log, g2_log: see configs/default.yaml
compute:
  artifacts_dir: "./artifacts"  # Colab: "/content/drive/MyDrive/ppfeddata/artifacts"
  runs_csv: "./results/runs.csv"   # v1.3: run ledger (Colab: a path on Drive)
  checkpoint_every_rounds: 5
  budget_hours: null            # 🔶 set by the user after step 7.4
```

Reference counts to check in Phase 1 (`expected_counts` in `configs/label_map.yaml`): total 39,000,601; Normal 24,955,877 (round R, Normal extracted with the authors' settings; the first extraction, every packet, had a total of 59,624,826 and Normal 45,580,102); Attack 14,044,724; BCF-DoS 5,825,026; BCF-DDoS 6,455,848; WILL-DoS 114,474; WILL-DDoS 156,055; DELAYED-DoS 291,988; DELAYED-DDoS 309,554; INVALID-DoS 109,150; INVALID-DDoS 181,190; SYN-DoS 103,828; SYN-DDoS 497,611.

---

## Phase 0 — Project set-up

**Work:** create the repository layout, `requirements.txt`, `pyproject.toml`/`setup.cfg` (editable install), standard logging, a `set_seed()` function, a `config_hash()` function, a working `pytest`.
**Tests:** `test_seed.py` (same seed → same numpy/torch random sequence); `test_config.py` (reads the yaml, stable hash).
**Gate:** `pytest -q` green; `python -m ppfeddata.cli --help` runs.

---

## Phase 1 — Raw-data inventory (`inventory.py`)

**Work:**
1. Scan `raw_root`: Attack follows the layout `<Scenario>/<DoS|DDoS>/CSV Files/*.csv`; Normal is in `normal_csv_dir`. Look at the real folder and file names, then write `configs/label_map.yaml` mapping them to the 5 scenario codes (`BCF, WILL, DELAYED, INVALID, SYN`) + `attack_type ∈ {DoS, DDoS, NONE}`.
2. For **every file** count the rows in chunks (without loading everything), the file size and the number of columns; skip empty files (`BF1_DoS_AD_11.csv`), **do not delete them**.
3. Define the **`group_id`**: Normal uses the `Capture_ID` column; Attack uses the file name. Look at the naming pattern: if the files are in fact consecutive slices of the same capture and the original capture can be identified, group by the original capture; otherwise use the file as the group and state the limitation clearly (adjacent slices can be quite similar).
4. Write `data/inventory/files.csv` (path, scenario, attack_type, class6, class11, group_id, rows, bytes) and `class_counts.csv`.

**Test/Gate:**
- Total rows = `expected_counts.total` (39,000,601 after round R) and every class matches `expected_counts` **exactly** (assert). A mismatch → stop and investigate.
- No file belongs to two groups; every file is assigned exactly one class.
- Report the number of groups per (scenario, attack_type). If a subclass has < 3 groups, write a warning (the block-split fallback will be used).

---

## Phase 2 — Schema harmonisation, format audit and presence matrix (`harmonize.py`)

**Work:** (every statistic is computed by streaming over the whole data and cached; `--recompute` recomputes, `--decisions-only` only regenerates the yaml)
1. Normalise the column names to snake_case; the 3 duplicated column names get a suffix by **position** (`qos_level_1/_2`, `frame_length_on_wire_1/_2`, `clean_session_flag_1/_2`); Normal's `.1` maps to `_2`. Read the header of every file: the 33 columns are in the same order in Normal and Attack (Normal adds `Label`, `Capture_ID`); a mismatch stops the run → `header_check.csv`.
2. **Presence matrix:** for every column × every subclass (11 classes), the share of non-empty cells. Write `presence_matrix.csv` and the heatmap `results/figures/presence_heatmap.png`.
3. **Format audit:** for every column × source (Normal vs Attack), count the "value kinds" with regexes: integer, float, hex, containing a comma, text, True/False, Set/Not set, empty → `format_audit.csv`. Goal: detect differences caused by the **extraction procedure**, not by network behaviour.
4. **Token audit** (`token_audit.csv`): count every value (multi-value cells split on commas) of the categorical/binary columns, per source. Confirmed encoding differences: (a) `message_type`, `qos_level_1/2`, `requested_qos`: Normal uses **numeric codes**, Attack uses Wireshark's **text labels** → mapping table `VALUE_MAPS` and `canonical_token` (truncated or run-together cells are matched by a unique prefix; anything that cannot be mapped goes to `OTHER`); (b) binary flags: Normal `True/False`, Attack `Set/Not set` → both mapped to 0/1.
5. **Duplicated column pairs** (`duplicate_pairs.csv`): `frame_length_on_wire_1/2`, `clean_session_flag_1/2` are exported from the same TShark field. If `_1` equals `_2` on every Attack row and Normal leaves `_1` empty, **drop `_1`, keep `_2`**.
6. **Multi-value cells:** compare the share of multi-value cells between the two sources. Normal was exported with TShark `-E occurrence=f` (first element only), so it has 0 % multi-value cells, while Attack has about 0.4 % to 1.6 % depending on the column. Hence `multi_policy: first_only` for every column and **`n_mqtt_msgs` is not used** (always ≤ 1 in Normal).
7. Generate `configs/feature_decisions.yaml` with the default rules (every column gets `action: drop|numeric|binary|categorical`, with a reason):
   - `drop`: `no`, `epoch_time`, `time_since_reference_or_first_frame`, `stream_index` (dropped from the features but **kept as metadata** `stream_id = (group_id, stream_index)`), `source`, `info`, `label`, `capture_id`, the `_1` copy of a duplicated column (step 5), every column empty ≥ 99.9 % in every class. (`requested_qos` is NOT dropped automatically: the > 99.9 % empty rule does not hold on the real data.)
   - `numeric`: `frame_length_on_wire_2`, `time_delta_from_previous_displayed_frame`, `irtt`, `time_since_first_frame_in_this_tcp_stream`, `tcp_segment_len`, `calculated_window_size`, `keep_alive`, `user_name_length`, `password_length`, `will_message_length`, `will_topic_length`, `topic_length`, `msg_len`.
   - `binary`: `syn`, `reset`, `acknowledgment`, `clean_session_flag_2`, `retain`, `will_retain`, `will_flag`. `categorical`: `protocol`, `message_type`, `qos_level_1`, `qos_level_2`, `requested_qos`.
   - **SUSPECT (flags raised from the numbers, drop by default, the user decides):** (a) a column empty in one source but present in the other, unless the *parent event* is present in Normal (`will_*` depends on `will_flag`, `requested_qos` depends on SUBSCRIBE; Normal is then empty because the feature is not used, not because of an extraction error); (b) an empty-share difference > 0.95 between Normal and an Attack class; (c) values **present only in Normal** (share ≥ `token_min_share` and count ≥ `token_min_count`; for example STP, LOOP, SSDP in `protocol`); (d) a strongly skewed time column (step 8). Values present only in Attack (for example `will_flag` = Set) are attack behaviour and are NOT flagged.
   - The user's decisions are not overwritten on a rerun: `harmonize.user_overrides`, `row_filters`, `g1_log`, `g2_log` in `default.yaml`. The row filter is stored in the yaml (`_row_filters`) and Phase 3 reads it from there.
8. **Comparability check of the time features** (`time_delta_from_previous_displayed_frame`, `irtt`, `time_since_first_frame_in_this_tcp_stream`): compare the percentiles (1, 25, 50, 75, 99 %) of the two sources on the **same packet kind** (TCP without MQTT) → `time_feature_audit.csv`. A column whose max/min ratio of p25, p50, p75 or p99 is > `time_ratio_suspect` → SUSPECT. Write in the limitations: the CVAE generates every packet independently, so the time sequence between packets is not kept.

**Tests:** `test_harmonize.py` (column-name mapping, the `.1` columns; format regexes on `"Publish Message,Publish Message"`, `"40,41"`, `""`; `canonical_token` for text labels, numeric codes and truncated cells; the parent-event rule, the count thresholds, the duplicated pairs, `user_overrides`; the header check).
**Gate 🛑 G1:** hand in `presence_matrix.csv`, `format_audit.csv`, `token_audit.csv`, `feature_decisions.yaml` with the SUSPECT list (`results/reports/g1_feature_review.md`, generated automatically). The user approves the feature set.
**G1 decisions settled (2026-10-03):** keep `will_message_length`, `will_topic_length`, `requested_qos`; keep `protocol` and filter the rows (both Normal and Attack) to `protocol ∈ {TCP, MQTT}`; `first_only` and drop `n_mqtt_msgs`; `time_since_first_frame_in_this_tcp_stream` is provisionally a `diagnostic` column.

---

## Phase 3 — Group split and streaming sampling (`split_sample.py`)

**Work:**
1. **Pass A:** every file is read once, taking only the row-filter columns, `Stream index` and (Normal) `Capture_ID` → a validity mask from `_row_filters` + the stream indices; check that the row counts match the inventory and that `Capture_ID` matches `group_id`. Cached on disk.
2. **Split:** every subclass (scenario × attack_type) gives `test_groups_per_subclass` groups to test, `val_groups_per_subclass` to val, the rest to train; chosen by `split_seed`, skipping the groups without enough rows for the quota (recorded). Normal uses `Capture_ID`.
3. **Block split** for a subclass with < `min_groups_for_group_split` groups: cut it into `n_blocks` consecutive blocks, dropping `gap_rows` buffer rows between blocks; every block is a "group" with `group_id = <file>#blkNN`. Because the gap cannot separate a TCP stream that lives through the whole capture, **drop every stream with valid rows in more than one block** (`drop_cross_block_streams`; write the number of streams and rows dropped in the manifest).
4. **Sampling:** the row count of every file is known, so random row indices are drawn in advance without replacement for every group/block (equivalent to a reservoir, one read pass, deterministic in the seed). Train is drawn evenly over the train groups (an even split; a group short of rows passes the remainder to the others). Label mode (`label_mode`):
   - `6class` (default): uses `quota`; every scenario is split evenly between DoS and DDoS (DoS gets the odd row).
   - `11class`: uses `quota_11class`. With the same `split_seed` both modes use the **same test and val groups/blocks**.
   - Binary Normal/Attack is not a separate mode; it is only reported as a side table by merging the multi-class predictions.
   - If there are not enough rows for the quota: reduce the quota and write a warning; if val or test of a class falls below `min_eval_rows`, **stop and ask the user**.
4b. (Optional, `train_sampling.max_rows_per_stream`) Cap the packets per `stream_id` in the train pool (reason: Phase 9, "Unit of protection"). Off by default; run as sensitivity analysis A5.
5. Write `data/interim/<label_mode>/{train,val,test}.parquet` (harmonised raw columns, as strings, with `label`, `class6`, `class11`, `split`, `data_source`, `group_id`, `stream_id`, `source_file`, `row_idx`; note that the raw column `source` is the sender address) and `split_manifest.json` (which group/block is in which split, eligible and sampled row counts, seed, `config_hash`, `git_commit`, `git_dirty`, python/numpy/pandas/pyarrow versions, sha256 of every Parquet, peak memory). A copy of the manifest is written to `results/manifests/split_manifest_<mode>.json` for committing; **rerun to produce the manifest only on a clean git tree** (`git_dirty: false`).

**Test/Gate:**
- **No group (or block) and no TCP stream appears in more than one split** (assert); no raw row is sampled twice; the row filter holds.
- Row counts match the quota (or carry a clear note when reduced for lack of data).
- A rerun with the same seed and the same library versions gives Parquet files with the same hash.
- Peak memory (psutil) < `max_peak_gb` (4 GB).
- Refuse to run when `feature_decisions.yaml` is missing (the row filter must not be dropped silently).

---

## Phase 4 — Preprocessing (`parse_multi.py`, `preprocess.py`)

**Work:**
1. `parse_multi(cell) -> list[str]`: split multi-value cells, handle empty/`nan`/whitespace. Under `multi_policy: first_only` every column uses the **first element** (`first_values`); `n_mqtt_msgs` is not produced.
2. A `Preprocessor` class (fitted on **train** only, saved with joblib; `transform` does not change the fitted state):
   - Numeric columns: fill 0 for not-applicable values, add a `<col>_is_na` flag when the empty share in train lies inside (`na_flag_range`); `log1p` when the skew > the threshold and the column is non-negative (computed on the filled column); mean/std standardisation (std = 0 → scale = 1); clip at ±`clip_sigma`. Time columns are clipped to ≥ 0 before `log1p` (there are negative values around -1e-6) and the clipped rows are counted. `preprocess.numeric_scale` multiplies a column by a constant before `log1p` (by default only `time_delta_from_previous_displayed_frame` × 1000, because `log1p` of a value < 1 s is almost the identity and the tail is clipped at 5σ); the original value range in the schema and `inverse_transform` stay in the original unit.
   - **Ultra-sparse columns** (empty ≥ the upper bound of `na_flag_range` but not 100 %, `ultra_sparse_fix: true`): the statistics are computed over the rows where the column applies, empty rows sit at 0 after standardisation, and **a flag is added**. The v1.1 rule (fill 0, then standardise over the whole column) clipped every applicable row at +5σ and merged distinct values (`will_message_length` = 2 and 3164).
   - Binary columns: 0/1 (`True/False`, `Set/Not set`, `0/1`), empty → 0 + an `_is_na` flag by the same rule.
   - Categorical columns: normalise the tokens with `canonical_token` (numeric codes and text labels treated alike), top-K by train frequency + `OTHER` (outside the top-K or not mappable) + `NONE` (empty cell); one-hot.
   - **`diagnostic` columns** (in `feature_decisions.yaml`): processed like numeric columns but exported separately (`X_diag`), not part of `D`, for Phase 5 to measure. The current configuration has no diagnostic column left (see G2).
3. Write `feature_schema.json`: contiguous column layout `[numeric][binary][na_flag][categorical groups]`, every block with `name`, `type`, `column`, `start`, `width`; the standardisation parameters, the category lists, the audit statistics (rows clipped at 5σ, cells that could not be parsed). `Preprocessor.inverse_transform_block` and `inverse_transform` serve the data generation (NaN when the flag = 1 or the category is `NONE`; a column without a flag keeps the fill value 0 = "not applicable").
4. Write `data/processed/<label_mode>/{train,val,test}.npz` (float32 `X`, `X_diag`, int `y`, `group_id`, `stream_id`, `class11`), `label_map.json` (order as in the config), `preprocessor.joblib`.

**Tests:**
- `parse_multi`: real examples, empty cells, cells with 3 elements.
- No `NaN/inf` in X; the dimension matches the schema; every one-hot group sums to exactly 1; contiguous layout.
- Numeric round trip: `inverse(transform(x))` relative error < 1e-4 (except the clipped part).
- Leakage guard: `transform`/`audit` on val/test do not change the fitted parameters; fitting on train+val does change them (control group).
- Constant columns (std = 0), negative values in time columns, ultra-sparse columns keep their values distinguishable.
**Gate:** print `D`, the number of `_is_na` flags, the class distribution of every split. Actual result (6 classes): D = 60, 12 flags, maximum relative round-trip error 1.2e-5, 0 values that could not be parsed.

---

## Phase 5 — Leakage and label-reliability checks (`checks/leakage.py`)

Random Forest with 100 trees, seeds {0,1,2}, macro-F1 measured on **val** (groups other than train). 🔶 Every threshold is a starting heuristic (`thresholds` in the config): the implementer only **reports and flags**, the user decides at G2. Train is very imbalanced while val is balanced, so the report always gives both the unweighted RF and the class-balanced RF.

| Code | Check | How to read the result (heuristic, the user decides) |
|---|---|---|
| C1 | Presence matrix per class (from Phase 2) | A column empty in one class and full in another → suspect an artifact; compare with the parent event |
| C2 | RF on the `_is_na` flags only | macro-F1 > `na_only_f1_flag` (far above chance 1/number of classes) → the model learns "the extraction procedure"; list the columns that contribute most |
| C3 | **Class-balanced** RF (`min_samples_leaf=5`) on **every raw column** (value + flag + one-hot of that column) | macro-F1 **or the best per-class F1** > `single_feature_f1_flag` → inspect its meaning. (The unweighted RF collapses to the majority class and hides the signal.) Plus leave-one-column-out from the full model |
| C4 | Split the train pool by random rows, by **stream**, by **group** (stratified CV) | Flag when (random − group) or (random − stream) > `leakage_gap_flag`. (random − val) is for reference only: the CV folds keep the imbalanced class distribution while val is balanced, so the two macro-F1 are not directly comparable |
| C5 | DoS vs DDoS within the same scenario (11-class data) | F1 < `dos_ddos_f1_flag` (≈ chance 0.5) → recommend 6 classes |

Also for the G2 decision: the reference model (unweighted and class-balanced), the ablations `+ diagnostic columns` and `without the time columns` (with per-class F1).

**Deliverables:** `results/reports/leakage_report.md` (tables, figures, flags) + `results/reports/leakage/*.csv`, `leakage_results.json`. Update `feature_decisions.yaml` only when the user decides at G2.
**Gate 🛑 G2:** the user settles (a) the final column list, (b) `label_mode` (6 or 11 classes). Then rerun Phase 4 (and Phase 3 if the row filter or `label_mode` changes).
**G2 decisions settled (2026-10-03):** (a) `time_since_first_frame_in_this_tcp_stream` joins the main feature set (adding it raises the class-balanced macro-F1 from 0.442 to 0.499, the gain being mostly in the attack classes: BCF +0.10, DELAYED +0.11, NORMAL +0.013, so it is not a Normal-vs-Attack artifact); keep `irtt`, `time_delta...`, `will_*`, `requested_qos`, `keep_alive`. (b) `label_mode = 6class`. (c) 3 new thresholds confirmed.
**Results after G2 (6 classes, D = 60, after `numeric_scale`):** val macro-F1 0.453 (unweighted) and 0.498 (class-balanced); C2 0.103 (below chance 0.167); C3 no column flagged (highest macro-F1 0.31, highest per-class F1 0.61); C4 random−group gap +0.007, random−stream +0.002; C5: BCF 0.593 (flagged), DELAYED 0.614, WILL 0.690, INVALID 0.788, SYN 0.962. **Key conclusion:** no leakage found, but the attack classes are hard to separate at the packet level (a random row split inside the train pool itself only reaches 0.53), so a low B0 is not a sign of a bug.

---

## Phase 6 — Baselines and the evaluation framework (`eval/`)

### 6.1 Three evaluation protocols (all tested on the **real test set**)
- **TRTR:** train on real data (train pool) → test on real data. This is **B0**.
- **TSTR:** train on generated data only (balanced, `syn_per_class`) → test on real data.
- **TAug:** train on (train pool + extra generated data so that every class reaches at least `target_per_class`; large classes are not reduced) → test on real data.

### 6.2 Classifiers
`RandomForestClassifier(n_estimators=200, n_jobs=-1)` and `MLPClassifier(hidden=(128,64), early_stopping=True)`. Save the **predictions** of every run (`artifacts/{run_id}/preds/`) for the paired bootstrap.

### 6.3 Baselines
- **B0:** real data only.
- **B1a:** RF with `class_weight="balanced"` (**v1.3:** RF only; sklearn's `MLPClassifier` has no `class_weight` and the code refuses that combination). **B1b:** SMOTE in the encoded space up to `target_per_class`, with post-processing (argmax for the one-hot groups, rounding of the binary flags). The runs are B0-rf, B0-mlp, B1a-rf, B1b-rf, B1b-mlp. **v1.3:** `run_id = {config}_{seed}` where `config` already includes the classifier (`B0-rf`); the 11-class mode adds `11class` to the id.
- **v1.3 (run ledger):** `eval/runs.py` (`RunLedger`) writes `results/runs.csv` from Phase 6 on: a finished run is skipped on a rerun, a run with the same id is replaced; the test predictions are stored in `artifacts/<run_id>/preds/test.npz` (Phase 12 reads them back for the bootstrap). The runner refuses to run if `feature_schema.json` does not record `fit = {split: train, n_rows}` matching the train row count.
- (Optional) **W:** a centralised WGAN-GP, only to measure time and parameter count, showing that "WGAN-GP is heavier than the CVAE".

### 6.4 Metrics
- **Utility:** macro-F1 (the main metric), per-class recall/precision/F1 (especially `WILL`, `INVALID`), balanced accuracy, macro OvR PR-AUC, confusion matrix, and a binary Normal/Attack side table derived by merging the multi-class predictions.
- **Fidelity** (`fidelity.py`): Wasserstein on the numeric columns (divided by the train std), Jensen-Shannon for the binary/categorical columns, correlation distance (mean |Δ Pearson|), **C2ST** (an RF telling real from fake, 5-fold, AUC; 0.5 is best). Computed per class.
- **Privacy** (`privacy.py`):
  - exact duplicate rate between synthetic and train (hash after rounding);
  - **DCR** (distance to the closest train record, in the standardised feature space) and the ratio `median DCR(syn→train) / median DCR(syn→val)`; ≈ 1 is good, ≪ 1 means copying;
  - **Simple MIA:** score = −DCR of a record to the generated set; members = train samples, non-members = val samples; AUC computed **per class** (balanced counts), then averaged. AUC ≈ 0.5 is good.
  - **Positive control:** train a deliberately overfitted CVAE (500 samples, no DP, many epochs); the MIA must give an AUC clearly > 0.5. If not → the MIA is too weak, and the report must say so.
- **Cost** (`overhead.py`): time per round and in total, bytes sent per round (measured or computed analytically), peak RAM (psutil).
- **Statistics** (`stats.py`): stratified bootstrap over the test set (1000 resamples) for the CI of macro-F1 and the **CI of the paired difference** between two configurations.

**Tests:** metrics on fake data with a known answer (perfect predictions → F1 = 1; random predictions → about 1/number of classes); the `Preprocessor` is not reused between splits; results repeat with the same seed.
**Gate 🛑 G3 (baseline sanity):** B0 and B1 run with all 3 seeds; std of macro-F1 across seeds < 0.02; B0 macro-F1 is not ≈ 1.000 in every class (if so: back to Phase 5, suspect leakage). The user looks at the B0/B1 table before going on. Realistic expectation measured in Phase 5 (RF 100 trees, val): macro-F1 ≈ 0.45 (unweighted) and ≈ 0.50 (class-balanced); a low B0 is a property of packet-level data, not a bug.

---

## Phase 7 — Centralised CVAE (B2) and Optuna

### 7.1 Architecture (`models/cvae.py`)
- Encoder: `[x ; onehot(y)] → Linear(hidden[0]) → ReLU → Linear(hidden[1]) → ReLU → (μ, logσ²)` (latent 16).
- Decoder: `[z ; onehot(y)] → reversed hidden → outputs per block`: logits for the binary flags/`_is_na` (BCE), logits for every categorical group (CE), numeric values (MSE). The model parameters are **one vector of fixed size** (so FedAvg and SecAgg apply unchanged).
- **No BatchNorm** (incompatible with Opacus); only `nn.Linear`, `ReLU`, optionally `LayerNorm`.
- Loss = sum of the block reconstructions + `β · KL`, averaged over samples; β rises linearly from 0 to `beta` over `beta_warmup_epochs`.
- Early stopping on the val loss.

### 7.2 Data generation (`models/generate.py`)
`generate(model, class_counts, seed)`: draw `z ~ N(0,I)` and labels for the requested counts; numeric columns → inverse standardisation/log1p → clip to the train [min, max] → round the integer columns; binary/`_is_na` columns are sampled from a Bernoulli; categorical columns are sampled from the softmax; if `_is_na = 1`, the matching numeric value is set to its default.
- **v1.3 (residual noise):** the MSE decoder outputs the mean, so the generated data shrink in variance; Gaussian noise is added with the **per-class standard deviation of the reconstruction residuals** (estimated on train, `generate.residual_noise: true`). This variant is the default for the configurations **without DP**. The noise scale is computed from the whole train pool, so it is **outside ε**: the DP configurations (M1, M3) have two evaluation variants, `plain` (the decoder alone, entirely within ε, the **main number**) and the variant with noise (for comparison only); B3 is also evaluated as `B3-plain` (the ε = ∞ point with the same decoder). Categories with 0 train rows are excluded (`dead_category_mask`). `snap_support` was tried and turned off.

### 7.3 Hyper-parameter optimisation (`tune.py`)
- Optuna TPE, `n_trials` = 30, stored in SQLite for resuming.
- Space: `latent ∈ {8,16,32}`, `hidden width ∈ {64,128,256}`, `β ∈ [0.1, 2]` (log), `lr ∈ [1e-4, 3e-3]` (log), fixed 30 epochs.
- **Fitness:** TSTR macro-F1 (RF 100 trees, `syn_per_class` = 5000) evaluated on the **real val set**. The test set is not used.
- Result: `configs/best_cvae.yaml`. State the limitation clearly: tuning is done on centralised data, without privacy. **v1.3:** tuning goes through the final generation path (with residual noise); training runs at most 30 epochs (exactly the tuned number) with early stopping `cvae.patience`.

**Tests:** input/output shapes; the loss falls after a few epochs on small data; `generate` returns the right count per class; values lie in the valid range; generated one-hots sum to exactly 1; reproducible by seed.
**Gate:** B2 runs with all 3 seeds. Minimum checks (🔶 thresholds from `thresholds`): C2ST AUC < `c2st_auc_max`, duplicate rate < `dup_rate_max`, DCR ratio > `dcr_ratio_min`. Report TSTR and TAug against B0, even without improvement. **v1.3:** the failed C2ST (≈ 1.000 for every generator) is reported as the limitation "low fidelity" and does not block the later Phases (the user's decision); the MIA positive control (section 6.4) runs in Phase 7, and **if it fails, the report must say clearly** that an AUC ≈ 0.5 only says that no generated row is a near copy of a train row; it does not prove the absence of memorisation.

### 7.4 Resource benchmark (🔶 mandatory before Phase 8)
1. On one client (n ≈ 18,000 rows), measure: the time of 1 plain CVAE epoch, the time of 1 DP-SGD epoch (Opacus), the peak RAM.
2. Extrapolate: time of 1 FL run ≈ `rounds × local_epochs × Σ_i t_epoch_i` (an upper bound when the clients run one after another; **v1.3:** in practice the clients run in parallel, so the measurement is below the formula, see section 1.2); the whole matrix = the sum of the runs in the Phase 11 table × 3 seeds; Optuna time = `n_trials` × the time of 1 trial. Write it to `results/reports/compute_budget.md` with the raw measurements.
3. If the total exceeds `compute.budget_hours` (set by the user) or does not fit the planned number of Colab sessions, cut **in this order**: (a) Optuna 30 → 15 trials; (b) `rounds` 30 → 20; (c) the seeds of M1-ε1 and M1-ε10 down to 2; (d) drop the extensions A1–A5. **Do not cut** the seeds of B0–B3, M1-ε5, M2, M3. Every cut must be stated in the final report.
4. Optuna stores its SQLite in `compute.artifacts_dir` so it can resume across Colab sessions.

---

## Phase 8 — Non-IID FL (B3) (`partition.py`, `fl/`)

**Work:**
1. `partition.py`: split the train pool over `num_clients` with a Dirichlet(α) over the labels; every client ≥ `min_client_size` (otherwise redraw the Dirichlet sample); store the indices in `data/partitions/alpha{α}_seed{s}.json`; write the per-client label distribution table and heatmap.
2. The Flower application (simulation, one machine): use `flwr new` to get the template of the installed version, **then replace the model with the CVAE**. Client `fit`: train `local_epochs` epochs on the local data, return the weights + the sample count. Server: FedAvg weighted by sample count. Common initialisation from the seed.
3. Every 5 rounds, evaluate the ELBO of the global model on val (for monitoring only; note that this is a simulation-side evaluation that does not exist in real FL). Save the final checkpoint and the per-round log.
4. After the last round: generate data and run the full evaluation (Phase 6).
5. **Checkpoint/resume:** every `checkpoint_every_rounds` rounds save to `artifacts/{run_id}/ckpt/`: the global weights, the round index, the RNG state, every client's DP step counter (Phase 9), the round log. The `--resume` option loads the latest checkpoint.
6. **v1.3 (how Flower is run):** use the message API of Flower 1.39 (`ServerApp`, `ClientApp`, `FedAvg.start`) launched with `run_simulation` on the Ray backend, every run in its own process (`python -m ppfeddata.fl.run`) so that seed/resume/spec are controlled in code. The client uses a fresh Adam every round, the beta warm-up follows the global epoch. **The replies are aggregated in client-id order** (the float32 summation order made two runs with the same seed differ; once it was fixed they are bit-identical). Stop immediately if a round misses a client reply (`require_all_replies`) and rerun automatically from the checkpoint at most 2 times (Ray on Windows has killed actors mid-run). The data are split **by row, not by TCP stream**, so a stream can sit on several clients. Every generation statistic (the per-class residual noise) is computed from the whole train pool at the simulated server and is not covered by DP.

**Test/Gate:**
- Partition: the index sets are pairwise disjoint, their union = the whole train pool.
- **FedAvg is correct:** with random weights, the aggregate equals the weighted mean computed by hand with numpy (error < 1e-6).
- **Sanity against centralised training:** 1 client (all the data) is equivalent to B2 (val loss differs by < 5 %).
- IID (large α) versus non-IID: plot the loss per round; non-IID is expected to converge more slowly or worse (not mandatory, but the opposite must be explained).
- All 30 rounds run without error; log the time and bytes per round.
- **Resume:** stopping at round 10 and resuming to round 20 gives valid weights, all 20 log lines, and DP step counters exactly equal to an uninterrupted run.

---

## Phase 9 — Client-side DP (M1) (`fl/dp_utils.py`) 🔶 Opacus API to be verified on the installed version

**Work:**
1. For every client `i` (size `n_i`), the sampling rate `q_i = batch_size / n_i`, the total number of steps `T_i = rounds × local_epochs × (n_i / batch_size)`. Compute `σ_i` once for the whole training with Opacus's utility (`get_noise_multiplier` with `target_epsilon`, `target_delta`, `sample_rate`, and the number of steps or epochs; check the function signature in the installed version). **v1.3 (Opacus 1.6.0):** the Poisson loader has `q = 1/ceil(n/B)` and `ceil(n/B)` steps per epoch (not the raw `B/n` and `n/B`; because `int(1/q)` sometimes truncates one step away, **the reported ε uses the counted steps**); changing the plan (number of rounds, ε, δ) on resume changes σ and is refused.
2. Every round, `PrivacyEngine.make_private(..., noise_multiplier=σ_i, max_grad_norm=C, poisson_sampling=True)`. **Note:** re-creating the engine every round **resets the accountant**, so the per-round `get_epsilon` is not used. Instead, count the total number of steps and compute the final ε with an RDP accountant from `(σ_i, q_i, T_i)`.
3. Run `ModuleValidator.validate` to make sure the model is compatible (no BatchNorm).
4. Sweep `ε ∈ {1, 5, 10}` (δ = 1e-5; make sure δ < 1/n_i for every client); add a "no DP" point (= B3).
5. **How ε is reported when the clients have different `n_i`:** calibrate `σ_i` so that every `ε_i ≤` the target. For every configuration report the table of the per-client `ε_i`; **the configuration's ε = max_i ε_i** (the worst case, used as the main number in the figures and in the recommendation rule), with the median. Meaning: ε_i protects a record of client i against every party that only sees that client's updates. No privacy amplification from SecAgg is claimed.
6. **Unit of protection (important limitation):** ε here is at the **record (packet) level**. Packets of the same TCP stream and the same capture are correlated, so an attack session made of many packets is protected much more weakly than the ε number says (by group privacy, the degradation is of the order of the number of packets in the group; 🔶 do not claim an exact formula unless it has been computed). **v1.3:** report the measured size of these two units (rows per stream and per capture group in train; in these data a stream has only ~1.0-1.2 rows because the rows are sampled at random, while a capture group holds tens to hundreds of rows of a rare class, so the correlated unit in practice is the capture). The report must state this clearly. Mitigation (optional): turn on `train_sampling.max_rows_per_stream` and run the sensitivity analysis A5 to see whether the results depend on within-stream correlation.
7. The labels (the condition) are not protected by the DP-SGD of the features; state it in the limitations.
8. **v1.3 (DP-specific hyper-parameters):** with C = 1 every per-sample gradient is clipped (norms 4-40), hence `tune_dp.py` (Optuna, simulating one client with 20 % of the data on the exact FL code path, fitness = val macro-F1 of TSTR-rf with the plain decoder, val only) and `verify-dp` (real FL runs for the best few candidates, with a selection rule fixed in advance: max val macro-F1, within 0.01 pick the lower val ELBO) → `configs/best_cvae_dp.yaml` (family `M1d-t21`). This family is the main DP family of the matrix; the Phase 7 hyper-parameter family (`M1-eps*`) is kept as a reference table. It must be stated: tuning uses the real val set without privacy and is not counted in ε, one seed per trial, tuned at ε = 5.
9. **v1.3 (decoder within ε):** every DP model has two evaluation variants: `plain` (the main number) and with residual noise (outside ε, for comparison only); see Phase 7.2. **ε is recomputed independently** with a hand-written RDP code (`eval/dp_check.py`, integer orders 2-256, the same conversion as Opacus) and compared with the reported ε (red flag F5).

**Test/Gate:**
- The achieved ε ≤ 1.02 × the target ε for every client (also after resuming from a checkpoint: the ε recomputed from the total steps must equal an uninterrupted run).
- σ increases as the target ε decreases.
- With σ = 0 and a very large C, the result is close to B3 (sanity).
- Expected trend: TSTR macro-F1 falls as ε falls; MIA AUC and the DCR ratio approach 0.5 and 1. If the trend does not appear, check the strength of the MIA (positive control) before concluding.

---

## Phase 10 — SecAgg (M2) and DP + SecAgg (M3) 🔶 Flower API to be verified on the installed version

**Work:**
1. Use `SecAggPlusWorkflow` (or `SecAggWorkflow`, a special case of SecAgg+) on the server and `secaggplus_mod` on the client; start from the example `flwr new @flwrlabs/flower-secure-aggregation`, then replace the model with the CVAE. Starting parameters: `num_shares` = the number of clients, `reconstruction_threshold` = 3 (read the documentation to confirm the meaning and constraints of these two parameters in the installed version); leave `clipping_range`, `quantization_range`, `modulus_range` at their defaults, then check them.
2. Check that `max |w|` of every parameter over every round is below `clipping_range`; if it exceeds it, raise `clipping_range` or warn (clipped values cause a bias).
3. M3 = client DP (Phase 9) + SecAgg together. Record the achieved ε.
4. **Fallback** if Flower's SecAgg does not run in the simulation of the installed version: write `secagg_sim.py` (Bonawitz-style pairwise masking: every pair of clients adds/subtracts the same mask from a shared seed, the masks cancel in the sum), stating clearly that this is a substitute simulation and how it differs from the full protocol (no dropout tolerance). **v1.3:** not needed; Flower 1.39.0's `SecAggPlusWorkflow` and `secaggplus_mod` run in the Ray simulation.
5. **v1.3 (how Flower 1.39.0 is used):** `SecAggPlusWorkflow` belongs to the old API (it needs `LegacyContext` and an old `Strategy`), so the B3 ServerApp loop (validation, log, checkpoint, resume) is kept and every round calls the workflow **unchanged** through a minimal old `Strategy` (`fl/secagg.py`); the client uses the same training function (plain or DP-SGD) wrapped in `secaggplus_mod`. `num_shares` = the number of secret shares of every client, `reconstruction_threshold` = the minimum number of shares needed to reconstruct (threshold < num_shares, num_shares > 2; 5/3 tolerates at most 2 dropped clients per round). **`max_weight` must be ≥ the largest `n_i`** (raised to 100,000) and `clipping_range` is 16 after the |w| check (item 2). Quantisation error bound = (number of clients) × 2·clip/2^22 ÷ Σ(n_i/max_weight). The `num_examples` and metrics of FitRes travel unencrypted in Flower (the server knows `n_i`, which is public in this study). A round with a missing client **stops the run**, which then resumes from the checkpoint (so that the DP plan and ε ledger stay as designed). SecAgg does not enter the ε ledger and **no privacy amplification is claimed**.

**Test/Gate:**
- **T-SA1 (correctness):** the SecAgg global model against plain FedAvg with the same seed: `max |Δ|` ≤ 1e-4 (quantisation error); the macro-F1 difference lies within the std across seeds.
- **T-SA2 (hiding, if the workflow's log can be hooked):** the masked vector the server receives has a Pearson correlation |ρ| < 0.05 with the true update and an approximately uniform distribution on [0, modulus) (KS test). If it cannot be hooked, write "not verified directly".
- **T-SA3 (optional):** drop 1 client in one round; the aggregation still completes if ≥ the threshold remain.
- Measure the SecAgg **overhead**: `time per SecAgg round ÷ time per plain round` and bytes per round (note: a one-machine simulation measures the cryptographic computation, not the network latency). **v1.3:** time = the median of rounds 2+; bytes counted on the wire (protobuf size), normalised by the parameter count because the DP-tuned model is smaller than the B3/M2 model; M2 is compared with B3 and M3 with M1 at the same DP configuration.
- **v1.3:** T-SA1, T-SA2, T-SA3 run on the real data (`secagg-check`) and are written to `artifacts/secagg_checks_<mode>.json`; report `m2_m3_secagg.md` (`secagg-report`). FL training is sensitive to noise of order 1e-5 (Adam re-initialised every round), so any two runs that are not bit-identical can differ by about ±0.03 TSTR macro-F1: the T-SA1 utility criterion is "within the std across seeds", and the error of a single aggregation must be ≤ the quantisation bound.

---

## Phase 11 — Experiment orchestration (`run_experiment.py`, `aggregate.py`)

**Matrix (every configuration × seed {0,1,2}):**

| ID | Configuration |
|---|---|
| B0 | Real data only |
| B1a / B1b | class_weight / SMOTE |
| B2 | Centralised CVAE |
| B3 | CVAE + FL |
| M1-ε1, M1-ε5, M1-ε10 | FL + DP |
| M2 | FL + SecAgg |
| M3 | FL + DP(ε=5) + SecAgg |

**v1.3:** M1 and M3 use the DP-specific hyper-parameter family (`M1d-t21`, `M3d-t21`; Phase 9) and are evaluated with the `plain` decoder as the main number. Outside the matrix there is a `reference` group (`B3-plain` = the ε = ∞ point; `M1p7-eps*` = the Phase 7 hyper-parameter family) and an `extension` group; every configuration has a file `configs/exp/<id>.yaml` (`kind`, `group`, `est_minutes`). A1 has a runner (`α ∈ {0.1, 10}`; α = 0.5 is B3 itself); A2-A5 need Phases 3-4 rerun (different quota, `split_seed`, `max_rows_per_stream`), so they are only configurations marked `needs_data`.

**Extensions (only if time remains, the first to be cut when over budget):** A1 non-IID skew `α ∈ {0.1, 0.5, 10}` on B3; A2 train-pool imbalance ratio `{10:1, 60:1, 200:1}`; A3 the 11-class mode; A4 **sensitivity to the test groups**: change `split_seed` (2 other choices of test/val groups), then rerun B0 and B3 to see whether the conclusions depend on the files chosen; A5 `max_rows_per_stream` on versus off on B3 and M1-ε5.

**Requirements:**
- Every run has `run_id = {config}_{seed}`; it writes one line to `results/runs.csv` (metrics, time, achieved ε, config hash). **Resume:** finished runs are skipped, and an unfinished run is reloaded from its checkpoint.
- `aggregate.py` produces `results/summary.csv` (mean ± std per configuration) and the figures: (1) macro-F1 per configuration (error bars); (2) per-class recall for the rare classes; (3) the **utility–privacy** curve (macro-F1 vs ε, with MIA AUC vs ε); (4) fidelity (Wasserstein, C2ST) vs ε; (5) overhead (time per round, bytes per round); (6) Pareto: macro-F1 vs MIA AUC, point size by overhead.
- `results/reports/final_report.md` is generated automatically from the tables; no number is written by hand.

**Gate 🛑 G4 (before the full run):** a trial run of 1 seed for every configuration, look at the logs; the full 3 seeds run only after the user agrees. **v1.3:** G4 is encoded in the tools: `run --stage trial` reruns the whole matrix from scratch for 1 seed in a separate workspace (`artifacts/_trial`, its own ledger), compares it with the existing results and writes `g4_trial.md`; `run --stage full` raises `PermissionError` without `--approve-g4`. `run_experiment.py` is the orchestration layer: `plan` (done / partial / todo), `execute` runs what is missing, cheapest first; one failing configuration does not stop the others. `aggregate` writes `summary.csv` (mean ± std, ddof = 0, with the median round time), 6 figures, `final_report.md` and (Phase 12) `interpretation.json` and the generated blocks of the README.

---

## Phase 12 — How the results are decided (interpretation rules)

**Questions and how they are answered:**

| Question | Measure | Concluded "it works" when |
|---|---|---|
| R1. Do the generated data improve the IDS? | ΔmacroF1 (TAug − B0), Δrecall of the rare classes | The paired-bootstrap CI of Δ excludes 0 **and** \|Δ\| > the std across seeds |
| R2. Is the CVAE better than the simple methods? | TAug(B2) against B1a, B1b | As above |
| R3. How much does non-IID FL lose? | B2 − B3 | Report the loss; no pass/fail threshold |
| R4. What does DP cost and what does it bring? | B3 − M1(ε); MIA AUC, DCR ratio, duplicates | A reasonably monotone utility–privacy curve; MIA AUC approaches 0.5 as ε falls (after the positive control) |
| R5. What does SecAgg cost? | \|M2 − B3\| in utility; overhead | The utility difference is within the seed noise; the overhead is quantified |
| R6. Recommended configuration? | All | By the rule below |

**v1.3 (definitions of the points the spec left open):**
- **R1, R2:** Δ = the mean over seeds of the macro-F1 (or recall) difference between two configurations with the same seed; CI = the 95 % percentile interval of that mean over `eval.bootstrap` resamples **shared by every run** (resampling the test rows stratified by class); "std across seeds" = the max of the std (ddof = 0) of the two configurations; "it works" when both conditions hold and the sign of Δ is on the same side as the CI. Mandatory sensitivity check: repeat with a **stream bootstrap** (whole streams drawn), marked † when the conclusion changes.
- **R3:** only report the loss B2 − B3. **R4:** compare DP with `B3-plain` (same decoder, the ε = ∞ point); "reasonably monotone curve" = a step to a larger ε does not get worse by more than the larger std of the two points. **R5:** "within the noise" = the conclusion "does not work" under the R1 rule.
- **R6:** B2 pools the real data, so it is **not eligible** (it stays in the table); the configurations "not worse than the best one under the R1 rule" count as **tied**, and ties are broken by the formal protection level (a finite ε, the smaller the stronger; then SecAgg). Also report the branch "DP guarantee required" (dropping only the overhead condition) and the ranking by TSTR next to TAug.
- **Red flags:** F1 (B0 ≥ `f1_near_one`), F2 (TSTR beats B0 under the R1 rule; if raised, investigate by comparing with the class-balanced real reference B1b, the class-balance signal in the recall, and that the test set was not used for fitting/tuning), F3 (duplicates > `dup_rate_max` or DCR ratio < `dcr_ratio_min`), F4 (std > `seed_std_redflag`; rows inside and outside the matrix are told apart), F5 (the reported ε differs from the independently recomputed ε by more than `eps_recompute_rtol`).
- **Report section 9 is generated from `interpretation.json`; no number is written by hand.** The report must have: (9.9) the justification for using the CVAE when R1, R2 are negative: the CVAE is the **design premise** of the topic, not the result of a generator comparison; R1, R2 test that premise; if it is not supported, give the negative numbers, the remaining context (raw data cannot be pooled, only TSTR remains) and **the alternatives not yet tested** (training the classifier by FL, federated class weights/SMOTE, other generators); never claim that the CVAE helps the IDS when the results do not say so. (9.10) a guide to **which configuration for which requirement** (M1, M2, M3): because R6 picks only one configuration while TAug does not separate the candidates, the requirement → configuration table sets the measured costs (time, bytes, TSTR, rare-class recall) next to what each configuration protects, with the measured particulars of IoT/MQTT data (rare classes thin at every client, the DP cost falling on the rare classes, the capture unit versus the record, constrained devices); claim no privacy benefit while the MIA is at chance and has not passed the positive control.

**Recommendation rule (transparent, thresholds set in the config):** (🔶 the thresholds are heuristics, read from `thresholds`) among the configurations meeting `MIA_AUC ≤ mia_auc_max` **and** `ε_max ≤ 5` (if DP is used) **and** time-per-round overhead ≤ `overhead_ratio_max` × plain FL, choose the one with the highest TAug macro-F1; if none qualifies, report the Pareto front and state the trade-off clearly. State clearly that SecAgg protects each client's update from the server, while DP protects against leakage from the model/generated data.

**Red flags (must be investigated, must not be reported as success):**
- macro-F1 ≈ 1.00 in every configuration including B0 → suspect leakage/artifact (back to Phase 5).
- TSTR clearly beats TRTR → suspect the generated data "leak" the label or the test set was used by mistake.
- Duplicates > 1 % or DCR ratio ≪ 1 → the model copies the data.
- Large swings across seeds (std > 0.05) → more seeds are needed or the stability must be checked.
- The reported ε is far from the independently recomputed ε.

---

## Phase 13 — Demo and packaging

- `demo/app.py` (Streamlit, run on the personal machine; Gradio may replace it when running on Colab): (1) overview of the data and the class distribution; (2) result tables/figures from `results/summary.csv`; (3) the utility–privacy–overhead chart and the recommendation; (4) sampling: choose a configuration + class + count → see the table of generated samples, download a CSV; (5) a page explaining the threat model and the limitations.
- `README.md`: installation, how to run every Phase, how to reproduce (`python -m ppfeddata.cli ...`), folder layout, library versions, reference running times.
- Save `pip freeze`, `configs/`, `split_manifest.json`, `feature_schema.json` with the results for reproduction.
- **v1.3:**
  - **Demo:** the logic lives in `demo_lib.py` (testable without Streamlit), `demo/app.py` only builds the interface (in Vietnamese). Page 3 adds "which configuration for which requirement" (section 9.10). Page 4 samples with exactly the `models.generate.generate` of the evaluation (same model, seed and count give bit-identical output to the stored `synthetic.npz`), brought back to the original units with `Preprocessor.inverse_transform`; **DP configurations generate with the plain decoder only**. No number is typed by hand: every number is read from `results/`.
  - **Two READMEs:** `README.md` (English, the GitHub default) and `README.vi.md` (Vietnamese), cross-linked, with the same structure and commands. The blocks `<!-- BEGIN GENERATED: results|libraries|times|limitations -->` are rewritten by `aggregate` from `interpretation.json`, the run ledger and the trial session (the `results` block is written in the language of each file since 2026-10-21; the other blocks are in English, identical in both files); they are not edited by hand. A test compares the READMEs with what the generator writes from the committed `interpretation.json`.
  - **Limitations** are written once in `limitations.py` (every item of the Definition of Done and a few additions), shared by section 10 of the report, the README and page 5 of the demo; the numbers come from the manifest, the schema, the config and `interpretation.json`, never typed by hand.
  - `ppfeddata package` writes `results/repro/` (`pip_freeze.txt`, a copy of `configs/` without `local.yaml`, `environment.json`, `MANIFEST.json` with SHA-256) and copies `feature_schema.json` to `results/manifests/feature_schema_<mode>.json`; `requirements-lock.txt` has the same content as `pip_freeze.txt`.
  - `ppfeddata accept` checks every Definition of Done box against the files and writes `results/reports/dod_checklist.md` (PASS / PARTIAL / MANUAL / FAIL; a missing file is FAIL, never PASS).

## Phase 14 — Optimisation round of M1, M2, M3 (O0-O5) (**v1.4**)

Goal (the user, 2026-10-07): optimise M1, M2, M3 so that the privacy-preserving data generation framework strengthens IoT/MQTT attack detection **on every metric at once**, and is "flexible" in the sense that **a configuration can be chosen by deployment requirement**. Allowed: re-tuning, better DP techniques, adding or changing the generator, changing the evaluation protocol. Budget: a few machine-days. One stage per turn, stopping at each Gate.

**Common rules (every stage O0-O5):**
- The real test set stays locked: it is read only for the final evaluation of a settled configuration; every choice (hyper-parameters, classifier, rounds, variants) uses **val**. Tuning on val is not private, as in Phases 7 and 9 (limitation recorded).
- **Scorecard** (`ppfeddata scorecard`, `scorecard.py`): for every protected configuration (B3, M1, M2, M3 and the new variants), utility on the real test set: 6-class TSTR macro-F1, binary F1, mean recall of the rare classes, the TAug gain (macro-F1 of real + generated minus B0 with the same classifier); the classifier (RF or MLP) of every protocol is **chosen by val macro-F1**. Privacy: the achieved ε (max over clients), with or without SecAgg, the AUC of the MIA with model access (calibrated). Cost: bytes per round, FL time.
- **Validity:** a row counts only when its number matches its name. A DP configuration with a component computed from the train data outside ε (for example the `residual_std` of the residual noise) is listed but is **not valid**: it does not enter the Pareto front and is not recommended.
- **Pareto front:** a dominates b when a is no worse than b on every axis and better on at least one. Utility: a difference counts only when it exceeds the seed std of both (inside that range they are equal); ε is compared with a 2 % tolerance, cost with 5 %; with SecAgg beats without. The classifier trained directly by FL (`fed_classifier.json`, key `protected`) is a **reference**: it releases no data, so it does not enter the Pareto front.
- **Baseline:** `scorecard --freeze-baseline` runs once at O0 (`results/scorecard_baseline.json`, never overwritten); the later stages are compared with it by the same seed rule. The final conclusion (O5) uses the full R1/R2 rule (paired bootstrap + seed std).
- Method changes are written in `SPEC_DEVIATIONS.md` (section O); the spec only holds the rules.

**Stages:**
- **O0** — scorecard, Pareto front, baseline. **Gate O0:** the scorecard is generated from `results/`, tests green, the user approves.
- **O1** — DP improvements to the CVAE: (a) the per-class `residual_std` computed **with DP** (Gaussian mechanism; this share of ε is added to the total ε by the accountant) so that the noisy variant becomes valid; (b) DP tuning **at full scale** (all clients, no 20 % proxy) **separately for every ε**, adding the number of rounds, the local epochs, the sampling rate, and widening the parameters that hit a bound; (c) class balancing at the client. Not done: keeping the encoder at the client so that only the decoder is under DP (an encoder without DP leaks data into the decoder's gradient, and ε no longer holds).
- **O2** — two M3 variants, both reported: **M3-local** (every client adds the full noise itself, as now) and **M3-distributed** (distributed DP through SecAgg: every client adds a share of the noise, the sum after SecAgg reaches the target ε when at least t clients are honest; also report ε when only one client is honest). Bandwidth optimisation of M2/M3 (number of quantisation bits).
- **O3** — a second generator: synthesis from DP marginals in a federated style (AIM/MST; noisy marginal count tables can be summed through SecAgg); DP-CTGAN if time allows. Check the novelty against the existing work.
- **O4** — `ppfeddata recommend` (and a demo page): the input is the deployment requirement (trust in the server, ε budget, bandwidth and compute of the MQTT broker/gateway, attack classes that must be caught), the output is a configuration on the Pareto front with its measurements.
- **O5** — rerun the best configuration with 3 seeds, update the final report, both READMEs, `accept`.

**v1.5 — change of direction after Gate O1 (approved by the user 2026-10-08)** and the new rules of O2-O5:
- **Main generator: FedDP-Marginal** (`models/marginal.py`). Three releases, each the sum over the clients of noisy count tables: (1) fine 1-way marginals per class; (2) every 2-way pair on coarse bins (coarse bins = quantiles of (1), post-processing) → a Chow-Liu tree from the noisy MI; (3) the tables (root, class) and (parent, child, class). One record adds 1 to one cell of every table: L1 sensitivity = m, L2 = √m with m tables. Sampling follows the tree for every class, through the same `postprocess` as the CVAE. The CVAE (M1, M2, M3, M1o, M3o) is the baseline.
- **Distributed DP:** valid only for one-shot summable releases (not for multi-step DP-SGD). **Skellam** noise (the difference of two Poissons; a sum of Skellams is a Skellam) split over K clients; RDP accounting following Agarwal, Kairouz, Liu (2021, Corollary 3.6), composed by adding the RDP, converted to (ε, δ) as in Opacus. Report ε when only h clients are honest (h = 1…K). The Gaussian variant uses zCDP. The **local** variant: every client adds the full noise.
- **Real SecAgg+:** the three releases go through Flower SecAgg+ (`fl/mg_app.py`): every client's weight equals `max_weight` (factor 1), `quantization_range` = 2 × `clipping_range` (integers are not rounded), the masked vector is sent as uint32; the secure sum must equal the exact sum (checked at every release); bytes are measured for real.
- **Selection on val with 3 seeds:** every choice of the generator (bins, budget split, structure, post-processing) is scored by the 3-seed mean on val; a change is kept only when it beats the reference by more than the seed spread. Post-processing of the released tables (consistency estimation, IPF, smoothing) costs no ε.
- **Not valid under record-level DP:** less noise for the rare classes only (rare-class records would be protected less than the published ε).
- **Robustness (O4):** Dirichlet α, number of clients, the 11-class mode (`robustness`). **Recommendation:** `recommend` filters the valid scorecard rows by the requirement (server trust, number of honest clients, ε, MB per round, total MB, number of rounds, priority, minimum rare-class recall, maximum MIA), and takes the Pareto front on the deployment axes.
- **Report:** sections 9.9b (why the generator was changed, the comparison with the CVAE, novelty against FLAIM/MST/AIM) and 9.10b (configuration by requirement) are generated from JSON; new limitations in `limitations.py`.

**Addition to v1.5 — the next round P1-P2 (approved by the user 2026-10-08):**
- **Group-level ε (optional, `privacy-units`):** the unit of protection is the TCP stream or the capture. Two conditions: every unit is held entirely by one client (Dirichlet partition by unit, `group_dp.py`) and every client uses at most m rows of each unit; the sensitivity of a release of T tables: Δ1 = m·T, Δ2 = m·√T. m is chosen on val with 3 seeds. The framework's default configuration keeps record-level ε; `recommend` drops the configurations whose unit is smaller than required.
- **Threshold of honest clients t (optional):** every client adds 1/t of the calibrated variance, so the target ε holds when at least t clients add their share (the rest may collude or drop out); also report ε when every client is honest. **Default t = K.** `recommend` chooses t by the requirement (minimum number of honest clients); the membership-inference attack on the released tables is measured for every t entered in the scorecard.
- **IDS side (outside the generator, `ids-followup`):** class-probability weights chosen on val (3-seed mean) and using every generated row; reported separately, the scorecard does not change. **Local augmentation:** every client adds generated rows for the classes it holds with fewer than 5,000 rows.
- **Report:** section 9.11 is generated from `results/privacy_units.json`, `results/ids_followup.json`, `results/recommend.json`.

**Addition to v1.5 — round Q1 (approved by the user 2026-10-12):** FedDP-Marginal without DP (`mg-nodp`, `mg_nodp.py`).
- **Two configurations:** `MGs-nodp` (the three releases through Flower SecAgg+, without noise; the counterpart of M2) and `MGp-nodp` (the same three releases through plain Flower rounds, without SecAgg; the counterpart of B3). This is the framework's ε = ∞ reference, not the default configuration. The coarse bins are chosen on val with 3 seeds; no post-processing (there is no noise to reconcile).
- **Mandatory check:** the two configurations must release the same tables and the same generated data at every seed (the secure sum equals the exact sum); only the cost (bytes, seconds, measured with the same counters on the Flower grid) may differ.
- **Measured:** the price of DP for FedDP-Marginal against `MGs-nodp` under the R1 rule (paired row bootstrap, rechecked by stream, seed std; both RF and MLP); the membership-inference attack on the released tables is also run for the configuration without DP.
- **Report:** not entered in the R1-R6 matrix (the spec's matrix is kept; two passes over the same data would be counted twice); present in the scorecard, `recommend` (new scenario `hidden-tables`: the server is not trusted, no ε needed) and section 9.12 generated from `results/mg_nodp.json`.

**Addition to v1.5 — raw-data provenance (approved by the user 2026-10-13):** command `check-extraction` (`data/extraction_check.py`); the script that extracted Normal is kept unchanged in `tools/extract_normal_pcap.py`.
- **Source:** the authors' release on Kaggle (`alaaalatram/dosddos-mqtt-iot`). The attack rows are the authors' CSV, used as they are. The release has Normal only as PCAP, so the team extracted it with TShark, keeping every packet, with the same 33 columns; those files are the `normal_csv_dir` of Phase 1.
- **Checks (only counts, shares, scores and hashes are written; no data rows):** (0) every file on the machine matches the path and size of the release; (1) rerun the script's own extraction function on the attack PCAP and compare every column with the authors' CSV after the parsers of `preprocess.py`; (2) the Normal CSV has frame numbers 1..N, each exactly once, N = the row count recorded at extraction; (3) N = the packet count of the PCAP (`capinfos`), and re-extracting a few captures gives byte-identical files; (4) add the authors' export settings (two-pass analysis `-2`, display filter `-Y`) and check that the frame numbers match the authors' CSV; (5) measure the effect: rescore the stored predictions of every run on the test rows sent by clients, and retrain the RF without the rows sent by the broker.
- **Decision (2026-10-13):** keep the existing Normal CSV and state the difference in the limitations list with the measurements of (4)-(5). **Replaced by round R (2026-10-15):** re-extract Normal with the authors' settings and rerun the whole study.

**Addition to v1.5 — round R (approved by the user 2026-10-15 and 2026-10-16):** the whole study rerun on the Normal CSV extracted with the authors' settings (details: `SPEC_DEVIATIONS.md` Phase 18).
- **Data:** `extract-normal` (`data/normal_extract.py`) runs `tools/extract_normal_pcap.py` with `-2 -Y ip.dst==<broker>` into `NormalData/CSV_Authors_Settings` (= `normal_csv_dir`); check with `check-extraction --stages normal_authors`. `expected_counts`: NORMAL 24,955,877, total 39,000,601. SYN_DDoS-AD_1 is used as released (the authors kept both directions).
- **FedDP-Marginal:** the O3.1 grid is chosen by the 3-seed mean on val; ε 0.5 / 2 / 20 added (MG only; the CVAE keeps 1 / 5 / 10). **The framework's main configuration at every ε** = MGr if the post-processing search on val keeps it at that ε, otherwise MGs (`tune_marginal.framework_name`); this rule replaces v1.5's "MGr at ε 1".
- **Extensions:** A2 (train quota of the four rare classes ×6 and ×0.3: NORMAL : WILL = 10 : 1 and 200 : 1) is a world of `sensitivity`; every world adds the framework's main configuration at ε 5; A3 for the CVAE (`b3 --label-mode 11class`, in `robustness.md`); federated SMOTE (`fed-baseline --only FedMLPsm --merge`: every client runs SMOTE on its own data, then FedAvg). AIM is not run (Private-PGM is not installed).
- **Configuration names** are read from the search files (`configs/best_*.yaml`); trial numbers are not hard-coded.

---

## Final acceptance checklist (Definition of Done)

- [ ] `pytest -q` green; every Phase has tests.
- [ ] The total row count and the per-class counts match `expected_counts`.
- [ ] No group appears in more than one split; the real test set is locked and not used for tuning.
- [ ] `leakage_report.md` covers C1–C5; the G1, G2 decisions are stored.
- [ ] B0, B1, B2, B3, M1×3, M2, M3 all have 3 seeds in `runs.csv`.
- [ ] The achieved DP ε is reported and matches the target; T-SA1 passes; the MIA has a positive control (**v1.3:** if the control fails with the CVAE, this box is a recorded FAIL with an explanation, and the report does not use the MIA as evidence of a DP benefit).
- [ ] `final_report.md` is generated automatically, every number has a source in `results/`.
- [ ] `compute_budget.md` has real measurements and records every cut (if any).
- [ ] The limitations list covers: packet-level data; **record-level ε, the packets of a stream and of a capture are correlated**; the time features depend on the export and the CVAE generates packets independently; non-private tuning; one-machine simulation (no network latency measured); standardisation uses centralised statistics; the labels are not protected by DP; the test set comes from few groups (see A4 if run); the evaluation thresholds are heuristics settled by the user; results on one dataset; the attack classes are hard to separate at the packet level (B0 ≈ 0.45-0.50), so a CVAE that generates independent packets cannot create separability that the features do not already hold; 8/11 subclasses have only one capture file, so their val/test are consecutive blocks of the same capture, and dropping the streams that span several blocks removes 1.2 % to 9.9 % of the eligible rows of each subclass (biased towards short connections); the filter `protocol ∈ {TCP, MQTT}` removes 0.50 % of the Normal rows and 128 RIPv2 rows of Attack; only the first element of a multi-value cell is used (the information of several messages in one packet is lost, partly kept in `tcp_segment_len`); the tail of `time_delta...` is still clipped at 5σ on 0.08 % of the train rows after the ×1000 scaling (1.04 % without it); ultra-sparse columns are handled differently from v1.1; the DoS/DDoS subclass label is merged in the 6-class mode; (**v1.5, data provenance; fixed in round R**) the Normal CSV was extracted by the team from the PCAP with the authors' export settings (display filter, two-pass analysis), the capture SYN_DDoS-AD_1 was exported by the authors in both directions, and the first study used the every-packet extraction.
- [ ] The README is enough for someone else to rerun everything from scratch (**v1.3:** both READMEs; the real test is an install on a clean machine/venv).

---

## Appendix — Opening prompt when the steps are handed to an AI agent

Use it only when the steps are handed to an AI agent; someone running them by hand skips this appendix and follows section 0 and the Phases.

> You are an ML engineer. Read `PP-FedData_Implementation_Spec.md`. Carry out **one Phase per turn**, in order, starting from Phase 0. For every Phase: (1) list a short plan, (2) write the code and the tests, (3) run the tests and the Gate checks, (4) summarise the results with the real numbers and the files created. Stop at every HUMAN GATE (G1–G4) and wait for confirmation. Do not modify the raw data. Do not invent numbers. If blocked, write `BLOCKERS.md` and stop.

---

## Change history

- **Documents (2026-10-21):** the spec and the log have an English (`.md`) and a Vietnamese (`.vi.md`) version, like the README; the wording is neutral about the implementer (running it by hand or with an AI agent); section 3 and Phase 1 give the round-R `normal_csv_dir` and `expected_counts` (until now only in the round-R block of Phase 14); the README `results` block is written in the language of each file; the rules do not change.
- **v1.5, round R (2026-10-15/16):** Normal re-extracted with the authors' settings and everything rerun; the MG grid with 3 seeds, ε 0.5 / 2 / 20 for MG, the main configuration chosen on val at every ε; A2, A3 for the CVAE, federated SMOTE; new command `extract-normal`.
- **v1.5, data-provenance addition (2026-10-13):** verification of the raw-data provenance (files against the Kaggle release, the Normal CSV extracted from the PCAP, the procedure against the authors' CSV, reproduction of the authors' export settings, effect on the results); new command `check-extraction`, folder `tools/`; one limitation added to the Definition of Done.
- **v1.5, round Q1 addition (2026-10-12):** FedDP-Marginal without DP, through SecAgg+ and through plain Flower rounds (the framework's ε = ∞ reference), section 9.12, the `hidden-tables` scenario of `recommend`; new command `mg-nodp`.
- **v1.5, addition (2026-10-08):** the next round P1-P2: group-level ε (stream, capture), the threshold of honest clients t (default t = K, chosen by the requirement in `recommend`), IDS-side options, local augmentation, section 9.11; new commands `privacy-units`, `ids-followup`.
- **v1.5 (2026-10-08):** the main generator moved to FedDP-Marginal after Gate O1 (the re-tuned CVAE did not beat the baseline); rules added for distributed DP (Skellam, real SecAgg+, ε by number of honest clients), selection on val with 3 seeds, robustness, the extended `recommend`, sections 9.9b and 9.10b; new commands `tune-marginal`, `mia-marginal`, `robustness` (with `taug-rare`, `tune-fedsgd`, `secagg-bits` written in O1-O2).
- **v1.4 (2026-10-07):** Phase 14 added (optimisation round of M1, M2, M3: O0-O5) following the user's goal: a multi-metric scorecard with the classifier chosen on val, the validity rule (a component outside ε makes a DP row invalid), the Pareto front under the seed rule, a frozen baseline, the O1-O5 plan, the `scorecard` command.
- **v1.3 (2026-10-05):** folded in the changes learnt while doing Phases 6-13 (details and evidence: `SPEC_DEVIATIONS.md` sections 6-13, kept as the log). Configuration: `patience`, `generate.residual_noise`, `secagg.clipping_range` 16 and `max_weight`, `eval.smote`, `compute.runs_csv`, 4 Phase 12 thresholds. Phase 6: B1a RF only, `run_id` includes the classifier, run ledger from Phase 6. Phase 7: per-class residual noise, dead categories, tuning through the final generation path, C2ST not blocking. Phase 8: how Flower 1.39 is run (Ray, aggregation in client-id order, stop on a missing reply). Phase 9: Opacus step count, the `plain` decoder as the main number, separate DP tuning (`tune-dp`, `verify-dp`), ε recomputed independently. Phase 10: `SecAggPlusWorkflow` used unchanged (`secagg_sim.py` not needed), `max_weight`, `clipping_range`, T-SA1-3 on the real data. Phase 11: the main DP family, the reference/extension groups, G4 encoded in the tools. Phase 12: definitions of R1-R6 and the red flags, the stream bootstrap, sections 9.9 (why the CVAE) and 9.10 (which configuration for which requirement). Phase 13: demo, two READMEs, `limitations.py`, `package`, `accept`. Corrected: the statement "packets of the same stream are strongly correlated" only holds at the capture level (the sample has ~1.0-1.2 rows per stream). `SPEC_DEVIATIONS.md` has a note at the top saying which rows were folded in.
- **v1.2 (2026-10-03):** folded in the changes learnt while doing Phases 0-5 (details and evidence: `SPEC_DEVIATIONS.md`). Phase 2: token audit, duplicated column pairs, header; text label ↔ numeric code mapping; SUSPECT rules generated from the numbers (parent event, count thresholds); `first_only` instead of first+count/sum and `n_mqtt_msgs` dropped; the `protocol` filter; the user's decisions stored in the config (`user_overrides`, `g1_log`, `g2_log`). Phase 3: row indices chosen in advance instead of a reservoir, `interim/<mode>/`, `#blkNN` groups, streams spanning several blocks dropped, a manifest with versions and a committable copy. Phase 4: ultra-sparse columns, block layout, `diagnostic` columns, `processed/<mode>/`. Phase 5: class-balanced C3 + per-class F1, C4 adds the stream split and flags only by group/stream, ablations; 3 new thresholds; the G1, G2 decisions; `configs/local.yaml` and `config_hash` without `paths`; additional limitations. After G2: `numeric_scale` for `time_delta`, `requirements.txt` drops the `numpy<2.0` constraint (checked by a dry-run install of `torch`, `flwr`, `opacus`: no conflict), `.md` reports and `.png` figures in `results/` may be committed.
- **v1.1:** added sections 1.1 (Colab/Drive, surviving session loss) and 1.2 (compute budget); added the 7.4 benchmark step; the 11-class quota table and no separate `binary` mode; the time-feature check (Phase 2); `stream_id` kept and the `max_rows_per_stream` option; the per-client ε reporting rule (max_i); checkpoint/resume for FL and DP; the 🔶 label and the `thresholds` block in the config; sensitivity analyses A4 (change the test groups) and A5; Definition of Done updated.
- **v1.0:** first version.
