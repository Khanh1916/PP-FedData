# PP-FedData — Đặc tả triển khai cho coding agent (v1.2)

> Đây là hợp đồng công việc cho agent lập trình. Đọc toàn bộ trước khi viết code. Làm **tuần tự theo Phase**; mỗi Phase có *Deliverables* và *Gate* (điều kiện nghiệm thu). Không sang Phase sau khi Gate chưa đạt. v1.2 gộp các thay đổi rút ra khi thực hiện Phase 0-5 (bằng chứng chi tiết ở `SPEC_DEVIATIONS.md`); các mục 🔶 và quyết định G1, G2 đã được người dùng chốt ghi rõ trong từng Phase.

**Đề tài:** Khung sinh dữ liệu bảo toàn quyền riêng tư trong học liên kết (FL) để tăng cường phát hiện tấn công MQTT DoS/DDoS trên IoT.
**Ý tưởng:** CVAE có điều kiện nhãn, huấn luyện bằng FL trên dữ liệu non-IID; so sánh ba cơ chế bảo vệ: **DP** (DP-SGD tại client), **SecAgg** (Flower), và **DP + SecAgg**; dữ liệu sinh ra dùng để cân bằng lớp cho bộ phân loại IDS (Random Forest, MLP).
**Mô hình đe dọa:** server honest-but-curious; client trung thực, không thông đồng.

---

## 0. Quy tắc vận hành cho agent

1. Làm theo thứ tự Phase. Có **4 HUMAN GATE (🛑)**: dừng lại, ghi báo cáo ngắn, chờ người dùng xác nhận.
2. **Không sửa/xóa dữ liệu gốc** (CSV, PCAP). Chỉ đọc.
3. Không hard-code đường dẫn, seed, siêu tham số: tất cả nằm trong `configs/*.yaml`.
4. Seed cố định `{0,1,2}` cho python/numpy/torch/sklearn. Mỗi kết quả ghi kèm `seed`, `config_hash`, `git_commit` (nếu có).
5. **Không bịa số liệu.** Mọi con số trong báo cáo phải đọc từ file kết quả. Kết quả xấu hoặc ngược kỳ vọng vẫn báo cáo nguyên trạng.
6. Mỗi module có unit test; `pytest -q` phải xanh trước khi qua Phase mới.
7. Trước khi dùng API của **Flower** và **Opacus**, chạy `pip show <pkg>` và đọc tài liệu đúng phiên bản đã cài. Không dựa vào trí nhớ. Ghim phiên bản trong `requirements.txt`.
8. Dữ liệu thô rất lớn (59,6 triệu dòng, khoảng 13 GB CSV): **luôn đọc theo chunk/streaming**, không nạp toàn bộ vào RAM.
9. Đường dẫn Windows có dấu cách: dùng `pathlib.Path`, không ghép chuỗi. Với multiprocessing trên Windows, đặt code chính dưới `if __name__ == "__main__":`.
10. Sau 2 lần sửa lỗi không thành công: ghi `BLOCKERS.md` (lỗi, lệnh, log, giả thuyết) rồi dừng, không làm tiếp theo hướng đoán.
11. **Nhãn 🔶 = CẦN XÁC MINH.** Gắn cho (a) ngưỡng/số liệu là heuristic do người soạn đặt (người dùng chốt, nằm ở `thresholds` trong config), (b) API hoặc hành vi thư viện chưa được kiểm chứng ở phiên bản cài. Với mục 🔶, agent phải kiểm tra/đo thực tế và ghi kết quả vào báo cáo, không coi là đã đúng.
12. Mọi bước chạy lâu (huấn luyện FL, Optuna, ma trận thí nghiệm) phải **ghi checkpoint và resume được** (xem mục 1.1).

---

## 1. Môi trường

Có hai giai đoạn chạy khác nhau:

| Giai đoạn | Nơi chạy | Việc |
|---|---|---|
| **A. Chuẩn bị dữ liệu** (Phase 1–5) | Máy cá nhân Windows (dữ liệu gốc nằm ở đây) | Thống kê, chia nhóm, lấy mẫu, tiền xử lý → xuất Parquet nhỏ (vài chục MB) |
| **B. Thực nghiệm** (Phase 6–13) | Colab CPU hoặc máy cá nhân | CVAE, FL, DP, SecAgg, đánh giá |

Thư viện (ghim phiên bản sau khi cài): `python>=3.10`, `numpy`, `pandas`, `pyarrow`, `polars` (tuỳ chọn, để đọc nhanh), `scikit-learn`, `imbalanced-learn`, `torch` (CPU), `flwr`, `opacus`, `optuna`, `scipy`, `matplotlib`, `psutil`, `pyyaml`, `joblib`, `pytest`, `streamlit` (demo).

### 1.1 Chuyển dữ liệu lên Colab và chống mất phiên
- Chỉ đưa lên Colab **dữ liệu đã xử lý** (`processed/*.npz`, `feature_schema.json`, `split_manifest.json`, `partitions/`), dung lượng nhỏ (ước vài chục MB 🔶). Không đưa dữ liệu thô.
- Cách chuyển: copy lên Google Drive, trên Colab mount Drive. Đặt `compute.artifacts_dir` trỏ vào Drive; **mọi** checkpoint, SQLite của Optuna và `results/runs.csv` ghi vào Drive, không ghi vào đĩa tạm của Colab (mất khi hết phiên).
- Mỗi phiên Colab bắt đầu bằng một cell setup duy nhất: cài `requirements.txt`, mount Drive, `pip install -e .`, in phiên bản thư viện.
- Mọi tiến trình dài phải resume được: FL lưu checkpoint mỗi `checkpoint_every_rounds` vòng; Optuna dùng SQLite trên Drive; `run_experiment.py` bỏ qua run đã hoàn tất.

### 1.2 Ngân sách tính toán (🔶 phải đo, không dựa vào ước lượng)
- Spec không cam kết thời gian chạy. Agent đo ở bước **7.4** rồi lập `results/reports/compute_budget.md`.
- Lưu ý chung 🔶: DP-SGD (gradient từng mẫu) thường chậm hơn huấn luyện thường nhiều lần; mô phỏng FL chạy client tuần tự trên một máy, nên thời gian một vòng ≈ tổng thời gian các client.

---

## 2. Cấu trúc repo

```
ppfeddata/
├── configs/
│   ├── default.yaml            # xem mục 3
│   ├── local.yaml              # riêng từng máy (đường dẫn dữ liệu thô), KHÔNG commit, merge sâu lên default.yaml
│   ├── label_map.yaml          # ánh xạ thư mục/file → lớp
│   ├── feature_decisions.yaml  # sinh ở Phase 2 từ số liệu + quyết định người dùng (G1, G2)
│   └── exp/                    # 1 file / cấu hình thí nghiệm (B0..M3)
├── src/ppfeddata/
│   ├── data/      inventory.py, harmonize.py, split_sample.py, parse_multi.py, preprocess.py, partition.py
│   ├── models/    cvae.py, generate.py
│   ├── fl/        client_app.py, server_app.py, dp_utils.py, secagg_sim.py (fallback)
│   ├── eval/      utility.py, fidelity.py, privacy.py, overhead.py, stats.py
│   ├── checks/    leakage.py, leakage_report.py
│   ├── tune.py, run_experiment.py, aggregate.py, cli.py
├── tests/
├── notebooks/     00_eda.ipynb ... (chỉ để xem, không chứa logic)
├── data/          inventory/, interim/<mode>/, processed/<mode>/, partitions/   (KHÔNG commit; <mode> = 6class | 11class)
├── artifacts/     {run_id}/ model.pt, synthetic.parquet, preds/
├── results/       runs.csv, summary.csv, figures/, reports/, manifests/   (manifests/split_manifest_<mode>.json được commit: chỉ có mã nhóm, số lượng, hash, phiên bản thư viện)
├── demo/          app.py
├── requirements.txt, README.md, BLOCKERS.md (nếu có), SPEC_DEVIATIONS.md (nhật ký bằng chứng cho các thay đổi spec)
```

CLI thống nhất: `python -m ppfeddata.cli {inventory|harmonize|sample|preprocess|check|baseline|tune|run|aggregate|demo} --config ...`.

---

## 3. Cấu hình mặc định (`configs/default.yaml`)

```yaml
seeds: [0, 1, 2]
paths:
  raw_root: "./raw/DoS-DDoS-MQTT-IoT_Dataset"   # đường dẫn thật đặt trong configs/local.yaml (không commit) hoặc biến môi trường PPFEDDATA_RAW_ROOT; trong dấu nháy
  normal_csv_dir: "NormalData/CSV_Extracted_Split"
  work_dir: "./data"
  shared_manifest_dir: "./results/manifests"   # bản sao split_manifest (không có dòng dữ liệu) để commit
label_mode: "6class"            # "6class" | "11class"  (chốt ở G2). Nhị phân Normal/Attack chỉ là bảng phụ suy ra từ dự đoán đa lớp
split:
  test_groups_per_subclass: 1
  val_groups_per_subclass: 1
  fallback_block_split: {n_blocks: 10, gap_rows: 1000}
  split_seed: 0                  # quyết định NHÓM/KHỐI nào vào val/test và dòng nào được lấy mẫu (A4 đổi giá trị này)
  min_groups_for_group_split: 3  # ít nhóm hơn -> block-split
  min_eval_rows: 1000            # val/test của một lớp dưới mức này -> dừng và hỏi người dùng
  max_peak_gb: 4.0               # Gate Phase 3: bộ nhớ đỉnh
  drop_cross_block_streams: true # block-split: bỏ các TCP stream vắt qua nhiều khối (khoảng đệm không tách được chúng)
quota:                          # [train, val, test] số dòng mỗi lớp (6 lớp)
  NORMAL:    [60000, 2000, 5000]
  BCF:       [20000, 2000, 5000]   # Basic Connect Flooding
  DELAYED:   [3000,  2000, 5000]
  SYN:       [3000,  2000, 5000]
  INVALID:   [1500,  2000, 5000]
  WILL:      [1000,  2000, 5000]
quota_11class:                  # [train, val, test]; dùng khi label_mode = 11class
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
  max_rows_per_stream: null     # null = tắt. Số nguyên (ví dụ 20) = giới hạn số packet mỗi stream_id trong train pool (xem Phase 3, Phase 9)
preprocess:
  multi_value: {message_type: first, numeric_multi: first}   # v1.2: export Normal chỉ giữ phần tử đầu (occurrence=f) nên chỉ phần tử đầu so sánh được
  categorical_top_k: 10
  log1p_skew_threshold: 2.0
  clip_sigma: 5
  na_flag_range: [0.005, 0.995]  # thêm cờ <cột>_is_na khi tỉ lệ trống trong train nằm TRONG khoảng này
  ultra_sparse_fix: true         # trống >= cận trên (nhưng không phải 100%): thống kê trên dòng áp dụng + có cờ (false = đúng quy tắc v1.1)
  numeric_scale: {time_delta_from_previous_displayed_frame: 1000}   # nhân trước log1p (giây -> ms): với cột thời gian < 1 s, log1p gần như đồng nhất nên 1,04% dòng bị cắt ở 5σ, sau khi nhân còn 0,08%
cvae: {latent_dim: 16, hidden: [128, 64], beta: 0.5, beta_warmup_epochs: 10,
       lr: 1.0e-3, batch_size: 256, epochs: 50, class_balanced_sampler: false}
fl: {num_clients: 5, rounds: 30, local_epochs: 2, dirichlet_alpha: 0.5, min_client_size: 500}
dp: {epsilons: [1, 5, 10], delta: 1.0e-5, max_grad_norm: 1.0}
secagg: {num_shares: 5, reconstruction_threshold: 3, clipping_range: 8.0}
generate: {target_per_class: 20000}
eval:
  rf: {n_estimators: 200}
  mlp: {hidden: [128, 64], max_iter: 100, early_stopping: true}
  bootstrap: 1000
tune: {n_trials: 30, syn_per_class: 5000, rf_trees: 100}
thresholds:                     # 🔶 heuristic khởi điểm — người dùng chốt, không phải chuẩn khoa học
  c2st_auc_max: 0.95            # dữ liệu sinh không được hiển nhiên là giả
  dup_rate_max: 0.01
  dcr_ratio_min: 0.5
  mia_auc_max: 0.55
  overhead_ratio_max: 3.0
  seed_std_max: 0.02
  leakage_gap_flag: 0.05        # chênh macro-F1 giữa chia ngẫu nhiên và chia theo nhóm (hoặc theo stream)
  single_feature_f1_flag: 0.9   # C3: một cột riêng lẻ, macro-F1 hoặc F1 tốt nhất theo lớp vượt mức này -> soi ngữ nghĩa
  na_only_f1_flag: 0.5          # C2: RF chỉ dùng cờ _is_na vượt mức này -> mô hình học cách trích xuất
  dos_ddos_f1_flag: 0.6         # C5: F1 phân biệt DoS/DDoS dưới mức này -> không tách được (ngẫu nhiên = 0,5)
leakage: {rf_trees: 100, cv_folds: 5}
harmonize:                      # 🔶 heuristic của Phase 2; quyết định của người dùng nằm ở user_overrides / row_filters / g1_log / g2_log
  suspect_presence_diff: 0.95
  absent_presence_max: 0.001
  present_presence_min: 0.01
  time_ratio_suspect: 10.0
  token_min_share: 0.001
  token_min_count: 100
  # user_overrides, row_filters (protocol: [TCP, MQTT]), g1_log, g2_log: xem configs/default.yaml
compute:
  artifacts_dir: "./artifacts"  # Colab: "/content/drive/MyDrive/ppfeddata/artifacts"
  checkpoint_every_rounds: 5
  budget_hours: null            # 🔶 người dùng đặt sau bước 7.4
```

Số liệu tham chiếu để kiểm chứng ở Phase 1 (`expected_counts`, từ báo cáo xử lý dữ liệu): tổng 59.624.826; Normal 45.580.102; Attack 14.044.724; BCF-DoS 5.825.026; BCF-DDoS 6.455.848; WILL-DoS 114.474; WILL-DDoS 156.055; DELAYED-DoS 291.988; DELAYED-DDoS 309.554; INVALID-DoS 109.150; INVALID-DDoS 181.190; SYN-DoS 103.828; SYN-DDoS 497.611.

---

## Phase 0 — Khởi tạo dự án

**Việc:** tạo cấu trúc repo, `requirements.txt`, `pyproject.toml`/`setup.cfg` (cài editable), logging chuẩn, hàm `set_seed()`, hàm `config_hash()`, `pytest` chạy được.
**Test:** `test_seed.py` (cùng seed → cùng dãy ngẫu nhiên numpy/torch); `test_config.py` (đọc yaml, hash ổn định).
**Gate:** `pytest -q` xanh; `python -m ppfeddata.cli --help` chạy.

---

## Phase 1 — Kiểm kê dữ liệu thô (`inventory.py`)

**Việc:**
1. Quét `raw_root`: Attack theo cấu trúc `<Scenario>/<DoS|DDoS>/CSV Files/*.csv`; Normal trong `normal_csv_dir`. Xem tên thư mục/file thật rồi viết `configs/label_map.yaml` ánh xạ về 5 mã kịch bản (`BCF, WILL, DELAYED, INVALID, SYN`) + `attack_type ∈ {DoS, DDoS, NONE}`.
2. Với **mỗi file** đếm số dòng bằng chunk (không nạp hết), kích thước file, số cột; bỏ qua file rỗng (`BF1_DoS_AD_11.csv`), **không xóa**.
3. Xác định **`group_id`**: Normal dùng cột `Capture_ID`; Attack dùng tên file. Xem quy luật đặt tên: nếu các file thực chất là lát cắt liên tiếp của cùng một capture và nhận diện được capture gốc, gom theo capture gốc; nếu không, dùng file làm nhóm và ghi rõ hạn chế (các lát kề nhau có thể khá giống nhau).
4. Xuất `data/inventory/files.csv` (path, scenario, attack_type, class6, class11, group_id, rows, bytes) và `class_counts.csv`.

**Test/Gate:**
- Tổng dòng = 59.624.826 và từng lớp khớp `expected_counts` **chính xác** (assert). Lệch → dừng, điều tra.
- Không có file nào thuộc hai nhóm; mỗi file được gán đúng một lớp.
- Báo cáo số nhóm mỗi (kịch bản, attack_type). Nếu có lớp con < 3 nhóm, ghi cảnh báo (sẽ dùng fallback block-split).

---

## Phase 2 — Hài hòa schema, kiểm toán định dạng và ma trận hiện diện (`harmonize.py`)

**Việc:** (mọi thống kê tính bằng streaming trên toàn bộ dữ liệu, kết quả lưu cache; `--recompute` để tính lại, `--decisions-only` để chỉ sinh lại yaml)
1. Chuẩn hóa tên cột sang snake_case; 3 cột trùng tên gán hậu tố theo **vị trí** (`qos_level_1/_2`, `frame_length_on_wire_1/_2`, `clean_session_flag_1/_2`); ánh xạ `.1` của Normal về `_2`. Đọc header từng file: 33 cột khớp thứ tự giữa Normal và Attack (Normal thêm `Label`, `Capture_ID`), sai lệch thì dừng → `header_check.csv`.
2. **Presence matrix:** với mỗi cột × mỗi lớp con (11 lớp), tỉ lệ ô không trống. Xuất `presence_matrix.csv` và heatmap `results/figures/presence_heatmap.png`.
3. **Format audit:** với mỗi cột × nguồn (Normal vs Attack), thống kê "kiểu giá trị" bằng regex: số nguyên, số thực, hex, chứa dấu phẩy, chuỗi chữ, True/False, Set/Not set, rỗng → `format_audit.csv`. Mục tiêu: phát hiện khác biệt do **quy trình trích xuất**, không phải hành vi mạng.
4. **Kiểm toán token** (`token_audit.csv`): đếm từng giá trị (ô nhiều giá trị được tách theo dấu phẩy) của cột phân loại/nhị phân, theo nguồn. Khác biệt mã hóa đã xác nhận: (a) `message_type`, `qos_level_1/2`, `requested_qos`: Normal dùng **mã số**, Attack dùng **nhãn chữ** của Wireshark → bảng ánh xạ `VALUE_MAPS` và `canonical_token` (ô bị cắt/dính chữ khớp theo tiền tố duy nhất, không ánh xạ được thì vào `OTHER`); (b) cờ nhị phân: Normal `True/False`, Attack `Set/Not set` → cùng ánh xạ về 0/1.
5. **Cặp cột trùng** (`duplicate_pairs.csv`): `frame_length_on_wire_1/2`, `clean_session_flag_1/2` xuất từ cùng một trường TShark. Nếu `_1` bằng `_2` ở mọi dòng Attack và Normal để trống `_1` thì **bỏ `_1`, giữ `_2`**.
6. **Ô nhiều giá trị:** so tỉ lệ ô nhiều giá trị giữa hai nguồn. Normal được xuất bằng TShark `-E occurrence=f` (chỉ phần tử đầu) nên có 0% ô nhiều giá trị, Attack có khoảng 0,4% đến 1,6% tùy cột. Vì vậy `multi_policy: first_only` cho mọi cột và **không dùng `n_mqtt_msgs`** (luôn ≤ 1 ở Normal).
7. Sinh `configs/feature_decisions.yaml` với luật mặc định (mỗi cột có `action: drop|numeric|binary|categorical`, kèm lý do):
   - `drop`: `no`, `epoch_time`, `time_since_reference_or_first_frame`, `stream_index` (bỏ khỏi đặc trưng nhưng **giữ làm metadata** `stream_id = (group_id, stream_index)`), `source`, `info`, `label`, `capture_id`, bản `_1` của cột trùng (bước 5), mọi cột trống ≥ 99,9% ở mọi lớp. (`requested_qos` KHÔNG tự bỏ: quy tắc trống > 99,9% không đúng với dữ liệu thật.)
   - `numeric`: `frame_length_on_wire_2`, `time_delta_from_previous_displayed_frame`, `irtt`, `time_since_first_frame_in_this_tcp_stream`, `tcp_segment_len`, `calculated_window_size`, `keep_alive`, `user_name_length`, `password_length`, `will_message_length`, `will_topic_length`, `topic_length`, `msg_len`.
   - `binary`: `syn`, `reset`, `acknowledgment`, `clean_session_flag_2`, `retain`, `will_retain`, `will_flag`. `categorical`: `protocol`, `message_type`, `qos_level_1`, `qos_level_2`, `requested_qos`.
   - **SUSPECT (cờ sinh từ số liệu, mặc định drop, người dùng quyết):** (a) cột trống ở một nguồn nhưng có ở nguồn kia — trừ khi *sự kiện cha* có mặt ở Normal (`will_*` phụ thuộc `will_flag`, `requested_qos` phụ thuộc SUBSCRIBE; khi đó Normal trống vì không dùng chức năng, không phải lỗi trích xuất); (b) tỉ lệ trống chênh > 0,95 giữa Normal và một lớp Attack; (c) giá trị **chỉ có ở Normal** (chiếm ≥ `token_min_share` và ≥ `token_min_count`; ví dụ STP, LOOP, SSDP trong `protocol`); (d) cột thời gian lệch mạnh (bước 8). Giá trị chỉ có ở Attack (ví dụ `will_flag` = Set) là hành vi tấn công, KHÔNG bị cờ.
   - Quyết định của người dùng không bị ghi đè khi chạy lại: `harmonize.user_overrides`, `row_filters`, `g1_log`, `g2_log` trong `default.yaml`. Bộ lọc dòng được lưu trong yaml (`_row_filters`) và Phase 3 đọc từ đó.
8. **Kiểm tra tính so sánh được của đặc trưng thời gian** (`time_delta_from_previous_displayed_frame`, `irtt`, `time_since_first_frame_in_this_tcp_stream`): so các phân vị (1, 25, 50, 75, 99%) giữa hai nguồn trên **cùng loại packet** (TCP không có MQTT) → `time_feature_audit.csv`. Cột có tỉ lệ max/min của p25, p50, p75 hoặc p99 > `time_ratio_suspect` → SUSPECT. Ghi vào hạn chế: CVAE sinh từng packet độc lập nên không giữ chuỗi thời gian giữa các packet.

**Test:** `test_harmonize.py` (mapping tên cột, cột `.1`; regex định dạng trên `"Publish Message,Publish Message"`, `"40,41"`, `""`; `canonical_token` cho nhãn chữ, mã số, ô cắt cụt; luật sự kiện cha, ngưỡng đếm, cặp trùng, `user_overrides`; kiểm tra header).
**Gate 🛑 G1:** nộp `presence_matrix.csv`, `format_audit.csv`, `token_audit.csv`, `feature_decisions.yaml` kèm danh sách SUSPECT (`results/reports/g1_feature_review.md`, sinh tự động). Người dùng duyệt bộ đặc trưng.
**Quyết định G1 đã chốt (2026-10-03):** giữ `will_message_length`, `will_topic_length`, `requested_qos`; giữ `protocol` và lọc dòng (cả Normal lẫn Attack) về `protocol ∈ {TCP, MQTT}`; `first_only` và bỏ `n_mqtt_msgs`; `time_since_first_frame_in_this_tcp_stream` tạm là cột `diagnostic`.

---

## Phase 3 — Chia theo nhóm và lấy mẫu streaming (`split_sample.py`)

**Việc:**
1. **Lượt quét (pass A):** mỗi file đọc một lần, chỉ lấy cột lọc dòng, `Stream index` và (Normal) `Capture_ID` → mặt nạ hợp lệ theo `_row_filters` + chỉ số stream; kiểm tra số dòng khớp inventory và `Capture_ID` khớp `group_id`. Có cache trên đĩa.
2. **Chia:** mỗi lớp con (kịch bản × attack_type) có `test_groups_per_subclass` nhóm cho test, `val_groups_per_subclass` cho val, còn lại train; chọn theo `split_seed`, bỏ nhóm không đủ dòng cho quota (ghi lại). Normal dùng `Capture_ID`.
3. **Block-split** cho lớp con có < `min_groups_for_group_split` nhóm: chia thành `n_blocks` khối liên tiếp, bỏ `gap_rows` dòng đệm giữa các khối; mỗi khối là một "nhóm" với `group_id = <file>#blkNN`. Vì khoảng đệm không tách được TCP stream sống suốt capture, **bỏ mọi stream có dòng hợp lệ ở nhiều hơn một khối** (`drop_cross_block_streams`; ghi số stream và số dòng bị bỏ vào manifest).
4. **Lấy mẫu:** số dòng từng file đã biết nên chọn trước chỉ số dòng ngẫu nhiên không hoàn lại cho từng nhóm/khối (tương đương reservoir, một lượt đọc, tất định theo seed). Train lấy đều qua các nhóm train (chia đều, nhóm thiếu dòng thì phần dư chuyển cho nhóm khác). Chế độ nhãn (`label_mode`):
   - `6class` (mặc định): dùng `quota`; mỗi kịch bản chia đều giữa DoS và DDoS (DoS nhận dòng lẻ).
   - `11class`: dùng `quota_11class`. Cùng `split_seed` thì hai chế độ dùng **cùng nhóm/khối** test và val.
   - Nhị phân Normal/Attack không phải mode riêng; chỉ báo cáo như bảng phụ bằng cách gộp dự đoán đa lớp.
   - Nếu không đủ dòng cho quota: giảm quota, ghi cảnh báo; nếu val hoặc test của một lớp xuống dưới `min_eval_rows` thì **dừng và hỏi người dùng**.
4b. (Tuỳ chọn, `train_sampling.max_rows_per_stream`) Giới hạn số packet mỗi `stream_id` trong train pool (lý do: Phase 9, mục "Đơn vị bảo vệ"). Mặc định tắt; chạy như độ nhạy A5.
5. Xuất `data/interim/<label_mode>/{train,val,test}.parquet` (cột thô đã harmonize, dạng chuỗi, kèm `label`, `class6`, `class11`, `split`, `data_source`, `group_id`, `stream_id`, `source_file`, `row_idx`; lưu ý cột thô `source` là địa chỉ gửi) và `split_manifest.json` (nhóm/khối ở split nào, số dòng đủ điều kiện và số dòng lấy, seed, `config_hash`, `git_commit`, `git_dirty`, phiên bản python/numpy/pandas/pyarrow, sha256 từng Parquet, bộ nhớ đỉnh). Bản sao manifest được ghi vào `results/manifests/split_manifest_<mode>.json` để commit; **chỉ chạy lại để tạo manifest khi cây git sạch** (`git_dirty: false`).

**Test/Gate:**
- **Không nhóm (hoặc khối) nào và không TCP stream nào xuất hiện ở hơn một split** (assert); không dòng thô nào bị lấy hai lần; bộ lọc dòng thỏa.
- Số dòng đúng quota (hoặc có ghi chú rõ nếu giảm vì thiếu dữ liệu).
- Chạy lại với cùng seed và cùng phiên bản thư viện cho file Parquet cùng hash.
- Bộ nhớ đỉnh (psutil) < `max_peak_gb` (4 GB).
- Từ chối chạy nếu thiếu `feature_decisions.yaml` (không được âm thầm bỏ bộ lọc dòng).

---

## Phase 4 — Tiền xử lý (`parse_multi.py`, `preprocess.py`)

**Việc:**
1. `parse_multi(cell) -> list[str]`: tách ô nhiều giá trị, xử lý rỗng/`nan`/khoảng trắng. Theo `multi_policy: first_only` mọi cột dùng **phần tử đầu** (`first_values`); không sinh `n_mqtt_msgs`.
2. Lớp `Preprocessor` (fit chỉ trên **train**, lưu bằng joblib; `transform` không đổi trạng thái đã fit):
   - Cột số: điền 0 cho giá trị không áp dụng, thêm cờ `<col>_is_na` khi tỉ lệ trống trong train nằm trong (`na_flag_range`); `log1p` nếu skew > ngưỡng và không âm (tính trên cột đã điền); chuẩn hóa mean/std (std = 0 thì scale = 1); cắt ±`clip_sigma`. Cột thời gian được cắt về ≥ 0 trước `log1p` (có giá trị âm cỡ -1e-6) và đếm số dòng bị cắt. `preprocess.numeric_scale` nhân một cột với hằng số trước `log1p` (mặc định chỉ `time_delta_from_previous_displayed_frame` × 1000, vì `log1p` của giá trị < 1 s gần như là hàm đồng nhất và đuôi bị cắt ở 5σ); khoảng giá trị gốc trong schema và `inverse_transform` vẫn tính bằng đơn vị gốc.
   - **Cột siêu thưa** (trống ≥ cận trên của `na_flag_range` nhưng không phải 100%, `ultra_sparse_fix: true`): thống kê tính trên dòng áp dụng, dòng trống nằm ở 0 sau chuẩn hóa, và **có cờ**. Quy tắc v1.1 (điền 0 rồi chuẩn hóa trên cả cột) làm mọi dòng áp dụng bị cắt ở +5σ và gộp các giá trị khác nhau (`will_message_length` = 2 và 3164).
   - Cột nhị phân: 0/1 (`True/False`, `Set/Not set`, `0/1`), trống → 0 + cờ `_is_na` theo cùng quy tắc.
   - Cột phân loại: chuẩn hóa token bằng `canonical_token` (mã số và nhãn chữ như nhau), top-K theo tần suất train + `OTHER` (ngoài top-K hoặc không ánh xạ được) + `NONE` (ô rỗng); one-hot.
   - **Cột `diagnostic`** (trong `feature_decisions.yaml`): xử lý như cột số nhưng xuất riêng (`X_diag`), không nằm trong `D`, để Phase 5 đo. Cấu hình hiện tại không còn cột diagnostic (xem G2).
3. Xuất `feature_schema.json`: bố cục cột liền khối `[numeric][binary][na_flag][categorical groups]`, mỗi khối có `name`, `type`, `column`, `start`, `width`; tham số chuẩn hóa, danh sách hạng mục, thống kê audit (số dòng bị cắt 5σ, số ô không parse được). `Preprocessor.inverse_transform_block` và `inverse_transform` phục vụ sinh dữ liệu (NaN khi cờ = 1 hoặc loại `NONE`; cột không có cờ giữ giá trị lấp 0 = "không áp dụng").
4. Xuất `data/processed/<label_mode>/{train,val,test}.npz` (float32 `X`, `X_diag`, int `y`, `group_id`, `stream_id`, `class11`), `label_map.json` (thứ tự theo config), `preprocessor.joblib`.

**Test:**
- `parse_multi`: các ví dụ thật, ô rỗng, ô có 3 phần tử.
- Không có `NaN/inf` trong X; số chiều khớp schema; one-hot mỗi nhóm cộng đúng 1; bố cục liền khối.
- Round-trip số: `inverse(transform(x))` sai lệch tương đối < 1e-4 (trừ phần bị clip).
- Chống rò rỉ: `transform`/`audit` trên val/test không đổi tham số đã fit; fit trên train+val thì tham số đổi (nhóm đối chứng).
- Cột hằng (std = 0), giá trị âm ở cột thời gian, cột siêu thưa giữ phân biệt được các giá trị.
**Gate:** in ra `D`, số cờ `_is_na`, phân bố lớp từng split. Kết quả thực tế (6 lớp): D = 60, 12 cờ, sai lệch tương đối round-trip tối đa 1,2e-5, 0 giá trị không parse được.

---

## Phase 5 — Kiểm tra rò rỉ và độ tin cậy nhãn (`checks/leakage.py`)

Dùng Random Forest 100 cây, seed {0,1,2}, đánh giá macro-F1 trên **val** (nhóm khác train). 🔶 Mọi ngưỡng là heuristic khởi điểm (`thresholds` trong config): agent chỉ **báo cáo và đánh dấu**, người dùng quyết định ở G2. Train rất mất cân bằng còn val cân bằng, nên bản báo cáo luôn đưa cả RF không trọng số và RF cân bằng lớp.

| Mã | Phép kiểm tra | Cách đọc kết quả (heuristic, người dùng quyết định) |
|---|---|---|
| C1 | Ma trận hiện diện theo lớp (từ Phase 2) | Cột trống ở lớp này mà đầy ở lớp kia → nghi artifact; đối chiếu với sự kiện cha |
| C2 | RF chỉ dùng các cờ `_is_na` | macro-F1 > `na_only_f1_flag` (xa mức ngẫu nhiên 1/số lớp) → mô hình đang học "cách trích xuất"; liệt kê cột góp nhiều nhất |
| C3 | RF **cân bằng lớp** (`min_samples_leaf=5`) trên **từng cột thô** (giá trị + cờ + one-hot của cột đó) | macro-F1 **hoặc F1 tốt nhất theo lớp** > `single_feature_f1_flag` → soi ngữ nghĩa. (RF không trọng số sụp về lớp đa số nên che mất tín hiệu.) Kèm bỏ-từng-cột khỏi mô hình đầy đủ |
| C4 | Chia train pool theo dòng ngẫu nhiên, theo **stream**, theo **nhóm** (CV có phân tầng) | Cờ khi (ngẫu nhiên − nhóm) hoặc (ngẫu nhiên − stream) > `leakage_gap_flag`. (ngẫu nhiên − val) chỉ để tham khảo: fold CV giữ phân phối lớp mất cân bằng còn val cân bằng nên hai macro-F1 không so trực tiếp được |
| C5 | Phân biệt DoS vs DDoS trong cùng kịch bản (dữ liệu 11 lớp) | F1 < `dos_ddos_f1_flag` (≈ đoán ngẫu nhiên 0,5) → khuyến nghị dùng 6 lớp |

Thêm cho quyết định G2: mô hình tham chiếu (không trọng số và cân bằng lớp), ablation `+ cột diagnostic` và `bỏ các cột thời gian` (kèm F1 theo lớp).

**Deliverables:** `results/reports/leakage_report.md` (bảng, hình, các cờ) + `results/reports/leakage/*.csv`, `leakage_results.json`. Cập nhật `feature_decisions.yaml` chỉ khi người dùng quyết định ở G2.
**Gate 🛑 G2:** người dùng chốt (a) danh sách cột cuối, (b) `label_mode` (6 hay 11 lớp). Sau đó chạy lại Phase 4 (và Phase 3 nếu đổi bộ lọc dòng hoặc `label_mode`).
**Quyết định G2 đã chốt (2026-10-03):** (a) `time_since_first_frame_in_this_tcp_stream` được đưa vào bộ đặc trưng chính (thêm vào làm macro-F1 cân bằng lớp từ 0,442 lên 0,499, mức tăng chủ yếu ở lớp tấn công: BCF +0,10, DELAYED +0,11, NORMAL +0,013, nên không phải artifact Normal-vs-Attack); giữ `irtt`, `time_delta...`, `will_*`, `requested_qos`, `keep_alive`. (b) `label_mode = 6class`. (c) 3 ngưỡng mới được xác nhận.
**Kết quả sau G2 (6 lớp, D = 60, sau `numeric_scale`):** macro-F1 val 0,453 (không trọng số) và 0,498 (cân bằng lớp); C2 0,103 (dưới mức ngẫu nhiên 0,167); C3 không cột nào bị cờ (macro-F1 cao nhất 0,31, F1 theo lớp cao nhất 0,61); C4 chênh ngẫu nhiên−nhóm +0,007, ngẫu nhiên−stream +0,002; C5: BCF 0,593 (bị cờ), DELAYED 0,614, WILL 0,690, INVALID 0,788, SYN 0,962. **Kết luận quan trọng:** không thấy rò rỉ, nhưng các lớp tấn công khó phân biệt ở mức từng packet (chia ngẫu nhiên theo dòng trong chính train pool cũng chỉ 0,53), nên B0 thấp không phải dấu hiệu lỗi.

---

## Phase 6 — Baseline và khung đánh giá (`eval/`)

### 6.1 Ba giao thức đánh giá (tất cả test trên **tập test thật**)
- **TRTR:** train trên dữ liệu thật (train pool) → test thật. Đây là **B0**.
- **TSTR:** train chỉ trên dữ liệu sinh (cân bằng `syn_per_class`) → test thật.
- **TAug:** train trên (train pool + dữ liệu sinh bổ sung để mỗi lớp đạt tối thiểu `target_per_class`; không giảm lớp lớn) → test thật.

### 6.2 Bộ phân loại
`RandomForestClassifier(n_estimators=200, n_jobs=-1)` và `MLPClassifier(hidden=(128,64), early_stopping=True)`. Lưu **dự đoán** từng run (`artifacts/{run_id}/preds/`) để bootstrap ghép cặp.

### 6.3 Baseline
- **B0:** chỉ dữ liệu thật.
- **B1a:** RF `class_weight="balanced"`. **B1b:** SMOTE trên không gian đã mã hóa tới `target_per_class`, hậu xử lý (argmax cho nhóm one-hot, làm tròn cờ nhị phân).
- (Tuỳ chọn) **W:** WGAN-GP tập trung, chỉ để đo thời gian và số tham số, chứng minh "WGAN-GP nặng hơn CVAE".

### 6.4 Metric
- **Tiện ích:** macro-F1 (chỉ số chính), recall/precision/F1 theo lớp (đặc biệt `WILL`, `INVALID`), balanced accuracy, PR-AUC macro OvR, confusion matrix, và bảng phụ nhị phân Normal/Attack suy ra bằng cách gộp dự đoán đa lớp.
- **Độ trung thực** (`fidelity.py`): Wasserstein trên cột số (chia cho std train), Jensen-Shannon cho cột nhị phân/phân loại, khoảng cách tương quan (trung bình |Δ Pearson|), **C2ST** (RF phân biệt thật/giả, 5-fold, AUC; 0,5 là tốt nhất). Tính theo từng lớp.
- **Riêng tư** (`privacy.py`):
  - tỉ lệ dòng trùng lặp chính xác giữa synthetic và train (băm sau khi làm tròn);
  - **DCR** (khoảng cách tới bản ghi train gần nhất, không gian đặc trưng đã chuẩn hóa) và tỉ số `median DCR(syn→train) / median DCR(syn→val)`; ≈ 1 là tốt, ≪ 1 nghĩa là sao chép;
  - **MIA đơn giản:** điểm = −DCR của một bản ghi tới tập sinh; thành viên = mẫu train, không thành viên = mẫu val; tính AUC **theo từng lớp** (cân bằng số lượng) rồi trung bình. AUC ≈ 0,5 là tốt.
  - **Đối chứng dương (positive control):** huấn luyện một CVAE cố ý overfit (500 mẫu, không DP, nhiều epoch); MIA phải cho AUC rõ ràng > 0,5. Nếu không → MIA quá yếu, phải nói rõ khi báo cáo.
- **Chi phí** (`overhead.py`): thời gian mỗi vòng và tổng, byte truyền mỗi vòng (đo hoặc tính giải tích), RAM đỉnh (psutil).
- **Thống kê** (`stats.py`): bootstrap phân tầng trên tập test (1000 lần) cho CI của macro-F1 và **CI của hiệu số ghép cặp** giữa hai cấu hình.

**Test:** metric trên dữ liệu giả có đáp án biết trước (dự đoán hoàn hảo → F1 = 1; dự đoán ngẫu nhiên → xấp xỉ 1/số lớp); `Preprocessor` không bị dùng lại giữa các split; kết quả lặp lại được với cùng seed.
**Gate 🛑 G3 (sanity baseline):** B0 và B1 chạy đủ 3 seed; std macro-F1 giữa các seed < 0,02; macro-F1 B0 không ≈ 1,000 ở mọi lớp (nếu có: quay lại Phase 5, nghi rò rỉ). Người dùng xem bảng B0/B1 trước khi tiếp tục. Kỳ vọng thực tế đã đo ở Phase 5 (RF 100 cây, val): macro-F1 ≈ 0,45 (không trọng số) và ≈ 0,50 (cân bằng lớp); B0 thấp là đặc tính của dữ liệu mức packet, không phải lỗi.

---

## Phase 7 — CVAE tập trung (B2) và Optuna

### 7.1 Kiến trúc (`models/cvae.py`)
- Encoder: `[x ; onehot(y)] → Linear(hidden[0]) → ReLU → Linear(hidden[1]) → ReLU → (μ, logσ²)` (latent 16).
- Decoder: `[z ; onehot(y)] → hidden ngược → các đầu ra theo khối`: logit nhị phân/cờ `_is_na` (BCE), logit từng nhóm phân loại (CE), giá trị số (MSE).
- **Không dùng BatchNorm** (không tương thích Opacus); chỉ `nn.Linear`, `ReLU`, tuỳ chọn `LayerNorm`.
- Loss = tổng recon các khối + `β · KL`, trung bình theo mẫu; β tăng tuyến tính từ 0 tới `beta` trong `beta_warmup_epochs`.
- Early stopping theo loss trên val.

### 7.2 Sinh dữ liệu (`models/generate.py`)
`generate(model, class_counts, seed)`: lấy `z ~ N(0,I)` và nhãn theo số lượng yêu cầu; cột số → nghịch đảo chuẩn hóa/log1p → cắt về [min, max] của train → làm tròn với cột nguyên; cột nhị phân/`_is_na` lấy mẫu Bernoulli; cột phân loại lấy mẫu từ softmax; nếu `_is_na = 1` thì đặt giá trị cột số tương ứng về mặc định.

### 7.3 Tối ưu siêu tham số (`tune.py`)
- Optuna TPE, `n_trials` = 30, lưu SQLite để resume.
- Không gian: `latent ∈ {8,16,32}`, `hidden width ∈ {64,128,256}`, `β ∈ [0.1, 2]` (log), `lr ∈ [1e-4, 3e-3]` (log), epoch cố định 30.
- **Fitness:** macro-F1 TSTR (RF 100 cây, `syn_per_class` = 5000) đánh giá trên **val thật**. Không dùng test.
- Kết quả: `configs/best_cvae.yaml`. Ghi rõ hạn chế: tuning làm trên dữ liệu tập trung, không riêng tư.

**Test:** shape vào/ra; loss giảm sau vài epoch trên dữ liệu nhỏ; `generate` trả đúng số lượng mỗi lớp; giá trị nằm trong khoảng hợp lệ; one-hot sinh ra cộng đúng 1; tái lập theo seed.
**Gate:** B2 chạy đủ 3 seed. Kiểm tra tối thiểu (🔶 ngưỡng lấy từ `thresholds`): C2ST AUC < `c2st_auc_max`, tỉ lệ trùng lặp < `dup_rate_max`, DCR ratio > `dcr_ratio_min`. Báo cáo TSTR và TAug so với B0, kể cả khi không cải thiện.

### 7.4 Benchmark tài nguyên (🔶 bắt buộc trước Phase 8)
1. Trên một client (n ≈ 18.000 dòng), đo: thời gian 1 epoch CVAE thường, thời gian 1 epoch DP-SGD (Opacus), RAM đỉnh.
2. Ngoại suy: thời gian 1 run FL ≈ `rounds × local_epochs × Σ_i t_epoch_i` (client chạy tuần tự); tổng ma trận = cộng các run theo bảng Phase 11 × 3 seed; thời gian Optuna = `n_trials` × thời gian 1 trial. Ghi vào `results/reports/compute_budget.md` kèm số đo gốc.
3. Nếu tổng vượt `compute.budget_hours` (người dùng đặt) hoặc không vừa số phiên Colab dự kiến, cắt giảm **theo thứ tự**: (a) Optuna 30 → 15 trial; (b) `rounds` 30 → 20; (c) seed của M1-ε1 và M1-ε10 còn 2; (d) bỏ phần mở rộng A1–A5. **Không cắt** seed của B0–B3, M1-ε5, M2, M3. Mọi cắt giảm phải ghi rõ trong báo cáo cuối.
4. Optuna lưu SQLite vào `compute.artifacts_dir` để resume giữa các phiên Colab.

---

## Phase 8 — FL non-IID (B3) (`partition.py`, `fl/`)

**Việc:**
1. `partition.py`: chia train pool cho `num_clients` bằng Dirichlet(α) theo nhãn; mỗi client ≥ `min_client_size` (nếu không, lấy lại mẫu Dirichlet); lưu chỉ số vào `data/partitions/alpha{α}_seed{s}.json`; xuất bảng phân bố nhãn theo client và heatmap.
2. Ứng dụng Flower (mô phỏng, một máy): dùng `flwr new` để lấy template đúng phiên bản đã cài, **rồi thay mô hình bằng CVAE**. Client `fit`: huấn luyện `local_epochs` epoch trên dữ liệu local, trả trọng số + số mẫu. Server: FedAvg có trọng số theo số mẫu. Khởi tạo chung từ seed.
3. Mỗi 5 vòng, đánh giá ELBO của mô hình toàn cục trên val (chỉ để theo dõi; ghi chú đây là đánh giá phía mô phỏng, không có trong FL thật). Lưu checkpoint cuối và log từng vòng.
4. Sau vòng cuối: sinh dữ liệu và đánh giá đầy đủ (Phase 6).
5. **Checkpoint/resume:** mỗi `checkpoint_every_rounds` vòng lưu vào `artifacts/{run_id}/ckpt/`: trọng số toàn cục, chỉ số vòng, trạng thái RNG, bộ đếm bước DP của từng client (Phase 9), log vòng. Tham số `--resume` nạp checkpoint mới nhất.

**Test/Gate:**
- Phân hoạch: các chỉ số đôi một rời nhau, hợp lại = toàn bộ train pool.
- **FedAvg đúng:** với trọng số ngẫu nhiên, kết quả gộp bằng đúng trung bình có trọng số tính tay bằng numpy (sai lệch < 1e-6).
- **Sanity so với tập trung:** 1 client (toàn bộ dữ liệu) tương đương B2 (loss val chênh < 5%).
- IID (α lớn) so với non-IID: vẽ đường loss theo vòng; kỳ vọng non-IID hội tụ chậm hơn hoặc kém hơn (không bắt buộc, nhưng nếu ngược thì phải giải thích).
- Chạy hết 30 vòng không lỗi; log thời gian và byte/vòng.
- **Resume:** dừng ở vòng 10 rồi resume tới vòng 20 cho trọng số hợp lệ, đủ 20 dòng log, bộ đếm bước DP bằng đúng chạy không ngắt.

---

## Phase 9 — DP tại client (M1) (`fl/dp_utils.py`) 🔶 API Opacus cần xác minh theo phiên bản cài

**Việc:**
1. Với mỗi client `i` (kích thước `n_i`), tần suất lấy mẫu `q_i = batch_size / n_i`, tổng số bước `T_i = rounds × local_epochs × (n_i / batch_size)`. Tính `σ_i` một lần cho cả quá trình huấn luyện bằng tiện ích của Opacus (`get_noise_multiplier` với `target_epsilon`, `target_delta`, `sample_rate`, và số bước hoặc epoch; kiểm tra chữ ký hàm ở phiên bản đã cài).
2. Mỗi vòng, `PrivacyEngine.make_private(..., noise_multiplier=σ_i, max_grad_norm=C, poisson_sampling=True)`. **Lưu ý:** tạo lại engine mỗi vòng sẽ **đặt lại accountant**, nên không dùng `get_epsilon` của từng vòng. Thay vào đó, đếm tổng số bước và tính ε cuối cùng bằng RDP accountant từ `(σ_i, q_i, T_i)`.
3. Chạy `ModuleValidator.validate` để chắc mô hình tương thích (không BatchNorm).
4. Quét `ε ∈ {1, 5, 10}` (δ = 1e-5; đảm bảo δ < 1/n_i cho mọi client); thêm mốc "không DP" (= B3).
5. **Cách báo cáo ε khi các client có `n_i` khác nhau:** hiệu chỉnh `σ_i` sao cho mỗi `ε_i ≤` mục tiêu. Với mỗi cấu hình báo cáo bảng `ε_i` từng client; **ε của cấu hình = max_i ε_i** (trường hợp xấu nhất, dùng làm số chính trong biểu đồ và quy tắc khuyến nghị), kèm trung vị. Ý nghĩa: ε_i bảo vệ một bản ghi thuộc client i trước mọi bên chỉ thấy cập nhật của client đó. Không tuyên bố hiệu ứng khuếch đại riêng tư do SecAgg.
6. **Đơn vị bảo vệ (hạn chế quan trọng):** ε ở đây là mức **bản ghi (packet)**. Các packet cùng TCP stream tương quan mạnh, nên một phiên tấn công gồm nhiều packet được bảo vệ yếu hơn nhiều so với con số ε (theo tính chất group privacy, độ suy giảm cỡ số packet trong nhóm; 🔶 không tuyên bố công thức chính xác nếu chưa tính). Phải nêu rõ trong báo cáo. Giảm nhẹ (tuỳ chọn): bật `train_sampling.max_rows_per_stream` và chạy độ nhạy A5 để xem kết quả có phụ thuộc vào tương quan nội-stream không.
7. Nhãn (điều kiện) không được bảo vệ bởi DP-SGD của đặc trưng; ghi rõ trong phần hạn chế.

**Test/Gate:**
- ε thực đạt ≤ 1,02 × ε mục tiêu cho mọi client (kể cả sau khi resume từ checkpoint: ε tính lại bằng tổng số bước phải bằng chạy không ngắt).
- σ tăng khi ε mục tiêu giảm.
- Với σ = 0 và C rất lớn, kết quả xấp xỉ B3 (sanity).
- Xu hướng kỳ vọng: TSTR macro-F1 giảm khi ε giảm; MIA AUC và tỉ lệ DCR tiến gần 0,5 và 1. Nếu không thấy xu hướng, kiểm tra độ mạnh của MIA (đối chứng dương) trước khi kết luận.

---

## Phase 10 — SecAgg (M2) và DP + SecAgg (M3) 🔶 API Flower cần xác minh theo phiên bản cài

**Việc:**
1. Dùng `SecAggPlusWorkflow` (hoặc `SecAggWorkflow`, trường hợp riêng của SecAgg+) phía server và `secaggplus_mod` phía client; bắt đầu từ ví dụ `flwr new @flwrlabs/flower-secure-aggregation` rồi thay mô hình bằng CVAE. Tham số khởi đầu: `num_shares` = số client, `reconstruction_threshold` = 3 (đọc tài liệu để xác nhận ngữ nghĩa/ràng buộc của hai tham số này ở phiên bản đã cài); `clipping_range`, `quantization_range`, `modulus_range` để mặc định rồi kiểm tra.
2. Kiểm tra `max |w|` của mọi tham số qua mọi vòng nhỏ hơn `clipping_range`; nếu vượt, tăng `clipping_range` hoặc báo cảnh báo (giá trị bị cắt gây sai lệch).
3. M3 = DP client (Phase 9) + SecAgg cùng lúc. Ghi ε thực đạt.
4. **Phương án dự phòng** nếu SecAgg của Flower không chạy được trong mô phỏng ở phiên bản đã cài: viết `secagg_sim.py` (masking cặp kiểu Bonawitz: mỗi cặp client cộng/trừ cùng một mask từ seed chung, mask triệt tiêu khi cộng), ghi rõ đây là mô phỏng thay thế và sai khác so với giao thức đầy đủ (không có phần chịu client rớt).

**Test/Gate:**
- **T-SA1 (đúng đắn):** global model của SecAgg so với FedAvg thường cùng seed: `max |Δ|` ≤ 1e-4 (sai số lượng tử hóa); hiệu số macro-F1 nằm trong std giữa các seed.
- **T-SA2 (che giấu, nếu có thể móc log của workflow):** vector đã mask mà server nhận có tương quan Pearson |ρ| < 0,05 với cập nhật thật và phân bố xấp xỉ đều trên [0, modulus) (KS test). Nếu không móc được, ghi rõ "không kiểm chứng trực tiếp".
- **T-SA3 (tuỳ chọn):** cho 1 client rớt ở một vòng, tổng hợp vẫn hoàn tất nếu còn ≥ ngưỡng.
- Đo **overhead** SecAgg: `thời gian/vòng SecAgg ÷ thời gian/vòng thường` và byte/vòng (ghi chú: mô phỏng một máy đo được phần tính toán mật mã, không đo được độ trễ mạng).

---

## Phase 11 — Điều phối thí nghiệm (`run_experiment.py`, `aggregate.py`)

**Ma trận (mỗi cấu hình × seed {0,1,2}):**

| ID | Cấu hình |
|---|---|
| B0 | Chỉ dữ liệu thật |
| B1a / B1b | class_weight / SMOTE |
| B2 | CVAE tập trung |
| B3 | CVAE + FL |
| M1-ε1, M1-ε5, M1-ε10 | FL + DP |
| M2 | FL + SecAgg |
| M3 | FL + DP(ε=5) + SecAgg |

**Mở rộng (chỉ làm nếu còn thời gian, bị cắt đầu tiên khi vượt ngân sách):** A1 độ lệch non-IID `α ∈ {0.1, 0.5, 10}` trên B3; A2 tỉ lệ mất cân bằng train pool `{10:1, 60:1, 200:1}`; A3 mode 11 lớp; A4 **độ nhạy theo nhóm test**: đổi `split_seed` (2 cách chọn nhóm test/val khác) rồi chạy lại B0 và B3 để xem kết luận có phụ thuộc vào file được chọn không; A5 `max_rows_per_stream` bật so với tắt trên B3 và M1-ε5.

**Yêu cầu:**
- Mỗi run có `run_id = {config}_{seed}`; ghi một dòng vào `results/runs.csv` (metric, thời gian, ε thực đạt, config hash). **Resume:** bỏ qua run đã hoàn tất, và run đang dở được nạp lại từ checkpoint.
- `aggregate.py` sinh `results/summary.csv` (mean ± std theo cấu hình) và các hình: (1) macro-F1 theo cấu hình (thanh lỗi); (2) recall theo lớp cho lớp hiếm; (3) đường **utility–privacy** (macro-F1 vs ε, cùng MIA AUC vs ε); (4) độ trung thực (Wasserstein, C2ST) vs ε; (5) overhead (thời gian/vòng, byte/vòng); (6) Pareto: macro-F1 vs MIA AUC, kích thước điểm theo overhead.
- Sinh tự động `results/reports/final_report.md` từ các bảng, không viết tay số liệu.

**Gate 🛑 G4 (trước khi chạy toàn bộ):** chạy thử 1 seed cho mọi cấu hình, xem log; người dùng đồng ý rồi mới chạy đủ 3 seed.

---

## Phase 12 — Cách xác định kết quả (quy tắc diễn giải)

**Câu hỏi và cách trả lời:**

| Câu hỏi | Chỉ số | Kết luận "có tác dụng" khi |
|---|---|---|
| R1. Dữ liệu sinh có cải thiện IDS? | ΔmacroF1 (TAug − B0), Δrecall lớp hiếm | CI bootstrap ghép cặp của Δ loại trừ 0 **và** \|Δ\| > std giữa các seed |
| R2. CVAE có hơn cách đơn giản? | TAug(B2) so với B1a, B1b | Như trên |
| R3. FL non-IID mất bao nhiêu? | B2 − B3 | Báo cáo mức mất; không có ngưỡng đạt/không đạt |
| R4. DP tốn bao nhiêu và đem lại gì? | B3 − M1(ε); MIA AUC, DCR ratio, trùng lặp | Có đường cong utility–privacy đơn điệu hợp lý; MIA AUC tiến gần 0,5 khi ε giảm (đã qua đối chứng dương) |
| R5. SecAgg tốn gì? | \|M2 − B3\| về utility; overhead | Utility chênh trong nhiễu giữa seed; overhead được định lượng |
| R6. Cấu hình khuyến nghị? | Tất cả | Theo quy tắc bên dưới |

**Quy tắc khuyến nghị (minh bạch, chỉnh ngưỡng trong config):** (🔶 các ngưỡng là heuristic, đọc từ `thresholds`) trong các cấu hình thỏa `MIA_AUC ≤ mia_auc_max` **và** `ε_max ≤ 5` (nếu có DP) **và** overhead thời gian/vòng ≤ `overhead_ratio_max` × FL thường, chọn cấu hình có macro-F1 TAug cao nhất; nếu không cấu hình nào thỏa, báo cáo Pareto và nêu rõ đánh đổi. Nêu rõ SecAgg bảo vệ cập nhật từng client khỏi server, còn DP bảo vệ khỏi rò rỉ từ mô hình/dữ liệu sinh.

**Cờ đỏ (phải điều tra, không được báo cáo như thành công):**
- macro-F1 ≈ 1,00 ở mọi cấu hình kể cả B0 → nghi rò rỉ/artifact (quay lại Phase 5).
- TSTR vượt TRTR đáng kể → nghi dữ liệu sinh "lộ" nhãn hoặc test bị dùng nhầm.
- Trùng lặp > 1% hoặc DCR ratio ≪ 1 → mô hình sao chép dữ liệu.
- Kết quả giữa các seed dao động lớn (std > 0,05) → cần thêm seed hoặc kiểm tra tính ổn định.
- ε báo cáo khác xa ε tính lại độc lập.

---

## Phase 13 — Demo và đóng gói

- `demo/app.py` (Streamlit, chạy trên máy cá nhân; có thể thay bằng Gradio nếu chạy trên Colab): (1) tổng quan dữ liệu và phân bố lớp; (2) bảng/hình kết quả từ `results/summary.csv`; (3) biểu đồ utility–privacy–overhead và khuyến nghị; (4) tạo mẫu: chọn cấu hình + lớp + số lượng → xem bảng mẫu sinh, tải CSV; (5) trang giải thích mô hình đe dọa và hạn chế.
- `README.md`: cách cài, cách chạy từng Phase, cách tái lập (`python -m ppfeddata.cli ...`), cấu trúc thư mục, phiên bản thư viện, thời gian chạy tham khảo.
- Lưu `pip freeze`, `configs/`, `split_manifest.json`, `feature_schema.json` cùng kết quả để tái lập.

---

## Danh mục nghiệm thu cuối (Definition of Done)

- [ ] `pytest -q` xanh; mọi Phase có test.
- [ ] Tổng dòng và số dòng từng lớp khớp `expected_counts`.
- [ ] Không nhóm nào xuất hiện ở hơn một split; test thật được khóa, không dùng để tuning.
- [ ] `leakage_report.md` có đủ C1–C5; quyết định G1, G2 được lưu.
- [ ] B0, B1, B2, B3, M1×3, M2, M3 đều có đủ 3 seed trong `runs.csv`.
- [ ] ε thực đạt của DP được báo cáo và khớp mục tiêu; T-SA1 đạt; MIA có đối chứng dương.
- [ ] `final_report.md` sinh tự động, mọi số có nguồn trong `results/`.
- [ ] `compute_budget.md` có số đo thực và ghi lại mọi cắt giảm (nếu có).
- [ ] Danh sách hạn chế nêu đủ: dữ liệu ở mức packet; **ε ở mức bản ghi, các packet trong stream tương quan**; đặc trưng thời gian phụ thuộc cách xuất và CVAE sinh packet độc lập; tuning không riêng tư; mô phỏng một máy (không đo độ trễ mạng); chuẩn hóa dùng thống kê tập trung; nhãn không được bảo vệ bởi DP; tập test lấy từ ít nhóm (xem A4 nếu đã chạy); các ngưỡng đánh giá là heuristic do người dùng chốt; kết quả trên một dataset; các lớp tấn công khó phân biệt ở mức từng packet (B0 ≈ 0,45-0,50) nên CVAE sinh packet độc lập không thể tạo thêm độ phân biệt không có sẵn trong đặc trưng; 8/11 lớp con chỉ có một file capture nên val/test của chúng là các khối liên tiếp trong cùng một capture, và việc bỏ stream vắt qua nhiều khối loại 1,2% đến 9,9% dòng đủ điều kiện mỗi lớp con (lệch về phía kết nối ngắn); lọc `protocol ∈ {TCP, MQTT}` bỏ 0,50% dòng Normal và 128 dòng RIPv2 của Attack; chỉ dùng phần tử đầu của ô nhiều giá trị (mất thông tin gộp nhiều bản tin trong một packet, phần nào còn trong `tcp_segment_len`); đuôi `time_delta...` vẫn bị cắt ở 5σ ở mức 0,08% dòng train sau khi nhân 1000 (1,04% nếu không); cột siêu thưa xử lý khác v1.1; thông tin nhãn lớp con DoS/DDoS bị gộp ở chế độ 6 lớp.
- [ ] README đủ để người khác chạy lại từ đầu.

---

## Phụ lục — Prompt mở đầu để dán cho agent

> Bạn là kỹ sư ML. Đọc `PP-FedData_Implementation_Spec.md`. Thực hiện **một Phase mỗi lượt**, theo thứ tự, bắt đầu từ Phase 0. Với mỗi Phase: (1) liệt kê kế hoạch ngắn, (2) viết code và test, (3) chạy test và các kiểm tra Gate, (4) tóm tắt kết quả kèm số liệu thật và các file đã tạo. Dừng tại mọi HUMAN GATE (G1–G4) và chờ xác nhận. Không sửa dữ liệu gốc. Không bịa số liệu. Nếu bị chặn, ghi `BLOCKERS.md` và dừng.

---

## Lịch sử thay đổi

- **v1.2 (2026-10-03):** gộp các thay đổi rút ra khi làm Phase 0-5 (chi tiết và bằng chứng: `SPEC_DEVIATIONS.md`). Phase 2: kiểm toán token, cặp cột trùng, header; ánh xạ nhãn chữ↔mã số; luật SUSPECT sinh từ số liệu (sự kiện cha, ngưỡng đếm); `first_only` thay cho first+count/sum và bỏ `n_mqtt_msgs`; lọc `protocol`; quyết định người dùng lưu ở config (`user_overrides`, `g1_log`, `g2_log`). Phase 3: chọn trước chỉ số dòng thay cho reservoir, `interim/<mode>/`, nhóm `#blkNN`, bỏ stream vắt qua nhiều khối, manifest có phiên bản và bản sao commit được. Phase 4: cột siêu thưa, bố cục khối, cột `diagnostic`, `processed/<mode>/`. Phase 5: C3 cân bằng lớp + F1 theo lớp, C4 thêm chia theo stream và chỉ cờ theo nhóm/stream, ablation; 3 ngưỡng mới; quyết định G1, G2; `configs/local.yaml` và `config_hash` bỏ `paths`; hạn chế bổ sung. Sau G2: `numeric_scale` cho `time_delta`, `requirements.txt` bỏ ràng buộc `numpy<2.0` (kiểm tra bằng dry-run cài `torch`, `flwr`, `opacus`: không xung đột), cho phép commit báo cáo `.md` và hình `.png` trong `results/`.
- **v1.1:** thêm mục 1.1 (Colab/Drive, chống mất phiên) và 1.2 (ngân sách tính toán); thêm bước 7.4 benchmark; bảng quota 11 lớp và bỏ mode `binary` riêng; kiểm tra đặc trưng thời gian (Phase 2); giữ `stream_id` và tuỳ chọn `max_rows_per_stream`; quy tắc báo cáo ε theo từng client (max_i); checkpoint/resume cho FL và DP; nhãn 🔶 và khối `thresholds` trong config; độ nhạy A4 (đổi nhóm test) và A5; cập nhật Definition of Done.
- **v1.0:** bản đầu tiên.
