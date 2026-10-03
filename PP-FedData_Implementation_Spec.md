# PP-FedData — Đặc tả triển khai cho coding agent (v1.1)

> Đây là hợp đồng công việc cho agent lập trình. Đọc toàn bộ trước khi viết code. Làm **tuần tự theo Phase**; mỗi Phase có *Deliverables* và *Gate* (điều kiện nghiệm thu). Không sang Phase sau khi Gate chưa đạt.

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
│   ├── label_map.yaml          # ánh xạ thư mục/file → lớp
│   ├── feature_decisions.yaml  # sinh ở Phase 2, người duyệt (G1)
│   └── exp/                    # 1 file / cấu hình thí nghiệm (B0..M3)
├── src/ppfeddata/
│   ├── data/      inventory.py, harmonize.py, split_sample.py, parse_multi.py, preprocess.py, partition.py
│   ├── models/    cvae.py, generate.py
│   ├── fl/        client_app.py, server_app.py, dp_utils.py, secagg_sim.py (fallback)
│   ├── eval/      utility.py, fidelity.py, privacy.py, overhead.py, stats.py
│   ├── checks/    leakage.py
│   ├── tune.py, run_experiment.py, aggregate.py, cli.py
├── tests/
├── notebooks/     00_eda.ipynb ... (chỉ để xem, không chứa logic)
├── data/          inventory/, interim/, processed/, partitions/   (KHÔNG commit)
├── artifacts/     {run_id}/ model.pt, synthetic.parquet, preds/
├── results/       runs.csv, summary.csv, figures/, reports/
├── demo/          app.py
├── requirements.txt, README.md, BLOCKERS.md (nếu có)
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
label_mode: "6class"            # "6class" | "11class"  (chốt ở G2). Nhị phân Normal/Attack chỉ là bảng phụ suy ra từ dự đoán đa lớp
split:
  test_groups_per_subclass: 1
  val_groups_per_subclass: 1
  fallback_block_split: {n_blocks: 10, gap_rows: 1000}
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
  multi_value: {message_type: first+count, numeric_multi: sum}
  categorical_top_k: 10
  log1p_skew_threshold: 2.0
  clip_sigma: 5
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
  leakage_gap_flag: 0.05        # chênh macro-F1 giữa chia ngẫu nhiên và chia theo nhóm
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

**Việc:**
1. Chuẩn hóa tên cột sang snake_case; 3 cột trùng tên gán hậu tố theo **vị trí** (`qos_level_1/_2`, `frame_length_on_wire_1/_2`, `clean_session_flag_1/_2`); ánh xạ `.1` của Normal về `_2`. Kiểm tra 33 cột khớp thứ tự giữa Normal và Attack.
2. **Presence matrix:** với mỗi cột × mỗi lớp con (11 lớp), tỉ lệ ô trống, tính trên toàn bộ dữ liệu bằng streaming (có thể dùng lại `data_profile_full.csv` nếu khớp cấu trúc, nhưng phải tính lại theo lớp con). Xuất `presence_matrix.csv` và heatmap.
3. **Format audit:** với mỗi cột × nguồn (Normal vs Attack), thống kê "kiểu giá trị" bằng regex: số nguyên, số thực, hex, chứa dấu phẩy, chuỗi chữ, True/False, Set/Not set, rỗng. Xuất `format_audit.csv`. Mục tiêu: phát hiện khác biệt do **quy trình trích xuất** (TShark tự chạy cho Normal, CSV có sẵn cho Attack) chứ không phải do hành vi mạng.
4. Sinh `configs/feature_decisions.yaml` với luật mặc định (mỗi cột có `action: drop|numeric|binary|categorical|multi`, kèm lý do):
   - `drop`: `no`, `epoch_time`, `time_since_reference_or_first_frame`, `stream_index` (bỏ khỏi đặc trưng nhưng **giữ làm metadata** `stream_id = (group_id, stream_index)`), `source`, `info`, `requested_qos` (nếu trống > 99,9% ở mọi lớp), mọi cột trống 100% ở mọi lớp.
   - `SUSPECT` (mặc định drop, cần người quyết): cột có tỉ lệ trống chênh > 0,95 giữa Normal và bất kỳ lớp Attack nào, hoặc có định dạng khác hẳn giữa hai nguồn.
   - `numeric`: `frame_length_on_wire` (bản còn thông tin), `time_delta_from_previous_displayed_frame`, `irtt`, `time_since_first_frame_in_this_tcp_stream`, `tcp_segment_len`, `calculated_window_size`, `keep_alive`, `user_name_length`, `password_length`, `will_message_length`, `will_topic_length`, `topic_length`, `msg_len`.
   - `binary`: `syn`, `reset`, `acknowledgment`, `clean_session_flag_*`, `retain`, `will_retain`, `will_flag`.
   - `categorical`: `protocol`, `message_type`, `qos_level_*`.
   - `multi` (ô nhiều giá trị): `message_type`, `msg_len`, `qos_level_1`, `retain`, `topic_length`.

5. **Kiểm tra tính so sánh được của đặc trưng thời gian** (`time_delta_from_previous_displayed_frame`, `irtt`, `time_since_first_frame_in_this_tcp_stream`). Giá trị "previous displayed frame" phụ thuộc bộ lọc hiển thị lúc xuất; nếu Normal (TShark do người dùng chạy) và Attack (CSV có sẵn) xuất với bộ lọc khác nhau thì giá trị không so sánh được. So sánh các phân vị (1, 25, 50, 75, 99%) của từng cột thời gian giữa hai nguồn trên **cùng loại packet** (ví dụ chỉ packet TCP không có MQTT), xuất `time_feature_audit.csv`. Cột lệch bất thường → SUSPECT. Ghi vào hạn chế: CVAE sinh từng packet độc lập nên không giữ được chuỗi thời gian giữa các packet.

**Test:** `test_harmonize.py` (mapping tên cột, cột `.1`; cột thời gian không âm); test regex định dạng trên ví dụ `"Publish Message,Publish Message"`, `"40,41"`, `""`.
**Gate 🛑 G1:** agent nộp `presence_matrix.csv`, `format_audit.csv`, `feature_decisions.yaml` kèm danh sách SUSPECT. **Người dùng duyệt bộ đặc trưng** trước khi sang Phase 3.

---

## Phase 3 — Chia theo nhóm và lấy mẫu streaming (`split_sample.py`)

**Việc:**
1. Với mỗi lớp con (kịch bản × attack_type), gán nhóm cho split theo seed: `test_groups_per_subclass` nhóm cho test, `val_groups_per_subclass` cho validation, phần còn lại cho train. Normal dùng `Capture_ID`. Đảm bảo mỗi nhóm được chọn đủ dòng cho quota; nếu không, chọn nhóm khác và ghi lại.
2. Lớp con có < 3 nhóm: dùng **block-split**: chia thành `n_blocks` khối liên tiếp theo thứ tự dòng, gán khối cho split, bỏ `gap_rows` dòng đệm giữa các khối.
3. Lấy mẫu bằng **reservoir sampling** theo (lớp, split) đến đúng quota; với chế độ 6 lớp, mỗi kịch bản chia đều quota giữa DoS và DDoS. Train lấy đều qua các nhóm train.
4. **Chế độ nhãn** (`label_mode`):
   - `6class` (mặc định): dùng `quota`; mỗi kịch bản chia đều giữa DoS và DDoS.
   - `11class`: dùng `quota_11class` (bảng ở mục 3, mỗi lớp con bằng một nửa quota kịch bản).
   - Nhị phân Normal/Attack **không phải mode riêng** (train pool nhị phân không còn bài toán mất cân bằng đáng kể); chỉ báo cáo như bảng phụ bằng cách gộp dự đoán đa lớp.
   - Nếu một nhóm không đủ dòng cho quota: giảm quota của lớp đó, ghi cảnh báo; nếu val hoặc test của một lớp xuống dưới 1000 dòng thì **dừng và hỏi người dùng** (không tự hạ tiếp).
4b. (Tuỳ chọn, `train_sampling.max_rows_per_stream`) Giới hạn số packet lấy từ mỗi `stream_id` trong train pool để giảm tương quan nội-stream (lý do: xem Phase 9, mục "Đơn vị bảo vệ"). Mặc định tắt; chạy như độ nhạy A5.
5. Xuất `data/interim/{train,val,test}.parquet` (dữ liệu thô đã harmonize + `label`, `group_id`, `stream_id`, `split`) và `split_manifest.json` (nhóm nào ở split nào, số dòng, seed).

**Test/Gate:**
- **Không nhóm nào xuất hiện ở hơn một split** (assert).
- Số dòng đúng quota (hoặc có ghi chú rõ nếu giảm vì thiếu dữ liệu).
- Chạy lại với cùng seed cho file Parquet cùng hash.
- Bộ nhớ đỉnh (psutil) < 4 GB.

---

## Phase 4 — Tiền xử lý (`parse_multi.py`, `preprocess.py`)

**Việc:**
1. `parse_multi(cell) -> list[str]`: tách ô nhiều giá trị, xử lý rỗng/`nan`/khoảng trắng. Cột `message_type`: sinh `message_type_first` (categorical) và `n_mqtt_msgs` (số phần tử). Cột số dạng multi: lấy **tổng** (độ dài) hoặc phần tử đầu, theo `feature_decisions.yaml`.
2. Lớp `Preprocessor` (fit chỉ trên **train**, lưu bằng joblib):
   - Cột số: điền 0 cho giá trị không áp dụng và thêm cờ `<col>_is_na` khi tỉ lệ trống trong train nằm trong (0,5%; 99,5%); `log1p` nếu skew > ngưỡng và không âm; chuẩn hóa mean/std; cắt ±`clip_sigma`.
   - Cột nhị phân: 0/1 (parse `True/False`, `Set/Not set`, `0/1`), trống → 0 + cờ `_is_na`.
   - Cột phân loại: top-K theo tần suất train + `OTHER` + `NONE`; one-hot.
3. Xuất `feature_schema.json`: danh sách khối cột (numeric/binary/categorical-group/na-flag) kèm chỉ số, tham số chuẩn hóa, danh sách hạng mục. `Preprocessor.inverse_transform_block` phục vụ sinh dữ liệu.
4. Xuất `data/processed/{train,val,test}.npz` (float32 X, int y, group_id) và ánh xạ nhãn.

**Test:**
- `parse_multi`: các ví dụ thật, ô rỗng, ô có 3 phần tử.
- Không có `NaN/inf` trong X; số chiều khớp schema; one-hot mỗi nhóm cộng đúng 1.
- Round-trip số: `inverse(transform(x))` sai lệch < 1e-4 (trừ phần bị clip).
- Test **chống rò rỉ**: đổi dữ liệu val/test không làm thay đổi tham số của Preprocessor đã fit.
**Gate:** in ra `D` (số chiều đầu vào), số cờ `_is_na`, và phân bố lớp của từng split.

---

## Phase 5 — Kiểm tra rò rỉ và độ tin cậy nhãn (`checks/leakage.py`)

Dùng Random Forest nhỏ (100 cây), seed cố định, đánh giá macro-F1 trên **val** (nhóm khác train). 🔶 Mọi ngưỡng trong bảng là heuristic khởi điểm (xem `thresholds` trong config): agent chỉ **báo cáo và đánh dấu**, người dùng quyết định ở G2. Với C3, đặc biệt soi các cột thời gian đã bị đánh dấu ở Phase 2 (mục 5).

| Mã | Phép kiểm tra | Cách đọc kết quả (heuristic, người dùng quyết định) |
|---|---|---|
| C1 | Ma trận hiện diện theo lớp (từ Phase 2) | Cột trống hoàn toàn ở nhóm này mà đầy đủ ở nhóm kia → nghi artifact |
| C2 | RF chỉ dùng các cờ `_is_na` | macro-F1 cao hơn xa mức ngẫu nhiên (1/số lớp) → mô hình đang học "cách trích xuất"; liệt kê cột góp nhiều nhất |
| C3 | RF/stump trên **từng đặc trưng riêng lẻ** | Đặc trưng đơn lẻ có macro-F1 > 0,9 → soi ngữ nghĩa |
| C4 | Chia theo nhóm vs chia ngẫu nhiên theo dòng (cùng train pool) | Chênh macro-F1 = mức rò rỉ; báo cáo con số này (kết quả đáng đưa vào báo cáo) |
| C5 | Phân biệt DoS vs DDoS trong cùng kịch bản (chạy ở mode 11 lớp) | Nếu F1 phân biệt DoS/DDoS thấp (≈ đoán ngẫu nhiên) → khuyến nghị dùng 6 lớp |

**Deliverables:** `results/reports/leakage_report.md` (bảng, hình, khuyến nghị) và cập nhật `feature_decisions.yaml` nếu cần bỏ thêm cột.
**Gate 🛑 G2:** người dùng chốt (a) danh sách cột cuối cùng, (b) `label_mode` (6 hay 11 lớp). Sau đó **chạy lại Phase 3–4** với quyết định mới.

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
**Gate 🛑 G3 (sanity baseline):** B0 và B1 chạy đủ 3 seed; std macro-F1 giữa các seed < 0,02; macro-F1 B0 không ≈ 1,000 ở mọi lớp (nếu có: quay lại Phase 5, nghi rò rỉ). Người dùng xem bảng B0/B1 trước khi tiếp tục.

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
- [ ] Danh sách hạn chế nêu đủ: dữ liệu ở mức packet; **ε ở mức bản ghi, các packet trong stream tương quan**; đặc trưng thời gian phụ thuộc cách xuất và CVAE sinh packet độc lập; tuning không riêng tư; mô phỏng một máy (không đo độ trễ mạng); chuẩn hóa dùng thống kê tập trung; nhãn không được bảo vệ bởi DP; tập test lấy từ ít nhóm (xem A4 nếu đã chạy); các ngưỡng đánh giá là heuristic do người dùng chốt; kết quả trên một dataset.
- [ ] README đủ để người khác chạy lại từ đầu.

---

## Phụ lục — Prompt mở đầu để dán cho agent

> Bạn là kỹ sư ML. Đọc `PP-FedData_Implementation_Spec.md`. Thực hiện **một Phase mỗi lượt**, theo thứ tự, bắt đầu từ Phase 0. Với mỗi Phase: (1) liệt kê kế hoạch ngắn, (2) viết code và test, (3) chạy test và các kiểm tra Gate, (4) tóm tắt kết quả kèm số liệu thật và các file đã tạo. Dừng tại mọi HUMAN GATE (G1–G4) và chờ xác nhận. Không sửa dữ liệu gốc. Không bịa số liệu. Nếu bị chặn, ghi `BLOCKERS.md` và dừng.

---

## Lịch sử thay đổi

- **v1.1:** thêm mục 1.1 (Colab/Drive, chống mất phiên) và 1.2 (ngân sách tính toán); thêm bước 7.4 benchmark; bảng quota 11 lớp và bỏ mode `binary` riêng; kiểm tra đặc trưng thời gian (Phase 2); giữ `stream_id` và tuỳ chọn `max_rows_per_stream`; quy tắc báo cáo ε theo từng client (max_i); checkpoint/resume cho FL và DP; nhãn 🔶 và khối `thresholds` trong config; độ nhạy A4 (đổi nhóm test) và A5; cập nhật Definition of Done.
- **v1.0:** bản đầu tiên.
