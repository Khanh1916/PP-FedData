# Nhật ký lệch so với PP-FedData_Implementation_Spec.md (v1.1)

File này ghi mọi chỗ code hoặc quyết định thực tế **khác** spec v1.1, kèm bằng chứng. Mục đích: sửa spec một lần
thành **v1.2 tại G2** mà không phải nhớ lại. Mỗi dòng mới thêm vào cuối bảng của Phase tương ứng. Số liệu lấy từ
`data/inventory/*.csv`, `configs/feature_decisions.yaml`, `data/interim/6class/split_manifest.json`.

> **Cập nhật 2026-10-03 (G2):** mọi dòng trạng thái `ĐÃ DUYỆT`/`TỰ QUYẾT` bên dưới đã được gộp vào spec v1.2 (xem mục Lịch sử thay đổi trong `PP-FedData_Implementation_Spec.md`). File này giữ làm nhật ký bằng chứng; không còn dòng `MỞ` nào (3.5, 3.6, 4.10, 4.11, 5.4, 5.5 đã được chốt hoặc xử lý ngày 2026-10-03; còn lại một việc theo dõi: bản khóa `pip freeze` ở Phase 7-8 và test OTHER/NONE ở Phase 7).

Trạng thái: `ĐÃ DUYỆT` = người dùng đồng ý (G1 hoặc trong hội thoại), `TỰ QUYẾT` = agent chọn, cần người dùng xác nhận
khi xem v1.2, `MỞ` = chưa chốt.

## Phase 2 - Hài hòa schema, kiểm toán định dạng

| # | Spec v1.1 nói | Thực tế / thay đổi | Bằng chứng | Trạng thái |
|---|---|---|---|---|
| 2.1 | Sinh `feature_decisions.yaml` bằng luật mặc định | Cờ SUSPECT được **sinh từ số liệu**, không viết cứng; thêm kiểm toán token (`token_audit.csv`), cặp cột trùng (`duplicate_pairs.csv`), kiểm tra thứ tự cột (`header_check.csv`) | Header: 522 file, 0 lệch thứ tự | ĐÃ DUYỆT |
| 2.2 | `requested_qos` bỏ nếu trống >99,9% mọi lớp | Điều kiện **không đúng**: INVALID_DoS có 16,6% giá trị. Giữ cột; Normal chỉ có 40 packet SUBSCRIBE | `presence_matrix.csv`, `token_audit.csv` | ĐÃ DUYỆT |
| 2.3 | `_1` và `_2` của cột trùng tên đều là đặc trưng | `frame_length_on_wire_1`, `clean_session_flag_1` là bản sao của `_2` (bằng nhau 14.044.724/14.044.724 và 2.028.233/2.028.233 dòng Attack); Normal để trống `_1`. Bỏ `_1`, giữ `_2` | `duplicate_pairs.csv` | ĐÃ DUYỆT |
| 2.4 | `multi_value: {message_type: first+count, numeric_multi: sum}` | **`first_only`**, bỏ `n_mqtt_msgs`. Script trích xuất Normal dùng `-E occurrence=f`, nên Normal có 0% ô nhiều giá trị, Attack có 1,58% (`message_type`) | `format_audit.csv` | ĐÃ DUYỆT (G1) |
| 2.5 | (không nhắc) | `message_type`, `qos_level_1/2`, `requested_qos`: Normal dùng **mã số**, Attack dùng **nhãn chữ** Wireshark. Cần bảng ánh xạ `VALUE_MAPS`/`canonical_token`; 27 token `message_type` không ánh xạ được (1,19e-06) | `token_audit.csv` | ĐÃ DUYỆT |
| 2.6 | (không nhắc) | Cờ nhị phân: Normal `True/False`, Attack `Set/Not set`; parser ánh xạ cả hai về 0/1 | `format_audit.csv` | ĐÃ DUYỆT |
| 2.7 | SUSPECT: cột trống lệch >0,95 hoặc định dạng khác hẳn | Thêm quy tắc "sự kiện cha" (`PARENT_EVENT`): cột con trống ở Normal vì Normal không dùng chức năng đó thì **không** là SUSPECT (`will_message_length`, `will_topic_length`, `requested_qos`); thêm ngưỡng đếm tối thiểu `token_min_count=100`; chỉ cờ giá trị **chỉ có ở Normal** (giá trị chỉ có ở Attack là hành vi tấn công) | `will_flag` hiện diện 0,075 dòng Normal, `will_flag` Normal 3.427.904 `False` | ĐÃ DUYỆT (G1) |
| 2.8 | Đặc trưng thời gian: so phân vị, cột lệch → SUSPECT | Luật dựa trên tỉ lệ p25/p50/p75/p99 giữa hai nguồn, ngưỡng 10. `time_since_first_frame_in_this_tcp_stream` bị cờ (p99 ×9883); xuất ở dạng cột **`diagnostic`** (loại khỏi bộ chính, Phase 4 vẫn xuất để C3 đo) | `time_feature_audit.csv` | ĐÃ DUYỆT (G1), quyết định cuối ở G2 |
| 2.9 | (không nhắc) | `protocol` của Normal có 14 giao thức không thấy ở Attack (STP, LOOP, SSDP, CDP...), 0,476% token, có ở cả 49 capture. **Bộ lọc `protocol ∈ {TCP, MQTT}` cho cả Normal và Attack** (Phase 3) | `token_audit.csv`, đo trên 49 capture | ĐÃ DUYỆT (G1) |
| 2.10 | (không nhắc) | Quyết định của người dùng lưu ở `harmonize.user_overrides`/`row_filters`/`g1_log` trong `default.yaml`; chạy lại `harmonize` không ghi đè | `configs/default.yaml` | ĐÃ DUYỆT |

## Phase 3 - Chia theo nhóm và lấy mẫu

| # | Spec v1.1 nói | Thực tế / thay đổi | Bằng chứng | Trạng thái |
|---|---|---|---|---|
| 3.1 | Reservoir sampling | Chọn trước chỉ số dòng ngẫu nhiên không hoàn lại (cùng phân phối, một lượt đọc, tất định); khả thi vì số dòng từng file đã có từ Phase 1 | `split_sample.py` | TỰ QUYẾT |
| 3.2 | Xuất `data/interim/{train,val,test}.parquet` | Xuất `data/interim/<label_mode>/...` để 6 lớp và 11 lớp không ghi đè nhau | `split_manifest.json` | TỰ QUYẾT |
| 3.3 | Block-split: chia khối, bỏ `gap_rows` dòng đệm | **Thêm** bỏ các stream TCP vắt qua nhiều khối (`split.drop_cross_block_streams`). Không có bước này: 27 stream (1.324/130.500 dòng) nằm ở nhiều split. Cái giá: loại 1,2%-9,9% dòng đủ điều kiện mỗi lớp con (SYN_DDoS: 49.375 dòng, 9,92%) | `split_manifest.json` (`cross_block`) | TỰ QUYẾT, cần ghi vào hạn chế |
| 3.4 | "Không nhóm nào ở hơn một split" | Với lớp con block-split, `group_id` = `<file>#blkNN`; file gốc ở `source_file`; thêm assert "không stream nào ở hơn một split" | manifest: `streams_in_multiple_splits = {}` | TỰ QUYẾT |
| 3.5 | Hạn chế phải nêu | Thêm: 8/11 lớp con chỉ có 1 file capture → test/val của chúng là khối liên tiếp trong cùng 1 capture (không phải capture khác); lọc `protocol` loại 128 dòng RIPv2 của Attack (SYN_DDoS) | `group_counts.csv`, manifest (đã vào danh sách hạn chế của spec v1.2) | ĐÃ DUYỆT |
| 3.6 | Thư viện ghim sau khi cài (mục 0.7) | Đã `pip install` thêm: pyarrow 25.0.1, psutil 7.2.2, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0. `numpy` thực tế 2.3.2 (pandas 2.3.1) trong khi `requirements.txt` ghim `numpy<2.0`: môi trường đã test khác môi trường khai báo. `torch`, `flwr`, `opacus` chưa cài | `pip` Xử lý 2026-10-03: dry-run cài cùng lúc torch 2.14.1, flwr 1.39.0, opacus 1.6.0, numpy 2.5.3, scikit-learn 1.9.1, pyarrow 25.0.1 không xung đột nên `requirements.txt` đổi thành `numpy>=1.26,<3`. Bản khóa `pip freeze` làm ở Phase 7-8 sau khi cài thật (dry-run không chứng minh API Flower/Opacus chạy đúng). | ĐÃ XỬ LÝ |
| 3.7 | `split_manifest.json` ở `data/` (không commit) | Bản sao chia sẻ được ghi thêm vào `results/manifests/split_manifest_<mode>.json` (git theo dõi): chỉ có mã nhóm, số lượng, hash, không có dòng dữ liệu hay đường dẫn máy | `paths.shared_manifest_dir` | TỰ QUYẾT |

| 3.8 | `raw_root` đặt thẳng trong `default.yaml` (spec mục 3 dùng đường dẫn máy cá nhân) | `default.yaml` chỉ có đường dẫn giữ chỗ; đường dẫn thật ở `configs/local.yaml` (git bỏ qua, merge sâu) hoặc biến môi trường `PPFEDDATA_RAW_ROOT`. Ví dụ đường dẫn trong chính spec cũng thay bằng giữ chỗ. Lịch sử git từ PR #2 vẫn còn đường dẫn cũ | `utils.load_config` | ĐÃ DUYỆT |
| 3.9 | `config_hash` (mục 0.4) | `config_hash` **bỏ mục `paths`** để cùng một thí nghiệm có cùng hash trên mọi máy | `utils.config_hash` | TỰ QUYẾT |
| 3.10 | (không nhắc) | `sample` từ chối chạy nếu thiếu `feature_decisions.yaml` (trước đó âm thầm bỏ bộ lọc); manifest ghi thêm phiên bản python/numpy/pandas/pyarrow vì hash Parquet phụ thuộc phiên bản | `split_sample.py` | TỰ QUYẾT |

## Phase 4 - Ghi chú đầu vào (rà soát dữ liệu interim 6 lớp, 130.500 dòng)

Chưa phải lệch spec, là điều Phase 4 phải xử lý (số liệu đo trên `data/interim/6class/*.parquet`). **Cả 4 mục đã được xử lý và có test** (`tests/test_preprocess.py`).

| # | Phát hiện | Việc Phase 4 phải làm |
|---|---|---|
| 4.1 | `time_delta_from_previous_displayed_frame` có 2 giá trị âm nhỏ (tối thiểu -1e-6) | Cắt về 0 trước `log1p` và ghi số dòng bị cắt (spec: "cột thời gian không âm") |
| 4.2 | Cột gần hằng số: `password_length` luôn bằng 8 khi có giá trị, `will_topic_length` luôn 15, `user_name_length` chỉ 3-9 | Chuẩn hóa phải xử lý std = 0 (đặt scale = 1), có test |
| 4.3 | Ô nhiều giá trị vẫn còn trong interim (`message_type` 4.071 ô, `topic_length` 3.912 ô, ...) vì Phase 3 giữ giá trị thô | `parse_multi` lấy phần tử đầu (`multi_policy: first_only`) |
| 4.4 | Khảo sát token: 0 token chưa ánh xạ trong mẫu 6 lớp; 11 mã `message_type` | Ô cắt cụt vẫn vào `OTHER` cho các lần chạy khác (mẫu 11 lớp, seed khác) |

## Phase 4 - Lệch spec và kết quả

| # | Spec v1.1 nói | Thực tế / thay đổi | Bằng chứng | Trạng thái |
|---|---|---|---|---|
| 4.5 | Cờ `<col>_is_na` chỉ khi tỉ lệ trống trong (0,5%; 99,5%); điền 0, rồi log1p và chuẩn hóa trên cả cột | Cột **siêu thưa** (trống ≥ 99,5% nhưng không phải 100%) dùng thống kê của các dòng áp dụng, dòng trống = 0 sau chuẩn hóa, và **có cờ**. Theo đúng quy tắc spec, cả 90 dòng train của `will_message_length` bị cắt ở +5σ nên hai giá trị 2 và 3164 (khác nhau giữa WILL_DDoS và WILL_DoS) bị gộp làm một. Công tắc: `preprocess.ultra_sparse_fix` (false = đúng spec). Áp dụng cho `will_message_length`, `will_topic_length` (trống 99,9%) | `feature_schema.json` (`ultra_sparse`), `test_ultra_sparse_keeps_distinct_values` | TỰ QUYẾT |
| 4.6 | `multi_value: first+count`, `sum` | `first_only` cho mọi cột (đã ở 2.4); không sinh `n_mqtt_msgs` | `parse_multi.first_values` | ĐÃ DUYỆT (G1) |
| 4.7 | Xuất `data/processed/{train,val,test}.npz` | Xuất `data/processed/<label_mode>/`, kèm `X_diag` (cột `diagnostic`), `stream_id`, `class11`, `feature_schema.json`, `label_map.json`, `preprocessor.joblib` | `preprocess.run_preprocess` | TỰ QUYẾT |
| 4.8 | `feature_schema.json` liệt kê khối numeric/binary/categorical/na-flag | Bố cục cột liền khối: `[numeric][binary][na_flag][categorical]`, mỗi khối có `start`, `width` để đầu ra CVAE tách theo loại (MSE, BCE, CE) | `feature_schema.json` | TỰ QUYẾT |
| 4.9 | `inverse_transform_block` | Thêm `inverse_transform(X)` cho cả bảng: NaN khi cờ = 1 hoặc loại `NONE`. Cột không có cờ (trống < 0,5%) giữ giá trị lấp 0, hiểu là "không áp dụng" | `preprocess.py` | TỰ QUYẾT |
| 4.10 | `log1p` nếu skew > 2 | Đúng spec, nhưng với các cột thời gian giây (≤ 0,6 s) `log1p` gần như là hàm đồng nhất (std = 0,022) nên 918/88.500 dòng train (1,04%) của `time_delta_from_previous_displayed_frame` bị cắt ở 5σ (giá trị > ~0,11 s gộp lại). Chưa sửa | `feature_schema.json` (`audit`) Xử lý 2026-10-03: `preprocess.numeric_scale` nhân `time_delta...` × 1000 trước `log1p`: dòng train bị cắt 918 (1,04%) còn 72 (0,08%), val 11, test 4; macro-F1 val RF cân bằng lớp 0,4978 → 0,4973 (trong nhiễu). `irtt` không nhân vì số dòng bị cắt tăng 126 → 136. | ĐÃ XỬ LÝ |
| 4.11 | one-hot top-K + OTHER + NONE | Đúng spec, nhưng có 6 cột luôn bằng 0 trong train (OTHER/NONE không xuất hiện: `protocol` OTHER và NONE, `qos_level_1/2` OTHER, `requested_qos` OTHER; cộng `will_topic_length` hằng 15 mà cờ đã mang thông tin). Giữ để val/test và dữ liệu sinh không lỗi; Phase 7 có thể che | `feature_schema.json` Chấp nhận: giữ 6 cột; Phase 7 phải có test dữ liệu sinh không chọn các hạng mục OTHER/NONE chưa từng xuất hiện. | ĐÃ DUYỆT |
| 4.12 | (không nhắc) | Mã `message_type` = 0 (Reserved) nằm ngoài top-10 nên thành `OTHER` (đúng 1 dòng train, `train_counts.OTHER = 1`); số dòng `message_type` = NONE (54.752) đúng bằng số dòng `protocol` = TCP (54.752), nên hai cột này dư thừa một phần. Round-trip số đo trên dữ liệu thật: sai lệch tương đối tối đa 3,3e-7 trước khi sửa cột siêu thưa và 1,2e-5 trên bộ cột cuối của G2 | đo trực tiếp | ĐÃ DUYỆT |

## Phase 5 - Kiểm tra rò rỉ (C1-C5)

| # | Spec v1.1 nói | Thực tế / thay đổi | Bằng chứng | Trạng thái |
|---|---|---|---|---|
| 5.1 | C3: RF/stump trên từng đặc trưng, cờ nếu macro-F1 > 0,9 | RF **cân bằng lớp**, `min_samples_leaf=5`, báo thêm **F1 tốt nhất theo lớp**; cờ nếu macro-F1 hoặc F1 theo lớp > 0,9. RF không trọng số sụp về lớp đa số (train 60.000 NORMAL so với 1.000 WILL) nên mọi cột đều trông vô hại, kể cả khi có thể lộ một lớp trên một tập con packet | `leakage_report.md`, `test_column_that_identifies_one_class_only...` | TỰ QUYẾT |
| 5.2 | C4: chênh macro-F1 giữa chia ngẫu nhiên và chia theo nhóm | Thêm chia theo **stream**. Cờ chỉ dựa vào chênh (ngẫu nhiên − nhóm) và (ngẫu nhiên − stream); chênh (ngẫu nhiên − val) chỉ để tham khảo, vì fold CV giữ phân phối lớp mất cân bằng còn val cân bằng. Kiểm chứng: cùng dự đoán, đánh giá trên bản cân bằng lớp cho ngẫu nhiên 0,3998, nhóm 0,3884, val 0,3946 | đo trực tiếp (xem báo cáo C4) | TỰ QUYẾT |
| 5.3 | (không nhắc) | Thêm mô hình tham chiếu cân bằng lớp, ablation `+diagnostic` và `bỏ cột thời gian` (kèm F1 theo lớp) để phục vụ quyết định G2 | `leakage_report.md` | TỰ QUYẾT |
| 5.4 | Ngưỡng ở `thresholds` | Thêm 3 ngưỡng heuristic (cần người dùng chốt): `single_feature_f1_flag=0.9`, `na_only_f1_flag=0.5`, `dos_ddos_f1_flag=0.6`; khối `leakage` (cây, số fold, đường dẫn) | `default.yaml` Xác nhận ở G2. | ĐÃ DUYỆT |
| 5.5 | B0 kỳ vọng không ≈ 1,000 (G3) | Đã thấy sớm: RF toàn bộ đặc trưng đạt macro-F1 val 0,3946 (không trọng số), 0,4420 (cân bằng lớp). Chia ngẫu nhiên theo dòng trong train pool cũng chỉ 0,4756. Không phải rò rỉ; các lớp tấn công khó phân biệt ở mức từng packet | `leakage_report.md` (đã vào danh sách hạn chế của spec v1.2) | ĐÃ DUYỆT |

## Phase 6 - Baseline và khung đánh giá

| # | Spec v1.2 nói | Thực tế / thay đổi | Bằng chứng | Trạng thái |
|---|---|---|---|---|
| 6.1 | B1a: RF `class_weight="balanced"` | B1a chỉ có cho RF (sklearn `MLPClassifier` không có `class_weight`; code từ chối tổ hợp này). Bộ chạy: B0-rf, B0-mlp, B1a-rf, B1b-rf, B1b-mlp | `eval/utility.make_classifier` | TỰ QUYẾT |
| 6.2 | `run_id = {config}_{seed}` | `config` đã gồm bộ phân loại (`B0-rf`, `B1b-mlp`); với chế độ 11 lớp id thêm `11class` để hai chế độ không đè nhau (`B0-rf_11class_0`) | `eval/runs.run_id` | TỰ QUYẾT |
| 6.3 | Đối chứng dương MIA: huấn luyện một CVAE cố ý overfit (500 mẫu) | CVAE chưa có (Phase 7). Phase 6 cung cấp `privacy.positive_control(generator, ...)` kèm test bằng bộ sinh sao chép (AUC > 0,9) và bộ sinh độc lập (≈ 0,5). Phase 7 phải gọi nó với CVAE overfit và báo cáo AUC | `tests/test_eval.py` | MỞ (làm ở Phase 7) |
| 6.4 | Đánh giá trên test thật | Runner từ chối chạy nếu `feature_schema.json` không ghi `fit = {split: train, n_rows}` khớp số dòng train (Phase 4 nay ghi trường này); val chỉ để báo cáo sanity | `baselines.check_fit_on_train` | TỰ QUYẾT |
| 6.5 | `run_experiment.py` ghi `results/runs.csv` (Phase 11) | Sổ chạy đã có từ Phase 6: `eval/runs.py` (`RunLedger`, resume bỏ qua run đã xong, thay thế run cùng id), dự đoán test lưu ở `artifacts/<run_id>/preds/test.npz`; đường dẫn `compute.runs_csv` (Colab trỏ vào Drive) | `eval/runs.py` | TỰ QUYẾT |
| 6.6 | Thư viện | Cài thêm `imbalanced-learn 0.14.2` (có trong `requirements.txt`); `scikit-learn` giữ 1.9.1 | `pip` | ĐÃ DUYỆT |
| 6.7 | W: WGAN-GP tập trung (tuỳ chọn) | Chưa làm (cần `torch`, chưa cài). Bỏ qua trừ khi muốn chứng minh "WGAN-GP nặng hơn CVAE" | | MỞ (tuỳ chọn) |
| 6.8 | Kỳ vọng G3 | 15 lượt chạy (5 cấu hình × 3 seed) xong trong khoảng 8 phút; MLP dừng sớm (n_iter 31-95 < 100). Bảng B0/B1 ở `results/reports/g3_baseline.md` | `results/runs.csv` | |


## Phase 7 - CVAE tập trung (B2), Optuna, benchmark

| # | Spec v1.2 nói | Thực tế / thay đổi | Bằng chứng | Trạng thái |
|---|---|---|---|---|
| 7.1 | Sinh: cột số lấy từ decoder (MSE) | Decoder MSE cho giá trị trung bình, nên dữ liệu sinh co phương sai và rơi giữa các giá trị rời rạc (`calculated_window_size`: 80 giá trị thật, 3.387 giá trị sinh; `keep_alive` 3 so với 357). Thêm **nhiễu Gauss theo độ lệch chuẩn phần dư tái tạo của từng lớp** (ước lượng trên train, `generate.residual_noise: true`): TSTR macro-F1 trên val của mô hình seed 0 từ 0,377 lên 0,415. Thử thêm gán về tập giá trị đã thấy (`snap_support`): val TSTR giảm còn 0,353 và C2ST không đổi, nên tắt (giữ làm tuỳ chọn) | `models/generate.py` (`GenStats`, `fit_gen_stats`), script chẩn đoán trong phiên | TỰ QUYẾT |
| 7.2 | `generate` trả về cột phân loại lấy từ softmax | Loại các hạng mục có 0 dòng train (OTHER/NONE chết) bằng `dead_category_mask`; có test | `tests/test_models.py::test_generation_is_valid_and_counts_exact` | TỰ QUYẾT |
| 7.3 | Optuna 30 trial, fitness TSTR macro-F1 trên val | Chạy hai lần. Lần 1 (không nhiễu): best 0,4098 (mặc định 0,3912), lưu `artifacts/best_cvae_nonoise.yaml`, `artifacts/optuna_cvae_6class_nonoise.db`. Lần 2 sau khi thêm nhiễu (đường sinh dữ liệu cuối): best 0,4141 tại trial 1 (mặc định 0,3653) -> `configs/best_cvae.yaml`. Chênh lệch giữa các trial tốt nhất nhỏ so với nhiễu 1 seed/trial; trial 0 là cấu hình mặc định để làm mốc | `tune.py`, `configs/best_cvae.yaml` | ĐÃ GHI |
| 7.4 | Early stopping theo loss val | Có (`cvae.patience: 5`, chỉ đếm sau giai đoạn warm-up beta), nhưng chạy tối đa 30 epoch (đúng số epoch đã tune) thay vì 50 của config gốc; `best_cvae.yaml` ghi `epochs: 30` | `models/train.py` | TỰ QUYẾT |
| 7.5 | TSTR: `syn_per_class` | Dùng `tune.syn_per_class` = 5000/lớp (cùng giá trị với fitness của Optuna); TAug nâng mỗi lớp lên `generate.target_per_class` = 20000 | `models/b2.py` | TỰ QUYẾT |
| 7.6 | Gate: C2ST < 0,95 | **KHÔNG ĐẠT**: C2ST ≈ 1,000 ở cả 3 seed (RF phân biệt gần hoàn hảo thật/giả). Từng cột đơn lẻ đã đạt 0,82-0,90 (cửa sổ TCP, độ dài frame, thời gian trong stream) vì tần suất giá trị rời rạc và phân bố thời gian không khớp. Dup rate (0,0001-0,0004 < 0,01) và DCR ratio (0,94-0,96 > 0,5) đạt | `results/reports/b2_cvae.md` | MỞ (chờ G4) |
| 7.7 | Đối chứng dương MIA bằng CVAE overfit (mục 6.3) | Làm xong: CVAE overfit trên 498 mẫu (1500 epoch, batch 32, beta 0,01, không DP) cho AUC 0,519, **không phát hiện được**. Thử beta 0, 600-1500 epoch: AUC 0,50-0,51. Đối chứng bằng bộ sao chép trên dữ liệu thật: nhiễu 0 -> 0,977; 0,05σ -> 0,669; 0,2σ -> 0,532. Kết luận: MIA chỉ phát hiện bản sao sát; AUC ≈ 0,5 của B2 chỉ nói không có dòng sinh nào gần như là bản sao của dòng train, không chứng minh không ghi nhớ | `results/reports/b2_cvae.md`, `artifacts/B2_positive_control_6class.json` | ĐÃ GHI (hạn chế phải nêu trong báo cáo cuối) |
| 7.8 | Thư viện | Cài `torch 2.14.1+cpu` (không có CUDA), `optuna 5.0.0`, `opacus 1.6.0`; `numpy 2.3.2` và các thư viện Phase 3-6 không đổi. Mô hình tương thích Opacus (`ModuleValidator` không lỗi, grad_sample đủ cho mọi tham số, có test). `flwr` và `ray` cài ở Phase 8, đã chốt `requirements-lock.txt` (xem 8.1) | `pip` | ĐÃ ĐÓNG ở Phase 8 |
| 7.9 | Benchmark 7.4 | Đo trên CPU (10 luồng): epoch DP-SGD đắt gấp 6,5-9,0 lần epoch thường giữa các lần đo (nhiễu máy; lần đo cuối 9,0); ngoại suy ma trận lõi + Optuna ≈ 3,5-3,8 giờ, không cần cắt giảm. `compute.budget_hours` chưa đặt, SecAgg chưa tính chi phí mật mã | `results/reports/compute_budget.md` | ĐÃ GHI |

## Phase 8 - FL non-IID (B3)

| # | Spec v1.2 nói | Thực tế / thay đổi | Bằng chứng | Trạng thái |
|---|---|---|---|---|
| 8.1 | Thư viện Flower; "chốt `pip freeze`" | `flwr 1.39.0` + `ray 2.59.0` (backend mô phỏng). Flower cảnh báo Ray trên Windows là thử nghiệm nhưng chạy ổn (mọi test và 20+ run đều xong). Khoá phiên bản: `requirements-lock.txt` (torch là bản CPU 2.14.1; numpy vẫn 2.3.2). Template lấy bằng `flwr new @flwrlabs/quickstart-pytorch` (cần `PYTHONUTF8=1` trên Windows) | `requirements-lock.txt` | ĐÃ ĐÓNG |
| 8.2 | "Dùng `flwr new` rồi thay mô hình bằng CVAE" | Dùng API tin nhắn của 1.39 (`ServerApp`, `ClientApp`, `FedAvg.start`) nhưng khởi chạy bằng `run_simulation` (Flower ghi là deprecated, chuyển sang `flwr run`) vì cần điều khiển seed/resume/spec từng run bằng mã. Mỗi run chạy trong một tiến trình riêng (`python -m ppfeddata.fl.run`) | `fl/app.py`, `fl/run.py` | TỰ QUYẾT |
| 8.3 | FedAvg có trọng số theo số mẫu | Đúng; client dùng Adam mới mỗi vòng (không giữ trạng thái), beta warm-up theo epoch toàn cục `(vòng-1)*local_epochs+e`. Phản hồi được gộp theo thứ tự mã client: không cố định thứ tự thì hai lần chạy cùng seed lệch nhau từ vòng 3 (thứ tự cộng float32), sau khi sửa hai lần chạy giống hệt từng bit (trọng số chênh 0,0) | `fl/app.py::RoundStrategy.aggregate_train`, `tests/test_fl.py` | ĐÃ SỬA |
| 8.4 | Đánh giá ELBO mỗi 5 vòng | Đúng (`eval_every=5`, kèm vòng 0 và vòng cuối); nhật ký từng vòng ở `artifacts/<run_id>/rounds.jsonl` (thời gian vòng, thời gian từng client, byte/vòng, RAM đỉnh của server, bộ đếm bước từng client) | `fl/app.py` | |
| 8.5 | Test "1 client tương đương B2 (loss val chênh < 5%)" | 1 client x 30 vòng x 1 epoch: val loss 1,7236 so với B2 1,7862, chênh 3,5 % (đạt). Lưu ý B2 dừng sớm còn FL dùng Adam khởi tạo lại mỗi vòng | `results/reports/b3_fl.md` | ĐÃ GHI |
| 8.6 | Kỳ vọng IID hội tụ nhanh hơn non-IID; nếu ngược thì giải thích | **Ngược**: val loss sau 30 vòng 1,9075 (alpha=100) so với 1,7638 (alpha=0,5). Giải thích đã kiểm: với alpha=0,5 client rất lệch kích thước (4.810 đến 40.106 dòng), client lớn vừa chạy nhiều bước hơn vừa có trọng số lớn hơn, nên mô hình gộp có 228 bước hiệu dụng mỗi vòng so với 140 của IID. Đối chứng IID với 4 epoch cục bộ (281 bước hiệu dụng) cho 1,7669, gần bằng non-IID. Phù hợp với giải thích; không tách riêng được ảnh hưởng của lệch nhãn | `results/reports/b3_fl.md`, `results/figures/fl_convergence.png` | ĐÃ GHI |
| 8.7 | Resume: dừng ở vòng 10, resume tới 20 | Đạt: trọng số chênh 0,0, 20 dòng log, bộ đếm bước từng client bằng nhau. Checkpoint 5 vòng/lần, giữ 2 bản mới nhất, kèm trạng thái RNG và trường `dp_steps` (để trống, Phase 9 sẽ điền) | `tests/test_fl.py`, báo cáo B3 | ĐÃ GHI |
| 8.8 | Thống kê sinh dữ liệu | **`residual_std` (nhiễu theo lớp, 7.1) và hạ tầng sinh dữ liệu B3 được tính từ toàn bộ train pool tại server (mô phỏng)**; trong FL thật đây là thống kê tổng hợp từ client (n, tổng bình phương phần dư theo lớp) và không nằm trong DP-SGD. Phase 9 phải xử lý: hoặc phát hành qua cơ chế DP riêng, hoặc dùng hằng số công khai | `models/generate.py::gen_stats_from_cfg` | MỞ (xử lý ở Phase 9) |
| 8.9 | Dữ liệu client | Chia theo hàng, không theo TCP stream, nên một stream có thể nằm ở nhiều client; kích thước client rất không đều (seed 1: một client giữ 79 % dòng) | `data/partitions/*.json`, báo cáo B3 | ĐÃ GHI (hạn chế) |
| 8.10 | Chi phí | Vòng 1 mất ~27 giây vì khởi động Ray; từ vòng 2 trung vị 1,6-2,4 giây/vòng (5 client chạy song song). Tổng thời gian client tuần tự đo được cao hơn công thức 7.4 khoảng 28 %. RAM đỉnh chỉ tính tiến trình server | `results/reports/b3_fl.md`, `compute_budget.md` | ĐÃ GHI |
