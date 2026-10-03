# Nhật ký lệch so với PP-FedData_Implementation_Spec.md (v1.1)

File này ghi mọi chỗ code hoặc quyết định thực tế **khác** spec v1.1, kèm bằng chứng. Mục đích: sửa spec một lần
thành **v1.2 tại G2** mà không phải nhớ lại. Mỗi dòng mới thêm vào cuối bảng của Phase tương ứng. Số liệu lấy từ
`data/inventory/*.csv`, `configs/feature_decisions.yaml`, `data/interim/6class/split_manifest.json`.

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
| 3.5 | Hạn chế phải nêu | Thêm: 8/11 lớp con chỉ có 1 file capture → test/val của chúng là khối liên tiếp trong cùng 1 capture (không phải capture khác); lọc `protocol` loại 128 dòng RIPv2 của Attack (SYN_DDoS) | `group_counts.csv`, manifest | MỞ (đưa vào danh sách hạn chế ở v1.2) |
| 3.6 | Thư viện ghim sau khi cài (mục 0.7) | Đã `pip install` thêm: pyarrow 25.0.1, psutil 7.2.2, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0. `numpy` thực tế 2.3.2 (pandas 2.3.1) trong khi `requirements.txt` ghim `numpy<2.0`: môi trường đã test khác môi trường khai báo. `torch`, `flwr`, `opacus` chưa cài | `pip` | MỞ (ghim lại khi cài torch/flwr/opacus ở Phase 7-8, vì Flower/Opacus có thể ràng buộc numpy) |
| 3.7 | `split_manifest.json` ở `data/` (không commit) | Bản sao chia sẻ được ghi thêm vào `results/manifests/split_manifest_<mode>.json` (git theo dõi): chỉ có mã nhóm, số lượng, hash, không có dòng dữ liệu hay đường dẫn máy | `paths.shared_manifest_dir` | TỰ QUYẾT |

| 3.8 | `raw_root` đặt thẳng trong `default.yaml` (spec mục 3 dùng đường dẫn máy cá nhân) | `default.yaml` chỉ có đường dẫn giữ chỗ; đường dẫn thật ở `configs/local.yaml` (git bỏ qua, merge sâu) hoặc biến môi trường `PPFEDDATA_RAW_ROOT`. Ví dụ đường dẫn trong chính spec cũng thay bằng giữ chỗ. Lịch sử git từ PR #2 vẫn còn đường dẫn cũ | `utils.load_config` | ĐÃ DUYỆT |
| 3.9 | `config_hash` (mục 0.4) | `config_hash` **bỏ mục `paths`** để cùng một thí nghiệm có cùng hash trên mọi máy | `utils.config_hash` | TỰ QUYẾT |
| 3.10 | (không nhắc) | `sample` từ chối chạy nếu thiếu `feature_decisions.yaml` (trước đó âm thầm bỏ bộ lọc); manifest ghi thêm phiên bản python/numpy/pandas/pyarrow vì hash Parquet phụ thuộc phiên bản | `split_sample.py` | TỰ QUYẾT |

## Phase 4 - Ghi chú đầu vào (rà soát dữ liệu interim 6 lớp, 130.500 dòng)

Chưa phải lệch spec, là điều Phase 4 phải xử lý (số liệu đo trên `data/interim/6class/*.parquet`).

| # | Phát hiện | Việc Phase 4 phải làm |
|---|---|---|
| 4.1 | `time_delta_from_previous_displayed_frame` có 2 giá trị âm nhỏ (tối thiểu -1e-6) | Cắt về 0 trước `log1p` và ghi số dòng bị cắt (spec: "cột thời gian không âm") |
| 4.2 | Cột gần hằng số: `password_length` luôn bằng 8 khi có giá trị, `will_topic_length` luôn 15, `user_name_length` chỉ 3-9 | Chuẩn hóa phải xử lý std = 0 (đặt scale = 1), có test |
| 4.3 | Ô nhiều giá trị vẫn còn trong interim (`message_type` 4.071 ô, `topic_length` 3.912 ô, ...) vì Phase 3 giữ giá trị thô | `parse_multi` lấy phần tử đầu (`multi_policy: first_only`) |
| 4.4 | Khảo sát token: 0 token chưa ánh xạ trong mẫu 6 lớp; 11 mã `message_type` | Ô cắt cụt vẫn vào `OTHER` cho các lần chạy khác (mẫu 11 lớp, seed khác) |

