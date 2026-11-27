# Hướng dẫn sử dụng PP-FedData

[English](USAGE.md) | **Tiếng Việt**

Từng bước, theo việc bạn muốn làm. [README](../README.vi.md) là tài liệu tra cứu: mọi lệnh kèm đầu ra (mục "Chạy từng Phase"), kết quả và bảng thuật ngữ tên cấu hình (B0 ... M3, MGs, MGr, MGd, MGl). Mọi thời gian và con số dưới đây lấy từ một file của repo hoặc từ một lần chạy, có ghi nguồn ngay cạnh. Chạy mọi lệnh từ thư mục gốc của repo.

## 1. Chọn lối đi

| Tôi muốn... | Cần dữ liệu thô | Thời gian trên máy của nghiên cứu | Mục |
|---|---|---|---|
| đọc kết quả và kết luận | không | không phải chạy gì | 3 |
| xem nghiên cứu trên trình duyệt (demo) | không; trang 2 và 4 cần kết quả chạy | chỉ cài đặt | 4 |
| tìm cấu hình cho triển khai của mình | không | vài giây | 5 |
| sinh gói tin tổng hợp bằng FedDP-Marginal | có | chuẩn bị dữ liệu khoảng 1 giờ, rồi các lần chạy | 6 |
| chạy lại toàn bộ nghiên cứu | có | riêng ma trận CVAE 28,0 phút mỗi seed; các lần tìm kiếm và các vòng tính thêm | 7 |
| đổi K, α, ε, hạn ngạch, seed | tùy bước | | 8 |
| dùng dữ liệu của mình | dữ liệu của mình | | 9 |
| sửa lỗi | | | 10 |

Nguồn của thời gian: mục 5 là một lần chạy khi viết hướng dẫn này (2026-11-27); chuẩn bị dữ liệu theo `data/README.md`; ma trận CVAE theo README, mục "Thời gian chạy tham khảo".

## 2. Cài đặt (một lần)

```bash
git clone https://github.com/Khanh1916/PP-FedData.git
cd PP-FedData
python -m venv .venv                               # tùy chọn; Windows: .venv\Scripts\activate   Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-lock.txt     # đúng phiên bản thư viện của các lần chạy (requirements.txt: bản mới nhất được phép)
python -m pip install -e .
python -m ppfeddata.cli --help                     # 41 lệnh
python -m pytest -q                                # 10-15 phút trên CPU laptop, không cần dữ liệu thô
```

- Python 3.10 trở lên; các lần chạy dùng Python 3.13.5 trên Windows 11, chỉ CPU.
- Không dùng môi trường ảo mà các gói nằm trong user site-packages thì đừng chạy Python với `-I`: chế độ cô lập ẩn chúng đi và `import numpy` báo lỗi.
- PyTorch bản chỉ CPU: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Nếu sẽ commit: `git config core.hooksPath .githooks` bật bước kiểm tra chặn dữ liệu và đầu ra của các lần chạy.

**Dữ liệu thô (lối D và E).** Tải DoS-DDoS-MQTT-IoT từ Kaggle (`alaaalatram/dosddos-mqtt-iot`, khoảng 12 GB CSV) rồi khai báo nơi đặt nó, trong `configs/local.yaml` (không bao giờ commit) hoặc biến môi trường `PPFEDDATA_RAW_ROOT`:

```yaml
paths:
  raw_root: "D:/path/to/DoS-DDoS-MQTT-IoT_Dataset"
```

Bản phát hành chỉ có lưu lượng Normal dạng PCAP. `python -m ppfeddata.cli extract-normal` trích nó theo thiết lập xuất của tác giả vào `NormalData/CSV_Authors_Settings` bên dưới `raw_root` (cần TShark; khoảng 30 phút). Lý do: README, mục "Nguồn gốc dữ liệu".

## 3. Lối A: đọc kết quả

Không cần cài gì. Mở trên GitHub:

| File | Trả lời câu hỏi gì |
|---|---|
| `results/reports/final_report.md` | mọi thứ: 9.0 trả lời ngắn, 9.9b vì sao FedDP-Marginal, 9.10b cấu hình nào cho yêu cầu nào, 10 hạn chế |
| `results/reports/recommend.md` | cấu hình được khuyến nghị cho 11 triển khai IoT mẫu (lab, cloud-aggregator, consortium, ...) |
| `results/reports/recommend.html` | cùng luật đó với bộ lọc do bạn đặt; tải file về và mở bằng trình duyệt (GitHub chỉ hiện mã nguồn) |
| `results/reports/scorecard.md` | mọi cấu hình trên mọi trục: tiện ích, ε, SecAgg, tấn công suy luận thành viên, chi phí |
| `results/figures/` | các hình của báo cáo |
| `results/reports/data_provenance.md` | dữ liệu thô đến từ đâu và đã được kiểm tra thế nào |

## 4. Lối B: demo

```bash
python -m ppfeddata.cli demo                       # mở http://localhost:8501
python -m ppfeddata.cli demo --port 8700 --headless
```

| Trang | Trên bản clone mới | Cần |
|---|---|---|
| 1. Data | chạy được | `results/manifests/` (đã commit) |
| 2. Results | báo cần chạy gì | `results/summary.csv`, do `aggregate` ghi sau các lần chạy (không commit) |
| 3. Trade-offs and recommendation | chạy được | `results/interpretation.json`, `results/scorecard.json`, báo cáo (đã commit) |
| 4. Generate samples | báo cần chạy gì | bộ sinh đã huấn luyện trong `artifacts/` và dữ liệu đã xử lý trong `data/processed/` (lối D hoặc E) |
| 5. Threat model and limitations | chạy được | các file đã commit |

Demo không gửi thống kê sử dụng (`.streamlit/config.toml`). Dừng bằng Ctrl+C.

## 5. Lối C: cấu hình cho một triển khai

`recommend` chỉ đọc `results/scorecard.json`, file đã commit: không cần dữ liệu thô, không phải chạy thí nghiệm. Mô tả triển khai bằng các tùy chọn:

| Tùy chọn | Ý nghĩa |
|---|---|
| `--untrusted-server` | máy chủ tổng hợp không được thấy cập nhật hay bảng của một gateway ở dạng rõ (tổng hợp an toàn hoặc DP cục bộ) |
| `--untrusted-clients` | các gateway có thể thông đồng với máy chủ: DP phân tán khi đó chỉ tính với một gateway trung thực |
| `--honest-clients 3` | DP phân tán phải đúng khi chỉ 3 gateway cộng phần nhiễu của mình (đường cong đã đo) |
| `--max-eps 5` | ngân sách ε ở mức bản ghi; không ghi thì không yêu cầu DP |
| `--max-mb-round 2`, `--max-mb-total 100`, `--max-rounds 3` | băng thông của các gateway |
| `--priority rare` | tối đa cái gì: `macro` (macro-F1 của 6 lớp, mặc định), `binary` (tấn công với bình thường), `rare` (recall của các tấn công hiếm) |
| `--min-rare-recall 0.2` | mức sàn cho recall của các lớp hiếm |
| `--max-mia 0.55` | tấn công suy luận thành viên đã hiệu chỉnh phải có AUC dưới mức này |

Ví dụ: các gateway của một công ty, tổng hợp trên cloud của bên thứ ba, ε ≤ 5:

```bash
python -m ppfeddata.cli recommend --untrusted-server --max-eps 5
```

Một lần chạy khi viết hướng dẫn này (2026-11-27) in ra:

```text
custom: MGd-eps5 (tstr_f1 0.426, epsilon 5); front ['MGd-eps5', 'MGs-t3-eps5', 'MGs-eps2', 'MGr-eps2', 'MGs-t3-eps1', 'MGs-eps1', 'MGs-eps0.5']; command: python -m ppfeddata.cli tune-marginal --final --eps 5
```

Đọc như sau: cấu hình tốt nhất (macro-F1 TSTR trên tập test thật, trung bình các seed), mặt Pareto dưới yêu cầu đó, và lệnh tạo lại nó. MGd là dạng mô phỏng của nhiễu phân tán (Gauss, cộng trong một tiến trình); MGs và MGr gửi cùng các lần phát hành đó qua Flower SecAgg+ (bảng thuật ngữ trong README).
Không có tùy chọn thì `recommend` chạy 11 triển khai mẫu và ghi lại `results/recommend.json`, `results/reports/recommend.md` và `results/reports/recommend.html`; cùng một scorecard thì các file này không đổi. Trang 3 của demo dùng cùng luật.

## 6. Lối D: sinh gói tin tổng hợp bằng FedDP-Marginal

Thiết lập mà nghiên cứu đã chọn trên tập validation đều đã commit (`configs/best_marginal.yaml`, `configs/best_marginal_refine.yaml`, `configs/best_marginal_bn.yaml`), nên không cần tìm kiếm lại.

**Bước 1. Chuẩn bị dữ liệu (Phase 1-4).** Khoảng một giờ trên máy của nghiên cứu, phần lớn là lần quét toàn bộ của `harmonize` (`data/README.md`).

```bash
python -m ppfeddata.cli extract-normal             # một lần: CSV Normal từ PCAP (cần TShark, khoảng 30 phút)
python -m ppfeddata.cli inventory                  # data/inventory/: mọi file thô, lớp và nhóm capture của nó
python -m ppfeddata.cli harmonize                  # schema chung và kiểm toán định dạng (Gate G1; các quyết định đã có trong configs/)
python -m ppfeddata.cli sample                     # chia theo nhóm capture, hạn ngạch lớp -> data/interim/6class/
python -m ppfeddata.cli preprocess                 # mã hoá đặc trưng -> data/processed/6class/
python -m ppfeddata.cli check                      # nên chạy: kiểm tra rò rỉ C1-C5 (Gate G2) -> results/reports/leakage_report.md
```

`sample` cũng ghi lại file đã commit `results/manifests/split_manifest_6class.json`. Xem các mục `parquet_sha256` của nó trong `git diff`: cùng mã băm nghĩa là cách chia của bạn trùng với nghiên cứu (các mục khác, như `git_commit`, đổi sau mỗi lần chạy). Nếu `git status` cho thấy một file đã commit khác bị đổi, đọc diff trước khi đi tiếp.

**Bước 2. Chạy FedDP-Marginal ở một mức ε.**

```bash
python -m ppfeddata.cli tune-marginal --final --eps 5 --seeds 0
```

Với thiết lập đã commit, lệnh này huấn luyện cho seed 0 các biến thể mà nghiên cứu giữ ở ε = 5: MGs-eps5 (nhiễu Skellam chia cho 5 gateway, ba lần phát hành qua Flower SecAgg+), MGr-eps5 (MGs cộng hậu xử lý chọn trên validation: cấu hình của khung ở ε = 5), MGd-eps5 và MGl-eps5 (nhiễu Gauss phân tán và DP cục bộ, mô phỏng trong một tiến trình) và MGb-eps5 (tùy chọn mạng Bayes được chọn ở ε = 5).
Mỗi biến thể được đánh giá như mọi bộ sinh (TSTR, TAug, độ trung thực, quyền riêng tư) và có các dòng trong `results/runs.csv`. Ba lần phát hành qua Flower mất tổng cộng 20 giây mỗi lần chạy (báo cáo cuối, bảng của 9.6); phần đánh giá tính thêm. Bỏ `--seeds 0` thì chạy đủ ba seed của nghiên cứu; `--eps 1 5 10` cho cả ba ngân sách.

Mỗi lần chạy để lại một thư mục trong `artifacts/`, ví dụ `MGr-eps5_0`: `mg_model.pkl` (các bảng có nhiễu đã phát hành), `synthetic.npz` (tập dữ liệu sinh đã được đánh giá, ở dạng mã hoá), `fidelity_privacy.json`, `mg_rounds.jsonl` (số byte và số giây của từng lần phát hành).

**Bước 3. Lấy các dòng ở đơn vị gốc.** Hoặc dùng trang 4 của demo: chọn MG-eps5, một lớp và số dòng mỗi lớp, bấm "Generate samples", tải CSV (đơn vị gốc hoặc đặc trưng đã mã hoá). Hoặc bằng Python, từ thư mục gốc của repo:

```python
from ppfeddata import demo_lib as dl
from ppfeddata.utils import load_config

cfg = load_config(None)                                              # configs/default.yaml + configs/local.yaml
g = next(x for x in dl.generators(cfg) if x.label == "MG-eps5")      # cấu hình của khung ở epsilon 5
s = dl.sample(cfg, g, train_seed=0, classes=None, n_per_class=1000, sample_seed=0)
s.table.to_csv("mg_eps5_seed0.csv", index=False)                    # đơn vị gốc, mỗi đặc trưng một cột, thêm cột `label`
```

Một lần chạy đoạn mã này (2026-11-27) cho 6.000 dòng, 1.000 dòng mỗi lớp, và 26 cột: `label` và 25 đặc trưng. Các dòng là hậu xử lý của bản phát hành có DP, nên lấy thêm dòng không tốn thêm ngân sách riêng tư; cùng mô hình, seed, lớp và số lượng thì cho cùng các dòng. Tối đa 50.000 dòng mỗi lớp mỗi lần gọi.

**Bước 4 (tùy chọn). Đưa các lần chạy vào báo cáo.** `aggregate` biến sổ chạy thành bảng tóm tắt, hình và báo cáo, và ghi đè các bản đã commit trong `results/`. Khi chỉ có một phần các lần chạy, ghi ra chỗ khác:

```bash
python -m ppfeddata.cli aggregate --out-dir preview
```

Với `--out-dir`, hai README không bị đụng tới.

## 7. Lối E: chạy lại toàn bộ nghiên cứu

Thứ tự và các gate là của spec; bảng "Chạy từng Phase" trong README liệt kê mọi lệnh kèm đầu ra.

1. Cài đặt bằng `requirements-lock.txt` và chuẩn bị dữ liệu thô (mục 2).
2. Dữ liệu: `check-extraction` (kiểm tra nguồn gốc, tùy chọn, cần TShark, khoảng 1,5 giờ), rồi mục 6, bước 1. Gate G1 (`harmonize`) và G2 (`check`) là bước duyệt của một người: đọc `results/reports/g1_feature_review.md` và `results/reports/leakage_report.md` trước khi đi tiếp.
3. Baseline (Phase 6, Gate G3): `baseline`, rồi đọc `results/reports/g3_baseline.md`.
4. Ma trận CVAE (Phase 7-11). `configs/best_cvae.yaml` và `configs/best_cvae_dp.yaml` đã commit nên có thể bỏ qua `tune`, `tune-dp` và `verify-dp`. `run --stage trial --dry-run` in ra kế hoạch; `run --stage trial` chạy cả ma trận cho một seed trong không gian làm việc riêng (`artifacts/_trial`, 28,0 phút trên máy của nghiên cứu); đọc `results/reports/g4_trial.md` (Gate G4), rồi mới chạy `run --stage full --approve-g4` (ba seed).
5. `aggregate` (khoảng 75 giây): bảng tóm tắt, hình, báo cáo, `results/interpretation.json`, các khối sinh tự động của hai README.
6. Các vòng sau (tối ưu O0-O4, tiếp theo P1-P2, Q1, R): các lệnh trong bảng của README, theo đúng thứ tự. Phần lớn tự viết báo cáo riêng; lệnh nào có `--help` ghi như vậy thì cần chạy `aggregate` và `scorecard` sau nó. Các file `best_*.yaml` đã commit cho phép bỏ qua các lần tìm kiếm và chỉ chạy các bước `--final`.
7. `scorecard`, `recommend` và `aggregate` lần nữa, rồi `accept` (Definition of Done, D1-D10) và `package --no-update-lock` (gói tái lập `results/repro/`).

Mỗi lần chạy được nhận diện bằng tên và seed: lần chạy đã xong sẽ được bỏ qua khi lặp lại lệnh, và `--no-resume` buộc chạy lại ở những lệnh có tùy chọn này.

## 8. Đổi thiết lập

Mọi tham số nằm trong `configs/default.yaml`; `configs/local.yaml` (không commit) được trộn đè lên, và `PPFEDDATA_RAW_ROOT` thay đường dẫn dữ liệu thô. Để thử giá trị khác mà không lẫn với các lần chạy của nghiên cứu, chép `configs/default.yaml` sang một file mới, đổi những gì cần, trỏ `compute.runs_csv` và `compute.artifacts_dir` (và `paths.work_dir` khi bước chuẩn bị dữ liệu thay đổi) tới chỗ mới, rồi đưa file đó vào bằng `--config`, đặt trước tên lệnh:

```bash
python -m ppfeddata.cli --config configs/my_run.yaml tune-marginal --final --eps 5 --seeds 0
```

Giữ sổ chạy riêng: lần chạy nào có tên và seed đã nằm trong sổ thì bị coi là đã xong và bỏ qua, kể cả khi thiết lập đã đổi.

| Khóa | Mặc định | Đổi cái gì |
|---|---|---|
| `seeds` | `[0, 1, 2]` | seed của mọi lần chạy (phân chia, nhiễu, lấy mẫu, bộ phân loại) |
| `paths.raw_root` | `./raw/DoS-DDoS-MQTT-IoT_Dataset` | nơi đặt CSV thô (khai báo trong `configs/local.yaml`) |
| `paths.work_dir` | `./data` | dữ liệu đã xử lý và phần chia cho các client |
| `compute.runs_csv`, `compute.artifacts_dir` | `./results/runs.csv`, `./artifacts` | sổ chạy; mô hình, tập dữ liệu sinh, dự đoán |
| `label_mode` | `6class` | 6 lớp, hoặc 11 lớp tách DoS và DDoS (chỉ bước chuẩn bị dữ liệu và `robustness`) |
| `split.split_seed` | `0` | nhóm capture nào vào validation và test, dòng nào được lấy mẫu |
| `quota.<class>` | `NORMAL: [60000, 2000, 5000]` ... | số dòng mỗi lớp ở train, validation, test; thứ tự các lớp là thứ tự nhãn |
| `fl.num_clients` | `5` | số gateway K |
| `fl.dirichlet_alpha` | `0.5` | các lớp phân bố lệch thế nào giữa các gateway (nhỏ hơn = lệch hơn) |
| `fl.rounds`, `fl.local_epochs` | `30`, `2` | chỉ đường CVAE; FedDP-Marginal luôn gửi ba lần phát hành |
| `dp.epsilons` | `[1, 5, 10]` | các giá trị ε mà một lệnh chạy khi không ghi `--eps` |
| `dp.delta` | `1.0e-5` | δ của mọi bảo đảm DP |
| `secagg.num_shares`, `secagg.reconstruction_threshold` | `5`, `3` | chia sẻ bí mật của SecAgg+ |
| `generate.target_per_class` | `20000` | số dòng mỗi lớp của mỗi tập dữ liệu sinh được đánh giá |
| `eval.rf.n_estimators`, `eval.bootstrap` | `200`, `1000` | kích thước random forest, số lần lấy mẫu bootstrap của báo cáo |
| `thresholds.mia_auc_max`, `thresholds.eps_max_recommend` | `0.55`, `5.0` | các ngưỡng của quy tắc R6 trong spec |

**Một mức ε khác cho FedDP-Marginal.** Các lần tìm kiếm đã commit phủ ε = 0,5, 1, 2, 5, 10 và 20. Với giá trị khác, tìm kiếm trước rồi mới chạy:

```bash
python -m ppfeddata.cli tune-marginal --eps 3                # lưới 4 số bin x 3 cách chia ngân sách, trung bình 3 seed trên validation
python -m ppfeddata.cli tune-marginal --final --eps 3
```

Các lần tìm kiếm ghi vào `configs/best_marginal.yaml` (và `configs/best_marginal_refine.yaml`, `configs/best_marginal_bn.yaml` khi có `--refine`, `--bn`) bất kể `--config` là gì: một lần tìm kiếm mới làm đổi các file đã commit này và giữ nguyên mục của các mức ε khác. Lần chạy cuối của một mức ε mới cho MGd, MGl và MGs; MGr và MGb chỉ có sau khi `--refine` và `--bn` chọn được gì đó ở mức ε đó. Demo chỉ liệt kê ε = 1, 5 và 10.

**K hoặc α khác.** Nghiên cứu đã chạy FedDP-Marginal với α = 0,1 và 10 và với 10 và 20 gateway qua `robustness`, dùng thiết lập chọn cho 5 gateway và α = 0,5 (`results/reports/robustness.md`). Với giá trị của bạn, đổi `fl.num_clients` hoặc `fl.dirichlet_alpha` trong một bản sao cấu hình có sổ chạy riêng rồi chạy `tune-marginal --final`; phần chia client được ghi vào `data/partitions/`, tên file có α, seed và K.

## 9. Dùng dữ liệu của mình

| Bước | Gắn với DoS-DDoS-MQTT-IoT? |
|---|---|
| `inventory` (Phase 1) | có: đọc `<raw_root>/<scenario folder>/<attack type>/CSV Files/*.csv` và các file Normal `Pcap_Files__NormalData<n>_part<k>.csv` dưới `paths.normal_csv_dir`; thư mục, lớp và số dòng mong đợi lấy từ `configs/label_map.yaml`, và ghi log "VERIFICATION FAILED" khi số dòng lệch |
| `harmonize`, `sample`, `preprocess` (Phase 2-4) | có: danh sách cột của bản xuất TShark (`SHARED_COLUMNS` trong `data/harmonize.py`), bộ lọc dòng TCP / MQTT và các quyết định của Gate G1 (`configs/feature_decisions.yaml`, `harmonize.user_overrides` trong `configs/default.yaml`) |
| bộ sinh, đánh giá, báo cáo (từ Phase 6) | không: chỉ đọc `data/processed/<mode>/` và cấu hình |

1. **Dữ liệu cùng loại** (CSV xuất từ TShark của lưu lượng MQTT với các cột này): dùng cùng cấu trúc thư mục, sửa `configs/label_map.yaml` (thư mục, lớp, số dòng mong đợi) và `quota` (mỗi lớp một mục), rồi chạy Phase 1-5 và đọc báo cáo của Gate G1 và G2 trước khi đi tiếp.
2. **Cột khác hoặc lĩnh vực khác**: hoặc sửa phần xử lý cột của Phase 2-4 (`SHARED_COLUMNS` và các bộ phân tích trong `data/harmonize.py` và `data/preprocess.py`), hoặc tự ghi `data/processed/<mode>/` theo định dạng của `preprocess`:
   - `train.npz`, `val.npz`, `test.npz` với `X` (đặc trưng đã mã hoá), `y` (chỉ số lớp), `X_diag`, `group_id`, `stream_id`, `class11`;
   - `feature_schema.json` với `blocks` (mỗi khối có `name`, `type` thuộc numeric / binary / na_flag / categorical, `column`, `start`, `width`), `n_features`, `label_map` và bản ghi `fit` (split `train` và số dòng của nó: các lệnh từ chối dữ liệu mà bộ tiền xử lý không được fit chỉ trên train);
   - `label_map.json`, và `preprocessor.joblib` để có các dòng ở đơn vị gốc.

   Nghiên cứu chưa thử cách này. Thêm bộ sinh mới: [ARCHITECTURE.vi.md](ARCHITECTURE.vi.md), mục 10.

## 10. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `import numpy` báo lỗi dù đã cài các gói | Python được chạy với `-I`, chế độ này ẩn user site-packages: chạy không có `-I`, hoặc cài vào môi trường ảo |
| một lần chạy Flower không tiến triển, CPU rảnh (Windows) | một worker của Ray bị treo: dừng lệnh bằng Ctrl+C rồi chạy lại; các lần chạy đã xong được bỏ qua và lần chạy Flower tiếp tục từ checkpoint gần nhất (mỗi 5 vòng) |
| một lần chạy Flower chết giữa chừng | vòng nào thiếu phản hồi của một client thì dừng ngay và lần chạy khởi động lại từ checkpoint tối đa hai lần; vẫn lỗi thì chạy lại lệnh |
| một lần chạy dài bị dừng sau khi chuyển xuống chạy nền | chạy các lệnh dài trong terminal riêng ngay từ đầu; chuyển một tác vụ đang chạy xuống nền từng làm chết các worker của Ray |
| thiếu `ray` trên Windows với Python 3.13 | `flwr[simulation]` không cài nó trên nền tảng đó; cài `requirements.txt`, file này có `ray` |
| `check-extraction` hoặc `extract-normal` không khởi động được | cả hai cần TShark: cài Wireshark và đưa `tshark` vào PATH |
| `inventory` ghi log "VERIFICATION FAILED" | các file thô khác bản phát hành, hoặc `paths.normal_csv_dir` trỏ tới một lần trích khác: so với `results/reports/data_provenance.md` |
| không tìm thấy file sau khi chuyển chỗ dữ liệu thô | sửa `paths.raw_root` và chạy lại `inventory`; khi đó `sample` đọc lại các file thô một lần nữa |
| `sample` dừng: "classes below 1000 rows ... stop and ask the user" | một lớp ở validation hoặc test nhỏ hơn `split.min_eval_rows`: phải đổi hạn ngạch hoặc cách chia, việc này do một người quyết định |
| `sample` dừng vì bộ nhớ đỉnh | lần chạy dùng quá `split.max_peak_gb` (4,0 GB) |
| "feature_schema.json has no `fit` record" hoặc "not fitted on the train split alone" | chạy lại `preprocess` |
| `KeyError: 'eps3'` từ `tune-marginal --final` | chưa có thiết lập nào được chọn ở mức ε đó: chạy `tune-marginal --eps 3` trước |
| một lệnh chỉ ghi log "skip ..." | lần chạy đã có trong `results/runs.csv`: dùng `--no-resume` ở lệnh có tùy chọn này, hoặc một `compute.runs_csv` khác |
| `run --stage full` từ chối chạy | cần `--approve-g4`, chỉ đưa vào sau khi đã đọc `results/reports/g4_trial.md` |
| trang 2 hoặc 4 của demo hiện lời nhắc thay vì dữ liệu | chúng cần `results/summary.csv` và `artifacts/` (mục 4) |
| `accept` thoát với mã 1 | một dòng của `results/reports/dod_checklist.md` không đạt: cột bằng chứng của nó ghi thiếu gì |
| `git commit` bị hook pre-commit từ chối | hook cố ý chặn dữ liệu, đầu ra của các lần chạy và file trên 5 MB: đừng đưa chúng vào commit |

Log của mỗi lần chạy Flower là `fl.log` trong thư mục của nó dưới `artifacts/`.
