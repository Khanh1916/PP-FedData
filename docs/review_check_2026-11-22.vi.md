# Đối chiếu bản review luận văn (22/11/2026) với code và luận văn hiện tại

**Đầu vào:** file review `Review_main_PPFedData_2026-11-22.md`; luận văn `Latex_FinalCapstone/main.pdf`. PDF này có cùng SHA-256 với bản người review dùng (`042aa791…`), commit luận văn `082e9bf`. Code ở commit `a400d7d`; người review dùng `ab0aa76`, và phần khác nhau giữa hai commit chỉ là mục 9.14 (split-check).

**Cách kiểm:** mở đúng dòng code mà review dẫn, đọc đúng trang luận văn (trang PDF = trang in + 15), và tính lại số từ `results/*.json`, `results/summary.csv`, cùng các file `artifacts/*/spec.json` trên máy này. Không chạy lại thí nghiệm nào.

## Tóm tắt

| Mã | Kết luận | Cần làm gì | Số liệu có đổi không |
|---|---|---|---|
| R01 | **Đúng**, nhưng chọn lại bằng validation vẫn ra **cùng 3 route** | Sửa 1 dòng code (chọn theo validation) + sửa chữ trang 86, 99 | Không |
| R02 | **Đúng** | Viết lại công thức SecAgg+ ở trang 70 | Không |
| R03 | **Đúng, bằng chứng còn mạnh hơn review** | Chạy lại t=2 với T_recon=4 (MG chạy nhanh) hoặc bỏ tuyên bố | Không (xem ghi chú) |
| R04 | **Đúng**, có thêm một lỗi nhỏ cùng chỗ | Sửa công thức trang 67, 69 | Không |
| R05 | **Đúng** | Thêm một đoạn giới hạn | Không |
| R06 | **Đúng một phần** | Đóng gói kết quả + viết lại Phụ lục C | Không |
| R07 | **Đúng** | Sửa hằng số và mô tả trang 63 | Không |
| R08 | **Đúng** | Đổi thành "đề xuất / chưa chạy", hoặc chạy thật | Không |
| R09 | **Đúng** (cả 4 ý) | Sửa chữ trang 66, 73–75, 90 | Không |
| R10 | **Đúng** | Sửa chữ trang 20, 79, 83 | Không |
| R11 | **Đúng** | Sửa chữ trang 98, 102 | Không |
| R12 | **Đúng**, có thêm một chỗ trộn số | Dùng cặp số so khớp ở tóm tắt và RQ2 | Không |
| R13 | **Đúng**, nhưng cần kiểm protocol trước khi thêm vào bảng | Thêm nhánh chỉ dùng dữ liệu tổng hợp vào Bảng 4.7 hoặc bàn riêng | Không |
| R14 | **Đúng**, luận văn tự mâu thuẫn | Sửa kết luận trang 26, 116, 126 | Không |
| R15 | **Đúng** | Viết lại vài câu Chương 2 | Không |
| R16 | **Đúng** | Thêm bảng metric → tập dữ liệu | Không |
| R17 | **Đúng**, Bảng 3.15 mâu thuẫn với chính Chương 3 | Sửa Hình 3.1 và Bảng 3.15 | Không |
| R18 | **Đúng** | Chỉnh dàn trang | Không |

**Kết luận chung:** không có mục nào sai. Cả 18 mục đều đúng, nhưng **không mục nào làm thay đổi con số kết quả**. Hầu hết chỉ cần sửa chữ. Ba mục cần làm thêm: R03 (chạy lại hoặc rút tuyên bố), R08 (rút tuyên bố hoặc chạy ablation), R06 (đóng gói).

## Chi tiết từng mục

### R01 Chọn route CVAE bằng điểm test
- **Code:** `src/ppfeddata/interpret_report.py:986` dùng `max(c, key=lambda r: r["tstr_f1"])`, trong đó `tstr_f1` là macro-F1 trên tập test. Bước chọn classifier trong từng route (`scorecard.py:64–74`, `_pick`) thì dùng `val_macro_f1_mean`.
- **Luận văn:** trang 86 (PDF 101) viết "the best valid CVAE route on validation". Trang 72 viết "test split … evaluated once". Trang 99 giữ M1-eps1 với lý do "does worse on the test split".
- **Phát hiện thêm:** mình tính lại bằng điểm validation từ `results/summary.csv`. Kết quả ra **cùng 3 route**:
  - ε=1: M1-eps1 val 0.1927 > M1o-eps1 0.1763 > M3o-eps1 0.1678
  - ε=5: M3o-eps5 val 0.2438 > M1o-eps5 0.2332
  - ε=10: M1o-eps10 val 0.2612 > M3o-eps10 0.2526
- **Sửa:** đổi `tstr_f1` thành điểm validation trong `_pivot_facts`, chạy lại báo cáo. Ghi rõ trong luận văn rằng route được chọn bằng validation và kết quả trùng với lựa chọn cũ. Sửa câu trang 99 để lý do dựa trên validation (0.1927 > 0.1763), không dựa trên test.

### R02 Công thức trọng số SecAgg+
- **Luận văn (trang 70):** p_k = n_k / Σn_j, server không chia lại, cho ra "exact weighted FedAvg".
- **Code và Flower 1.39.0 (đã cài trên máy):**
  - Client: `secaggplus_mod.py:443–463` tính `ratio = num_examples / max_weight`, lượng tử hóa ratio rồi nhân vào tham số.
  - Server: `secaggplus_workflow.py:640–650` chia cho tổng ratio.
  - Docstring ở `secagg.py:104–109` cũng ghi như vậy.
- **Tự mâu thuẫn:** cùng trang đó, T-SA1 lại ghi "deviation ≤ 10⁻⁴", tức là không chính xác tuyệt đối. Sai số đo được là 3.48e-5, cận trên 4.31e-5 (`results/reports/m2_m3_secagg.md`).

### R03 Ngưỡng T_recon = 4 cho t = 2
- **Luận văn:** trang 65 (PDF 80) viết "Evaluating t = 2 therefore requires T_recon = 4". Trang 73 và 74 cũng ghi "t = 2 with T_recon = 4".
- **Bằng chứng mạnh hơn review:** mình quét các file `artifacts/*/spec.json`. **Cả 12 lần chạy có `honest_t = 2` đều có `reconstruction_threshold: 3`.** Như vậy t=2 đã được chạy với ngưỡng 3, không phải 4. Code không có chỗ nào đổi ngưỡng theo t (`privacy_units.py:98`, `mg_app.py:285–290`, `secagg.py:44`).
- **Số đúng:** MGr-t2-eps5 và MGs-t2-eps5 có ε khi chỉ còn 1 gateway trung thực là **7.63** (scorecard, Bảng B.1).
- **Ghi chú:** với route marginal, SecAgg+ cho tổng nguyên chính xác. Nếu chạy lại với T_recon=4, bảng đếm và điểm số sẽ **giống hệt**; chỉ có tuyên bố an ninh được chứng minh thêm. MG chạy mỗi lần khoảng 1 giây cộng thời gian Flower, nên chi phí rẻ. Lưu ý: ngưỡng 4 trên 5 share chỉ chịu được 1 client rớt mạng.

### R04 Tỉ lệ lấy mẫu DP-SGD
- **Luận văn (trang 67, Eq. 3.17):** q_k = B_k/n_k với B_k = min(B, n_k), và B_k được dùng làm mẫu số.
- **Code:** `fl/dp_utils.py:31–36` dùng `q = 1/ceil(n/B)`. Opacus 1.6.0 (`privacy_engine.py:440–441`) cũng dùng `1/len(loader)` và expected batch `int(n·q)`.
- **Ví dụ của review đúng:** n=3666, B=256 cho q=1/15=0.0667 (luận văn ra 0.0698); expected batch là 244.
- **Lỗi nhỏ thêm, cùng đoạn:** trang 67 viết "clients with very few rows (n_k ≤ 500 ≤ B_batch) have q_k = 1". Config dùng batch 256 < 500, nên client có 500 dòng có q = 1/2, không phải 1. Cần kiểm lại batch của các cấu hình đã tune trước khi sửa câu này.

### R05 Nhiễu tạo từ seed công khai
- **Code:**
  - `fl/mg_app.py:76` tạo nhiễu từ `default_rng([seed, 911, stage, pid])`.
  - `fl/core.py:98–107` seed generator nhiễu của Opacus; dòng 104 còn tắt cảnh báo "Secure RNG turned off".
  - `mg_app.py:104` lưu vector có nhiễu của từng client ra `client{pid}_{stage}.npy`, và dòng 235 đọc lại các file đó.
- **Luận văn:** trang 80 (PDF 95) nói seed điều khiển cả bộ sinh nhiễu, nhưng không có câu nào phân biệt "kế toán ε lý thuyết" với "phát hành thật cần nhiễu bí mật".
- **Sửa:** thêm một đoạn giới hạn. Câu mẫu của review dùng được.

### R06 Gói tái lập
- **Trên máy này:** cả 41 file trong `results/repro/MANIFEST.json` (nay ở commit `2ef961e`) **khớp hash tuyệt đối**, kể cả `runs.csv` và `summary.csv`.
- **Vấn đề thật:**
  - `.gitignore` loại `results/*.csv`, `data/`, `artifacts/` (chứa `preds/`) và `Latex_FinalCapstone/`. Người clone từ GitHub sẽ không có các file đó.
  - Người review gặp lỗi CRLF vì `core.autocrlf=true` và `.gitattributes` chỉ có quy tắc cho một file.
- **Phụ lục C (trang 135):**
  - Ghi "the code and the result files are in the project repository", nhưng các file CSV thì không có.
  - Không có bước cài đặt, inventory, tiền xử lý, train.
  - Thiếu `mia-marginal`, mà đây là lệnh tạo Bảng 4.15.
- **`analysis/mia_bootstrap.json`:** `make_thesis_figures.py` chỉ đọc file này. Không có script nào trong repo hay thư mục luận văn tạo ra nó.

### R07 Hằng số clip của route marginal
- **Luận văn (trang 63):** record-level dùng `clipping_range = 100,000`, group-level dùng `N + 6σ`, server khôi phục bằng centred lift.
- **Code:** `mg_app.py:39–45` dùng `CLIP = 2^20`, `quantization_range = 2^21`, và dừng chương trình nếu |giá trị| ≥ CLIP (dòng 79). Server lấy trung bình Flower nhân với số client (dòng 230), không làm centred lift.

### R08 Ablation MI và độ ổn định cây
- **Luận văn:** trang 60, 73, 74 mô tả so sánh plug-in MI, Miller–Madow MI, NMI, và Jaccard trên các cạnh của cây.
- **Repo:** không có code hay kết quả nào. Mình grep "miller", "jaccard", "nmi" trong `src`, `tools`, `results`, `notebooks`, `docs` và các script luận văn: không thấy. `git log --all -S` cũng **chưa từng có trong lịch sử**.
- **Sửa:** ghi là ablation đề xuất, chưa chạy. Nếu muốn giữ thì phải chạy thật.

### R09 Chọn cap và luật sàng lọc
- **Cap:** trang 66 viết "target privacy policies, not tuned model parameters". Nhưng `privacy_units.py:69` chọn `max(rows, key=macro_f1_mean)` trên validation, và `configs/best_group_dp.yaml` ghi `rule: "mean of 3 seeds on validation"`.
- **Eq. 3.21:** luận văn ghi ddof=1 và dùng std của baseline. Code `tune_marginal.py:101, 109, 156, 163` dùng `np.std` (ddof=0) và `max(std ứng viên, std baseline)`.
- **Con số 11%:** trang 75 tính dưới giả định các ứng viên độc lập, nhưng các ứng viên dùng chung seed. Review nói đúng rằng đây chỉ là xấp xỉ minh họa.
- **"Trọng số một packet = 1/(6×2000) = 0.00008":** có ở trang 90. Macro-F1 không tuyến tính, nên con số này không chính xác.

### R10 Bootstrap và H1/H2
- **Code:** `eval/compare.py:134–151` lấy mẫu lại các dòng test (hoặc stream), còn 3 run đã fit thì giữ nguyên. Độ dao động giữa seed chỉ vào luật quyết định qua `sigma`.
- **Luận văn:** trang 79 và 83 viết interval "reflects the sampling of the test rows and three training seeds".
- **H1/H2:** trang 20 hứa dùng paired bootstrap. Trang 99 thì kiểm H1/H2 bằng giá trị điểm và std giữa seed.

### R11 Quy chênh lệch cho kiến trúc
- **Trang 98:** viết "the difference comes from the generator". Nhưng ở ε=1, MGl-eps1 dùng RF (0.267) còn M1-eps1 dùng MLP (0.176).
- **Trang 102:** con số 3.76/1.95 = 1.93 so M1 đã tune với B3. Benchmark có so khớp (`results/reports/compute_budget.md`) cho **5.77×**.
- **Đã kiểm:** chênh lệch MGl − CVAE là +0.092, +0.143, +0.131, đúng như review.

### R12 Số liệu truyền thông
- **Đã kiểm:** 191 MB là **M2 (không DP)**; 1.33 MB là **MGs-cap25-eps1** (bảo vệ theo capture); 8.4 MB là M1o-eps5.
- **Ba cặp so khớp:** MG 0.476–0.530 MB, CVAE 13.63–72.37 MB; tỉ lệ 31.8, 26.6, 151.9. Cả ba khớp review.
- **Phát hiện thêm:** tóm tắt và trang 98 viết "+0.092 to +0.191". Ở đây **0.092 là MGl** (DP cục bộ), còn 0.191 là MGs (DP phân tán). Nếu chỉ tính DP phân tán thì khoảng là +0.158 đến +0.191. Đây chính là chỗ review cảnh báo "do not mix".
- **Đơn vị:** `mg_app.py:244–245, 305` cộng byte gửi lên và gửi xuống của cả liên mạng, không phải băng thông của một gateway.

### R13 Thiếu nhánh chỉ dùng dữ liệu tổng hợp ở Bảng 4.7
- **Đã kiểm `results/ids_followup.json` → `item6.summary`:**
  - local 0.3839
  - local+MG-eps10 0.4272, chỉ MG-eps10 **0.4456**
  - local+MG-eps5 0.4256, chỉ MG-eps5 0.4268
- **Cần kiểm thêm trước khi đưa vào cùng bảng:** std của nhánh "only:" rất nhỏ (0.006–0.019), trong khi các nhánh local là 0.043–0.055. Có thể nhánh này được đánh giá khác (một mô hình chung thay vì 15 mô hình client). Mình chưa đọc code của item6 để xác nhận.

### R14 Kết luận "ngang nhau khi không có DP"
- **Luận văn tự mâu thuẫn:** trang 26 (PDF 41) viết "the two routes are level without DP". Nhưng trang 97 viết "0.467 against 0.459 … +0.008 [+0.004, +0.012] passes the rule".
- **Chưa kiểm:** các con số RF +0.0012 và MLP +0.0378.

### R15 Câu tuyệt đối ở Chương 2
Đã tìm thấy đúng các câu review nhắc tới:
- Trang 13: "Blockchain preserves data privacy", "HE … ensuring absolute", "DP is exceptionally lightweight".
- Trang 14: "SecAgg … comprehensively".
- Trang 15: "fundamentally ill-suited", "zero-day attack variants".
- Trang 17: "synthesis in the cloud violates the decentralized privacy principle".

### R16 Tập dữ liệu dùng cho từng metric
- **Luận văn (trang 75):** viết "All scores come from the same locked real test split".
- **Code:** `models/b2.py:91–92` gọi `fidelity_report(Xtr, …)` trên dữ liệu train, và `privacy_report(…, Xtr, …, val…)` trên train và validation.

### R17 Số cột và nhãn MGb
- **Hình 3.1** (`Chapter3.tex:76`) ghi "Standardize 60 columns". Manifest ghi `n_features: 59`, `blocks: 37`, `n_na_flags: 12`.
- **Bảng 3.15** (`Chapter3.tex:939`) ghi MGb chạy qua "SecAgg+ over Z_2^32". Nhưng `Chapter3.tex:151` lại viết "MGd and MGb simulate the distributed sum inside one process". Code `tune_marginal.py:225, 240–244` fit `bn.fit` trong cùng process và ước lượng byte (`bytes_estimated: True`).

### R18 Dàn trang
- **Trang gần trống:** trang in 117 (PDF 132) chỉ có 3 dòng, trang 118 có 6 dòng, trang 124 có 6 dòng.
- **Build log:** `main.log:2463` có cảnh báo "Float too large for page by 12.6pt" trong `Chapter5.tex`.
- **Tiêu đề mẫu:** mục 5.1 vẫn tên "Restate the Research Problem or Objectives".

## Kiểm tra lại phần số của review

| Kiểm | Kết quả |
|---|---|
| 3 cặp so sánh chính (macro-F1, chênh lệch, MB/run, tỉ lệ byte) | Khớp scorecard đến 6 chữ số |
| MIA trên bảng đếm (0.5733 / 0.5390 / 0.5249 / 0.5147) | Khớp. Giá trị 0.5249 ở ε=5 là của **MGr-eps5** (MGs-eps5 là 0.5275); luận văn cũng dùng MGr-eps5 |
| ε khi 1 gateway trung thực của MGr-eps5 = 14.22 | Khớp |
| Kích thước đặc trưng 59 / 37 / 12 | Khớp manifest |

## Chưa kiểm

- Đủ 52 giá trị của Bảng 4.5 (mới kiểm các hàng chính).
- RF +0.0012 / MLP +0.0378 ở R14.
- Khác biệt cấu hình giữa B3 và M1 ở R11.
- Phép tính 11% ở R09.
- Danh mục tài liệu tham khảo.

## Thứ tự sửa đề xuất

1. **Sửa chữ, không cần chạy gì:** R02, R04, R05, R07, R09, R10, R11, R12, R14, R15, R16, R17. Đây là phần lớn công việc.
2. **R01:** sửa 1 dòng code và tạo lại báo cáo; số liệu không đổi.
3. **R03:** quyết định chạy lại t=2 với `reconstruction_threshold: 4` (rẻ, số không đổi) hay rút tuyên bố "chịu 3 gateway thông đồng".
4. **R08:** rút tuyên bố (nhanh) hay chạy ablation thật.
5. **R13:** đọc code item6 trước, rồi thêm nhánh chỉ dùng dữ liệu tổng hợp.
6. **R06:** đóng gói kết quả (Release/Zenodo), viết lại Phụ lục C, thêm script tạo `mia_bootstrap.json`.
7. **R18:** dàn trang, làm sau cùng.

Mỗi thay đổi phương pháp ghi vào cả hai log `docs/` (EN + VI), theo quy ước của dự án.

## Trạng thái sau khi sửa (2026-11-23)

- **Code:** R01 sửa `interpret_report._pivot_facts` (chọn route CVAE theo validation; vẫn ra M1-eps1, M3o-eps5, M1o-eps10). R03 thêm `mg_app.recon_threshold` (T_recon = max(3, K − t + 1)) và chạy lại 12 lần chạy t = 2 với T_recon = 4: macro-F1 giống hệt, chỉ số byte lệch vài chục byte. Thêm 1 test trong `tests/test_marginal.py`. Ghi dòng R.13 trong cả hai log `docs/SPEC_DEVIATIONS*.md`.
- **Luận văn:** sửa chữ cho R02, R04, R05, R07–R17 trong Abstract, Ch1–Ch6 và phụ lục; thêm bảng metric → tập dữ liệu (Bảng 3.x), nhánh chỉ dùng dữ liệu tổng hợp vào Bảng 4.7, dòng "DP-FedSGD recipe, no noise" vào Bảng 4.8; viết lại Phụ lục C theo thứ tự chạy; thêm `scripts/make_mia_bootstrap.py`.
- **Dàn trang (R18):** bỏ các `\clearpage` thừa ở Ch5 và trước mục 4.4 và 2.3, đổi bảng giới hạn Ch5 sang `longtable`, đổi tên Ch4 thành "Experiments and Results" và hai tiêu đề mẫu ở Ch5. Build lại bằng MiKTeX: không còn cảnh báo "Float too large", không có tham chiếu lỗi.
- **Chưa làm:** không đóng gói lại `results/repro/` (lệnh `package` nên chạy sau khi commit, như các lần trước); chưa commit gì.
