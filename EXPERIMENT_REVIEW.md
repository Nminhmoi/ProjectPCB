# TỔNG HỢP THỰC NGHIỆM & HƯỚNG TỐI ƯU

> Tài liệu dành cho cả nhóm: tóm tắt kết quả 3 script thực nghiệm đã chạy full **693 cặp** PKU-Market-PCB, các nhận xét rút ra, và danh sách việc cần tối ưu kèm chỉ tiêu nghiệm thu.
>
> Cập nhật: 15/11/2026. Số liệu lấy từ chính các file CSV do script sinh ra (không chép tay).

---

## 1. Phạm vi & cách tái lập

| Hạng mục | Thông tin |
| :--- | :--- |
| Dữ liệu | 693 cặp template–test, 6 lớp lỗi, 2.953 bbox GT (`data/PKU-Market-PCB/`) |
| Môi trường | `py -3.14`, OpenCV 5.0.0, NumPy, Matplotlib |
| Detector mặc định | ORB, `max_features = 2000`, `keep_percentage = 0.8`, RANSAC 3 px |
| Blur mặc định | Gaussian 3x3 |
| Nắn lệch giả lập | ±3°, ±40 px, ±3% (chỉ khi đo chất lượng căn chỉnh) |
| Ghép bbox với GT | "chạm nhau" với dung sai 5 px (`match_boxes(..., tol=5)`) |

**Lệnh chạy full 693 cặp:**

```powershell
# 1. ORB vs SIFT (có nắn lệch giả lập, ~20–30 phút)
$env:PYTHONIOENCODING='utf-8'
py -3.14 -u experiments/exp_feature_matching.py --output-dir experiments

# 1b. AKAZE (cần opencv-contrib-python trong venv .venv, ~23 phút)
.\.venv\Scripts\python.exe -u experiments\exp_feature_matching.py `
    --detectors AKAZE --visualize 0 --output-dir experiments\akaze

# 2. Khảo sát max_features (6 mức, 1 lần/cặp, ~40 phút)
py -3.14 -u experiments/exp_max_features.py `
    --values 500 1000 2000 3000 5000 10000 --repeats 1 --output-dir experiments

# 3. Khảo sát kernel blur (4 kernel + 3 seed nhiễu, ~65 phút)
py -3.14 -u experiments/exp_blur_kernel.py --output-dir experiments
```

**File kết quả:**

| Script | File sinh ra |
| :--- | :--- |
| `exp_feature_matching.py` | `experiments/summary.csv`, `experiments/per_pair.csv`, `experiments/comparison_chart.png`, `experiments/vis/*.jpg` |
| `exp_feature_matching.py --detectors AKAZE` | `experiments/akaze/summary.csv`, `experiments/akaze/per_pair.csv` (chạy trong venv `.venv` vì cần `opencv-contrib-python`) |
| `exp_max_features.py` | `experiments/max_features_results.csv`, `experiments/charts/max_features_study.png` |
| `exp_blur_kernel.py` | `experiments/blur_kernel_results.csv`, `experiments/charts/blur_comparison.png`, `experiments/charts/blur_visual.png` |

> Lưu ý về thời gian: một vài cặp bị nhiễu do máy bị chiếm CPU (có cặp 453 s, 888 s trong thí nghiệm `max_features`). Bảng 2.2 dùng **trung vị** để loại ngoại lai; bảng 2.1 và 2.3 dùng **trung bình** (khớp cột trong `summary.csv` / `blur_kernel_results.csv`).

---

## 2. Kết quả thực nghiệm

### 2.1. So sánh bộ dò đặc trưng (693 cặp, `max_features = 2000`)

| Detector | Align TB (ms) | Keypoints | Matches | Inliers (%) | Reprojection (px) | Corner median (px) | Corner p95 (px) | Cặp > 1 px | Recall | Precision | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ORB** | **232.7** | 2000 | 1402 | 94.7 | 0.661 | 0.280 | 0.934 | 30 | 0.9611 | 0.9839 | 0.9723 |
| **SIFT** | 1485.1 | 1980 | 1458 | 98.2 | **0.085** | **0.035** | **0.089** | 3 | **0.9672** | 0.9833 | **0.9752** |
| **AKAZE** | 2065.9 | 2000 | 1514 | 96.9 | 0.096 | 0.049 | 0.099 | 4 | **0.9672** | 0.9833 | **0.9752** |

- ORB nhanh gấp **6.4 lần** SIFT và **8.9 lần** AKAZE; F1 chỉ thấp hơn 0.003 → giữ ORB làm mặc định.
- SIFT và AKAZE chính xác hơn hẳn về hình học (corner p95 0.089 / 0.099 px so với 0.934 px), nhưng không đổi được F1 cuối.
- AKAZE chậm hơn cả SIFT trên ảnh độ phân giải cao (2066 ms so với 1485 ms) → không có lợi thế tốc độ.
- AKAZE cần `opencv-contrib-python` (OpenCV 5.x đặt ở `cv2.xfeatures2d`); `max_features` với AKAZE chỉ là cắt bớt sau khi detect. Kết quả lưu riêng tại `experiments/akaze/`.

### 2.2. Ảnh hưởng của `max_features` (693 cặp)

| `max_features` | Keypoints | Matches | Inliers | Reprojection (px) | Corner median (px) | Corner p95 (px) | Cặp > 1 px | Thời gian (ms) | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 500 | 500 | 351 | 268 | 0.657 | 0.410 | 1.234 | 64 | **90.7** | 0.9641 |
| 1000 | 1000 | 706 | 537 | 0.661 | 0.373 | 1.006 | 36 | 94.4 | 0.9666 |
| **2000 (đang dùng)** | 2000 | 1402 | 1063 | 0.657 | 0.280 | 0.934 | 30 | 107.4 | 0.9668 |
| 3000 | 2984 | 2079 | 1572 | 0.661 | 0.266 | 0.847 | 19 | 124.5 | 0.9676 |
| 5000 | 4847 | 3354 | 2530 | 0.659 | 0.243 | 0.814 | 20 | 174.2 | 0.9691 |
| 10000 | 9432 | 6403 | 4817 | 0.667 | 0.239 | 0.820 | 17 | 377.3 | 0.9699 |

- **Reprojection gần như không đổi (~0.66 px)** ở mọi mức → không dùng chỉ số này để chọn tham số.
- 500 quá thấp (64 cặp > 1 px). Từ 3000 trở lên lợi ích bão hòa; 2000 là điểm cân bằng (107 ms, corner 0.280 px).

### 2.3. Kernel Gaussian blur (693 cặp; 1 ảnh gốc + 3 seed nhiễu σ = 6.0 mỗi cặp)

| Kernel | Recall±sd (gốc) | F1±sd (gốc) | Recall nét mảnh | F1±sd (nhiễu) | SNR (nhiễu) | Blur TB (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1x1 (không blur) | 0.950±0.168 | 0.967±0.118 | 0.940 | 0.948±0.147 | 1.71 | 3.89 |
| **3x3 (đang dùng)** | 0.967±0.144 | 0.975±0.106 | 0.964 | 0.980±0.102 | 2.89 | **3.07** |
| 5x5 | 0.970±0.142 | 0.976±0.106 | 0.966 | 0.981±0.101 | 3.15 | 3.80 |
| 7x7 | 0.967±0.149 | 0.975±0.110 | 0.964 | 0.982±0.101 | 3.31 | 4.59 |

- 3x3 / 5x5 / 7x7 chênh nhau 0.001–0.002 F1 (nằm trong độ lệch chuẩn) → **tương đương thống kê**.
- 6x6 không dùng được: `cv2.GaussianBlur` từ chối kernel chẵn, và kernel chẵn làm ảnh trôi 0.50 px mỗi chiều.
- Chọn 3x3 vì nhanh nhất + giữ tương phản nét 2 px tốt nhất về lý thuyết (75% so với 62% / 50%), **không phải vì F1 cao hơn**.

### 2.4. Chất lượng phát hiện lỗi theo lớp (ORB, 693 cặp)

| Lớp lỗi | Số GT | Recall | Precision | F1 |
| :--- | :---: | :---: | :---: | :---: |
| Missing_hole | 497 | 1.000 | 0.996 | 0.998 |
| Mouse_bite | 492 | 0.961 | 0.974 | 0.967 |
| Open_circuit | 482 | 0.981 | 0.986 | 0.984 |
| Short | 491 | 0.974 | 0.982 | 0.978 |
| **Spur** | 488 | **0.928** | 0.979 | 0.953 |
| **Spurious_copper** | 503 | **0.922** | 0.986 | 0.953 |

- 2 lớp yếu nhất đều là lỗi **mảnh, nhỏ** → nghi bước morphology opening 3x3 ăn mất nét.
- 46/693 cặp có ít nhất 1 blob không khớp bbox GT nào (48 blob); có thể là lỗi giả hoặc phát hiện trùng một lỗi. Precision tổng 0.984 vẫn tốt.

---

## 3. Nhận xét tổng hợp

1. **Chất lượng đã bão hòa ở F1 ≈ 0.97.** Tăng `max_features` gấp 5 lần chỉ thêm 0.003 F1; đổi kernel blur không đổi F1. Nút thắt không còn nằm ở căn chỉnh/tiền xử lý nữa.
2. **Reprojection error là chỉ số dễ gây hiểu nhầm.** Có cặp corner lớn tới 47 px nhưng reprojection vẫn 0.64 px, vì RANSAC chỉ tối ưu nội bộ tập inlier. **Corner error mới là chỉ số đúng** — nhưng phải bù dịch chuyển có sẵn giữa template và test (`H_pre @ H_true_inv`); 3 cặp 47/28/7 px thực chất là **dịch chuyển có sẵn** của dữ liệu, không phải căn chỉnh sai (xem CODE_REVIEW.md mục 2.2).
3. **ORB là lựa chọn đúng cho thời gian thực**, SIFT/AKAZE là phương án dự phòng khi cần độ chính xác hình học (AKAZE không nhanh hơn SIFT trên ảnh lớn).
4. **Lỗi còn lại mang tính hệ thống**, không phải ngẫu nhiên:
   - 3 cặp từng bị coi là "căn chỉnh sai nặng" thực ra căn chỉnh đúng (corner 0.2–0.5 px sau khi bù dịch chuyển); không có cặp nào lệch > 5 px trên 693 cặp.
   - 7 cặp bỏ sót toàn bộ GT; trong đó 3 cặp căn chỉnh rất tốt (corner 0.20–0.46 px) → lỗi ở ngưỡng/morphology.
   - 2 lớp Spur, Spurious_copper có recall thấp nhất (~0.92).
5. **F1 giữa 3 script chênh 0.003–0.008 do khác điều kiện đo**, không phải mâu thuẫn:
   - `exp_blur_kernel`: căn chỉnh thật, **không** nắn lệch giả lập → F1 0.975.
   - `exp_feature_matching`: có nắn lệch, `max_features = 2000` → F1 0.972.
   - `exp_max_features`: có nắn lệch, phát hiện đo trên ảnh đã căn lại → F1 0.967.
   → Khi trích dẫn báo cáo phải ghi rõ điều kiện đo.
6. **Chỉ số F1 hiện tại dùng ghép bbox "chạm nhau, tol 5 px"** — khá dễ dãi. Muốn thuyết phục hơn nên báo cáo thêm F1 theo IoU.

---

## 4. Điểm cần tối ưu & hướng làm

### P0 — Đúng đắn (làm trước, ảnh hưởng trực tiếp chỉ số)

#### P0.1. Kiểm chứng homography bằng điểm ảnh (NCC)

- **Vấn đề ban đầu:** 3 cặp có corner error lớn (7–47 px) vẫn lọt qua mọi kiểm tra hình học.
- **Đính chính (CODE_REVIEW.md mục 2.2):** 3 cặp đó **không phải căn chỉnh sai** — H của ORB khớp nội dung
  (NCC 0.996–0.999), chúng chỉ có dịch chuyển 7/28/47 px **có sẵn** giữa template và test. Thước đo corner
  cũ (so với `H_true_inv`) mới là cái sai. Sau khi sửa (mốc `H_pre @ H_true_inv`), cả 3 cặp còn 0.2–0.5 px.
  Vì vậy NCC **không thể** là thứ chặn 3 cặp này.
- **Bằng chứng cũ (NCC toàn ảnh, chưa mask):**

  | Cặp | Corner (px) | NCC | MAE |
  | :--- | :---: | :---: | :---: |
  | 06_spur_08 | 47.09 | 0.798 | 2.40 |
  | 04_spurious_copper_11 | 28.08 | 0.813 | 2.31 |
  | 06_spurious_copper_04 | 7.18 | 0.948 | 0.89 |
  | 05_open_circuit_08 | 1.25 | 0.995 | 0.79 |
  | 01_spur_04 | 0.41 | 1.000 | 0.36 |
  | 01_missing_hole_01 | 0.20 | 0.998 | 0.34 |

- **Hướng làm (đã triển khai):**
  1. Sau khi warp, resize ảnh xám template và ảnh aligned về cùng chiều rộng ~400 px.
  2. Chỉ tính trên **vùng hợp lệ** (che phần đen do warp bằng mask đã erode).
  3. Tính hệ số tương quan Pearson (NCC) và MAE.
  4. Thêm `stats["appearance_score"]` vào `align_images_with_stats`; từ chối căn chỉnh nếu NCC dưới ngưỡng.
- **Kết quả đo trên 693 cặp (điều kiện thật):** NCC min **0.9950**, p1 0.9967, median 0.9998; **0 cặp bị từ chối**
  ở ngưỡng 0.97, 0 lần dùng SIFT dự phòng.
- **Chốt lại (sau review):** NCC chỉ là **chốt chặn thảm hoạ**, không phải ngưỡng tinh chỉnh →
  `DEFAULT_MIN_NCC = 0.90`, `DEFAULT_FALLBACKS = ()` (bật SIFT bằng `--fallback SIFT` khi cần).
- **Chi phí thật:** **~38 ms/cặp** (~14% thời gian căn chỉnh), không phải ~5 ms.

#### P0.2. Sửa tiêu chí đề xuất của `exp_max_features.py`

- **Vấn đề:** script xét `corner max ≤ 1 px`, nên chỉ một cặp lệch (dù do dữ liệu) là không mức nào được đề xuất.
- **Hướng làm:** dùng **corner p95 ≤ 1 px** để đề xuất, vẫn in `max` như cảnh báo; thêm cột `time_p95_ms`. Corner đã được đo lại với mốc `H_pre @ H_true_inv` nên không còn cặp 47 px giả tạo.
- **Chỉ tiêu:** script đề xuất được 2000 (hoặc 3000 nếu p95 của 2000 vượt 1 px).

#### P0.3. Tìm nguyên nhân 3 cặp bỏ sót dù căn chỉnh tốt

- **Cặp:** `01_spur_04` (corner 0.41, GT 3, pred 0), `04_spurious_copper_05` (0.46, GT 3, pred 0), `01_mouse_bite_19` (0.20, GT 2, pred 1).
- **Nghi vấn:** sàn ngưỡng `max(Otsu, 30)` + morphology opening 3x3 xóa nét mảnh.
- **Hướng làm:** xem trực tiếp bản đồ vi sai và mask của 3 cặp; thử (a) hạ sàn ngưỡng, (b) opening 2x2 hoặc bỏ opening, (c) giảm `min_area` 20 → 10.
- **Chỉ tiêu:** recall 3 cặp này = 1.0 mà **không** tăng báo động giả quá 10%.

### P1 — Nâng chất lượng phát hiện

#### P1.1. Tăng recall cho Spur & Spurious_copper (0.92 → ≥ 0.96)

- Hai lớp này là lỗi mảnh; hướng làm giống P0.3 nhưng đánh giá trên toàn bộ 693 cặp.
- Theo dõi đồng thời precision để tránh đổi recall lấy nhiễu.
- **Chỉ tiêu:** recall từng lớp ≥ 0.96, F1 toàn bộ ≥ 0.975, số blob giả/ảnh ≤ 0.10.

#### P1.2. Đo F1 chặt hơn bằng IoU

- Bổ sung chế độ ghép bbox theo **IoU ≥ 0.3 và ≥ 0.5** bên cạnh "chạm nhau tol 5 px".
- Báo cáo cả 2 để thấy độ nhạy của chỉ số.
- **Chỉ tiêu:** có bảng F1@IoU0.3 và F1@IoU0.5 trên 693 cặp.

#### P1.3. AKAZE — đã hoàn tất
- Đã cài `opencv-contrib-python==5.0.0.93` trong venv `.venv` (không đụng môi trường chính) và chạy full 693 cặp.
- Kết quả: F1 0.9752, recall 0.9672, corner median 0.049 px, thời gian 2066 ms — chính xác ngang SIFT nhưng chậm hơn.
- Hướng tiếp theo (nếu cần): chỉ dùng AKAZE khi ưu tiên bất biến tỉ lệ và chấp nhận >2 s/cặp; không dùng cho pipeline thời gian thực.

### P2 — Kỹ thuật & báo cáo

| # | Việc | Hướng làm | Chỉ tiêu |
| :---: | :--- | :--- | :--- |
| P2.1 | Số đo thời gian bị nhiễu | Đo lại khi máy rảnh; báo cáo median + p95; tách thời gian detect/match/RANSAC/warp | Sai lệch giữa 2 lần chạy < 10% |
| P2.2 | Thống nhất điều kiện đo F1 | Ghi điều kiện (warp / không warp, detector, max_features, dung sai ghép bbox) ngay trong bảng báo cáo | Không còn 3 số F1 "mâu thuẫn" |
| P2.3 | Docstring thiếu lệnh full 693 | Thêm mục "Chạy full 693 cặp" vào cả 3 script | Chạy lại được chỉ bằng copy-paste |
| P2.4 | File kết quả chưa commit | Commit CSV + biểu đồ; chỉ giữ 1–2 ảnh `vis/` tiêu biểu (thư mục hiện ~2.2 MB) | `git status` sạch sau commit |
| P2.5 | `max_features`/kernel hard-code trong báo cáo cũ | Đã đồng bộ `DEFAULT_MAX_FEATURES = 2000`, kernel 3x3 ở `module_preprocess.py`, `main.py`, README, PROGRESS_REPORT | Không còn chỗ ghi 5000 |

---

## 5. Chỉ tiêu KPI đề xuất cho giai đoạn tiếp theo

> **Cập nhật 17/11:** nhóm chốt lại KPI phát hiện là `recall ≥ 0,95` (ngưỡng sàn) và `đốm giả ≤ 0,05/ảnh` (mục tiêu nội bộ); bỏ mốc F1 ≥ 0,975 và recall lớp yếu ≥ 0,96 khỏi tiêu chí chốt. Bảng dưới giữ nguyên làm bản đề xuất ban đầu; số chốt hiện tại xem `experiments/detection_defaults.md` mục 4 và `PROGRESS_REPORT.md` mục 5.6.

| Chỉ số | Hiện tại | Mục tiêu |
| :--- | :---: | :---: |
| Recall phát hiện lỗi (toàn bộ) | 0.961 | ≥ 0.97 |
| F1 phát hiện lỗi (tol 5 px) | 0.972 | ≥ 0.975 |
| Recall Spur / Spurious_copper | 0.928 / 0.922 | ≥ 0.96 |
| Báo động giả | 0.07 blob/ảnh | ≤ 0.05 blob/ảnh |
| Corner p95 (ORB) | 0.934 px | ≤ 0.85 px (đạt được ở mức 3000–5000 nếu cần) |
| Số cặp lệch > 5 px sau khi bù dịch chuyển có sẵn | 0/693 | 0/693 (NCC 0.90 chỉ là chốt chặn thảm hoạ) |
| Thời gian căn chỉnh ORB 2000 (median, `exp_max_features`) | 107.4 ms | ≤ 150 ms |
| Thời gian toàn pipeline (median, `exp_feature_matching`) | 378.8 ms | ≤ 450 ms |

---

## 6. Thứ tự thực hiện đề xuất

1. **P0.1 NCC** → đã chốt: `DEFAULT_MIN_NCC = 0.90` (chốt chặn thảm hoạ), bỏ SIFT dự phòng khỏi mặc định.
2. **P0.3 + P1.1** → chỉnh ngưỡng/morphology, đo lại recall theo lớp.
3. **P1.2** → thêm IoU vào bộ đo, cập nhật bảng báo cáo.
4. **P0.2, P2.*** → hoàn thiện script, số liệu và repo.

> Nguyên tắc khi tối ưu: mỗi thay đổi chỉ được chấp nhận nếu **không làm giảm precision** và được đo lại trên **cùng 693 cặp** với cùng điều kiện; mọi số liệu phải sinh từ script, không chép tay.

---

## 7. Trạng thái triển khai các điểm tối ưu (15/11/2026)

> Code đã xong và đã chạy thử trên dữ liệu giả lập cùng cấu trúc PKU. Ba mặc định đã đổi theo kết quả đo:
> kiểm chứng NCC (P0.1: 0.97 → 0.90, bỏ SIFT dự phòng) và mask vùng hợp lệ trước khi tính Otsu.
> Hai mặc định nhị phân hóa (floor 30 / open 3 / area 20) **chưa đổi** — chờ chạy full 693 cặp (mục 6.3 CODE_REVIEW.md).

| Mục | Đã làm | Việc còn lại (chạy trên 693 cặp) |
| :--- | :--- | :--- |
| **P0.1** NCC | `module_preprocess.appearance_score()`: NCC cường độ + NCC gradient + MAE ở độ rộng 400 px, có mask vùng hợp lệ, làm mờ 3x3 sau khi thu nhỏ (~38 ms/cặp, đo trên 693 cặp). `align_images_with_stats(min_ncc=0.90)` từ chối khi NCC thấp (`stats["failed_check"] = "appearance"`). `align_images_robust()`: ORB; **mặc định không có dự phòng**, bật SIFT bằng `--fallback SIFT`; nếu không đạt NCC thì chọn H có NCC cao nhất (`verified=False`). `main.py` dùng bản robust (`--fallback`, `--min-ncc`, `--no-verify`). | Đã đo 693 cặp: NCC min 0.9950, 0 cặp bị từ chối → ngưỡng 0.90 là chốt chặn thảm hoạ, không cần chốt lại. Chỉ cần đưa số liệu vào báo cáo. |
| **P0.2** p95 | `exp_max_features.py` đề xuất theo **corner p95 ≤ 1 px**; in thêm c.max (cảnh báo), số cặp > 1 px, NCC, thời gian trung vị + p95, F1@IoU; ghi `max_features_summary.csv`. | Chạy lại lệnh ở mục 1 → script phải đề xuất 2000 (hoặc 3000). |
| **P0.3** 3 cặp sót | `binarize_difference()` có tham số `threshold_floor`, `open_ksize`, `hysteresis_low` (mặc định = pipeline cũ, đã kiểm tra cho kết quả giống hệt). `exp_detection_params.py`: chẩn đoán từng lỗi bị sót mất ở bước **ngưỡng / opening / min_area**, lưu ảnh `experiments/inspect/`. | `py -3.14 experiments/exp_detection_params.py --pairs 01_spur_04 04_spurious_copper_05 01_mouse_bite_19 --inspect` |
| **P1.1** Spur / Spurious_copper | `exp_detection_params.py` quét lưới floor × opening × hysteresis × min_area (48 cấu hình), recall theo lớp, đốm giả/ảnh, đánh dấu ĐẠT KPI; chỉ đề xuất cấu hình **không làm giảm precision** ở mọi điều kiện, hòa thì giữ cấu hình gần gốc nhất. | `py -3.14 -u experiments/exp_detection_params.py --output-dir experiments` → nếu đề xuất khác gốc, sửa các hằng `DEFAULT_*` trong `module_preprocess.py`. |
| **P1.2** IoU | `evaluation.py` (dùng chung): ghép "chạm nhau tol 5" + ghép 1-1 theo IoU ≥ 0.3 / 0.5. F1@IoU có trong `exp_feature_matching`, `exp_max_features`, `exp_detection_params`. | Đưa cột `f1_iou30`, `f1_iou50` vào bảng báo cáo. |
| **P2.2** Điều kiện đo | `summary.csv` của `exp_feature_matching` và `max_features_summary.csv` có các cột `cond_*` (warp, max_features, NCC, tiêu chí ghép bbox). | — |
| **P2.3** Lệnh full 693 | Có trong docstring của cả 5 script thực nghiệm. | — |
| P2.1, P2.4 | Không làm được bằng code (cần máy rảnh / commit kết quả). | Đo lại thời gian khi máy rảnh; commit CSV + biểu đồ mới. |

### 7.1. Phát hiện mới cần đưa vào báo cáo

1. **Recall tụt trên ảnh warp là do dải biên đen, không phải nội suy.** Bản đầu của `exp_max_features.py` đo phát hiện trên ảnh đã nắn lệch rồi căn lại và cho recall chỉ **~0.23**. Kiểm chứng lại: Otsu tính trên toàn ảnh bị dải biên đen (rộng bằng độ dịch) chi phối, đẩy ngưỡng từ 2–31 lên 68–85 và xoá sạch lỗi mảnh. Khi chỉ lấy mẫu Otsu trong **vùng hợp lệ** (48 cặp, 8 cặp/lớp), recall 0.375 → **0.875**, precision 0.071 → 0.985, F1 0.119 → 0.927; ảnh gốc không đổi (0/48 cặp thay đổi kết quả). `exp_detection_params.py` vì vậy đo ở **2 điều kiện**: `clean` (ảnh gốc) và `resample` (ảnh đã nội suy), cả hai đều dùng mask trước khi tính ngưỡng.
2. **F1@IoU0.5 gần 0** trên dữ liệu thử: bbox của pipeline chỉ khoanh lõi vi sai, nhỏ hơn nhãn GT nhiều. Khi báo cáo F1 phải ghi rõ tiêu chí ghép bbox; muốn đạt F1@IoU cao cần thêm bước gộp đốm / nới bbox (chưa làm).
3. **Ngưỡng NCC 0.90 là chốt chặn thảm hoạ, không phải ngưỡng tinh chỉnh.** Trên 693 cặp PKU, NCC min 0.9950 (p1 0.9967, median 0.9998) nên ngưỡng 0.97 không chặn được cặp nào và cũng không nên dùng để phân loại "căn chỉnh tốt/xấu" — 3 cặp từng bị coi là ngoại lai thực ra có NCC 0.996–0.999. Ngưỡng 0.90 giữ biên an toàn ~0.09 dưới mức thấp nhất đo được, chỉ để bắt trường hợp H khớp nham sang hoạ văn lặp lại. Đo độ nhạy trên ảnh thử: lệch 4 px 0.941, 7 px 0.857 — dưới 0.90 là lệch ≥ ~6 px, đã ngoài dung sai ±2 px của bước trừ vi sai.
