# CODE REVIEW — Code tối ưu P0.1/P0.2/P0.3/P1.1/P1.2 (PR #6)

> Người review: kiểm tra độc lập sau khi pull commit `7efbf84` (Merge PR #6).
> Ngày: 15/11/2026. Mọi số liệu dưới đây **đo lại trên 693 cặp PKU-Market-PCB** bằng script tạm, không chép từ tài liệu.
> Phạm vi: `module_preprocess.py`, `main.py`, `evaluation.py`, `experiments/exp_alignment_check.py`,
> `experiments/exp_detection_params.py`, diff của `exp_feature_matching.py`, `exp_max_features.py`,
> `exp_blur_kernel.py`, mục 7 mới trong `EXPERIMENT_REVIEW.md`.

---

## 1. Đánh giá tổng quan

Code mới bám sát roadmap: thêm kiểm chứng NCC (P0.1), đổi tiêu chí đề xuất sang p95 (P0.2),
tham số hóa ngưỡng/morphology (P0.3), đo F1@IoU (P1.2), ghi điều kiện đo vào bảng (P2.2).
Chất lượng code tốt, có docstring, có CSV, tái lập được.

**Điểm tốt**

| Hạng mục | Nhận xét |
| :--- | :--- |
| `evaluation.py` | Dùng chung thật sự: `match_touch` giữ nguyên logic cũ nên không làm lệch số liệu đã chốt; `match_iou` ghép 1-1 tham lam theo IoU giảm dần đúng chuẩn detection |
| `appearance_score()` | Cài đặt đúng: resize 400 px, mask vùng hợp lệ + erode, NCC cường độ **và** NCC gradient, MAE, `valid_fraction`; toán học scale homography (`S_t @ H @ S_s⁻¹`) chính xác |
| `exp_max_features.py` | Chuyển tiêu chí sang `corner p95`, vẫn in `max` như cảnh báo, thêm `ncc_rejects`, `time_p95`, cột `cond_*` → giải quyết đúng P0.2 + P2.2 |
| `binarize_difference()` | Tham số hóa nhưng mặc định giữ đúng hành vi cũ (đã đối chiếu floor 30 / open 3 / hysteresis None) |
| `exp_detection_params.py` | Có chẩn đoán "lỗi mất ở bước nào" và đo 2 điều kiện clean/resample — hướng rất đúng |
| `EXPERIMENT_REVIEW.md` §7 | Trung thực: ghi rõ các ngưỡng **phải chốt lại trên 693 cặp**, không nhận là đã xong |

**Kết luận ngắn:** phần code thì tốt, nhưng **tiền đề của P0.1 và cách đo corner error đang sai**, và
có một lỗi thật (mask biên sau warp) đang bị chẩn đoán nhầm thành "nhạy với nội suy ảnh".

---

## 2. Kiểm chứng bằng số liệu

### 2.1. NCC 0.97 không chặn được cặp nào — P0.1 hiện là no-op

Quét full 693 cặp, điều kiện thật (không nắn lệch giả lập), pipeline mặc định mới:

| Chỉ số | Kết quả |
| :--- | :--- |
| Thất bại | **0 / 693** |
| Bị NCC từ chối (`verified=False`) | **0** |
| Dùng detector dự phòng (SIFT) | **0** |
| NCC min · p1 · p5 · median | **0.9950** · 0.9967 · 0.9985 · 0.9998 |
| Số cặp NCC < 0.97 | **0** |
| Chi phí kiểm chứng (`time_verify_ms`) | **38.3 ms/cặp** (mean), max 42.8 ms |

- Ngưỡng `DEFAULT_MIN_NCC = 0.97` **chưa bao giờ chạy** trên dữ liệu này.
- Comment trong `module_preprocess.py` ghi chi phí "~6 ms" → thực tế **38 ms**, tức ~14% thời gian căn chỉnh.
- Comment ghi "nguong chot bang experiments/exp_alignment_check.py tren 693 cap" là **chưa đúng**: script chưa được chạy full.
- Fallback SIFT cũng chưa từng chạy, nên **F1 0.9723 và recall 0.9611 trong báo cáo vẫn còn hiệu lực**;
  chỉ có latency là thay đổi (+38 ms/cặp).

### 2.2. 3 "cặp ngoại lai homography" không phải căn chỉnh sai

Đo trực tiếp: warp ảnh test bằng H ước lượng rồi so với template.

| Cặp | H ước lượng lệch "GT" (corner) | NCC sau warp **bằng H ước lượng** | NCC nếu warp bằng **H "GT"** |
| :--- | :---: | :---: | :---: |
| `06_spur_08` | 47.09 px | **0.9988** | 0.4586 |
| `04_spurious_copper_11` | 28.08 px | 0.9962 | – |
| `06_spurious_copper_04` | 7.18 px | 0.9974 | – |

→ H của ORB **khớp nội dung ảnh**, còn `H_true_inv` mới là cái sai. Nguyên nhân: 3 cặp này **có sẵn**
dịch chuyển 7/28/47 px giữa template và test (cùng kích thước, khác vị trí chụp), trong khi
`evaluation.synthetic_warp` / `corner_error` giả định template và test trùng khít nhau.

Bằng chứng bổ sung — quét 693 cặp, corner error so với **identity** ở điều kiện thật:

| Chỉ số | Kết quả |
| :--- | :--- |
| corner median / p95 | 0.000 / 0.000 px |
| Số cặp corner > 1 px | **3** (đúng 3 cặp trên) |
| Số cặp corner > 5 px | **3** |

→ Toàn bộ 690 cặp còn lại căn chỉnh **hoàn hảo**. Vậy "3 cặp ngoại lai" là **dữ liệu bất thường**,
không phải lỗi kiểm tra homography.

**Hệ quả tài liệu cần sửa:**
- `EXPERIMENT_REVIEW.md` mục 3.2 ("reprojection error dễ gây hiểu nhầm… có cặp lệch 47 px nhưng reprojection 0.64 px")
  → kết luận đúng là "reprojection không phản ánh sai số toàn ảnh", nhưng **không phải vì H sai** — H đúng, chỉ thước đo corner sai.
- Mục 4 P0.1 (bảng NCC case study 0.798–0.948) và chỉ tiêu "3 cặp ngoại lai bị NCC chặn" **không thể đạt**.
- Mục 7 dòng P0.1 ghi "kiểm tra 3 cặp ngoại lai bị chặn và SIFT sửa được" → phải bỏ.

### 2.3. Nguyên nhân thật của "recall tụt còn 0.19–0.23" trên ảnh warp: Otsu bị dải biên đen chi phối

Đo 48 cặp (8 cặp/lớp), cùng pipeline mặc định (floor 30 / open 3 / area 20):

| Điều kiện | Otsu tính trên | Recall | Precision | F1 |
| :--- | :--- | :---: | :---: | :---: |
| Không warp | toàn ảnh | 0.882 | 0.985 | 0.931 |
| Có warp + căn lại | **toàn ảnh** (code hiện tại) | **0.215** | 0.970 | **0.352** |
| Có warp + căn lại | **chỉ vùng hợp lệ** | **0.875** | 0.985 | **0.927** |

Ví dụ ngưỡng Otsu:

| Cặp | Otsu toàn ảnh | Otsu chỉ vùng hợp lệ |
| :--- | :---: | :---: |
| `06_spur_08` | 68 | **2** |
| `04_spurious_copper_11` | 85 | **19** |
| `06_spurious_copper_04` | 84 | **31** |
| `01_missing_hole_01` | 51 | 51 |

Dải biên đen do warp (rộng đúng bằng độ dịch) chiếm diện tích lớn với tương phản cực đại → Otsu bị đẩy
lên 68–85, xoá sạch lỗi mảnh. Khi chỉ lấy mẫu Otsu trong vùng hợp lệ, recall **quay về ngang ảnh gốc**.

**Hệ quả:** kết luận §7.1 #1 ("phát hiện lỗi rất nhạy với nội suy ảnh, recall thực tế có thể rất thấp")
**sai nguyên nhân**. Vấn đề không phải nội suy mà là **ngưỡng bị nhiễu bởi vùng biên không hợp lệ**.
Đây cũng là rủi ro lớn nhất cho ảnh camera thật: ảnh thật luôn phải warp → luôn có dải biên.

### 2.4. 3 cặp bỏ sót dù căn chỉnh tốt — xác nhận do opening / sàn ngưỡng / min_area

Đo trên ảnh gốc, căn chỉnh như `main.py`, mask biên:

| Cặp | Hiện tại (floor 30, open 3, area 20) | floor 20 + open 2x2 + area 10 |
| :--- | :---: | :---: |
| `01_spur_04` | 0/3 GT, 0 box | **3/3 GT, 3 box** |
| `04_spurious_copper_05` | 0/3 GT, 0 box | **3/3 GT, 3 box** |
| `01_mouse_bite_19` | 0/2 GT, 1 box | **2/2 GT, 3 box** |
| `01_missing_hole_01` (đối chứng) | 3/3 GT, 3 box | 3/3 GT, 3 box |

→ Đúng như nghi vấn: **opening 3x3 + min_area 20 + sàn ngưỡng 30** xoá nét mảnh.
Cần chạy `exp_detection_params` trên 693 cặp để chốt lại, đồng thời theo dõi precision.

---

## 3. Lỗi và điểm cần sửa trong code

| # | Vị trí | Vấn đề | Mức |
| :---: | :--- | :--- | :---: |
| 1 | `main.py:142-157`, `module_preprocess.difference_map` | **Không mask vùng biên sau warp** (không truyền `test_mask`, `binarize_difference` không có tham số `valid`). Trên ảnh camera thật sẽ dính đúng lỗi 2.3: recall ~0.2 + 1 blob dải biên khổng lồ — **ĐÃ SỬA** (xem mục 6) | ~~Cao~~ |
| 2 | `module_preprocess.py:29,36` | `DEFAULT_MIN_NCC = 0.97` bật mặc định khi chưa chốt; trên PKU min NCC 0.9950 nên vô dụng, trên ảnh thật dễ từ chối oan; kèm `DEFAULT_FALLBACKS` SIFT làm cặp bị từ chối tốn thêm ~1.5 s — **ĐÃ SỬA** (0.90, bỏ SIFT dự phòng; xem mục 6.4) | ~~Cao~~ |
| 3 | `module_preprocess.py:409` | `time_warp_ms = (t_end - t_homo)*1000 - time_verify_ms` gộp cả kiểm tra hình học + warp, có thể **âm**; làm hỏng phân rã thời gian khi báo cáo — **ĐÃ SỬA** (đo trực tiếp giữa `t_verify_end` và `t_end`) | ~~TB~~ |
| 4 | `module_preprocess.py:425-427` | `align_images` truyền `min_ncc` **positional** thứ 6 → dễ vỡ im lặng nếu đổi chữ ký; nên dùng keyword — **ĐÃ SỬA** (cả `align_images` và `align_images_robust`) | ~~TB~~ |
| 5 | `exp_alignment_check.py:171` | `final = max(cands)[1]` — hòa điểm thì chọn corner **lớn hơn** (xấu hơn); phải chọn corner nhỏ hơn — **ĐÃ SỬA** (`max(key=(score, -corner))`) | ~~Thấp~~ |
| 6 | `exp_alignment_check.py:190-194` | `suggest()` loại cặp fail hình học khỏi nhóm "bad" (nhưng `simulate` lại tính) → có thể báo "tách được" sai — **ĐÃ SỬA** (đếm riêng `bad_geom_fail` + in rõ) | ~~Thấp~~ |
| 7 | `exp_max_features.py:495` | Help của `--max-corner-px` vẫn ghi "sai số góc xấu nhất" trong khi code dùng **p95** — **ĐÃ SỬA** (ghi "corner p95 tối đa") | ~~Thấp~~ |
| 8 | `exp_detection_params.py:40` | Ước lượng "20–40 phút" quá lạc quan: 693 × 2 × 49 cấu hình = **67.9k lần** Otsu + morphology + contours; nên ghi rõ, gợi ý `--per-class` trước — **ĐÃ SỬA** (ghi ~1.5–3 h + gợi ý) | ~~TB~~ |
| 9 | `exp_detection_params.py:136-141` | Điều kiện `clean` không mask biên; mask `valid` erode 31 px cứng; `diagnose_missed` sẽ báo "ngưỡng" quá nhiều vì dính lỗi 2.3 — **ĐÃ SỬA** (xem mục 6.3) | ~~TB~~ |
| 10 | `exp_feature_matching.py:166` | `match_boxes` thành **dead code** (đã chuyển sang `evaluation`); `read_labels` vẫn trùng lặp ở 2 script — **ĐÃ SỬA** (xóa `_touch`/`match_boxes`/`read_labels` cục bộ ở 2 script, import từ `evaluation`) | ~~Thấp~~ |
| 11 | `.gitignore` | Chưa ignore `experiments/inspect/`, `experiments/charts/detection_params.png`, `experiments/alignment_check*.csv` — **ĐÃ SỬA** | ~~Thấp~~ |

---

## 4. Có phải chạy lại 693 cặp?

| Mục | Cần chạy lại? | Lý do |
| :--- | :---: | :--- |
| F1 / recall / precision | **Không** | 0 cặp bị từ chối, 0 lần dùng SIFT → mặc định mới cho kết quả phát hiện y hệt |
| Latency | **Nên** | +38 ms/cặp do NCC (align ~270 ms thay vì 232.7 ms); `exp_feature_matching` giờ tính NCC cả khi `--min-ncc` tắt |
| Corner error | **Phải sửa thước đo rồi đo lại** | 3 cặp bị tính sai là "ngoại lai" |
| Tham số ngưỡng/opening | **Bắt buộc**, sau khi sửa mục 1 | Đây mới là nguồn recall còn thiếu |

---

## 5. Thứ tự thực hiện đề xuất

1. **[Đã xong]** **Mask vùng không hợp lệ trước khi tính Otsu.** Truyền `valid` vào `binarize_difference`
   (chỉ lấy mẫu Otsu trong vùng hợp lệ) và `detect_difference`; `main.py` tính mask từ `H`.
   Kết quả: recall trên ảnh warp 0.375 → 0.875, precision không đổi (mục 6.1).
2. **[Đã xong]** **Viết lại thước đo corner error:** mốc thành `H_pre @ H_true_inv`
   (`H_pre` = H nội dung của cặp) thay vì loại 3 cặp khỏi thống kê (mục 6.2).
3. **[Đã xong]** **Hạ P0.1 xuống "chốt chặn thảm hoạ":** `DEFAULT_MIN_NCC = 0.90`, `DEFAULT_FALLBACKS = ()`
   (bật SIFT bằng `--fallback SIFT`); giữ `appearance_score` (mục 6.4).
4. Chạy `exp_detection_params` trên 693 cặp **sau khi sửa (1)** để chốt `DEFAULT_*`
   (dự kiến open 2, min_area 10, floor ~20) — bắt buộc kiểm tra precision không giảm.
5. Sửa các mục 3–11 và rút gọn `main.py` (6 cờ CLI mới chưa chốt giá trị).

> Nguyên tắc giữ nguyên: mỗi thay đổi chỉ được chấp nhận nếu **không làm giảm precision** và được đo lại
> trên **cùng 693 cặp**, cùng điều kiện; mọi số liệu phải sinh từ script.

---

## 6. Trạng thái triển khai sau review

### 6.1. Việc #1 — mask vùng không hợp lệ trước khi tính Otsu: **ĐÃ SỬA**

Thay đổi:

| File | Nội dung |
| :--- | :--- |
| `module_preprocess.py` | Thêm `warp_valid_mask(img_test, H, out_shape, erode_border=0)`: mask 255 = pixel thật, 0 = vùng trống do warp |
| `module_preprocess.binarize_difference()` | Thêm tham số `valid`. Otsu chỉ lấy **mẫu trong vùng hợp lệ**; pixel ngoài vùng bị đặt 0 **trước** hysteresis và opening |
| `module_preprocess.detect_difference()` | Thêm tham số `valid`, chuyển tiếp xuống `binarize_difference` |
| `main.py` | Tính `valid_mask` từ `align_stats["H"]` (`warp_valid_mask(..., erode_border=3)`) và truyền vào `detect_difference` |

Tương thích ngược: `valid=None` (mặc định) giữ nguyên hành vi cũ; 48/48 cặp ảnh gốc cho **kết quả giống hệt**.

Kiểm chứng (48 cặp, 8 cặp/lớp, cùng tham số mặc định floor 30 / open 3 / area 20):

| Điều kiện | Recall | Precision | F1 |
| :--- | :---: | :---: | :---: |
| Ảnh gốc, trước và sau fix | 0.875 | 0.985 | 0.927 |
| Ảnh warp + căn lại, **trước fix** (không mask) | 0.375 | 0.071 | 0.119 |
| Ảnh warp + căn lại, **sau fix** | **0.875** | **0.985** | **0.927** |

Ba cặp từng bị dịch chuyển 7/28/47 px (trước đây bỏ sót toàn bộ GT):

| Cặp | Trước fix | Sau fix |
| :--- | :---: | :---: |
| `06_spur_08` | 0/5 GT | **2/5 GT** |
| `04_spurious_copper_11` | 0/3 GT | **3/3 GT** |
| `06_spurious_copper_04` | 0/5 GT | **5/5 GT** |
| `01_missing_hole_01` (đối chứng) | 3/3 GT | 3/3 GT |

### 6.2. Việc #2 — sửa thước đo corner error: **ĐÃ SỬA**

Vấn đề: `corner_error` lấy `H_true_inv` (chỉ gồm phép nắn lệch giả lập) làm mốc, trong khi một số cặp
PKU vốn đã lệch 7/28/47 px. H đúng vẫn bị chấm là "căn chỉnh lỗi".

| File | Nội dung |
| :--- | :--- |
| `evaluation.py` | `corner_error(..., H_pre=None)` — mốc thành `H_pre @ H_true_inv`; thêm `estimate_content_homography()` ước lượng H nội dung trên ảnh test **chưa** nắn lệch, tự refine bằng SIFT nếu lệch identity > 1 px (chỉ vài cặp) |
| `experiments/exp_feature_matching.py` | Bỏ bản `corner_error` cục bộ (đang **che** hàm dùng chung), import từ `evaluation`; tính `H_pre` khi bật nắn lệch, truyền vào `corner_error` |
| `experiments/exp_max_features.py` | Như trên |
| `experiments/exp_alignment_check.py` | Tính `H_pre` cho cả 2 điều kiện (kể cả `--no-synthetic-warp`), truyền vào `corner_error`; sửa thông điệp đề xuất đang khẳng định 3 cặp là ngoại lai |

Kiểm chứng (điều kiện thật, 3 cặp từng bị coi là ngoại lai):

| Cặp | corner cũ | corner mới |
| :--- | :---: | :---: |
| `06_spur_08` | 47.1 px | **0.38 px** |
| `04_spurious_copper_11` | 28.1 px | **0.53 px** |
| `06_spurious_copper_04` | 7.2 px | **0.23 px** |

Chạy `--per-class 2` (12 cặp, có nắn lệch giả lập): corner 0.13–0.30 px, **0 cặp ngoại lai (>5 px)**.
`exp_feature_matching --limit 5` và `exp_max_features --limit 5` chạy OK, corner 0.20–0.25 px.

Hệ quả tài liệu: mục 3.2 / 3.4 / 4 P0.1 / §7.1 #1 của `EXPERIMENT_REVIEW.md` cần viết lại theo kết luận
"3 cặp là dữ liệu bất thường (dịch chuyển có sẵn), H của ORB đúng" — chưa sửa trong lượt này.

### 6.3. `exp_detection_params.py` — chuyển sang mask trước khi tính ngưỡng: **ĐÃ SỬA**

- `prepare()`: tạo `valid` cho **mọi** điều kiện có `H` (trước chỉ tạo cho `resample`), erode `VALID_ERODE = 3` px.
- `detect()` và `diagnose_missed()`: truyền `valid=valid` vào `binarize_difference`; bỏ đoạn
  `bitwise_and` sau nhị phân hóa.
- `diagnose_missed()`: Otsu lấy mẫu trong đúng vùng hợp lệ (khớp với `binarize_difference`).
- Docstring: ghi rõ chi phí thật (~1.5–3 h cho 693 cặp) và lý do mask trước ngưỡng.

Chạy thử 3 cặp trọng điểm (24 cấu hình): chẩn đoán cấu hình gốc giờ báo `opening: 5, ngưỡng: 3`
thay vì dồn hết vào "ngưỡng"; cấu hình `floor20_open1_area10` cho recall 1.00 trên cả clean và resample.

### 6.4. Việc #3 — hạ P0.1 + sửa phân rã thời gian + gọi hàm bằng keyword: **ĐÃ SỬA**

| File | Nội dung |
| :--- | :--- |
| `module_preprocess.py` | `DEFAULT_MIN_NCC = 0.90` (chốt chặn thảm hoạ); `DEFAULT_FALLBACKS = ()` (không dùng SIFT mặc định); sửa comment sai "~6 ms" → ~38 ms và bỏ mô tả sai "3 cặp lệch cho NCC 0.80–0.95" |
| `module_preprocess.align_images_with_stats()` | `time_warp_ms` đo trực tiếp `t_end - t_verify_end` và kẹp `max(0, ...)`, không còn gộp kiểm tra hình học vào warp (trước đây có thể âm) |
| `module_preprocess.align_images()` | Truyền `min_ncc` và các tham số bằng **keyword** |
| `module_preprocess.align_images_robust()` | Truyền `min_ncc`/`test_mask` bằng keyword; cập nhật docstring (mặc định không dự phòng) |
| `main.py` | Help `--min-ncc`/`--fallback` phản ánh đúng mặc định mới; docstring `process_pair` |
| `EXPERIMENT_REVIEW.md` | Viết lại mục P0.1, KPI 214, mục 6 bước 1, bảng §7, §7.1 #1 và #3 theo kết luận đã kiểm chứng |

Kiểm chứng (693 cặp + 1 cặp ngưỡng cao):

| Kiểm tra | Kết quả |
| :--- | :--- |
| Quét 693 cặp | **693/693 chạy được**, NCC min 0.9950 · p1 0.9967 · median 0.9998, **0 cặp bị từ chối ở 0.90** |
| `time_warp_ms` | min 5.0 · median 10.6 · max 51.4 ms, **không âm**; 0/693 cặp lệch tổng thành phần > 100 ms |
| Thời gian căn chỉnh | median 243.5 ms/cặp (gồm NCC) |
| `align_images()` | Chạy OK, trả `(1586, 3034, 3)` |
| `--min-ncc 0.999` | Bị chặn đúng: `failed_check="appearance"`, NCC 0.9982, có `time_verify_ms` và `H_candidate` |
| `align_images_robust(min_ncc=0.999)` | Trả `verified=False` với H tốt nhất (hành vi cũ, đã ghi trong docstring) |
| CLI mặc định | `min_ncc=0.9`, `fallback=[]`, `no_verify=False` |
| `py_compile` | OK (module + 4 script) |

### 6.5. Việc #5 — các lỗi nhỏ còn lại: **ĐÃ SỬA**

| # | File | Nội dung |
| :---: | :--- | :--- |
| 5 | `exp_alignment_check.py` | Hòa điểm khi chọn H "tốt nhất" giờ lấy corner **nhỏ hơn** (`max(key=(score, -corner))`) |
| 6 | `exp_alignment_check.py` | `suggest()` đếm riêng `bad_geom_fail` (cặp hỏng hình học luôn bị chặn, không ràng buộc ngưỡng NCC) và in rõ trong bảng phân bố |
| 7 | `exp_max_features.py` | Help `--max-corner-px` ghi đúng "corner p95 tối đa" |
| 8 | `exp_detection_params.py` | Docstring ghi chi phí thật ~1.5–3 h (đã sửa ở lượt trước) |
| 10 | `exp_feature_matching.py`, `exp_max_features.py` | Xóa `_touch`/`match_boxes` (dead code) + `read_labels` cục bộ; dùng `evaluation.read_labels` |
| 11 | `.gitignore` | Ignore `experiments/inspect/`, `experiments/alignment_check*.csv`, `experiments/charts/detection_params.png` |

### 6.6. Checkpoint `--resume` cho `exp_detection_params.py` (phục vụ việc #4): **ĐÃ THÊM**

Chạy full 693 cặp mất 1.5–3 h, trước đây đứt giữa chừng là mất toàn bộ (kết quả chỉ ghi CSV ở cuối).
Nay script ghi checkpoint sau mỗi `--checkpoint-every` cặp (mặc định 20) vào
`<output-dir>/detection_params_state.pkl` và tự xóa khi chạy xong.

- `--resume`: nạp lại `rows`/`diag`/danh sách cặp đã xong, chỉ chạy các cặp còn lại.
- Checkpoint có **chữ ký** (data dir, conditions, danh sách cấu hình, tham số warp, danh sách cặp):
  nếu khác lần chạy trước thì cảnh báo và chạy lại từ đầu, tránh trộn kết quả không tương thích.
- Ghi nguyên tử (`.tmp` rồi `replace`) để không hỏng file nếu bị kill đúng lúc ghi.

Kiểm chứng: chạy 4 cặp (48 cấu hình, `--checkpoint-every 1`), kill sau cặp 1 → resume → hoàn tất 4/4 cặp:

| Kiểm tra | Kết quả |
| :--- | :--- |
| Thông báo resume | `đã xong 1/4 cặp, tiếp tục 3 cặp còn lại` |
| CSV từng cặp | 4 cặp × 96 dòng (48 cấu hình × 2 điều kiện), không trùng/thiếu |
| Checkpoint sau khi xong | Đã tự xóa |
| Exit code | 0 |

### 6.7. Việc tiếp theo (chưa làm)

- **Chạy full 693 cặp với `--resume`** rồi chốt `DEFAULT_*` (việc #4). Lệnh:
  `& .\.venv\Scripts\python.exe -u experiments\exp_detection_params.py --output-dir experiments --resume`
- Bổ sung cột F1@IoU vào bảng báo cáo chính (`PROGRESS_REPORT.md`).

