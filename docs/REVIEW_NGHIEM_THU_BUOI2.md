# BIÊN BẢN RÀ SOÁT & VIỆC CẦN LÀM — NGHIỆM THU BUỔI 2

> Mục đích: biến các nhận xét rà soát thành task cụ thể cho từng thành viên, kèm
> cách kiểm chứng (Definition of Done). Không đổi chữ ký hàm đã giao trong README
> để tránh vỡ tích hợp.

---

## 0. Tiêu chí nghiệm thu Buổi 2 (theo README mục 6)

| # | Tiêu chí | Trạng thái hiện tại | Ghi chú |
|---|---|---|---|
| 1 | `python main.py` chạy không lỗi | ❌ FAIL | crash `UnicodeEncodeError` trên console không UTF-8 |
| 2 | Ảnh trừ vi sai không có viền trắng quanh đường mạch | ❌ FAIL | 9/16 box là vệt viền dài – mỏng |
| 3 | Recall ≥ 95% với lỗi nhìn thấy rõ | ⚠️ CHƯA ĐO | chưa có script/ground truth |
| 4 | Crop lỗi lưu đầy đủ vào `crops/` | ✅ ĐẠT | nhưng cần xoá file cũ mỗi lần chạy |

**Bổ sung bắt buộc cho báo cáo:** phải có Precision/F1, không chỉ Recall. Trên cặp
mẫu hiện tại ước tính `P ≈ 0.31, R = 1.0, F1 ≈ 0.48` (16 box, ~5–6 lỗi thật).

**Bằng chứng đã đo được (dùng để giải thích cho nhóm):**

| Hiện tượng | Số liệu |
|---|---|
| Ảnh mẫu 92000000 | 16 box, ~0.30 s |
| Box dài – mỏng dọc viền (false positive) | 9/16 |
| Nén lại JPEG q=75 một ảnh **không lỗi** rồi so sánh | 46 false positive |
| Spur rộng 1 px | `contourArea = 0` → bị lọc mất |
| Mở morphology 7×7 thay 3×3 (thử nhanh) | còn 7 box, hết vệt mỏng |

**Công cụ đã có trong repo:**
- `tools/test_pipeline.py` — smoke test. Hiện `6 pass, 2 expected failure`.
- `tools/evaluate_deeppcb.py` — đo P/R/F1 @ IoU 0.33 trên DeepPCB.

---

## 1. Việc chung của cả nhóm (chốt trước khi chia việc)

| ID | Việc | Cách làm | DoD |
|---|---|---|---|
| C-01 | Giữ nguyên API | Không đổi tên/kiểu trả về của 4 hàm đã giao | `main.py` vẫn chạy, không sửa chữ ký |
| C-02 | Chạy test trước khi push | `py -3.14 tools/test_pipeline.py -v` | 6 pass + 2 expected failure; sau khi sửa P0 phải 8/8 |
| C-03 | Chuẩn bị 10 cặp DeepPCB | Đặt trong `data/deeppcb10/`, giữ nguyên `*_temp.jpg`, `*_test.jpg`, `*.txt` | `evaluate_deeppcb.py --data data/deeppcb10` chạy được |
| C-04 | Không commit file sinh ra | Thêm `.gitignore`: `crops/`, `results/`, `__pycache__/`, `*.pyc`, `data/deeppcb*/` | `git status` sạch sau khi chạy |
| C-05 | Báo cáo phải có số | Mỗi lần sửa phải kèm bảng P/R/F1 + thời gian + ảnh overlay | Có file `results/eval*/` và bảng trong báo cáo |

**Quy ước nhánh:** mỗi người một nhánh `fix/<tên>-<id>` (VD `fix/namanh-p0`), PR
nhỏ, mô tả kèm "trước/sau" số liệu.

---

## 2. Nam Anh — `module_preprocess.py`

### NA-01 (P0) — Otsu sinh false positive trên ảnh không lỗi
**Hiện trạng:** `detect_difference()` dùng Otsu toàn cục trên `absdiff` (dòng ~85).
Otsu luôn chọn một ngưỡng, kể cả khi ảnh không có lỗi. Đã kiểm chứng: nén JPEG lại
một ảnh **defect-free** → 46 box.

**Cách sửa:** đặt ngưỡng sàn.
```python
otsu_val, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thr = max(int(otsu_val), 30)          # 30 là giá trị khởi điểm, tune trên 50 ảnh
_, diff_binary = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
```
**DoD:** `test_jpeg_noise_does_not_create_boxes` pass (bỏ `expectedFailure`).

---

### NA-02 (P0) — Viền ảnh sau warp tạo box rác
**Hiện trạng:** `warpPerspective(..., BORDER_REPLICATE)` (dòng ~57), sau đó không
loại vùng biên. Các box #1, #10, #13 nằm sát rìa.

**Cách sửa (ưu tiên):** warp thêm một mask hợp lệ rồi AND vào kết quả.
```python
valid = np.full(gray_template.shape, 255, np.uint8)
valid = cv2.warpPerspective(valid, H, (w, h), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
diff_cleaned = cv2.bitwise_and(diff_cleaned, valid)
```
Hoặc tối thiểu: xoá 8 px sát biên trong `detect_difference` (ghi chú rõ trong báo cáo
là đánh đổi, không phát hiện lỗi nằm sát mép ảnh).

**DoD:** không còn box nào có `x < 8` hoặc `x + w > 632` (tương tự cho y) trên 10 cặp test.

---

### NA-03 (P1) — Viền mạch vẫn lọt sau `MORPH_OPEN 3×3`
**Hiện trạng:** sai lệch cục bộ 3–6 px giữa template/test tạo vệt dọc rộng 3–8 px,
dài 100–250 px. Open 3×3 không xoá được.

**Hướng sửa (chọn 1, đo bằng ablation):**
1. **So sánh có dung sai vị trí** (khuyến nghị):
```python
k = np.ones((5, 5), np.uint8)  # dung sai ±2 px
missing = cv2.subtract(gray_test, cv2.dilate(gray_template, k))  # test sáng hơn → mất đồng
extra   = cv2.subtract(cv2.erode(gray_template, k), gray_test)   # test tối hơn → thừa đồng
diff = cv2.max(missing, extra)
```
2. Tăng open lên 5×5 hoặc 7×7 **nhưng phải kiểm tra recall spur mảnh**.

**DoD:** số box trên ảnh mẫu giảm từ 16 xuống ≤ 8 mà vẫn giữ đủ 5–6 lỗi thật; ghi
bảng so sánh 3×3 / 5×5 / 7×7 / tolerant vào báo cáo.

---

### NA-04 (P2) — Căn chỉnh nên đơn giản hoá & kiểm tra chặt hơn
- DeepPCB đã căn chỉnh sẵn: thử `cv2.phaseCorrelate` hoặc `findTransformECC`
  (translation/affine) trước, chỉ dùng homography khi thật cần. Mục tiêu nhanh hơn
  và ít sai lệch cục bộ hơn.
- Kiểm tra `H` gần identity: reprojection error < 3 px; nếu lệch nhiều thì cảnh báo.
- `keep_percentage=0.8` không phải Lowe ratio — nêu rõ trong báo cáo hoặc đổi sang
  ratio test `0.75`.

**DoD:** có bảng so sánh ORB-homography vs phase-correlate về thời gian + P/R trên 50 ảnh.

---

### NA-05 (P1) — Hợp đồng lỗi rõ ràng
Ghi rõ trong docstring: `align_images` ném `ValueError` khi thiếu đặc trưng,
`RuntimeError` khi không đủ match / homography xấu. Không tự fallback bên trong module.
**DoD:** `main.py` bắt và log được từng loại lỗi (xem H-02).

---

## 3. Minh — `module_detection.py`

### M-01 (P0) — Bỏ sót lỗi mỏng do `contourArea`
**Hiện trạng:** `extract_bounding_boxes()` lọc `cv2.contourArea(cnt)` (dòng ~11).
Đường rộng 1 px có `contourArea = 0` dù chứa 51 pixel → bị loại.

**Cách sửa:** đếm pixel thật trong bounding box.
```python
x, y, w, h = cv2.boundingRect(cnt)
area = int(cv2.countNonZero(clean_mask[y:y+h, x:x+w]))
if area < min_area:
    continue
```
**DoD:** `test_thin_spur_is_detected` pass; báo cáo ghi rõ `min_area` giờ là **số pixel
mask**, khớp với README ("Area ≥ 20 px").

---

### M-02 (P1) — Thứ tự box không ổn định
**Cách sửa:** `boxes.sort(key=lambda b: (b[1], b[0]))` trước khi return.
**DoD:** chạy 2 lần cùng ảnh, danh sách box và nhãn `Defect #` giống hệt.

---

### M-03 (P1) — Crop cũ còn sót giữa các lần chạy
Hiện `crop_defect_patches` chỉ `makedirs(exist_ok=True)` và ghi đè `defect_001.png...`.
Nếu lần trước có 16 lỗi, lần này 5 lỗi → còn 11 file cũ.

**Cách sửa:** thống nhất để `main.py` xoá sạch thư mục trước khi gọi (xem H-04),
hoặc thêm tham số `clean=True` cho hàm crop. Không đổi chữ ký bắt buộc → ưu tiên xoá ở `main.py`.

**DoD:** chạy lại sau khi sửa thuật toán, `crops/` chỉ còn đúng số file của lần chạy mới.

---

### M-04 (P2) — Validate & thông báo lỗi
- Sửa message padding: `"Padding phải >= 0"` (hiện ghi "lớn hơn hoặc bằng 0" nhưng
  chỉ chặn `< 0`).
- Kiểm tra `len(box) >= 4`, các giá trị là số; bỏ qua box `NaN`.
- Cân nhắc gộp box giao nhau > 50% (spur + mousebite đôi khi tách làm 2).
**DoD:** không crash với box rỗng/`None`; có test cho padding âm (đã có sẵn).

---

## 4. Hinh — `main.py`, tích hợp & thử nghiệm

### H-01 (P0) — Crash Unicode trên Windows
**Hiện trạng:** `print("KẾT QUẢ")` (dòng ~142) ném `UnicodeEncodeError` với console
cp932/cp1258 → vi phạm tiêu chí #1.

**Cách sửa:** đầu file.
```python
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
```
**DoD:** chạy `py -3.14 main.py` trên PowerShell/CMD mặc định (không set
`PYTHONIOENCODING`) → không crash, in đủ log.

---

### H-02 (P0) — Nuốt lỗi căn chỉnh
**Hiện trạng:** `try/except Exception: img_test_aligned = img_test_color` (dòng ~118).
Đây chính là nguồn false positive lớn nhất và không ai biết.

**Cách sửa:**
```python
try:
    img_test_aligned = align_images(img_temp_color, img_test_color)
except Exception as exc:
    print(f"[CẢNH BÁO] Căn chỉnh thất bại: {exc} — dùng ảnh chưa căn chỉnh")
    img_test_aligned = img_test_color
```
**DoD:** log có cảnh báo rõ; báo cáo có dòng "số ảnh align lỗi".

---

### H-03 (P0) — Import fallback gây lỗi khó hiểu
**Hiện trạng:** import lỗi → gán `None` → gọi hàm `None` → `TypeError`.
**Cách sửa:** import trực tiếp, fail fast:
```python
from module_preprocess import align_images, detect_difference
from module_detection import extract_bounding_boxes, crop_defect_patches
```
**DoD:** thiếu module → báo lỗi rõ ràng ngay khi khởi động.

---

### H-04 (P1) — Chạy được batch + dọn output
Thêm `argparse`:
```powershell
py -3.14 main.py --test data/92000000_test.jpg --temp data/92000000_temp.jpg
py -3.14 main.py --pair-dir data/deeppcb10 --output results/batch
```
Trong `main`: xoá `crops/` và file kết quả cũ trước khi ghi; đặt tên output theo tên
ảnh test; in số crop thực tế (`len(crop_paths)`), sửa dòng log thừa dấu `)`.

**DoD:** chạy 1 ảnh và chạy cả thư mục đều được; không còn file crop cũ.

---

### H-05 (P1) — Thử nghiệm 10 cặp + số liệu
**Đây là việc bắt buộc của Buổi 2 theo README mục 4.**
```powershell
py -3.14 tools/evaluate_deeppcb.py --data data/deeppcb10 --save-dir results/eval10
```
Nộp kèm:
- bảng P/R/F1 @ IoU 0.33, thời gian/ảnh;
- ảnh overlay (đỏ = TP, cam = FP, xanh dương = FN) cho 10 cặp;
- danh sách ca fail + nguyên nhân (align, viền, bỏ sót).

**DoD:** bảng số + overlay + nhận xét ngắn trong báo cáo.

---

### H-06 (P2) — Dọn code & tài liệu
- Bỏ đọc ảnh gray không dùng (dòng ~46–47).
- `requirements.txt`: pin version (`opencv-python==4.12.*`, `numpy>=2`), bỏ `matplotlib`
  nếu không dùng.
- Cập nhật README: cấu trúc `data/92000000_*`, thư mục `tools/`, tiêu chí Precision,
  trạng thái "chưa phân loại 6 lớp".
- Thêm `.gitignore` (C-04).

---

## 5. Thứ tự thực hiện đề xuất

**Đợt 1 — P0 (chặn nghiệm thu):**
1. H-01 (Unicode) → chạy được `main.py`.
2. NA-01 (ngưỡng sàn) + NA-02 (mask viền).
3. M-01 (area bằng `countNonZero`).
4. H-02, H-03 (không nuốt lỗi, import rõ).
5. Chạy `tools/test_pipeline.py` → mục tiêu 8/8 pass.

**Đợt 2 — P1 (chất lượng):**
6. NA-03 (giảm false positive viền) + ablation 3×3/5×5/7×7/tolerant.
7. M-02, M-03 (sort box, dọn crop) + H-04 (argparse, dọn output).
8. H-05: chạy 10 cặp, lập bảng P/R/F1.

**Đợt 3 — P2 (hoàn thiện báo cáo):**
9. NA-04 (align), M-04 (validate), H-06 (README/requirements/.gitignore).
10. Chạy full 500 ảnh test DeepPCB (nếu kịp) để có số liệu mạnh hơn.

---

## 6. Checklist nộp Buổi 2

- [ ] `py -3.14 main.py` không crash trên console mặc định.
- [ ] `tools/test_pipeline.py` → **8/8 pass** (đã bỏ 2 `expectedFailure`).
- [ ] Ảnh `results/*.jpg` không còn khung đỏ dọc viền mạch.
- [ ] `crops/` có đúng số file của lần chạy mới nhất.
- [ ] Bảng P/R/F1 + thời gian trên 10 cặp DeepPCB.
- [ ] Ảnh overlay TP/FP/FN kèm nhận xét.
- [ ] README cập nhật cấu trúc thư mục & trạng thái chức năng.
- [ ] `git status` sạch, không commit `crops/`, `results/`, dataset.

---

## 7. Lưu ý để không mất điểm

- **Không tune riêng cho cặp 92000000.** Mọi ngưỡng phải chọn trên 50 ảnh (tune),
  báo cáo trên 10 cặp/500 ảnh test.
- **Không đổi API** đã giao trong README. Nếu buộc phải đổi, báo cả nhóm + cập nhật README
  trong cùng PR.
- **Không thêm deep learning** — đề bài Milestone 2 yêu cầu classical CV.
- **Nêu rõ giới hạn:** chưa phân loại 6 lớp, bỏ qua lỗi sát biên nếu mask viền, phụ thuộc
  chất lượng căn chỉnh. Trung thực trong báo cáo tốt hơn che giấu.
