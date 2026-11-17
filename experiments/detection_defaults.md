# Chốt tham số phát hiện (bước nhị phân hoá + lọc đốm)

Nguồn số liệu: `experiments/detection_params_summary.csv` — lưới **60 cấu hình**
(sàn ngưỡng 10/15/20/25/30 × opening 1/2/3 × hysteresis 0/0.5 × min_area 10/20),
chạy trên **693 cặp** × **2 điều kiện** (`clean` = ảnh gốc, `resample` = nắn lệch
rồi căn ngược, ảnh nội suy như camera thật).

Tiêu chí ghép: "chạm nhau" dung sai 5 px (cột `f1/precision/recall`).

## 1. Cấu hình đang dùng trong code (trước khi chốt)

`DEFAULT_THRESHOLD_FLOOR = 30`, `DEFAULT_OPEN_KSIZE = 3`,
`DEFAULT_HYSTERESIS_LOW = None`, `DEFAULT_MIN_AREA = 20`
→ tên trong lưới: `floor30_open3_single_area20` (cột `is_baseline=True`).

| Điều kiện | F1 | Precision | Recall | FP / ảnh | Cặp sót toàn bộ | Recall Spur+SpCu |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0,97643 | 0,98240 | 0,97054 | 0,0765 | 3 | 0,9435 |
| resample | 0,97442 | 0,98355 | 0,96546 | 0,0707 | 3 | 0,9304 |

## 2. Ứng viên thay thế

`floor20_open2_single_area20` — sàn ngưỡng 20, opening 2×2, không hysteresis,
min_area 20 (nằm giữa lưới, không ở biên).

| Điều kiện | F1 | Precision | Recall | FP / ảnh | Cặp sót toàn bộ | Recall Spur+SpCu |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0,98948 | 0,98278 | 0,99627 | 0,0765 | 0 | 0,9889 |
| resample | 0,98852 | 0,98219 | 0,99492 | 0,0794 | 0 | 0,9859 |

So với baseline:

| Điều kiện | ΔF1 | ΔPrecision | ΔRecall | ΔFP/ảnh | Sót toàn bộ |
|---|---:|---:|---:|---:|---:|
| clean | **+0,01305** | +0,00038 | +0,02573 | 0 | 3 → 0 |
| resample | **+0,01410** | **−0,00136** | +0,02946 | +0,0087 | 3 → 0 |

Quy ra số đốm: resample có `n_pred` 3.089 (baseline 2.978) → khoảng **4 dự đoán
sai thêm** trên ~3.000 dự đoán, đổi lấy **~87 lỗi GT phát hiện thêm**.

## 3. Các lựa chọn khác đã loại

| Cấu hình | Lý do loại |
|---|---|
| `hyst0.5` (mọi biến thể) | `clean` đẹp nhưng `resample` sụp: F1 0,955–0,970, precision 0,916–0,944, FP 0,27–0,42 đốm/ảnh (baseline 0,07) |
| `area10` (mọi biến thể) | FP/ảnh tăng ~0,086, precision giảm; `floor20_open1_single_area10` F1 0,9892 nhưng precision 0,9809 < baseline ở cả 2 điều kiện |
| `floor20_open1_single_area20` | Tương đương `open2` nhưng ΔP resample −0,00163 (xấu hơn `open2`) |

Chỉ **1/60** cấu hình không làm giảm precision ở cả 2 điều kiện: chính baseline.
Nói cách khác, "không giảm precision" và "tăng recall đáng kể" loại trừ nhau trên
lưới này; muốn tăng recall phải chấp nhận đánh đổi precision rất nhỏ.

## 4. KPI mục 5 (EXPERIMENT_REVIEW.md)

`KPI = {recall ≥ 0,97, f1 ≥ 0,975, weak_recall ≥ 0,96, false_blobs ≤ 0,05/ảnh}`

**0/60 cấu hình đạt KPI** trên cả 2 điều kiện — nút thắt là `false_blobs`:
mọi cấu hình đều 0,07–0,09 đốm giả/ảnh (ngưỡng 0,05). Ghi rõ khi báo cáo, không
được viết "đạt KPI".

## 5. Quyết định

- [ ] Giữ `floor30_open3_single_area20` (không giảm precision ở cả 2 điều kiện,
      nhưng còn 3 cặp sót toàn bộ và recall Spur+SpCu ~0,93)
- [ ] Đổi sang `floor20_open2_single_area20` (F1 +0,014, hết cặp sót toàn bộ,
      đổi lại precision resample −0,0014 ≈ 4 dự đoán sai thêm)

Sau khi chốt: sửa `DEFAULT_*` trong `module_preprocess.py`, chạy
`experiments/verify_defaults.py` (so lại đúng bảng trong tài liệu này), rồi chạy
lại `build_classifier_dataset.py` + `exp_classify.py` vì bbox đổi theo.
