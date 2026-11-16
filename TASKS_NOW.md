# VIỆC LÀM ĐƯỢC NGAY — XẾP THEO THỨ TỰ ƯU TIÊN

> Cập nhật: **16/11/2026** (rev. 2). Mốc: **17/11** hết giai đoạn 3 · **18/11** tổng duyệt · **19/11** thông Thầy.
> Bối cảnh: job `experiments/exp_detection_params.py` (693 cặp × 2 điều kiện × 60 cấu hình) đang chạy nền.
> **Toàn bộ việc dưới đây không phụ thuộc kết quả job.** Việc bị chặn bởi job ghi ở mục 4.

---

## 1. Ràng buộc chung (đọc trước khi làm)

| Không được | Vì sao |
| :--- | :--- |
| Sửa `module_preprocess.py`, `experiments/exp_detection_params.py` | Đổi `BASELINE`/chữ ký checkpoint → mất tiến độ resume của job đang chạy |
| Chạy full 693 cặp (`exp_feature_matching`, `exp_max_features`, `main.py`…) | Tranh CPU với job; chỉ dùng `--limit 3` hoặc `--per-class 1` để test code |
| Xóa `experiments/detection_params_state.pkl` | Mất checkpoint 40 cặp |
| Commit trực tiếp lên `main` | Mỗi người một nhánh: `feature/classify` (Minh), `feature/gui` (Hinh), `feature/numbers` (Nam Anh) |

Quy ước nhánh + test nhanh:

```powershell
git switch -c feature/<ten-nhanh>
& .\.venv\Scripts\python.exe experiments\<script>.py --limit 3
```

---

## 2. Thứ tự ưu tiên

Ưu tiên theo **đường găng**: việc nào chặn người khác hoặc chặn bước chốt tham số tối nay thì làm trước.

| Ưu tiên | # | Việc | Người | Effort | Xong khi (DoD) | Chặn ai / bước nào |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| **P0** | 1 | **Chốt interface phân loại** trong `module_classification.py`: `CLASS_NAMES`, `FEATURE_NAMES`, `load_model()`, `classify()` (bản stub trả `"chưa phân loại"`) | Minh | 30–45' | Hinh `import` được và gọi `classify()` trên 3 cặp không lỗi | **Chặn #6 (GUI)** |
| **P0** | 2 | **`module_visualize.py`**: tách `draw_feature_matches`, `draw_homography_verification` (checkerboard + alpha overlay), `draw_results` từ `experiments/exp_feature_matching.py` và `main.py` | Hinh | 3 h | Xuất 3 ảnh mẫu vào `experiments/vis/`, ảnh giống bản cũ (không đổi logic) | **Chặn #6 và ảnh cho §2.3** |
| **P1** | 3 | **`extract_features()` thật**: area, perimeter, aspect, extent, solidity, circularity, 7 Hu (log-scale), `mean_diff`; vector `float32`, không NaN | Minh | 3–4 h | Test hình chữ nhật 20×10 → area 200, aspect 2.0, extent ≈ solidity ≈ 1.0 | Chặn #4 |
| **P1** | 4 | **`experiments/build_classifier_dataset.py`**: chạy pipeline trên 693 cặp, khớp blob ↔ GT (**tâm blob nằm trong GT** hoặc IoU ≥ 0.1), nhãn lấy từ `class_name` của cặp, thêm lớp `background` | Minh | 2 h | `--limit 3` chạy được; in phân bố 7 lớp + tỉ lệ blob khớp GT | Chặn #5 |
| **P1** | 5 | **`experiments/exp_classify.py`**: chia train/test **theo `pair_id`** (GroupKFold), chuẩn hoá đặc trưng, SVM (RBF) vs Random Forest (`cv2.ml`), F1 macro + per-class, confusion matrix 6×6 (+ background), lưu `model_classifier.pkl` và `experiments/charts/confusion_matrix.png` | Minh | 2–3 h | Chạy được trên 3 cặp (số chỉ để kiểm code); full 693 chạy sau khi job xong | Chặn deliverable giai đoạn 3 |
| **P1** | 6 | **`gui_app.py`** (tkinter): chọn cặp từ manifest → chạy → 4 panel (template / test / vi sai / mask+box) → nhãn dự đoán → thời gian + NCC → nút xuất ảnh | Hinh | 4–5 h | Chạy được trên 3 cặp với nhãn stub; khi có model chỉ đổi 1 dòng `classify()` | — |
| **P1** | 7 | **`experiments/verify_defaults.py`**: chạy lại **đúng cấu hình đã chốt** trên 693 cặp × 2 điều kiện, so với `detection_params_summary.csv` đã lưu, sai lệch phải = 0 | Minh | 30' code + 4' chạy | Code viết được ngay (đọc cấu hình từ CLI, không hardcode); **chạy sau khi job xong** | Cổng chặn hồi quy cho bước chốt `DEFAULT_*` |
| **P2** | 8 | **Kiểm nhánh dự phòng SIFT**: 3 cặp với `--min-ncc 0.999 --fallback SIFT`, ghi NCC + thời gian + kết luận khi nào nên bật | Nam Anh | 30' | Có ghi chú ngắn (kèm số) đưa được vào báo cáo | — |
| **P2** | 9 | **Rà số liệu §2.1 / 2.2 / 2.4** của `PROGRESS_REPORT.md` + dựng script vẽ lại `charts/max_features_study.png` từ CSV | Nam Anh | 2 h | Mọi số trong bảng truy được về `experiments/*.csv`; script vẽ chạy lại được sau khi có số mới | — |
| **P2** | 10 | **`evaluate_metrics.py`**: CLI bọc `evaluation.py` (IoU / Precision / Recall / F1 trên 693) | Hinh | 1 h | `--limit 3` in bảng số đúng định dạng báo cáo | — |
| **P2** | 11 | **`tests/test_contracts.py`**: ~6 assert (xem mục 6) — hợp đồng hàm + toàn vẹn dữ liệu, chạy 2 giây, không cần ảnh thật | Minh | 30' | `python tests/test_contracts.py` in PASS/FAIL rõ ràng, exit code 0 | — |

**Nếu chỉ làm được 3 việc hôm nay:** #1 → #2 → #3 (mở khoá GUI, rồi vào đường găng phân loại).

> **Vì sao không dùng "smoke test 3 cặp" (assert F1)**: đó là benchmark chứ không phải unit test —
> 3 cặp không chứng minh được gì về chất lượng chung, và dễ bị hỏi ngược. Phép đo chất lượng đúng là
> bộ 693 cặp cố định + CSV tái lập. Test tự động chỉ giữ phần **hợp đồng hàm** và **toàn vẹn dữ liệu**
> (mục 6), còn chống hồi quy tham số dùng `verify_defaults.py` (#7).

---

## 3. Hợp đồng interface (chốt ở việc #1)

```python
# module_classification.py  — Minh cung cấp, Hinh chỉ gọi
CLASS_NAMES = ["Missing_hole", "Mouse_bite", "Open_circuit",
               "Short", "Spur", "Spurious_copper", "background"]

FEATURE_NAMES = ["area", "perimeter", "aspect", "extent", "solidity",
                 "circularity", "hu1", ..., "hu7", "mean_diff"]

def extract_features(gray, mask, boxes) -> np.ndarray      # (N, D) float32
def load_model(path="model_classifier.pkl") -> dict         # {"scaler":..., "model":..., "classes":...}
def classify(gray, mask, boxes, model=None) -> list[str]    # lớp "background" -> trả "không phải lỗi"
```

Và `main.py:process_pair()` nên thêm tham số tuỳ chọn `extra_out: dict | None = None`
(điền `mask`, `gray`, `boxes` khi truyền vào) — **giữ nguyên 5 giá trị trả về** nên không phá caller nào.

---

## 4. Việc bị chặn bởi job (chỉ ghi để nhớ, KHÔNG làm bây giờ)

- Chốt `DEFAULT_THRESHOLD_FLOOR / DEFAULT_OPEN_KSIZE / DEFAULT_MIN_AREA` từ `detection_params_summary.csv`.
- Chạy `build_classifier_dataset.py` và `exp_classify.py` **full 693**; chạy lại `exp_feature_matching.py` / `exp_max_features.py`.
- Chạy `verify_defaults.py` (#7) — cần CSV của job để so.
- Chạy `main.py` để smoke test với mặc định mới.
- Chép dòng baseline cũ (`floor30_open3_hyst0_area20`) vào tài liệu **trước khi** sửa `DEFAULT_*`.

---

## 5. Checklist nhanh theo người

**Minh**
- [x] #1 interface + stub `classify()`
- [x] #3 `extract_features()`
- [x] #4 `build_classifier_dataset.py`
- [x] #5 `exp_classify.py`
- [ ] #7 `verify_defaults.py`
- [ ] #11 `tests/test_contracts.py`

**Hinh**
- [ ] #2 `module_visualize.py`
- [ ] #6 `gui_app.py`
- [ ] #10 `evaluate_metrics.py`

**Nam Anh**
- [ ] #8 test SIFT fallback
- [ ] #9 rà §2.1 / 2.2 / 2.4 + script vẽ chart

---

## 6. Nội dung `tests/test_contracts.py` (việc #11)

Chỉ assert những gì **chắc chắn đúng** và không phụ thuộc dữ liệu thật:

1. `binarize_difference` trên ảnh tổng hợp: giữ nguyên `shape`, `dtype == uint8`, giá trị ⊆ {0, 255}.
2. `binarize_difference(valid=...)`: mọi pixel nằm ngoài `valid` bằng 0.
3. `iou(a, a) == 1.0`; `iou` của 2 box rời nhau `== 0.0`.
4. `prf_from_counts` với `n_gt=10, gt_hit=8, n_pred=9, pred_hit=8` → precision/recall/F1 khớp số học.
5. `extract_bounding_boxes`: blob diện tích 9 px bị loại khi `min_area=10`, được giữ khi `min_area=5`.
6. `dataset_utils.find_pairs`: đếm đúng số cặp trong manifest và mọi `template_path` tồn tại
   (nếu thiếu ảnh thì **skip** kèm thông báo, không fail cứng).

Với thầy, cách mô tả an toàn: *"Kiểm thử hợp đồng API + toàn vẹn dữ liệu; chất lượng đo bằng bộ 693 cặp
cố định và lưu CSV để tái lập."*
