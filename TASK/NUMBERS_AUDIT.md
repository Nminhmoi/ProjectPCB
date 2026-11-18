# RÀ SOÁT SỐ LIỆU BÁO CÁO + KIỂM NHÁNH SIFT (việc #8, #9 — Nam Anh)

> Nhánh `feature/numbers` · 16/11/2026 · Không sửa `module_preprocess.py`, `exp_detection_params.py`; không chạy full 693.

## 1. Việc #9 — mọi số trong §2.1 / §2.2 / §2.4 truy được về CSV

Script mới **`experiments/report_tables.py`** tính lại 3 bảng bằng chính hàm tổng hợp của script thí nghiệm
(`exp_max_features.summarize`, `exp_blur_kernel.summarize`) và đối chiếu với `PROGRESS_REPORT.md`:

```powershell
& .\.venv\Scripts\python.exe experiments\report_tables.py --check   # exit 1 nếu có ô lệch
& .\.venv\Scripts\python.exe experiments\report_tables.py --write   # ghi đè bảng khi có CSV mới
```

Kết quả đối chiếu trước khi sửa: **14 ô lệch**, còn lại khớp.

| Bảng | Ô lệch | Nguyên nhân | Đã sửa |
| :--- | :--- | :--- | :--- |
| §2.2 cột F1 (6 ô) | Báo cáo 0.9641 / 0.9666 / **0.9668** / 0.9676 / 0.9691 / 0.9699 — CSV 0.9683 / 0.9718 / **0.9723** / 0.9725 / 0.9729 / 0.9750 | Không tái lập được từ `max_features_results.csv` (không khớp cả micro lẫn macro) → số từ một lần chạy cũ. Số CSV ở mức 2000 (0.9723) **trùng** F1 ORB của §2.1, tức hai thí nghiệm giờ nhất quán | Thay bằng số CSV; sửa lời phân tích + kết luận (F1 0.9668 → 0.9723) |
| §2.1 + §2.2 corner p95 (7 ô) | Lệch 0.001–0.005 px | Báo cáo dùng phân vị kiểu "nearest", script dùng `numpy.percentile` mặc định (tuyến tính) | Thống nhất theo script (tuyến tính) |
| §2.2 corner trung vị 10000 | 0.239 → 0.238 | Làm tròn (0.2385) | Theo CSV |
| §2.4 | 0 | — | Thêm dòng nguồn |

Sửa lời trong báo cáo (không đổi kết luận chọn tham số):
- §2.1, §2.2: bỏ nhận định "3 cặp ngoại lai do kiểm tra homography quá lỏng" — 3 cặp `06_spur_08`, `04_spurious_copper_11`, `06_spurious_copper_04` **lệch sẵn** so với template, H đúng (CODE_REVIEW §2.2; ORB/SIFT/AKAZE cho cùng H, chênh < 0.1 px).
- Thêm cảnh báo: CSV §2.1/§2.2 chạy **trước** 2 bản sửa ngày 15/11 (thước đo corner, mask vùng hợp lệ) → chạy lại sau khi job xong, rồi `report_tables.py --write`.
- Dưới mỗi bảng ghi **nguồn CSV + quy ước** (trung bình hay trung vị, cách tính p95, F1 micro). §2.1 dùng thời gian **trung bình**, §2.2 dùng **trung vị**, và là hai lần chạy khác nhau → không so thời gian chéo.
- §2.4: bỏ lý do "3x3 nhanh nhất" — chênh < 1.5 ms và có nhiễu đo (1x1 không blur đo **chậm hơn** 3x3: 3.89 so với 3.07 ms).

**Biểu đồ `max_features_study.png` cũ sai ô thời gian:** vẽ theo trung bình nên có đỉnh **~777 ms đúng tại mức 2000** (vài cặp đo lúc máy bận, tối đa ~453 s/cặp), trong khi bảng ngay dưới ghi 107 ms.
Script mới **`experiments/plot_max_features.py`** vẽ lại từ CSV bằng `exp_max_features.plot_study`, ô thời gian dùng **trung vị** (87 → 104 → 120 → 168 → 373 ms, khớp bảng). Đề xuất theo tiêu chí p95: **2000** — khớp lựa chọn của nhóm.

## 2. Việc #8 — nhánh dự phòng SIFT

Script mới **`experiments/check_sift_fallback.py`** (3 cặp mặc định: `01_missing_hole_01`, `06_spur_08`, `04_spurious_copper_05`):

```powershell
& .\.venv\Scripts\python.exe experiments\check_sift_fallback.py
```

Với mỗi cặp: căn bằng ORB và SIFT (NCC, thời gian, độ chênh hai H, số GT phát hiện), rồi chạy
`align_images_robust` như `main.py` ở chế độ mặc định và chế độ ép `--min-ncc 0.999 --fallback SIFT`.
Tự ghi bảng + kết luận vào `experiments/sift_fallback_note.md` để dán vào báo cáo.

**Chưa có số thật** (cần ảnh PKU). Kết luận dự kiến dựa trên số liệu đã có:

- 693 cặp (`summary.csv`): SIFT corner trung vị 0.035 so với 0.280 px, nhưng chậm **6.4 lần** (1485 so với 233 ms); F1 chỉ **+0.0029**.
- NCC của ORB trên PKU ≥ 0.995 (CODE_REVIEW §2.1) → với ngưỡng 0.90 nhánh dự phòng **không bao giờ chạy**.
- → **Không bật mặc định.** Bật khi ORB hỏng thật (thất bại hình học hoặc NCC < 0.90: ảnh camera ít chi tiết, mờ, lệch sáng, biến dạng lớn). Cần độ chính xác cao offline thì dùng thẳng `--detector SIFT`.

Chạy thử trên ảnh giả lập: ép 0.999 làm 1/3 cặp chạy thêm SIFT (+~710 ms) nhưng không cặp nào đạt ngưỡng → vẫn dùng ORB; H ORB và SIFT lệch nhau 0.01 px; số GT phát hiện giống hệt.

## 3. Phát hiện mới cho nhóm: nhãn GT chưa được chiếu qua H

`exp_feature_matching.py`, `exp_max_features.py`, `exp_detection_params.py`, `exp_blur_kernel.py` đều so bbox phát hiện
(toạ độ **template**, sau warp) với nhãn GT (toạ độ **ảnh test**) mà **không chiếu nhãn qua H**.
Với 690 cặp trùng khít thì không sao; với 3 cặp lệch sẵn 7 / 28 / 47 px thì phát hiện đúng bị chấm là "trượt + đốm giả".

- Kiểm chứng (ảnh giả, dịch 30 px): chiếu nhãn → **4/4 GT**; không chiếu → **0/4 GT, 4 đốm giả**.
- Có thể giải thích vì sao `06_spur_08` chỉ đạt 2/5 GT sau bản sửa mask (CODE_REVIEW §6.1).
- Ảnh hưởng: 13 GT trên tổng 2953 → recall ORB có thể lên tối đa **0.9655 thay vì 0.9611** (CSV cũ); xếp hạng cấu hình không đổi vì sai như nhau ở mọi cấu hình.
- Cách sửa: `check_sift_fallback.project_boxes(gt, H, shape)` → chuyển vào `evaluation.py` và gọi trong các script **sau khi job xong** (không sửa `exp_detection_params.py` lúc này).

## 4. File thay đổi

| File | Loại |
| :--- | :--- |
| `experiments/report_tables.py` | mới — sinh lại / đối chiếu / ghi đè bảng §2.1, §2.2, §2.4 |
| `experiments/plot_max_features.py` | mới — vẽ lại `max_features_study.png` từ CSV |
| `experiments/check_sift_fallback.py` | mới — việc #8 |
| `experiments/charts/max_features_study.png` | vẽ lại (thời gian theo trung vị) |
| `experiments/exp_max_features.py` | chỉ sửa lời cảnh báo cũ "lỗi kiểm tra homography" |
| `PROGRESS_REPORT.md` | sửa 14 ô số + lời nhận định §2.1 / §2.2 / §2.4, thêm dòng nguồn |
| `NUMBERS_AUDIT.md` | file này |

Việc còn lại sau khi job xong: chạy lại `exp_feature_matching.py` + `exp_max_features.py`, rồi
`report_tables.py --write` và `plot_max_features.py`; chạy `check_sift_fallback.py` (vài phút) và dán `sift_fallback_note.md` vào báo cáo.
