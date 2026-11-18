# Notebook thực nghiệm — thứ tự demo trước Thầy

Mỗi notebook trả lời **một** câu hỏi "vì sao nhóm chọn cách làm này", và **không** tự sinh
số liệu chính thức: số chính thức luôn đọc từ CSV trong `experiments/`, do script `.py` sinh ra.

| # | File | Câu hỏi | Nguồn số liệu | Thời gian chạy |
| --- | --- | --- | --- | --- |
| 1 | `rgb2binary.ipynb` | Vì sao tiền xử lý bằng Gray + CLAHE thay vì HSV/Lab? | `rgb2binary_results.csv` | ~15 s |
| 2 | `feature_matching.ipynb` | Vì sao chọn ORB thay vì SIFT/AKAZE? | `summary.csv`, `per_pair.csv` | ~40 s |
| 3 | `detection_params.ipynb` | Vì sao chốt `floor20_open2` và chấp nhận đánh đổi gì? | `detection_params_summary.csv` | ~20 s |
| 4 | `classification.ipynb` | Vì sao Random Forest, và vì sao báo cáo 7 lớp lẫn 6 lớp? | `classifier_dataset.csv`, `classify_results*.csv` | ~20 s |

## Cách chạy

```powershell
# Cài Jupyter một lần (hiện .venv chưa có)
.\.venv\Scripts\python.exe -m pip install jupyterlab ipykernel

# Mở notebook
.\.venv\Scripts\python.exe -m jupyter lab experiments\notebooks
```

Trước khi chụp ảnh báo cáo: **Kernel → Restart & Run All** để output khớp code hiện tại.

## Nguyên tắc chung

1. **Một nguồn logic**: notebook import hàm từ `experiments/exp_*.py` và các module gốc,
   không chép lại thuật toán.
2. **Chỉ đọc dữ liệu**: không ghi đè `data/`, `labels/`, `results/`, `crops/` hay CSV chính thức.
   Hình vẽ ghi ra `%TEMP%/pcb_notebook_charts/`.
3. **Số chính thức lấy từ CSV**, không lấy từ output cell — notebook chạy demo trên vài cặp,
   còn bảng trong `PROGRESS_REPORT.md` lấy từ lần chạy đủ 693 cặp.
4. **Xoá output trước khi commit**:

```powershell
jupyter nbconvert --clear-output --inplace experiments\notebooks\*.ipynb
```

## Sai lệch dữ liệu đã biết (nói thật khi bị hỏi)

* `experiments/summary.csv` chỉ có **ORB và SIFT** — số AKAZE full-693 nằm ở
  `PROGRESS_REPORT.md` §2.1 (lần chạy riêng, chưa lưu per-pair). `feature_matching.ipynb`
  chạy AKAZE trên 5 cặp để so cùng điều kiện và ghi rõ nhãn "demo".
  Muốn CSV đủ 3 bộ dò:
  `python experiments/exp_feature_matching.py --detectors ORB SIFT AKAZE` (~35 phút).
* Các số recall/precision/F1 trong `summary.csv` đo bằng default **cũ** `floor30/open3`;
  các chỉ số căn chỉnh (thời gian, inliers, reprojection, corner error) không phụ thuộc default.
* `detection_params_summary.csv` giữ `is_baseline=True` ở `floor30_open3` — đó là cấu hình gốc
  **lúc chạy lưới**, không phải default hiện tại (`floor20_open2`).
