# Kết quả Giai đoạn 1 – gửi Nam Anh đồng bộ báo cáo

Chào Nam Anh, mình đã chốt xong phần RGB → Binary trên bộ PKU-Market-PCB. Tóm tắt để cậu cập nhật báo cáo chung:

## 1. Dữ liệu
- Bộ gốc PKU-Market-PCB (10 bo mạch: 01, 04–12; 6 lớp lỗi) đặt tại `data/PKU-Market-PCB/raw/`.
- Giữ nguyên cấu trúc gốc, không nhân bản ảnh; script `experiments/build_pairs.py` sinh manifest + nhãn.
- Manifest: `data/PKU-Market-PCB/pairs_manifest.csv` (693 cặp, 2.953 bbox).
- Nhãn: `data/PKU-Market-PCB/labels/<pair_id>.txt` (`x y w h class_id`), bảng lớp: `classes.csv`.
- Lưu ý: chỉ `raw/` (~1.9 GB) nằm trong `.gitignore`; manifest + labels được commit. Cậu chỉ cần bộ ảnh gốc, không cần copy ảnh qua chat.

## 2. Kết quả khảo sát (693 cặp)

| Phương pháp | Thời gian (ms) | Otsu eta | Foreground (%) | Bảo toàn biên cạnh | IoU lệch sáng |
| :--- | :---: | :---: | :---: | :---: | :---: |
| gray_otsu | **9.03** | 0.727 | 12.42 | 0.246 | 0.786 |
| lab_b_otsu | 20.54 | 0.768 | 67.60 | 0.285 | **0.934** |
| lab_a_otsu | 20.87 | **0.751** | 20.99 | 0.194 | 0.866 |
| clahe_otsu | 17.12 | 0.638 | 17.55 | **0.356** | 0.870 |

**Phương pháp thắng: CLAHE (clipLimit 2.0, tile 8×8) + Gaussian 3×3 + Otsu.**
Lý do: bảo toàn biên cạnh cao nhất (0.356, +45% so với gray), IoU lệch sáng 0.870, thời gian 17.12 ms vẫn đạt thời gian thực. Eta thấp hơn do CLAHE kéo giãn histogram cục bộ, không phản ánh chất lượng bóc tách.

## 3. Tích hợp & kiểm chứng
- Đã thêm `_gray_clahe()` vào `module_preprocess.py` và dùng trong `detect_difference()`.
- `main.py` có thêm `--pair`, `--limit`, `--data-dir`.
- Chạy thử `py -3.14 main.py --pair 01_short_01`: phát hiện **3/3 lỗi Short**, **1.04 s/cặp**, bounding box đúng vị trí Ground Truth (hộp lõi nhỏ hơn GT).

## 4. File cần lấy số liệu
- `experiments/rgb2binary_results.csv` – số liệu thô từng cặp.
- `experiments/charts/rgb2binary_metrics.png` – biểu đồ 4 chỉ số.
- `experiments/charts/rgb2binary_visual.png` – mask nhị phân trực quan.
- `PROGRESS_REPORT.md` mục 3 – bảng + nhận xét đã điền.

## 5. Việc cần cậu đồng bộ
- Mục 2.1 / 2.2 (ORB vs SIFT vs AKAZE, khảo sát `max_features`): điền số liệu bên cậu và dùng cùng bộ 693 cặp để so sánh công bằng.
- Mục 2.4 (kernel 3×3 vs 5×5): nếu có ảnh so sánh thì gửi mình chèn vào báo cáo.
- Xác nhận giúp mình hạn Giai đoạn 1 (10/11) đã trễ 1 ngày; mình đang chốt trong hôm nay 11/11.
