# HỆ THỐNG KIỂM TRA CHẤT LƯỢNG & PHÁT HIỆN LỖI BO MẠCH PCB (SURFACE DEFECT DETECTION)

---

## 1. MỤC TIÊU HỆ THỐNG
Xây dựng pipeline thị giác máy tính tự động so sánh cặp ảnh bo mạch in PCB: **Ảnh chuẩn (Template/Defect-free)** và **Ảnh kiểm tra (Test/Defective)** chụp từ camera công nghiệp nhằm:
1. **Phát hiện bất thường (Anomaly Detection):** Nhận diện có tồn tại sai khác/khuyết tật trên bề mặt bo mạch hay không.
2. **Định vị lỗi (Defect Localization):** Xác định chính xác tọa độ Bounding Box $[x, y, w, h]$ của từng vết lỗi bằng phương pháp trừ vi sai và hình thái học.
3. **Phân loại khuyết tật (Defect Classification):** Sử dụng **Học máy truyền thống (Classical Machine Learning: SVM / Random Forest)** kết hợp trích xuất đặc trưng hình thái học (Morphological Features, Hu Moments, HOG) bằng OpenCV để phân loại chính xác vào **6 nhóm khuyết tật** theo tiêu chuẩn IPC-A-600.
4. **Đánh giá thực nghiệm khoa học (Scientific Benchmarking):** Có đầy đủ số liệu thực nghiệm, chỉ số định lượng (Precision, Recall, F1-Score, IoU, Reprojection Error) và biểu đồ trực quan hóa để chứng minh cơ sở lựa chọn thuật toán/tham số.

---

## 2. GIỚI HẠN PHẠM VI DỰ ÁN

### Trong phạm vi (In-Scope):
* **Đối tượng kiểm tra:** Bo mạch in trần (**Bare PCB**) 2D phẳng, chưa hàn gắn linh kiện điện tử (không bao gồm IC, tụ, mối hàn PCBA).
* **Bộ dữ liệu chính:** **PKU-Market-PCB Dataset**
  * Ảnh màu thực tế chụp từ camera công nghiệp (độ phân giải cao, có góc xoay, lệch tịnh tiến, ánh sáng và phản xạ kim loại thực tế).
  * Đầy đủ nhãn Bounding Box và phân loại 6 loại lỗi.
* **Bộ dữ liệu đối chứng (Baseline / Sanity Check):** **DeepPCB Dataset** (Cặp ảnh $640 \times 640\text{ px}$ nhị phân đã tiền xử lý căn chỉnh chuẩn trong phòng lab để kiểm chứng thuật toán).
* **6 nhóm khuyết tật mục tiêu (theo IPC-A-600):**
  1. Hở mạch (Open circuit)
  2. Chập mạch (Short circuit)
  3. Lỗ chuột gặm (Mousebite)
  4. Gai đồng (Spur)
  5. Đồng thừa / Thiếu đồng (Spurious copper)
  6. Lỗi lỗ khoan (Pin-hole / Missing hole)
* **Phương pháp kỹ thuật:** Thuần túy **Thị giác máy tính cổ điển & Học máy truyền thống (Classical CV & Traditional ML)** bằng OpenCV và Scikit-Learn. Tuyệt đối **không sử dụng Deep Learning** (theo yêu cầu giảng viên để đảm bảo tính giải thích cao, gọn nhẹ và tối ưu thời gian xử lý thực tế).

### Ngoài phạm vi (Out-of-Scope):
* Không kiểm tra bo mạch đã hàn linh kiện (PCBA/SMT 3D).
* Không điều khiển cơ khí băng chuyền vật lý trong phạm vi đồ án.

---

## 3. SƠ ĐỒ PIPELINE XỬ LÝ HỆ THỐNG

Pipeline xử lý ảnh màu end-to-end gồm chuỗi các bước tuần tự:

```text
[Ảnh Mẫu (Template - RGB)] ──┐
                             ├──► [1. Tiền xử lý ảnh màu & Khử phản quang]
[Ảnh Kiểm Tra (Test - RGB)] ──┘      (Chuyển đổi HSV/Lab, Cân bằng CLAHE)
                                                    │
                                                    ▼
                              [2. Trích xuất & Ghép đặc trưng (ORB / SIFT)]
                                                    │
                                                    ▼
                              [3. Trực quan hóa Khớp đặc trưng (Visual Matching Lines)]
                                                    │
                                                    ▼
                              [4. Ước lượng Homography (RANSAC) & Nắn ảnh]
                                                    │
                                                    ▼
                              [5. Trực quan hóa độ khít (Checkerboard / Alpha Blend)]
                                                    │
                                                    ▼
                              [6. Lọc mờ Gaussian (3x3) & Trừ vi sai đa ngưỡng]
                                                    │
                                                    ▼
                              [7. Nhị phân hóa Otsu & Khử nhiễu Morphology Opening]
                                                    │
                                                    ▼
                              [8. Dò biên Contours & Định vị Bounding Boxes]
                                                    │
                                                    ▼
                              [9. Cắt đốm lỗi (Crops) & Trích xuất đặc trưng hình thái]
                                                    │
                                                    ▼
                              [10. Phân loại khuyết tật bằng Machine Learning (SVM/RF)]
                                                    │
                                                    ▼
                              [11. Xuất báo cáo, Hiển thị Giao diện & Đánh giá định lượng]
```

---

## 4. KẾ HOẠCH THỰC NGHIỆM ĐỐI SÁNH & CƠ SỞ KHOA HỌC (BENCHMARKING)

Nhóm xây dựng các kịch bản thực nghiệm định lượng để bảo vệ các quyết định kỹ thuật:

### 1. Đối sánh bộ dò đặc trưng: ORB vs. SIFT (vs. AKAZE)
* **Mục tiêu:** Chứng minh lý do chọn ORB thay vì SIFT bằng số liệu định lượng.
* **Chỉ số đo đạc:**
  * Thời gian trích xuất & ghép cặp (Latency - ms).
  * Số lượng điểm đặc trưng (Keypoints) và số cặp khớp hợp lệ (Valid Matches).
  * Tỷ lệ điểm khớp đúng RANSAC (Inlier Ratio % = $\frac{\text{Inliers}}{\text{Matches}}$).
  * Sai số tái chiếu (Reprojection Error tính bằng pixel qua phép chiếu ngược).
* **Luận điểm:** SIFT cho độ chính xác cao nhưng tốc độ chậm; ORB sử dụng descriptor nhị phân (Hamming distance) tính toán cực nhanh, phù hợp camera công nghiệp thời gian thực và là thuật toán mã nguồn mở tự do.

### 2. Khảo sát số lượng đặc trưng (`max_features = 5000`)
* **Mục tiêu:** Thể hiện bằng số liệu thực nghiệm tại sao 5000 là điểm tối ưu (Ablation Study).
* **Phương pháp:** Chạy thử nghiệm trên dải giá trị: `[500, 1000, 2000, 3000, 5000, 8000, 10000]`.
* **Biểu đồ cần xuất:** Đồ thị quan hệ 2 trục $Y$ giữa **Sai số tái chiếu (Reprojection Error)** và **Thời gian xử lý (Execution Time)** theo trục $X$ (`max_features`) để chỉ ra điểm cân bằng Pareto tại 5000.

### 3. Phương pháp tiền xử lý ảnh màu công nghiệp $\rightarrow$ Ảnh nhị phân
* **Mục tiêu:** Giải quyết bài toán ảnh màu chụp từ camera công nghiệp bị bóng phản xạ từ bề mặt kim loại/vết hàn.
* **Đối sánh 3 phương pháp:**
  1. *Baseline:* RGB $\rightarrow$ Grayscale $\rightarrow$ Otsu thông thường (dễ bị nhiễu bóng kim loại).
  2. *Không gian màu HSV/Lab:* Tách kênh màu để phân tách đường đồng/mối hàn khỏi lớp phủ xanh (Solder Mask).
  3. *Thích nghi CLAHE:* Cân bằng sáng cục bộ thích nghi để triệt tiêu ảnh hưởng ánh sáng không đồng đều từ đèn chiếu camera.

### 4. Tối ưu kích thước Kernel làm mờ: Tại sao 3x3 mà không phải 5x5 hay 6x6?
* **Cơ sở toán học:** Kernel tích chập làm mờ bắt buộc phải có kích thước **lẻ** ($3 \times 3, 5 \times 5, 7 \times 7$) để đảm bảo có **tâm đối xứng (center pixel)** rõ ràng. Kích thước chẵn như $6 \times 6$ làm lệch tâm tọa độ và méo ảnh.
* **Thực nghiệm đối chứng $3 \times 3$ vs $5 \times 5$:** Đường mạch PCB có độ rộng chi tiết rất nhỏ (micro-traces), các lỗi li ti như lỗ rỗ (pin-hole) hoặc gai đồng nhỏ chỉ rộng vài pixel. Kernel $5 \times 5$ làm mờ quá mức sẽ "nuốt chửng" các lỗi nhỏ hoặc gây viền giả ở biên. Kernel $3 \times 3$ là kích thước tối ưu bảo toàn biên cạnh (Edge Preservation) và loại bỏ tốt nhiễu hạt cảm biến.

### 5. Trực quan hóa căn chỉnh Homography
* Xuất ảnh nối cặp đặc trưng (`drawMatches`) với phân biệt Inlier (xanh lá) và Outlier (đỏ).
* Xuất ảnh bàn cờ (`checkerboard_blend`) và ảnh phủ mờ (`alpha_blend`) để người dùng nhìn thấy trực tiếp độ khớp chính xác sau khi nắn Homography.

---

## 5. PHÂN CHIA NHIỆM VỤ THÀNH VIÊN & LỊCH TRÌNH THEO DÕI TIẾN ĐỘ

Để vừa đảm bảo hoàn thiện hệ thống chạy thật (End-to-End), vừa có đủ số liệu thực nghiệm khoa học kịp buổi thông tiến độ với Thầy (**Thứ 5 tuần sau - 19/11/2026**), công việc được phân công cụ thể và theo dõi theo mốc thời gian (Milestones) như sau:

### 👤 5.1. Phân công chi tiết theo Thành viên

#### 1. Nam Anh — Tiền xử lý, Căn chỉnh & Khảo sát Đặc trưng
* **Trách nhiệm Hệ thống:**
  * Xây dựng pipeline tiền xử lý ảnh màu trong `module_preprocess.py`: nạp ảnh màu, chuyển đổi không gian màu (HSV/Lab), áp dụng CLAHE cân bằng sáng.
  * Hoàn thiện hàm căn chỉnh ảnh màu `align_images()` bằng ma trận Homography và RANSAC.
* **Trách nhiệm Thực nghiệm & Báo cáo:**
  * Viết script thực nghiệm `exp_feature_matching.py`: Đối sánh định lượng **ORB vs SIFT vs AKAZE** (thời gian, inlier ratio, reprojection error).
  * Viết script khảo sát `exp_max_features.py`: Chạy từ 500 $\rightarrow$ 10.000 điểm đặc trưng và xuất biểu đồ Pareto tìm điểm tối ưu 5000.
  * Viết script khảo sát kích thước Kernel làm mờ Gaussian ($3 \times 3$ vs $5 \times 5$), giải thích toán học về kernel lẻ/chẵn.

#### 2. Minh — Định vị khuyết tật & Phân loại Machine Learning
* **Trách nhiệm Hệ thống:**
  * Tinh chỉnh `module_detection.py`: Trừ vi sai có dung sai vị trí trên ảnh màu đã xử lý, nhị phân hóa Otsu, Morphology Opening lọc nhiễu, trích xuất Bounding Box chính xác.
  * Phát triển `module_classification.py`:
    * Trích xuất đặc trưng hình thái đốm lỗi: Diện tích (Area), Độ tròn/Độ lồi (Solidity, Extent, Aspect Ratio), Hu Moments (7 bất biến Hu), HOG đặc trưng biên cạnh.
    * Huấn luyện mô hình học máy truyền thống (**SVM / Random Forest** qua OpenCV ML hoặc Scikit-Learn) phân loại 6 loại lỗi.
* **Trách nhiệm Thực nghiệm & Báo cáo:**
  * Thực nghiệm đối sánh các phương pháp chuyển đổi RGB $\rightarrow$ Binary (Grayscale thô vs HSV/Lab vs CLAHE).
  * Đánh giá độ chính xác phân loại của mô hình học máy: Vẽ Ma trận nhầm lẫn (Confusion Matrix), tính Accuracy trên 6 nhóm khuyết tật.

#### 3. Hinh — Tích hợp, Trực quan hóa, Giao diện & Đánh giá Chỉ số
* **Trách nhiệm Hệ thống:**
  * Xây dựng module trực quan hóa kết quả:
    * `draw_feature_matches`: Vẽ các đường nối điểm khớp đặc trưng.
    * `draw_homography_verification`: Xuất ảnh bàn cờ (Checkerboard) và ảnh phủ mờ (Alpha Overlay) để trực quan hóa độ khít sau căn chỉnh.
    * `draw_results`: Đóng khung chữ nhật đỏ kèm nhãn phân loại khuyết tật.
  * Xây dựng giao diện ứng dụng Desktop trực quan (**GUI Demo** bằng Tkinter / PyQt) cho phép người dùng chọn ảnh mẫu, ảnh test, bấm nút kiểm tra và hiển thị toàn bộ kết quả trực quan.
* **Trách nhiệm Thực nghiệm & Báo cáo:**
  * Xây dựng module đánh giá định lượng `evaluate_metrics.py`: Đọc nhãn Ground Truth (tọa độ file `.txt` của dataset), tính toán các chỉ số: **IoU, Precision, Recall, F1-Score**.
  * Chạy đánh giá đo đạc trên tập ảnh màu thực tế PKU-Market-PCB.
  * Quản lý cập nhật số liệu và hình ảnh vào file `PROGRESS_REPORT.md`.

---

### 📅 5.2. Lịch trình chi tiết theo dõi tiến độ (Roadmap đến Thứ 5 tuần sau)

| Giai đoạn & Thời gian | Công việc cần hoàn thành | Phụ trách | Sản phẩm đầu ra (Deliverables) |
| :--- | :--- | :---: | :--- |
| **Giai đoạn 1: Nền tảng & Thực nghiệm cốt lõi**<br>*(Thứ 2 - Thứ 3: 09/11 - 10/11)* | - Đưa ảnh màu PKU-Market-PCB vào thư mục `data/`.<br>- Viết script benchmark **ORB vs SIFT vs AKAZE**.<br>- Khảo sát tham số **`max_features`** (500 $\rightarrow$ 10.000).<br>- Thử nghiệm kernel Gaussian Blur ($3 \times 3$ vs $5 \times 5$).<br>- Thử nghiệm chuyển RGB $\rightarrow$ Binary (Grayscale vs HSV vs CLAHE). | **Nam Anh**<br>**Minh** | - Script `exp_feature_matching.py` + Bảng số liệu.<br>- Biểu đồ `max_features_study.png` trong `experiments/charts/`.<br>- Điền kết quả vào Mục 2.1, 2.2, 2.4 của `PROGRESS_REPORT.md`. |
| **Giai đoạn 2: Trực quan hóa & Bóc tách đốm lỗi**<br>*(Thứ 4 - Thứ 6: 11/11 - 13/11)* | - Hoàn thiện hàm vẽ đường match và ảnh bàn cờ Checkerboard.<br>- Căn chỉnh Homography trơn tru trên ảnh màu PKU-Market-PCB.<br>- Bóc tách và cắt các đốm lỗi lưu vào thư mục `crops/`.<br>- Viết hàm trích xuất vector đặc trưng hình thái (Hu Moments, HOG). | **Hinh**<br>**Nam Anh**<br>**Minh** | - Ảnh `feature_matches.jpg` & `checkerboard_alignment.jpg`.<br>- Thư mục `crops/` chứa các đốm lỗi đã cắt.<br>- Module `module_classification.py` (phần extract features).<br>- Cập nhật Mục 2.3 của `PROGRESS_REPORT.md`. |
| **Giai đoạn 3: Huấn luyện Machine Learning & Đánh giá định lượng**<br>*(Thứ 7 - Thứ 3: 14/11 - 17/11)* | - Huấn luyện mô hình **SVM / Random Forest** phân loại 6 dạng lỗi.<br>- Viết script `evaluate_metrics.py` đọc Ground Truth và tính IoU, Precision, Recall, Confusion Matrix.<br>- Dựng giao diện Desktop GUI cơ bản để bấm nút chạy demo. | **Minh**<br>**Hinh** | - File mô hình phân loại (`model_classifier.pkl`).<br>- Biểu đồ Confusion Matrix & Bảng IoU/Precision/Recall.<br>- Giao diện `gui_app.py` hoạt động được. |
| **Giai đoạn 4: Tổng duyệt & Chuẩn bị thông**<br>*(Thứ 4: 18/11)* | - Rà soát toàn bộ file `PROGRESS_REPORT.md` (đảm bảo đầy đủ số liệu, ảnh biểu đồ, minh chứng trực quan).<br>- Test chạy luồng `main.py` và `gui_app.py` demo trực tiếp trên máy không phát sinh lỗi ngoại lệ. | **Cả nhóm** | - Báo cáo `PROGRESS_REPORT.md` hoàn thiện 100%.<br>- Hệ thống sẵn sàng chạy demo trực tiếp trước mặt Thầy. |

---

## 6. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
ProjectPCB/
├── README.md                   # Tài liệu tổng quan, kiến trúc và phân công nhiệm vụ
├── requirements.txt            # Danh sách thư viện (opencv-python, numpy, scikit-learn, matplotlib)
├── main.py                     # Luồng chạy kiểm tra hoàn chỉnh từ đầu đến cuối (CLI)
├── gui_app.py                  # Giao diện người dùng Desktop thao tác trực quan
├── module_preprocess.py        # Tiền xử lý ảnh màu, CLAHE, căn chỉnh ảnh Homography
├── module_detection.py         # Trừ vi sai, nhị phân hóa, bóc tách Bounding Box & cắt ảnh
├── module_classification.py    # Trích xuất đặc trưng hình học/Hu Moments và mô hình ML (SVM/RF)
├── data/                       # Dữ liệu ảnh bo mạch
│   └── PKU-Market-PCB/         # Dataset chính: ảnh màu công nghiệp độ phân giải cao kèm nhãn
├── experiments/                # Các script thực nghiệm đối sánh khoa học
│   ├── exp_feature_matching.py # Đo đạc đối sánh ORB vs SIFT vs AKAZE
│   ├── exp_max_features.py     # Khảo sát tham số max_features (500 -> 10000)
│   ├── exp_blur_kernel.py      # So sánh kernel 3x3 vs 5x5
│   ├── evaluate_metrics.py     # Đo Precision, Recall, F1, IoU dựa trên Ground Truth
│   └── charts/                 # Thư mục lưu các biểu đồ tự động sinh ra phục vụ báo cáo
├── crops/                      # Thư mục lưu các đốm lỗi vừa cắt ra
└── results/                    # Ảnh kết quả trực quan hóa (khung đỏ, đường match, checkerboard)
```

---

## 7. TIÊU CHUẨN NGHIỆM THU

### Tiêu chuẩn Kỹ thuật (Hệ thống):
1. **Pipeline hoàn chỉnh:** Chạy trơn tru trên ảnh màu thực tế của PKU-Market-PCB từ khâu nạp ảnh $\rightarrow$ căn chỉnh $\rightarrow$ phát hiện $\rightarrow$ phân loại đốm lỗi $\rightarrow$ xuất kết quả.
2. **Căn chỉnh chính xác:** Sai số tái chiếu trung vị $< 1.5\text{ px}$, không bị lỗi bóng ma hay viền trắng khi trừ vi sai.
3. **Độ nhạy phát hiện:** Đạt độ bao phủ phát hiện lỗi $\text{Recall} \ge 90\%$ trên tập kiểm thử thực tế.
4. **Phân loại khuyết tật:** Mô hình học máy truyền thống phân loại chính xác các dạng lỗi cơ bản với F1-score $\ge 80\%$.
5. **Giao diện người dùng:** Ứng dụng Desktop chạy ổn định, trực quan, hỗ trợ hiển thị ảnh gốc, ảnh căn chỉnh, đốm lỗi và nhãn khuyết tật.

### Tiêu chuẩn Báo cáo & Phản biện (Khoa học):
1. **Bảng số liệu đối sánh ORB vs SIFT:** Có đầy đủ chỉ số về thời gian (ms), tỷ lệ inliers (%), sai số tái chiếu (px) để bảo vệ lý do chọn ORB.
2. **Biểu đồ chứng minh `max_features = 5000`:** Biểu đồ Pareto thể hiện điểm đánh đổi tối ưu giữa độ chính xác và tốc độ.
3. **Minh chứng toán học & thực nghiệm Blur:** Giải thích rõ lý do dùng kernel lẻ ($3 \times 3$ vs $5 \times 5$), chứng minh bảo toàn các khuyết tật vi mô.
4. **Trực quan hóa sinh động:** Đầy đủ ảnh minh họa đường nối feature matching và ảnh bàn cờ checkerboard chứng minh độ khít của Homography.
5. **Bảng số liệu định lượng:** Bảng đo IoU, Precision, Recall, Confusion Matrix trên tập dữ liệu chuẩn.
