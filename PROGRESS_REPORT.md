# BÁO CÁO TIẾN ĐỘ THỰC HIỆN ĐỒ ÁN (BUỔI 3)

---

## 1. TIẾP THU VÀ GIẢI QUYẾT CÁC GÓP Ý TỪ THẦY (BUỔI 1)

| STT | Vấn đề / Câu hỏi của Thầy | Hướng giải quyết & Minh chứng của Nhóm |
| :---: | :--- | :--- |
| **1** | Đánh giá chỉ số các phương pháp khác nhau, giải thích tại sao dùng ORB mà không dùng SIFT? | Xây dựng script benchmark đối sánh **ORB vs SIFT vs AKAZE** đo đạc định lượng: thời gian thực thi (ms), số lượng inlier, sai số tái chiếu (reprojection error). |
| **2** | Tại sao lại dùng `max_features = 5000`, có số liệu chứng minh không? | Thực hiện khảo sát tham số (Ablation Study) trên dải 500 đến 10.000 điểm đặc trưng, vẽ biểu đồ Pareto để chỉ ra 5000 là điểm tối ưu cân bằng tốc độ và độ chính xác. |
| **3** | Dữ liệu đầu vào: Xử lý đối với ảnh màu (chụp bằng camera công nghiệp) $\rightarrow$ chuyển sang ảnh nhị phân? | Chuyển đổi toàn bộ sang dataset ảnh màu công nghiệp **PKU-Market-PCB**, áp dụng kỹ thuật tách không gian màu (HSV/Lab) kết hợp cân bằng sáng CLAHE để khử phản quang kim loại. |
| **4** | Chạy xong phải có trực quan hóa (visual lines) các kết quả, sau khi homography khớp như thế nào? | Bổ sung 2 hình thức trực quan hóa: ảnh vẽ đường nối cặp đặc trưng (`drawMatches`) và ảnh bàn cờ (`checkerboard composite`) kiểm tra độ khít của đường mạch. |
| **5** | Làm mờ tại sao dùng 3x3, có dùng 5x5 hay 6x6 được không? Nêu điểm tối ưu? | Chứng minh cơ sở toán học (kernel phải lẻ để có tâm đối xứng) và thực nghiệm chỉ ra kernel 3x3 bảo toàn vết lỗi vi mô (micro-defects) tốt hơn kernel 5x5. |
| **6** | Quan trọng là phương pháp để làm, dữ liệu đánh giá và mô hình phân loại? | Đánh giá định lượng trên nhãn Ground Truth (IoU, Precision, Recall). Áp dụng **Học máy truyền thống (SVM / Random Forest)** qua đặc trưng hình thái học OpenCV (thay vì Deep Learning) để đảm bảo tính giải thích cao và tốc độ thời gian thực. |

---

## 2. KẾT QUẢ THỰC NGHIỆM ĐỐI SÁNH KHOA HỌC (BENCHMARKING)

### 2.1. Đối sánh bộ dò đặc trưng: ORB vs. SIFT
Thử nghiệm trên tập ảnh màu thực tế của PKU-Market-PCB:

| Bộ dò đặc trưng | Thời gian trích xuất & ghép (ms) | Số Keypoints trung bình | Số cặp Matches | Tỷ lệ Inliers RANSAC (%) | Sai số tái chiếu (Reprojection Error - px) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SIFT** | *... ms* | *...* | *...* | *... %* | *... px* |
| **ORB** | *... ms* | *...* | *...* | *... %* | *... px* |
| **AKAZE** | *... ms* | *...* | *...* | *... %* | *... px* |

> **Nhận xét & Kết luận:**  
> - ORB nhanh gấp **... lần** so với SIFT do sử dụng bộ mô tả nhị phân (Binary Descriptor) và khoảng cách Hamming, phù hợp với yêu cầu xử lý thời gian thực của camera công nghiệp.  
> - Sai số tái chiếu của ORB đạt **... px** (hoàn toàn đáp ứng tiêu chuẩn nắn chỉnh mạch dưới 1.5 px).  
> - ORB là thuật toán mở (free, open-source), không vướng bản quyền thương mại.

---

### 2.2. Khảo sát tham số `max_features = 5000` (Ablation Study)
Khảo sát ảnh hưởng của số lượng đặc trưng tối đa lên sai số căn chỉnh và thời gian thực thi:

*(Chèn biểu đồ thực nghiệm)*  
![Biểu đồ khảo sát max_features](experiments/charts/max_features_study.png)

* **Phân tích số liệu:**
  * Dưới 2000 features: Sai số tái chiếu lớn (> 2.5 px), một số vùng bo mạch ít chi tiết bị fail RANSAC.
  * Từ 3000 – 5000 features: Sai số giảm nhanh và tiệm cận mức bão hòa (< 1.0 px).
  * Trên 5000 features: Thời gian tính toán tăng mạnh tuyến tính nhưng sai số tái chiếu hầu như không giảm thêm.
* **Kết luận:** Lựa chọn `max_features = 5000` là điểm cân bằng Pareto tối ưu giữa tốc độ và độ chính xác căn chỉnh.

---

### 2.3. Trực quan hóa kết quả căn chỉnh Homography
Để kiểm tra trực quan độ chính xác của quá trình nắn ảnh (Homography):

#### a. Đường nối các cặp đặc trưng (Feature Matching Lines)
*(Hiển thị các cặp điểm khớp nhau giữa ảnh mẫu và ảnh kiểm tra)*  
![Feature Matches](results/feature_matches.jpg)

#### b. Kiểm tra độ khít bằng ảnh bàn cờ (Checkerboard Composite)
*(Ghép xen kẽ các ô cờ giữa ảnh mẫu và ảnh kiểm tra đã nắn Homography. Nếu căn chỉnh chuẩn, các đường mạch sẽ nối liền liên tục qua ranh giới các ô cờ mà không bị gãy hoặc lệch)*  
![Checkerboard Alignment](results/checkerboard_alignment.jpg)

---

### 2.4. Khảo sát kích thước Kernel làm mờ Gaussian (Blur)
* **Về mặt toán học (Tại sao không dùng kernel chẵn 6x6):**
  * Bộ lọc không gian (Spatial Filter) bắt buộc phải có kích thước **lẻ** ($3 \times 3, 5 \times 5$) để xác định duy nhất **tọa độ pixel tâm (Anchor Point)**: $\text{center} = (\lfloor k/2 \rfloor, \lfloor k/2 \rfloor)$. 
  * Kernel chẵn như $6 \times 6$ không có điểm tâm đối xứng, làm lệch pha không gian (spatial shift) và gây méo biên đường mạch.
* **Về mặt kích thước ($3 \times 3$ vs $5 \times 5$):**
  * Các khuyết tật trên PCB như lỗ rỗ (*pin-hole*), cầu chì đứt vi mô hay gai đồng (*spur*) chỉ có kích thước bề ngang từ $2 - 4\text{ px}$.
  * Kernel $5 \times 5$ làm nhòe quá mạnh, dẫn đến hiện tượng "nuốt chửng" các lỗi nhỏ hoặc tạo viền giả dày ở các cạnh đường mạch khi trừ vi sai.
  * Kernel $3 \times 3$ vừa đủ để triệt tiêu nhiễu hạt cảm biến camera công nghiệp mà vẫn bảo toàn nguyên vẹn cạnh vi sai của các lỗi nhỏ.

*(Ảnh so sánh trực quan lỗi vi mô khi dùng kernel 3x3 và 5x5)*  
![So sánh Blur](results/blur_comparison.png)

---

## 3. XỬ LÝ ẢNH MÀU CÔNG NGHIỆP & BÓC TÁCH ĐỐM LỖI
* **Xử lý ảnh màu (RGB $\rightarrow$ Binary):**
  * Áp dụng cân bằng sáng thích nghi **CLAHE** để triệt tiêu hiện tượng lóa sáng do đèn chiếu camera phản xạ trên bề mặt mối hàn kim loại.
  * Chuyển đổi sang không gian màu **HSV / Lab** để tách biệt lớp đồng với lớp phủ bảo vệ bo mạch (Solder Mask).
* **Kết quả bóc tách:** Toàn bộ các đốm khuyết tật được khoanh vùng Bounding Box và cắt lưu tự động vào thư mục `crops/`.

---

## 4. HƯỚNG TRIỂN KHAI TIẾP THEO (PHÂN LOẠI MACHINE LEARNING TRUYỀN THỐNG)
* **Trích xuất đặc trưng hình thái đốm lỗi (OpenCV):**
  * Đặc trưng hình học: Diện tích, Chu vi, Tỷ lệ khung (Aspect Ratio), Độ lồi/Độ đặc (Solidity, Extent).
  * Đặc trưng bất biến: 7 mô-men Hu (Hu Moments) để nhận diện hình dạng bất biến với phép xoay và tỷ lệ.
  * Đặc trưng kết cấu: Histogram of Oriented Gradients (HOG) cho biên cạnh khuyết tật.
* **Huấn luyện mô hình:** Sử dụng **SVM (Support Vector Machine)** hoặc **Random Forest** phân loại vào đúng 6 nhóm khuyết tật IPC-A-600.
* **Xây dựng Giao diện người dùng (GUI Demo):** Ứng dụng Desktop bấm nút nạp ảnh và hiển thị kết quả kiểm tra trực quan.
