# HỆ THỐNG KIỂM TRA CHẤT LƯỢNG & PHÁT HIỆN LỖI BO MẠCH PCB (SURFACE DEFECT DETECTION)

---

## 1. MỤC TIÊU HỆ THỐNG
Xây dựng pipeline xử lý ảnh số tự động so sánh cặp ảnh bo mạch in PCB: **Ảnh chuẩn (Template/Defect-free)** và **Ảnh kiểm tra (Test/Defective)** nhằm:
1. **Phát hiện bất thường (Anomaly Detection):** Nhận diện có tồn tại lỗi trên bề mặt bo mạch hay không.
2. **Định vị lỗi (Defect Localization):** Xác định chính xác tọa độ Bounding Box $[x, y, w, h]$ của từng vết lỗi.
3. **Phân loại chi tiết nhãn lỗi (Defect Classification):**
   * Phân loại chính xác từng vết lỗi vừa định vị vào đúng 1 trong **6 nhóm khuyết tật** theo tiêu chuẩn công nghiệp IPC-A-600: *Hở mạch (Open), Chập mạch (Short), Lỗ chuột gặm (Mousebite), Gai đồng (Spur), Đồng thừa (Spurious copper), Lỗi lỗ khoan (Missing hole)*.

---

## 2. GIỚI HẠN PHẠM VI DỰ ÁN

### Trong phạm vi (In-Scope):
* **Đối tượng kiểm tra:** Bo mạch in trần (**Bare PCB**) 2D phẳng, chưa gắn linh kiện điện tử (không bao gồm chip, tụ, mối hàn PCBA).
* **Bộ dữ liệu chuẩn:** **DeepPCB Dataset** (Cặp ảnh kích thước $640 \times 640\text{ px}$, nhị phân/grayscale đã khử nhiễu môi trường).
* **Dataset phụ:**  PKU-Market-PCB(Ảnh màu độ phân giải cao). dùng ở giai đoạn cuối để kiểm tra thuật toán nhà làm.
* **6 nhóm khuyết tật mục tiêu (theo IPC-A-600):**
  1. Hở mạch (Open circuit)
  2. Chập mạch (Short circuit)
  3. Lỗ chuột gặm (Mousebite)
  4. Gai đồng (Spur)
  5. Đồng thừa / Thiếu đồng (Spurious copper)
  6. Lỗi lỗ khoan (Pin-hole / Missing hole)
* **Phương pháp kỹ thuật (Milestone 2):** Thuần túy **Xử lý ảnh số cổ điển (Classical Computer Vision)** bằng OpenCV, sử dụng machine learning để phân loại.

### Ngoài phạm vi (Out-of-Scope):
* Không kiểm tra bo mạch đã hàn linh kiện (PCBA/SMT 3D).
* Không tích hợp phần cứng camera công nghiệp hay băng chuyền vật lý trong phạm vi đồ án này.
* Không yêu cầu thời gian thực cực cao ($< 5\text{ms}$), ưu tiên độ chính xác định vị ($\text{IoU} \ge 0.33$).

---

## 3. SƠ ĐỒ PIPELINE XỬ LÝ

Pipeline xử lý ảnh gồm chuỗi 8 bước tuần tự:
```
[Ảnh Mẫu (Template)] ────────────────────────┐
                                             ▼
[Ảnh Kiểm Tra (Test)] ──► [1. ORB + Homography] ──► [2. Căn chỉnh ảnh]
                                                          │
          ┌───────────────────────────────────────────────┘
          ▼
[3. Gaussian Blur (3x3)] ──► [4. Trừ vi sai ] ──► [5. Nhị phân hóa Otsu]
                                                                  │
          ┌───────────────────────────────────────────────────────┘
          ▼
[6. Morphology Opening (3x3)] ──► [7. Contours & Lọc diện tích (Area >= 20px)]
                                         │
                                         ▼
                 [8. Trích xuất Bounding Box & Cắt ảnh đốm lỗi (Crops)]
```

---

## 4. PHÂN CHIA NHIỆM VỤ 

### Buổi 2:
#### Nam Anh: Tiền xử lý & Trừ vi sai (`module_preprocess.py`)
* `align_images(img_template, img_test) -> img_test_aligned`
  * Dò đặc trưng ORB, khớp BFMatcher, lọc RANSAC tính ma trận Homography để nắn ảnh kiểm tra khớp với ảnh mẫu.
* `detect_difference(img_template, img_test_aligned) -> diff_binary`
  * Làm mờ Gaussian Blur ($3 \times 3$), trừ vi sai để khử viền giả do lệch căn chỉnh, nhị phân hóa ngưỡng, áp dụng phép mở `cv2.morphologyEx(MORPH_OPEN)` với Kernel $3 \times 3$ để tẩy sạch nhiễu hạt.

#### Minh: Định vị & Trích xuất đốm lỗi (`module_detection.py`)
* `extract_bounding_boxes(diff_binary, min_area=20) -> list[tuple(x, y, w, h)]`
  * Dò biên bằng `cv2.findContours`, tính diện tích từng đốm, loại bỏ đốm nhiễu hạt bụi $< 20\text{ px}$, trích xuất danh sách Bounding Box.
* `crop_defect_patches(img_test, boxes, output_dir="crops") -> list[str]`
  * Tự động cắt từng đốm lỗi kèm lề an toàn (padding 5px) và lưu thành file ảnh trong thư mục `crops/`.

#### Hinh: Tích hợp, Hiển thị & Thử nghiệm (`main.py`)
* `draw_results(img_test, boxes) -> img_annotated`
  * Đóng khung chữ nhật màu đỏ (`cv2.rectangle`) quanh từng đốm lỗi, đánh số thứ tự Defect #1, #2...
* Viết luồng chạy chính `main.py`, in log chi tiết ra Terminal (số lượng lỗi, tọa độ, thời gian xử lý) và lưu ảnh kết quả vào `results/`.
* Chạy kiểm thử trên 10 cặp ảnh DeepPCB để chuẩn bị báo cáo tiến độ.

### Buổi 3:
 #### Nam Anh — Script Đánh giá Định lượng & Đo đạc Thực nghiệm, Thử nghiệm & Thích ứng trên Dataset PKU-Market-PCB:
* Viết script `evaluate.py` tự động chạy hàng loạt trên 500 ảnh test của DeepPCB.
  * Đo đạc các chỉ số: IoU trung bình, Precision, Recall, Miss Rate.
  * Lập bảng số liệu và biểu đồ ma trận nhầm lẫn (Confusion Matrix) để đưa vào báo cáo tiến độ.pth`) và viết hàm `predict_defect(crop_img) -> label_name`.
  * Bổ sung bước tiền xử lý ảnh màu (chuyển đổi HSV/Lab, cân bằng sáng CLAHE để khử bóng phản xạ bề mặt kim loại).
  * Chạy thử nghiệm và phân tích ưu/nhược điểm của thuật toán khi gặp ảnh bo mạch thực tế.
  
 #### Minh — Dữ liệu & Huấn luyện Mô hình Phân loại (`module_classification.py`), Đóng gói ứng dụng Desktop:
  * Thu thập và gán nhãn các đốm lỗi từ tập train để tạo bộ dữ liệu huấn luyện
  * Trích xuất đặc trưng và huấn luyện mô hình phân loại 6 loại lỗi (CNN)
  * Xuất file mô hình đã huấn luyện (VD: `model.pkl`).
  * Tối ưu hóa kích thước mô hình và tốc độ nạp ảnh.
  * Đóng gói mã nguồn hoàn chỉnh (nếu có thể thì đóng gói file .exe (PyInstaller))
  
  #### Hinh — Cắt lỗi & Xây dựng Giao diện Desktop (`gui_app.py`):
  * Thiết kế và lập trình giao diện Desktop: kết nối pipeline từ tiền xử lý, định vị đến phân loại lên giao diện bấm nút.
  * Rà soát, chuẩn hóa toàn bộ mã nguồn của nhóm, bổ sung chú thích rõ ràng.
 #### Hinh - Đức: Tổng kết 
  * Hoàn thiện Báo cáo (trình bày đầy đủ cơ sở lý thuyết, kiến trúc pipeline và bảng số liệu thực nghiệm).
  * Thiết kế Slide báo cáo.

## 5. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
Project-PCB/
├── README.md                 # Tổng quan và phân việc
├── requirements.txt          # Danh sách thư viện (opencv-python, numpy, matplotlib)
├── data/                     # Dataset 
│   ├── PCBData/              # DeepPCB
│   └── PKU-Market-PCB/       # Dataset phụ (ảnh màu)
├── crops/                    # Thư mục lưu các đốm lỗi vừa cắt ra
├── results/                  # Ảnh kết quả đã đóng khung đỏ
├── module_preprocess.py      
├── module_detection.py       
├── main.py                   
```


---

## 6. TIÊU CHUẨN NGHIỆM THU BUỔI 2 
1. Chạy lệnh Terminal `python main.py` không phát sinh lỗi ngoại lệ.
2. Căn chỉnh ảnh chính xác, ảnh trừ vi sai không bị viền trắng bao quanh đường mạch.
3. Không bỏ sót lỗi mắt thường nhìn thấy rõ (Đảm bảo $\text{Recall} \ge 95\%$).
4. Các đốm lỗi được cắt chính xác và lưu đầy đủ vào thư mục `crops/`.
## 7, TIÊU CHUẨN NGHIỆM THU BUỔI 3
1. File mô hình phân loại (`model.pkl` hoặc tương đương) đạt độ chính xác cao trên các đốm lỗi cắt ra.
2. Ứng dụng Desktop chạy ổn định, thao tác chọn ảnh và bấm nút kiểm tra trực tiếp.
3. Bảng số liệu đo đạc thực nghiệm định lượng chi tiết trên 500 ảnh Test của DeepPCB.
4. Mã nguồn hoàn hiện
5. Bảng đánh giá và phân tích so sánh trên 2 tập dữ liệu (DeepPCB và PKU-Market-PCB).
6. Báo cáo và slide