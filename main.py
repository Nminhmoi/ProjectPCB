# Hinh: Tích hợp, Hiển thị & Thử nghiệm (main.py)
# Output: Hiển thị ảnh gốc với các ô chữ nhật màu đỏ bao quanh lỗi kèm báo cáo log chi tiết

import os
import time
import cv2
import numpy as np
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 1. Import module tiền xử lý
from module_preprocess import align_images, detect_difference

# 2. Import module phát hiện & cắt đốm lỗi
from module_detection import extract_bounding_boxes, crop_defect_patches


def load_and_verify_images(
    test_filename: str = '92000000_test.jpg',
    temp_filename: str = '92000000_temp.jpg',
    directory: str = 'data'
):
    if not os.path.exists(directory):
        if os.path.exists('/content/sample_data'):
            directory = '/content/sample_data'
        elif os.path.exists('data'):
            directory = 'data'

    test_path = os.path.join(directory, test_filename)
    temp_path = os.path.join(directory, temp_filename)

    # Kiểm tra sự tồn tại trên ổ đĩa vật lý
    if not os.path.exists(test_path) or not os.path.exists(temp_path):
        raise FileNotFoundError(
            f"\n[LỖI] Không tìm thấy file ảnh tại thư mục: '{directory}'\n"
        )

    # Đọc ảnh gốc ở dạng grayscale để xử lý vi sai
    img_test_gray = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    img_temp_gray = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)

    # Đọc thêm ảnh màu để phục vụ việc căn chỉnh hoặc vẽ khung
    img_test_color = cv2.imread(test_path)
    img_temp_color = cv2.imread(temp_path)

    if img_test_gray is None or img_temp_gray is None:
        raise ValueError("Không thể giải mã dữ liệu ảnh. File có thể bị hỏng!")

    return (img_test_gray, img_temp_gray), (img_test_color, img_temp_color)


# đóng khung đỏ đốm lỗi
def draw_results(img_test: np.ndarray, boxes: list) -> np.ndarray:
    # Đảm bảo ảnh vẽ là ảnh màu 3 kênh BGR
    if len(img_test.shape) == 2:
        img_annotated = cv2.cvtColor(img_test, cv2.COLOR_GRAY2BGR)
    else:
        img_annotated = img_test.copy()

    red_color = (0, 0, 255)  

    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        # 1. Vẽ ô chữ nhật màu đỏ bao quanh lỗi (độ dày nét = 2)
        cv2.rectangle(img_annotated, (x, y), (x + w, y + h), red_color, 2)

        # 2. Tạo nhãn chữ Defect #1, Defect #2...
        label = f"Defect #{idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thick = 1

        # Tính kích thước khung chứa nhãn
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
        label_y = max(y - 5, text_h + 5)
        label_x = x

        # Vẽ nền nhãn màu đỏ
        cv2.rectangle(
            img_annotated,
            (label_x, label_y - text_h - 2),
            (label_x + text_w + 4, label_y + baseline - 2),
            red_color,
            -1
        )
        # Viết chữ màu trắng lên nền đỏ
        cv2.putText(
            img_annotated,
            label,
            (label_x + 2, label_y - 2),
            font,
            font_scale,
            (255, 255, 255),
            font_thick,
            cv2.LINE_AA
        )
    return img_annotated

# hàm dọn output
def clean_output_directories(directories: list = None) -> int:
    """
    Xóa sạch các file ảnh trong thư mục output (mặc định: 'crops', 'results')
    trước khi chạy pipeline để tránh nhầm lẫn dữ liệu giữa các lần chạy.
    """
    if directories is None:
        directories = ['crops', 'results']

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    deleted_count = 0

    for directory in directories:
        if not os.path.exists(directory):
            continue

        for filename in os.listdir(directory):
            if filename.lower().endswith(valid_extensions):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_count += 1
                except Exception as exc:
                    print(f"[CẢNH BÁO] Không thể xóa file '{file_path}': {exc}")

    return deleted_count


# hàm main
def main():
    start_total_time = time.time()

    # Dọn dẹp output cũ trước khi chạy
    clean_output_directories(['crops', 'results'])

    # 1. Tải và kiểm tra ảnh đầu vào
    (img_test_gray, img_temp_gray), (img_test_color, img_temp_color) = load_and_verify_images(
        test_filename='92000000_test.jpg',
        temp_filename='92000000_temp.jpg',
        directory='data'
    )

    # 2. Căn chỉnh ảnh kiểm tra theo ảnh mẫu (ORB & Homography)
    try:
        img_test_aligned = align_images(img_temp_color, img_test_color)
    except Exception as exc:
        print(f"[CẢNH BÁO] Căn chỉnh thất bại: {exc} — dùng ảnh chưa căn chỉnh")
        img_test_aligned = img_test_color

    # 3. Trừ vi sai và nhị phân hóa Otsu, khử nhiễu Morphology
    diff_mask = detect_difference(img_temp_color, img_test_aligned)

    # 4. Tìm Contours và trích xuất Bounding Boxes
    boxes = extract_bounding_boxes(diff_mask, min_area=20)

    # 5. Cắt và lưu các ảnh con đốm lỗi (Crops)
    crop_dir = "crops"
    crop_paths = crop_defect_patches(img_test_aligned, boxes, output_dir=crop_dir, padding=5)

    # 6. Đóng khung đỏ và lưu ảnh kết quả vào thư mục 'results/'
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    img_result = draw_results(img_test_aligned, boxes)
    output_result_path = os.path.join(results_dir, "result_92000000.jpg")
    cv2.imwrite(output_result_path, img_result)

    # In báo cáo kết quả
    total_time = time.time() - start_total_time
    print(" KET QUA:    ")
    print(f"- Tổng số khuyết tật phát hiện: {len(boxes)}")
    print(f"- Tổng thời gian xử lý:         {total_time:.3f} giây\n")
    print(f"{'STT':<5} | {'Tọa độ (x, y)':<15} | {'Kích thước (w, h)':<19} | {'Diện tích':>10}")
    print("-" * 58)
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        coord_str = f"({x:>3}, {y:>3})"
        size_str = f"{w:>3} x {h:>3} px"
        area_str = f"{w * h:>5} px"
        print(f"#{idx:<4} | {coord_str:<15} | {size_str:<19} | {area_str:>10}")
    print("-" * 58)
    print(f"Đã lưu {len(crop_paths)} ảnh đốm lỗi trong thư mục '{crop_dir}/'")
    print(f"Ảnh kết quả trực quan hóa: {output_result_path}")
    print("=" * 65)


if __name__ == '__main__':
    main()

