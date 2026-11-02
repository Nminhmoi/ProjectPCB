# Hinh: Tích hợp, Hiển thị & Thử nghiệm (main.py)
# Output: Hiển thị ảnh gốc với các ô chữ nhật màu đỏ bao quanh lỗi kèm báo cáo log chi tiết

import os
import time
import cv2
import numpy as np

# 1. Import module tiền xử lý
try:
    from module_preprocess import align_images, detect_difference
except ImportError:
    align_images = None
    detect_difference = None

# 2. Import module phát hiện & cắt đốm lỗi của Minh
try:
    from module_detection import extract_bounding_boxes, crop_defect_patches
except ImportError:
    import module_detection
    extract_bounding_boxes = getattr(module_detection, 'extract_bounding_boxes', None)
    crop_defect_patches = getattr(module_detection, 'crop_defect_patches', None)


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


# hàm main
def main():
    start_total_time = time.time()

    # 1. Tải và kiểm tra ảnh đầu vào
    (img_test_gray, img_temp_gray), (img_test_color, img_temp_color) = load_and_verify_images(
        test_filename='92000000_test.jpg',
        temp_filename='92000000_temp.jpg',
        directory='data'
    )

    # 2. Căn chỉnh ảnh kiểm tra theo ảnh mẫu (ORB & Homography)
    try:
        img_test_aligned = align_images(img_temp_color, img_test_color)
    except Exception:
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
    print(" KẾT QUẢ:    ")
    print(f"- Tổng số khuyết tật phát hiện: {len(boxes)}")
    print(f"- Tổng thời gian xử lý:         {total_time:.3f} giây\n")
    print(f"{'STT':<6} | {'Tọa độ (x, y)':<16} | {'Kích thước (w, h)':<18} | {'Diện tích':<12}")
    print("-" * 65)
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        coord_str = f"({x}, {y})"
        size_str = f"{w} x {h} px"
        area_str = f"{w * h} px"
        print(f"#{idx:<5} | {coord_str:<16} | {size_str:<18} | {area_str:<12}")

    print("-" * 65)
    print(f"Đã lưu ảnh trong '{crop_dir}/'):")
    print(f"\nẢnh kết quả trực quan hóa: {output_result_path}")
    print("=" * 65)


if __name__ == '__main__':
    main()

