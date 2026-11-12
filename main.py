import argparse
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

# 3. Import loader dùng chung (đọc pairs_manifest.csv + labels/)
from dataset_utils import find_pairs, load_image


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


def process_pair(pair: dict) -> tuple:
    """
    Chạy toàn bộ pipeline trên 1 cặp ảnh.

    `pair` là dict do `dataset_utils.find_pairs` trả về, gồm:
    pair_id, test_path, template_path, class_name, annotation_path, ...

    Trả về: (boxes, crop_paths, output_path, thời_gian_giây)
    """
    start = time.time()

    # 1. Tải ảnh màu (test + template)
    test_path = pair["test_path"]
    temp_path = pair["template_path"]
    img_id = pair["pair_id"]

    img_test_color = load_image(test_path)
    img_temp_color = load_image(temp_path)

    # 2. Căn chỉnh ảnh kiểm tra theo ảnh mẫu (ORB & Homography)
    try:
        img_test_aligned = align_images(img_temp_color, img_test_color)
    except Exception as exc:
        print(f"    [CẢNH BÁO] Căn chỉnh thất bại: {exc} — dùng ảnh chưa căn chỉnh")
        img_test_aligned = img_test_color

    # 3. Trừ vi sai và nhị phân hóa Otsu, khử nhiễu Morphology
    diff_mask = detect_difference(img_temp_color, img_test_aligned)

    # 4. Tìm Contours và trích xuất Bounding Boxes
    boxes = extract_bounding_boxes(diff_mask, min_area=20)

    # 5. Cắt và lưu các ảnh con đốm lỗi (mỗi ảnh 1 thư mục riêng)
    crop_dir = os.path.join('crops', str(img_id))
    crop_paths = crop_defect_patches(
        img_test_aligned, boxes, output_dir=crop_dir, padding=5
    )

    # 6. Đóng khung đỏ và lưu ảnh kết quả (tên file theo mã ảnh)
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    img_result = draw_results(img_test_aligned, boxes)
    output_path = os.path.join(results_dir, f'result_{img_id}.jpg')
    cv2.imwrite(output_path, img_result)

    elapsed = time.time() - start
    return boxes, crop_paths, output_path, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy pipeline căn chỉnh + phát hiện lỗi trên cặp ảnh PCB."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Thư mục chứa pairs_manifest.csv (mặc định: data).",
    )
    parser.add_argument(
        "--pair",
        default=None,
        help="Chỉ chạy một cặp theo pair_id trong manifest, ví dụ: 01_short_01.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Chỉ chạy N cặp đầu tiên (0 = chạy tất cả).",
    )
    return parser.parse_args()


# hàm main
def main():
    """Tự động chạy pipeline trên các cặp ảnh tìm thấy trong data/."""
    args = parse_args()
    start_total_time = time.time()

    # Dọn dẹp output cũ MỘT LẦN trước khi chạy
    clean_output_directories(['crops', 'results'])
    os.makedirs('results', exist_ok=True)

    # Tìm mọi cặp ảnh từ manifest
    pairs = find_pairs(args.data_dir)
    if args.pair:
        pairs = [pair for pair in pairs if pair["pair_id"] == args.pair]
    if args.limit > 0:
        pairs = pairs[:args.limit]

    if not pairs:
        raise FileNotFoundError(
            f"\n[LỖI] Không tìm thấy cặp ảnh nào trong '{args.data_dir}'.\n"
            "       Cần pairs_manifest.csv (chạy experiments/build_pairs.py trước).\n"
        )

    print(f" Tìm thấy {len(pairs)} cặp ảnh trong '{args.data_dir}'")
    print("=" * 65)

    total_defects = 0
    total_crops = 0
    done = 0

    for idx, pair in enumerate(pairs, start=1):
        img_id = pair["pair_id"]
        print(f"\n[{idx}/{len(pairs)}] {img_id}")

        try:
            boxes, crop_paths, output_path, elapsed = process_pair(pair)
        except Exception as exc:
            print(f"    [LỖI] {exc}")
            continue

        done += 1
        total_defects += len(boxes)
        total_crops += len(crop_paths)

        print(f"    Số lỗi phát hiện : {len(boxes)}")
        print(f"    Số crop đã lưu  : {len(crop_paths)}")
        print(f"    Thời gian        : {elapsed:.3f} giây")
        print(f"    Ảnh kết quả      : {output_path}")

        # Nếu chỉ có 1 cặp thì in chi tiết toạ độ như bản cũ
        if len(pairs) == 1:
            print(f"\n{'STT':<5} | {'Tọa độ (x, y)':<15} | "
                  f"{'Kích thước (w, h)':<19} | {'Diện tích':>10}")
            print("-" * 58)
            for i, (x, y, w, h) in enumerate(boxes, start=1):
                coord_str = f"({x:>3}, {y:>3})"
                size_str = f"{w:>3} x {h:>3} px"
                area_str = f"{w * h:>5} px"
                print(f"#{i:<4} | {coord_str:<15} | {size_str:<19} | {area_str:>10}")
            print("-" * 58)

    # Tổng kết
    total_time = time.time() - start_total_time
    print("\n" + "=" * 65)
    print(" KET QUA TONG HOP:")
    print(f"- Số cặp chạy được      : {done}/{len(pairs)}")
    print(f"- Tổng số lỗi phát hiện : {total_defects}")
    print(f"- Tổng số crop đã lưu   : {total_crops}")
    print(f"- Tổng thời gian        : {total_time:.3f} giây")
    print(f"- Trung bình            : {total_time / max(done, 1):.3f} giây/cặp")
    print("=" * 65)


if __name__ == '__main__':
    main()
