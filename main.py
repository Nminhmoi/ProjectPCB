import argparse
import os
import time
import cv2
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Tiền xử lý: căn chỉnh ảnh, trừ vi sai, nhị phân hoá
from module_preprocess import (
    DETECTORS,
    DEFAULT_DETECTOR,
    DEFAULT_FALLBACKS,
    DEFAULT_HYSTERESIS_LOW,
    DEFAULT_MAX_FEATURES,
    DEFAULT_MIN_AREA,
    DEFAULT_MIN_NCC,
    DEFAULT_OPEN_KSIZE,
    DEFAULT_THRESHOLD_FLOOR,
    AlignmentError,
    align_images_robust,
    detect_difference,
    warp_valid_mask,
)

# Phát hiện đốm lỗi và cắt ra ảnh riêng
from module_detection import extract_bounding_boxes, crop_defect_patches

# Đọc cặp ảnh từ pairs_manifest.csv và nhãn trong labels/
from dataset_utils import find_pairs, load_image

# Trực quan hoá: đường nối đặc trưng, bàn cờ, đóng khung
from module_visualization import (
    draw_feature_matches,
    draw_homography_verification,
    draw_results,
)


# Xoá output cũ trước mỗi lần chạy
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


def process_pair(pair: dict, detector: str = DEFAULT_DETECTOR,
                 max_features: int = DEFAULT_MAX_FEATURES,
                 fallbacks=DEFAULT_FALLBACKS,
                 min_ncc=DEFAULT_MIN_NCC,
                 threshold_floor: int = DEFAULT_THRESHOLD_FLOOR,
                 open_ksize: int = DEFAULT_OPEN_KSIZE,
                 hysteresis_low=DEFAULT_HYSTERESIS_LOW,
                 min_area: int = DEFAULT_MIN_AREA,
                 extra_out: "dict | None" = None,
                 save_outputs: bool = True,
                 visualize: bool = False) -> tuple:
    """
    Chạy toàn bộ pipeline trên 1 cặp ảnh.

    `pair` là dict do `dataset_utils.find_pairs` trả về, gồm:
    pair_id, test_path, template_path, class_name, annotation_path, ...

    `detector` chọn bộ đặc trưng căn chỉnh: ORB / SIFT / AKAZE.
    `fallbacks` detector dự phòng khi detector chính bị từ chối (mặc định: không có).
    `min_ncc` ngưỡng NCC kiểm chứng căn chỉnh bằng điểm ảnh — chốt chặn thảm hoạ,
    không dùng để tinh chỉnh (mặc định 0.90; None = tắt).
    `extra_out` dict tuỳ chọn: nếu truyền vào, hàm điền thêm các khoá
    template/aligned/gray/mask/valid/boxes để GUI hoặc bước phân loại dùng lại,
    khỏi phải chạy lại pipeline.
    `save_outputs=False`: không ghi `crops/` và `results/` (dùng khi chạy
    dataset/benchmark số lượng lớn).
    `visualize=True`: ghi thêm ảnh nối đặc trưng và kiểm định căn chỉnh.

    Trả về: (boxes, crop_paths, output_path, thời_gian_giây, align_stats)
    """
    start = time.time()

    # 1. Đọc ảnh màu test và template
    test_path = pair["test_path"]
    temp_path = pair["template_path"]
    img_id = pair["pair_id"]

    img_test_color = load_image(test_path)
    img_temp_color = load_image(temp_path)

    # 2. Căn chỉnh ảnh test theo template (detector + Homography), kiểm chứng bằng
    #    NCC; nếu bị từ chối thì thử detector dự phòng.
    try:
        img_test_aligned, align_stats = align_images_robust(
            img_temp_color, img_test_color,
            max_features=max_features, detector=detector,
            fallbacks=fallbacks, min_ncc=min_ncc,
        )
    except AlignmentError as exc:
        print(f"    [CẢNH BÁO] Căn chỉnh thất bại: {exc} — dùng ảnh chưa căn chỉnh")
        img_test_aligned = img_test_color
        align_stats = exc.stats

    # 3. Trừ vi sai rồi nhị phân hoá Otsu + khử nhiễu.
    #    Mask vùng hợp lệ: sau warp, mép ảnh còn dải đen; nếu để nguyên,
    #    dải này kéo Otsu lên và xoá mất lỗi mảnh (CODE_REVIEW.md mục 2.3).
    valid_mask = None
    if align_stats.get("success") and align_stats.get("H") is not None:
        valid_mask = warp_valid_mask(
            img_test_color, align_stats["H"], img_temp_color.shape[:2], erode_border=3
        )
    diff_mask = detect_difference(
        img_temp_color, img_test_aligned,
        threshold_floor=threshold_floor, open_ksize=open_ksize,
        hysteresis_low=hysteresis_low, valid=valid_mask,
    )

    # 4. Tìm contour và lấy bounding box
    boxes = extract_bounding_boxes(diff_mask, min_area=min_area)

    # 5. Cắt từng đốm lỗi ra ảnh riêng
    crop_paths = []
    output_path = None
    if save_outputs:
        crop_dir = os.path.join('crops', str(img_id))
        crop_paths = crop_defect_patches(
            img_test_aligned, boxes, output_dir=crop_dir, padding=5
        )

        # 6. Đóng khung đỏ, lưu ảnh kết quả theo mã ảnh
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        img_result = draw_results(img_test_aligned, boxes)
        output_path = os.path.join(results_dir, f'result_{img_id}.jpg')
        cv2.imwrite(output_path, img_result)

        # 7. Vẽ đường nối đặc trưng + kiểm định căn chỉnh (khi bật --visualize)
        if visualize and align_stats.get("success"):
            cv2.imwrite(
                os.path.join(results_dir, f'matches_{img_id}.jpg'),
                draw_feature_matches(img_test_color, img_temp_color, align_stats),
            )
            cv2.imwrite(
                os.path.join(results_dir, f'verify_{img_id}.jpg'),
                draw_homography_verification(img_temp_color, img_test_aligned),
            )

    elapsed = time.time() - start
    if extra_out is not None:
        # Trả kết quả trung gian cho GUI / bước phân loại, không chạy lại pipeline.
        extra_out.update(
            template=img_temp_color,
            aligned=img_test_aligned,
            gray=cv2.cvtColor(img_test_aligned, cv2.COLOR_BGR2GRAY),
            mask=diff_mask,
            valid=valid_mask,
            boxes=boxes,
        )
    return boxes, crop_paths, output_path, elapsed, align_stats


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
    parser.add_argument(
        "--detector",
        choices=list(DETECTORS),
        default=DEFAULT_DETECTOR,
        help=f"Bộ đặc trưng căn chỉnh (mặc định: {DEFAULT_DETECTOR}).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        help=f"Số đặc trưng tối đa cho mỗi ảnh (mặc định: {DEFAULT_MAX_FEATURES}).",
    )
    default_fb = " ".join(d for d, _ in DEFAULT_FALLBACKS) or "none"
    parser.add_argument(
        "--fallback",
        nargs="+",
        default=[d for d, _ in DEFAULT_FALLBACKS],
        help=f"Detector dự phòng khi căn chỉnh bị từ chối (mặc định: {default_fb}; "
             "ví dụ 'SIFT' — chậm hơn ~6 lần, chỉ dùng khi cần; 'none' = không dự phòng).",
    )
    parser.add_argument(
        "--min-ncc",
        type=float,
        default=DEFAULT_MIN_NCC,
        help=f"Ngưỡng NCC kiểm chứng căn chỉnh — chốt chặn thảm hoạ, không phải "
             f"ngưỡng tinh chỉnh (mặc định: {DEFAULT_MIN_NCC}; PKU thấp nhất 0.9950).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Tắt kiểm chứng NCC (hành vi cũ, không khuyến nghị).",
    )
    parser.add_argument("--threshold-floor", type=int, default=DEFAULT_THRESHOLD_FLOOR,
                        help=f"Ngưỡng sàn khi nhị phân hóa (mặc định: {DEFAULT_THRESHOLD_FLOOR}).")
    parser.add_argument("--open-ksize", type=int, default=DEFAULT_OPEN_KSIZE,
                        help=f"Kernel morphology opening, 1 = bỏ (mặc định: {DEFAULT_OPEN_KSIZE}).")
    parser.add_argument("--hysteresis-low", type=float, default=DEFAULT_HYSTERESIS_LOW,
                        help="Tỷ lệ ngưỡng thấp cho hysteresis, vd 0.5 (mặc định: tắt).")
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA,
                        help=f"Diện tích tối thiểu của đốm lỗi (mặc định: {DEFAULT_MIN_AREA}).")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Xuất thêm ảnh nối điểm đặc trưng (matches) và kiểm định căn chỉnh (checkerboard/overlay).",
    )
    return parser.parse_args()


def main():
    """Tự động chạy pipeline trên các cặp ảnh tìm thấy trong data/."""
    args = parse_args()
    start_total_time = time.time()

    # Dọn output cũ một lần trước khi chạy
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

    fallbacks = [] if [f.lower() for f in args.fallback] == ["none"] else \
        [(f.upper(), args.max_features) for f in args.fallback]
    min_ncc = None if args.no_verify else args.min_ncc

    print(f" Bộ đặc trưng căn chỉnh : {args.detector}"
          + (f" (dự phòng: {', '.join(f for f, _ in fallbacks)})" if fallbacks else ""))
    print(f" Kiểm chứng NCC          : {'tắt' if min_ncc is None else f'>= {min_ncc}'}")
    print(f" Tìm thấy {len(pairs)} cặp ảnh trong '{args.data_dir}'")
    print("=" * 65)

    total_defects = 0
    total_crops = 0
    done = 0
    n_fallback = 0
    n_unverified = 0

    for idx, pair in enumerate(pairs, start=1):
        img_id = pair["pair_id"]
        print(f"\n[{idx}/{len(pairs)}] {img_id}")

        try:
            boxes, crop_paths, output_path, elapsed, align_stats = process_pair(
                pair, detector=args.detector, max_features=args.max_features,
                fallbacks=fallbacks, min_ncc=min_ncc,
                threshold_floor=args.threshold_floor, open_ksize=args.open_ksize,
                hysteresis_low=args.hysteresis_low, min_area=args.min_area,
                visualize=args.visualize,
            )
        except Exception as exc:
            print(f"    [LỖI] {exc}")
            continue

        done += 1
        total_defects += len(boxes)
        total_crops += len(crop_paths)

        if align_stats.get("success"):
            print(f"    Căn chỉnh {align_stats['detector']:<6} : {align_stats['inliers']} inliers "
                  f"({align_stats['inlier_ratio'] * 100:.0f}%), "
                  f"reproj {align_stats['reprojection_error']:.2f} px, "
                  f"NCC {align_stats.get('ncc', float('nan')):.3f}, "
                  f"{align_stats.get('time_all_attempts_ms', align_stats['time_total_ms']):.0f} ms")
            if align_stats.get("fallback_used"):
                n_fallback += 1
                print(f"    [DỰ PHÒNG] {' -> '.join(align_stats.get('attempts', []))}")
            if not align_stats.get("verified", True):
                n_unverified += 1
                print("    [CẢNH BÁO] Không phương án nào vượt kiểm chứng NCC — "
                      "dùng H có NCC cao nhất, nên kiểm tra lại ảnh.")
        else:
            print(f"    Căn chỉnh        : THẤT BẠI — {align_stats.get('error', '?')}")
        print(f"    Số lỗi phát hiện : {len(boxes)}")
        print(f"    Số crop đã lưu  : {len(crop_paths)}")
        print(f"    Thời gian        : {elapsed:.3f} giây")
        print(f"    Ảnh kết quả      : {output_path}")

        # Chỉ có 1 cặp thì in chi tiết toạ độ
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
    print(f"- Dùng detector dự phòng: {n_fallback} cặp; chưa kiểm chứng được: {n_unverified} cặp")
    print(f"- Tổng số crop đã lưu   : {total_crops}")
    print(f"- Tổng thời gian        : {total_time:.3f} giây")
    print(f"- Trung bình            : {total_time / max(done, 1):.3f} giây/cặp")
    print("=" * 65)


if __name__ == '__main__':
    main()
