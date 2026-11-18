"""
Module Trực quan hóa Kết quả (module_visualization.py)
Phụ trách: Hinh — Tích hợp & Trực quan hóa

Chức năng:
1. `draw_feature_matches`: Vẽ các đường nối điểm khớp đặc trưng (inliers) giữa ảnh test và ảnh template.
2. `draw_checkerboard`: Tạo ảnh bàn cờ xen kẽ ô giữa template và aligned để soi đường mạch.
3. `draw_alpha_overlay`: Tạo ảnh phủ mờ (Alpha Overlay / False Color) kiểm tra độ trùng khít.
4. `draw_homography_verification`: Xuất ảnh kiểm định căn chỉnh (kết hợp Checkerboard & Overlay).
5. `draw_results`: Đóng khung chữ nhật màu đỏ bao quanh các vết khuyết tật kèm nhãn.
"""

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np


def _fit_width(image: np.ndarray, max_width: int) -> Tuple[np.ndarray, float]:
    """Co giãn ảnh nếu chiều rộng vượt quá max_width để hiển thị vừa màn hình."""
    h, w = image.shape[:2]
    scale = min(1.0, max_width / float(w))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image, scale


def _ensure_color(image: np.ndarray) -> np.ndarray:
    """Đảm bảo ảnh đầu ra có 3 kênh màu BGR."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


# =====================================================================
# 1. Vẽ đường nối điểm khớp đặc trưng
# =====================================================================
def draw_feature_matches(
    img_test: np.ndarray,
    img_template: np.ndarray,
    stats_or_matches: Union[dict, list],
    kp_test: Optional[List[cv2.KeyPoint]] = None,
    kp_template: Optional[List[cv2.KeyPoint]] = None,
    max_lines: int = 150,
    max_width: int = 1100,
) -> np.ndarray:
    """
    Vẽ các đường nối điểm khớp đặc trưng (inliers) giữa ảnh test và template.

    Tham số:
        img_test: Ảnh kiểm tra.
        img_template: Ảnh mẫu chuẩn.
        stats_or_matches:
            - Dict `align_stats` trả về từ `align_images_with_stats` (chứa inlier_pts_test, inlier_pts_template, ...)
            - HOẶC danh sách `cv2.DMatch` (khi đó cần truyền thêm `kp_test` và `kp_template`).
        max_lines: Số lượng đường nối tối đa hiển thị (tránh rối mắt).
        max_width: Giới hạn chiều rộng mỗi ảnh con trên canvas.

    Trả về:
        Ảnh canvas ghép đôi (test bên trái, template bên phải) kèm các đường nối inliers.
    """
    left = _ensure_color(img_test)
    right = _ensure_color(img_template)

    left_scaled, s1 = _fit_width(left, max_width)
    right_scaled, s2 = _fit_width(right, max_width)

    h_left, w_left = left_scaled.shape[:2]
    h_right, w_right = right_scaled.shape[:2]
    canvas_h = max(h_left, h_right)
    canvas_w = w_left + w_right

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h_left, :w_left] = left_scaled
    canvas[:h_right, w_left:] = right_scaled

    pts_test = None
    pts_template = None
    banner_text = "Feature Matching Visualization"

    if isinstance(stats_or_matches, dict):
        stats = stats_or_matches
        pts_test = stats.get("inlier_pts_test")
        pts_template = stats.get("inlier_pts_template")
        detector = stats.get("detector", "DETECTOR")
        inliers = stats.get("inliers", 0)
        ratio = stats.get("inlier_ratio", 0.0) * 100.0
        reproj = stats.get("reprojection_error", float("nan"))
        t_ms = stats.get("time_total_ms", 0.0)
        reproj_str = f"{reproj:.2f}px" if np.isfinite(reproj) else "N/A"
        banner_text = f"{detector}: {inliers} inliers ({ratio:.1f}%), Reproj Err: {reproj_str}, Time: {t_ms:.0f}ms"
    elif isinstance(stats_or_matches, list) and kp_test is not None and kp_template is not None:
        matches = stats_or_matches
        pts_test = np.float32([kp_test[m.queryIdx].pt for m in matches])
        pts_template = np.float32([kp_template[m.trainIdx].pt for m in matches])
        banner_text = f"Matches: {len(matches)} pairs"

    if pts_test is not None and pts_template is not None and len(pts_test) > 0:
        total_pts = len(pts_test)
        indices = np.linspace(0, total_pts - 1, min(max_lines, total_pts)).astype(int)
        rng = np.random.default_rng(42)

        for i in indices:
            pt1 = (int(pts_test[i][0] * s1), int(pts_test[i][1] * s1))
            pt2 = (int(pts_template[i][0] * s2) + w_left, int(pts_template[i][1] * s2))
            color = tuple(int(c) for c in rng.integers(60, 256, size=3))

            cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, color, -1)
            cv2.circle(canvas, pt2, 3, color, -1)

    # Thanh tiêu đề trên đỉnh canvas
    banner_h = 36
    header = np.zeros((banner_h, canvas_w, 3), dtype=np.uint8)
    cv2.putText(
        header, banner_text, (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA
    )
    # Nhãn Test (Trái) và Template (Phải)
    cv2.putText(header, "[Test Image]", (canvas_w - 300, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(header, "[Template]", (canvas_w - 140, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1, cv2.LINE_AA)

    return np.vstack([header, canvas])


# =====================================================================
# 2. Ảnh bàn cờ (Checkerboard) & Phủ mờ (Alpha Overlay)
# =====================================================================
def draw_checkerboard(
    img_template: np.ndarray,
    img_aligned: np.ndarray,
    tiles: int = 8,
    show_grid: bool = True,
    grid_color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """
    Tạo ảnh bàn cờ xen kẽ ô giữa ảnh mẫu (Template) và ảnh đã căn chỉnh (Aligned).
    Nếu đường mạch chạy liền lạc không gãy khúc qua các mép ô cờ -> Căn chỉnh chuẩn xác.
    """
    tmpl = _ensure_color(img_template)
    alg = _ensure_color(img_aligned)

    h, w = tmpl.shape[:2]
    if alg.shape[:2] != (h, w):
        alg = cv2.resize(alg, (w, h), interpolation=cv2.INTER_LINEAR)

    ys = (np.arange(h) * tiles // h)[:, None]
    xs = (np.arange(w) * tiles // w)[None, :]
    mask = ((ys + xs) % 2 == 0)

    board = alg.copy()
    board[mask] = tmpl[mask]

    if show_grid:
        step_x = w / float(tiles)
        step_y = h / float(tiles)
        for i in range(1, tiles):
            cv2.line(board, (int(i * step_x), 0), (int(i * step_x), h), grid_color, 1, cv2.LINE_AA)
            cv2.line(board, (0, int(i * step_y)), (w, int(i * step_y)), grid_color, 1, cv2.LINE_AA)

    return board


def draw_alpha_overlay(
    img_template: np.ndarray,
    img_aligned: np.ndarray,
    alpha: float = 0.5,
    false_color: bool = True,
) -> np.ndarray:
    """
    Tạo ảnh phủ mờ (Alpha Overlay) kiểm tra độ trùng khít giữa Template và Aligned.

    Tham số:
        alpha: Tỷ lệ hòa trộn giữa Template (alpha) và Aligned (1 - alpha).
        false_color:
            - True: Kênh Đỏ = Template, Kênh Lục = Aligned (Trùng khít ra màu vàng, lệch lộ viền đỏ/lục).
            - False: Trộn màu tuyến tính thông thường (`cv2.addWeighted`).
    """
    tmpl = _ensure_color(img_template)
    alg = _ensure_color(img_aligned)

    h, w = tmpl.shape[:2]
    if alg.shape[:2] != (h, w):
        alg = cv2.resize(alg, (w, h), interpolation=cv2.INTER_LINEAR)

    if false_color:
        gray_tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        gray_alg = cv2.cvtColor(alg, cv2.COLOR_BGR2GRAY)
        # B = 0, G = Aligned, R = Template
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:, :, 1] = gray_alg
        overlay[:, :, 2] = gray_tmpl
        return overlay

    beta = 1.0 - alpha
    return cv2.addWeighted(tmpl, alpha, alg, beta, 0.0)


def draw_homography_verification(
    img_template: np.ndarray,
    img_aligned: np.ndarray,
    mode: str = "both",
    tiles: int = 8,
    alpha: float = 0.5,
    max_width: int = 1600,
) -> np.ndarray:
    """
    Xuất ảnh kiểm định căn chỉnh Homography.

    Tham số:
        mode:
            - "checkerboard": Chỉ xuất ảnh bàn cờ.
            - "overlay": Chỉ xuất ảnh phủ mờ (Alpha/False-color Overlay).
            - "both": Ghép ảnh bàn cờ (trái) và phủ mờ (phải) cạnh nhau.
        tiles: Số lượng ô bàn cờ mỗi chiều.
        alpha: Trọng số phủ mờ nếu dùng overlay.
        max_width: Chiều rộng tối đa của ảnh đầu ra (tự co giãn nếu vượt quá).

    Trả về:
        Ảnh trực quan hóa kiểm định độ khít sau căn chỉnh.
    """
    board = draw_checkerboard(img_template, img_aligned, tiles=tiles)
    overlay = draw_alpha_overlay(img_template, img_aligned, alpha=alpha, false_color=True)

    mode_lower = mode.lower()
    if mode_lower == "checkerboard":
        result = board
    elif mode_lower == "overlay":
        result = overlay
    else:  # "both"
        h, w = board.shape[:2]
        canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
        canvas[:, :w] = board
        canvas[:, w:] = overlay

        # Thêm nhãn phụ đề cho 2 nửa
        banner_h = 32
        header = np.zeros((banner_h, w * 2, 3), dtype=np.uint8)
        cv2.putText(
            header, f"Checkerboard Alignment ({tiles}x{tiles})", (15, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            header, "False-Color Overlay (Red: Template, Green: Test Aligned)", (w + 15, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA
        )
        result = np.vstack([header, canvas])

    result, _ = _fit_width(result, max_width)
    return result


# =====================================================================
# 3. Đóng khung kết quả khuyết tật (Bounding Boxes)
# =====================================================================
def draw_results(
    img_test: np.ndarray,
    boxes: list,
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Đóng khung chữ nhật màu đỏ xung quanh các khuyết tật kèm nhãn số thứ tự hoặc nhãn phân loại.

    Tham số:
        img_test: Ảnh bo mạch (thường là ảnh đã căn chỉnh).
        boxes: Danh sách tọa độ Bounding Box [(x, y, w, h), ...].
        labels: Danh sách nhãn khuyết tật tương ứng (tùy chọn). Mặc định là 'Defect #1', 'Defect #2'...
        color: Màu viền BGR (mặc định đỏ: (0, 0, 255)).
        thickness: Độ dày đường viền.

    Trả về:
        Ảnh đã được vẽ các Bounding Box và nhãn.
    """
    img_annotated = _ensure_color(img_test)

    for idx, box in enumerate(boxes, start=1):
        x, y, w, h = box[:4]
        cv2.rectangle(img_annotated, (x, y), (x + w, y + h), color, thickness)

        if labels is not None and idx - 1 < len(labels):
            label_text = f"#{idx} {labels[idx - 1]}"
        else:
            label_text = f"Defect #{idx}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thick = 1

        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)
        label_y = max(y - 5, th + 5)
        label_x = max(0, x)

        # Nền nhãn đỏ
        cv2.rectangle(
            img_annotated,
            (label_x, label_y - th - 2),
            (label_x + tw + 4, label_y + baseline - 2),
            color, -1
        )
        # Chữ nhãn trắng
        cv2.putText(
            img_annotated, label_text, (label_x + 2, label_y - 2),
            font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA
        )

    return img_annotated


# =====================================================================
# DEMO CHẠY THỬ ĐỘC LẬP (CLI TEST)
# =====================================================================
if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from MODULE.module_preprocess import align_images_with_stats

    print("=" * 65)
    print("DEMO TRỰC QUAN HÓA KẾT QUẢ (module_visualization.py)")
    print("=" * 65)

    os.makedirs("results", exist_ok=True)

    # 1. Tạo cặp ảnh mô phỏng bo mạch PCB
    h, w = 600, 800
    template = np.full((h, w, 3), (30, 90, 30), dtype=np.uint8)
    rng = np.random.default_rng(42)

    # Vẽ các pad linh kiện và đường mạch để tạo keypoints đặc trưng phong phú
    for _ in range(120):
        x = int(rng.integers(50, w - 50))
        y = int(rng.integers(50, h - 50))
        size = int(rng.integers(10, 25))
        color = (int(rng.integers(0, 50)), int(rng.integers(180, 255)), int(rng.integers(200, 255)))
        cv2.rectangle(template, (x, y), (x + size, y + size), color, -1)
        cv2.circle(template, (x + size // 2, y + size // 2), max(2, size // 4), (0, 0, 0), -1)

    for i in range(10):
        cv2.line(template, (50, 50 + i * 50), (w - 50, 50 + i * 50), (0, 220, 240), 2)

    # Test: Bị nắn xoay lệch 1.5 độ và dịch (dx=10, dy=5) + có vết khuyết tật
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 1.5, 1.0)
    M[0, 2] += 10.0
    M[1, 2] += 5.0
    test = cv2.warpAffine(template, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # Thêm 1 vết hở mạch lỗi
    cv2.rectangle(test, (280, 190), (330, 210), (30, 90, 30), -1)

    print("1. Đang căn chỉnh ảnh bằng ORB & RANSAC Homography...")
    aligned, stats = align_images_with_stats(template, test, detector="ORB")
    print(f"   -> Khớp {stats['inliers']} inliers ({stats['inlier_ratio']*100:.1f}%), "
          f"Reproj Error: {stats['reprojection_error']:.2f}px")

    print("\n2. Đang tạo các ảnh trực quan hóa:")
    # A. Đường nối điểm khớp đặc trưng
    matches_vis = draw_feature_matches(test, template, stats)
    p_matches = "results/demo_matches.jpg"
    cv2.imwrite(p_matches, matches_vis)
    print(f"   [✓] Đường nối đặc trưng : {p_matches}")

    # B. Kiểm định căn chỉnh Homography (Checkerboard & Alpha Overlay)
    verify_vis = draw_homography_verification(template, aligned, mode="both", tiles=8)
    p_verify = "results/demo_verification.jpg"
    cv2.imwrite(p_verify, verify_vis)
    print(f"   [✓] Kiểm định căn chỉnh : {p_verify}")

    # C. Đóng khung kết quả phát hiện lỗi
    demo_boxes = [(240, 195, 40, 15)]
    results_vis = draw_results(aligned, demo_boxes, labels=["Open Circuit"])
    p_results = "results/demo_results.jpg"
    cv2.imwrite(p_results, results_vis)
    print(f"   [✓] Đóng khung lỗi      : {p_results}")

    print("\nHoàn tất! Các ảnh trực quan hóa đã được lưu trong thư mục 'results/'.")
    print("=" * 65)

