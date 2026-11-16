"""Trực quan hoá: đường nối đặc trưng, kiểm chứng homography, khung đốm lỗi.

Tách từ `exp_feature_matching.py` / `main.py`, không đổi logic vẽ.
Ảnh vào là BGR hoặc xám (uint8), ảnh ra là BGR.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]

COLOR_DEFECT = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_GRID = (0, 0, 255)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Xám -> BGR; ảnh màu trả về bản copy."""
    if image is None:
        raise ValueError("Ảnh đầu vào là None.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def fit_width(image: np.ndarray, max_width: int) -> Tuple[np.ndarray, float]:
    """Thu nhỏ nếu rộng hơn `max_width`; trả (ảnh, tỉ lệ)."""
    h, w = image.shape[:2]
    scale = min(1.0, max_width / float(w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return image, scale


_fit_width = fit_width  # alias cho code cũ


def draw_feature_matches(
    img_test: np.ndarray, img_template: np.ndarray, stats: dict,
    max_lines: int = 150, max_width: int = 1100,
) -> np.ndarray:
    """Ghép test | template và nối các cặp inlier trong `stats`."""
    left, s1 = fit_width(img_test, max_width)
    right, s2 = fit_width(img_template, max_width)
    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, left.shape[1] + right.shape[1], 3), np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1]:] = right

    pts_test = stats.get("inlier_pts_test")
    pts_tmpl = stats.get("inlier_pts_template")
    if pts_test is not None and len(pts_test):
        idx = np.linspace(0, len(pts_test) - 1,
                          min(max_lines, len(pts_test))).astype(int)
        rng = np.random.default_rng(0)
        for i in idx:
            color = tuple(int(c) for c in rng.integers(60, 256, size=3))
            p1 = (int(pts_test[i][0] * s1), int(pts_test[i][1] * s1))
            p2 = (int(pts_tmpl[i][0] * s2) + left.shape[1],
                  int(pts_tmpl[i][1] * s2))
            cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, color, -1)
            cv2.circle(canvas, p2, 3, color, -1)

    text = (f"{stats['detector']}: {stats['inliers']} inliers "
            f"({stats['inlier_ratio'] * 100:.1f}%), reproj "
            f"{stats['reprojection_error']:.2f}px, {stats['time_total_ms']:.0f}ms")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


draw_inlier_matches = draw_feature_matches  # tên cũ


def checkerboard(img_template: np.ndarray, img_aligned: np.ndarray,
                 tiles: int = 8) -> np.ndarray:
    """Ghép xen kẽ ô cờ: mạch liền qua ranh giới ô = căn chuẩn."""
    h, w = img_template.shape[:2]
    ys = (np.arange(h) * tiles // h)[:, None]
    xs = (np.arange(w) * tiles // w)[None, :]
    mask = ((ys + xs) % 2 == 0)
    out = img_aligned.copy()
    out[mask] = img_template[mask]
    step_x, step_y = w / tiles, h / tiles
    for i in range(1, tiles):
        cv2.line(out, (int(i * step_x), 0), (int(i * step_x), h), COLOR_GRID, 1)
        cv2.line(out, (0, int(i * step_y)), (w, int(i * step_y)), COLOR_GRID, 1)
    return out


def draw_homography_verification(
    img_template: np.ndarray, img_aligned: np.ndarray,
    tiles: int = 8, alpha: float = 0.0, diff_threshold: int = 30,
) -> np.ndarray:
    """Ảnh bàn cờ; `alpha > 0` tô đỏ vùng hai ảnh lệch quá `diff_threshold`."""
    template = ensure_bgr(img_template)
    aligned = ensure_bgr(img_aligned)
    if aligned.shape[:2] != template.shape[:2]:
        aligned = cv2.resize(aligned, (template.shape[1], template.shape[0]))

    board = checkerboard(template, aligned, tiles=tiles)

    if alpha and alpha > 0.0:
        diff = cv2.absdiff(template, aligned).max(axis=2)
        heat = np.zeros_like(board)
        heat[diff >= int(diff_threshold)] = COLOR_DEFECT
        a = float(min(max(alpha, 0.0), 1.0))
        board = cv2.addWeighted(board, 1.0 - a, heat, a, 0.0)
    return board


def draw_results(img_test: np.ndarray, boxes: Sequence[Box],
                 labels: Optional[Sequence[str]] = None,
                 color: Tuple[int, int, int] = COLOR_DEFECT) -> np.ndarray:
    """Đóng khung từng bbox; có `labels` thì ghi tên lớp, không thì "Defect #n"."""
    img_annotated = ensure_bgr(img_test)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thick = 1

    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(img_annotated, (x, y), (x + w, y + h), color, 2)

        if labels is not None and idx - 1 < len(labels):
            label = str(labels[idx - 1])
        else:
            label = f"Defect #{idx}"

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
        label_y = max(y - 5, text_h + 5)
        label_x = min(max(x, 0), max(img_annotated.shape[1] - text_w - 4, 0))

        cv2.rectangle(
            img_annotated,
            (label_x, label_y - text_h - 2),
            (label_x + text_w + 4, label_y + baseline - 2),
            color,
            -1,
        )
        cv2.putText(
            img_annotated,
            label,
            (label_x + 2, label_y - 2),
            font,
            font_scale,
            COLOR_TEXT,
            font_thick,
            cv2.LINE_AA,
        )
    return img_annotated
