"""
Tiện ích đánh giá dùng chung cho các script thực nghiệm.

- Đọc nhãn GT (labels/*.txt: "x y w h class_id").
- Ghép bbox dự đoán với GT theo 2 tiêu chí:
    * "chạm nhau" với dung sai `tol` px (tiêu chí cũ, dễ dãi) - nhiều-nhiều;
    * IoU >= ngưỡng, ghép 1-1 tham lam theo IoU giảm dần (chuẩn detection).
- Nắn lệch giả lập bằng homography đã biết + sai số góc so với phép biến đổi thật.

Mọi script nên dùng chung các hàm này để số liệu giữa các thí nghiệm so được
với nhau (EXPERIMENT_REVIEW.md, P1.2 và P2.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # (x, y, w, h)

# Mô tả điều kiện đo, ghi kèm mọi bảng tổng hợp (P2.2)
MATCH_RULE_TOUCH = "touch_tol5"
IOU_THRESHOLDS = (0.3, 0.5)


# ---------------------------------------------------------------------
# Nhãn
# ---------------------------------------------------------------------
def read_labels(label_path: Optional[Path]) -> List[Box]:
    """Đọc labels/<pair>.txt -> danh sách (x, y, w, h)."""
    if not label_path or not Path(label_path).is_file():
        return []
    boxes: List[Box] = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 4:
            x, y, w, h = (int(float(v)) for v in parts[:4])
            boxes.append((x, y, w, h))
    return boxes


# ---------------------------------------------------------------------
# Ghép bbox
# ---------------------------------------------------------------------
def boxes_touch(a: Box, b: Box, tol: int = 5) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + tol < bx or bx + bw + tol < ax
                or ay + ah + tol < by or by + bh + tol < ay)


def match_touch(pred: Sequence[Box], gt: Sequence[Box], tol: int = 5) -> Tuple[int, int]:
    """(số GT được chạm, số pred chạm GT) - tiêu chí cũ của các script."""
    gt_hit = sum(1 for g in gt if any(boxes_touch(p, g, tol) for p in pred))
    pred_hit = sum(1 for p in pred if any(boxes_touch(p, g, tol) for g in gt))
    return gt_hit, pred_hit


def iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_iou(pred: Sequence[Box], gt: Sequence[Box], threshold: float) -> int:
    """Số cặp ghép 1-1 có IoU >= threshold (tham lam theo IoU giảm dần).

    Ghép 1-1 nên TP của GT = TP của pred; một lỗi bị tách thành nhiều đốm chỉ
    được tính 1 lần, phần còn lại thành báo động giả - đúng chuẩn detection.
    """
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            v = iou(p, g)
            if v >= threshold:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    return tp


def detection_counts(pred: Sequence[Box], gt: Sequence[Box],
                     iou_thresholds: Iterable[float] = IOU_THRESHOLDS,
                     tol: int = 5) -> Dict[str, int]:
    """Số đếm thô cho 1 ảnh: cộng dồn qua nhiều ảnh rồi gọi `prf_from_counts`."""
    gt_hit, pred_hit = match_touch(pred, gt, tol)
    out = {"n_gt": len(gt), "n_pred": len(pred), "gt_hit": gt_hit, "pred_hit": pred_hit,
           "false_blobs": len(pred) - pred_hit}
    for t in iou_thresholds:
        out[f"tp_iou{int(round(t * 100))}"] = match_iou(pred, gt, t)
    return out


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _f1(p: float, r: float) -> float:
    if np.isnan(p) or np.isnan(r) or p + r == 0:
        return 0.0 if not (np.isnan(p) or np.isnan(r)) else float("nan")
    return 2 * p * r / (p + r)


def prf_from_counts(rows: Iterable[Dict[str, object]],
                    iou_thresholds: Iterable[float] = IOU_THRESHOLDS) -> Dict[str, float]:
    """Recall/precision/F1 (micro) từ các dòng có số đếm của `detection_counts`."""
    rows = list(rows)
    n_gt = sum(int(r.get("n_gt", 0)) for r in rows)
    n_pred = sum(int(r.get("n_pred", 0)) for r in rows)
    recall = _safe_div(sum(int(r.get("gt_hit", 0)) for r in rows), n_gt)
    precision = _safe_div(sum(int(r.get("pred_hit", 0)) for r in rows), n_pred)
    out = {"n_gt": n_gt, "n_pred": n_pred, "recall": recall, "precision": precision,
           "f1": _f1(precision, recall),
           "false_blobs_per_img": _safe_div(sum(int(r.get("false_blobs", 0)) for r in rows),
                                            len(rows))}
    for t in iou_thresholds:
        key = f"tp_iou{int(round(t * 100))}"
        if not rows or key not in rows[0]:
            continue
        tp = sum(int(r.get(key, 0)) for r in rows)
        r_ = _safe_div(tp, n_gt)
        p_ = _safe_div(tp, n_pred)
        suffix = f"iou{int(round(t * 100))}"
        out.update({f"recall_{suffix}": r_, f"precision_{suffix}": p_, f"f1_{suffix}": _f1(p_, r_)})
    return out


# ---------------------------------------------------------------------
# Nắn lệch giả lập
# ---------------------------------------------------------------------
def random_homography(width: int, height: int, rng: np.random.Generator,
                      max_angle: float, max_shift: float, max_scale: float) -> np.ndarray:
    """Homography đã biết: xoay + co giãn quanh tâm ảnh rồi tịnh tiến."""
    angle = rng.uniform(-max_angle, max_angle)
    scale = 1.0 + rng.uniform(-max_scale, max_scale)
    tx, ty = rng.uniform(-max_shift, max_shift, size=2)
    rot = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    H = np.vstack([rot, [0.0, 0.0, 1.0]])
    H[0, 2] += tx
    H[1, 2] += ty
    return H


def corner_error(H_est: np.ndarray, H_true_inv: np.ndarray, width: int, height: int) -> float:
    """Sai số trung bình (px) tại 4 góc ảnh giữa H ước lượng và H thật."""
    corners = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    ).reshape(-1, 1, 2)
    est = cv2.perspectiveTransform(corners, H_est)
    true = cv2.perspectiveTransform(corners, H_true_inv)
    return float(np.mean(np.linalg.norm(est - true, axis=2)))


def synthetic_warp(image: np.ndarray, pair_id: str, seed: int, max_angle: float,
                   max_shift: float, max_scale: float):
    """Nắn lệch ảnh test như các script thực nghiệm (cùng seed -> cùng phép lệch).

    Trả về (ảnh_lệch, mask_pixel_thật, H_true).
    """
    import zlib

    h, w = image.shape[:2]
    rng = np.random.default_rng(zlib.crc32(f"{pair_id}:{seed}".encode()))
    H_true = random_homography(w, h, rng, max_angle, max_shift, max_scale)
    warped = cv2.warpPerspective(image, H_true, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H_true, (w, h),
                               flags=cv2.INTER_NEAREST)
    return warped, mask, H_true
