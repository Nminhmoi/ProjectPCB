"""
Module Đánh giá Định lượng & Nhãn Ground Truth (evaluation.py)
Cung cấp các hàm đọc nhãn Ground Truth, tính toán chỉ số phát hiện (Precision, Recall, F1, IoU)
dùng chung cho GUI (gui_app.py), đánh giá (evaluate_metrics.py) và các thực nghiệm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def read_labels(label_path: Optional[Union[str, Path]]) -> List[Tuple[int, int, int, int]]:
    """
    Đọc danh sách Bounding Box Ground Truth từ file nhãn txt.
    Định dạng mỗi dòng: x y w h [class_id]
    Trả về danh sách các tuple: [(x, y, w, h), ...]
    """
    if not label_path:
        return []
    p = Path(label_path)
    if not p.is_file():
        return []

    boxes: List[Tuple[int, int, int, int]] = []
    try:
        content = p.read_text(encoding="utf-8").strip()
        if not content:
            return []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                x = int(float(parts[0]))
                y = int(float(parts[1]))
                w = int(float(parts[2]))
                h = int(float(parts[3]))
                if w > 0 and h > 0:
                    boxes.append((x, y, w, h))
    except Exception:
        return []

    return boxes


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Tính chỉ số IoU (Intersection over Union) giữa hai hộp giới hạn (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = float(boxA_area + boxB_area - inter_area)

    return inter_area / union_area if union_area > 0 else 0.0


def _touch(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], tol: int = 5) -> bool:
    """Kiểm tra xem hai hộp giới hạn có giao nhau hoặc chạm nhau trong khoảng dung sai tol (px)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + tol < bx or
        bx + bw + tol < ax or
        ay + ah + tol < by or
        by + bh + tol < ay
    )


def match_boxes(pred_boxes: List[Tuple[int, int, int, int]],
                gt_boxes: List[Tuple[int, int, int, int]],
                tol: int = 5) -> Tuple[int, int]:
    """
    So sánh danh sách box dự đoán và Ground Truth.
    Trả về: (số box GT được phát hiện chạm vào, số box dự đoán chạm trúng GT).
    """
    if not pred_boxes or not gt_boxes:
        return 0, 0

    gt_hit = sum(1 for g in gt_boxes if any(_touch(p, g, tol=tol) for p in pred_boxes))
    pred_hit = sum(1 for p in pred_boxes if any(_touch(p, g, tol=tol) for g in gt_boxes))
    return gt_hit, pred_hit


def detection_counts(pred_boxes: List[Tuple[int, int, int, int]],
                     gt_boxes: List[Tuple[int, int, int, int]],
                     tol: int = 5) -> Dict[str, object]:
    """
    Tính toán số lượng đối sánh và các chỉ số đo đạc:
    n_gt, n_pred, gt_hit (Recall tương ứng), pred_hit (Precision tương ứng), tp_iou50.
    """
    n_gt = len(gt_boxes) if gt_boxes else 0
    n_pred = len(pred_boxes) if pred_boxes else 0

    if n_gt == 0:
        return {
            "n_gt": 0, "n_pred": n_pred,
            "gt_hit": 0, "pred_hit": 0,
            "tp_iou50": 0,
            "recall": 0.0, "precision": 0.0, "f1": 0.0
        }

    gt_hit, pred_hit = match_boxes(pred_boxes, gt_boxes, tol=tol)

    # Đếm số True Positive với IoU >= 0.5 (tiêu chuẩn PASCAL VOC)
    tp_iou50 = 0
    matched_gt = set()
    for p in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for idx, g in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = compute_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou >= 0.5 and best_gt_idx >= 0:
            tp_iou50 += 1
            matched_gt.add(best_gt_idx)

    recall = (gt_hit / n_gt) if n_gt > 0 else 0.0
    precision = (pred_hit / n_pred) if n_pred > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "gt_hit": gt_hit,
        "pred_hit": pred_hit,
        "tp_iou50": tp_iou50,
        "recall": recall,
        "precision": precision,
        "f1": f1
    }

