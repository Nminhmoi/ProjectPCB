"""
Module Phân loại Khuyết tật Bo mạch PCB (module_classification.py)
Phụ trách: Minh — Định vị khuyết tật & Phân loại Machine Learning

Phân loại 6 nhóm khuyết tật theo tiêu chuẩn IPC-A-600:
1. Open_circuit   (Hở mạch)
2. Short          (Chập mạch)
3. Mouse_bite     (Lỗ chuột gặm)
4. Spur           (Gai đồng)
5. Spurious_copper(Đồng thừa)
6. Missing_hole   (Lỗi lỗ khoan)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

# 6 nhóm khuyết tật mục tiêu
DEFECT_CLASSES = [
    "Open_circuit",
    "Short",
    "Mouse_bite",
    "Spur",
    "Spurious_copper",
    "Missing_hole",
]

MODEL_PATH = Path(__file__).resolve().parent / "model_classifier.pkl"


def extract_defect_features(patch: np.ndarray) -> np.ndarray:
    """
    Trích xuất vector đặc trưng hình thái của đốm lỗi:
    - Kích thước & hình học: Area, Aspect Ratio, Extent, Solidity
    - Hu Moments: 7 bất biến Hu mô tả hình dạng
    """
    if patch is None or patch.size == 0:
        return np.zeros(11, dtype=np.float32)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    h, w = gray.shape[:2]

    # Nhị phân hóa Otsu để tách đốm lỗi khỏi nền
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(11, dtype=np.float32)

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    x, y, bw, bh = cv2.boundingRect(cnt)

    # 1. Tỷ lệ khung (Aspect Ratio)
    aspect_ratio = float(bw) / bh if bh > 0 else 1.0

    # 2. Extent: Tỷ lệ diện tích đốm lỗi / diện tích bounding box
    rect_area = float(bw * bh)
    extent = (area / rect_area) if rect_area > 0 else 0.0

    # 3. Solidity: Tỷ lệ diện tích đốm lỗi / diện tích bao lồi (Convex Hull)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = (area / float(hull_area)) if hull_area > 0 else 0.0

    # 4. Hu Moments (7 đặc trưng bất biến với phép tịnh tiến, tỉ lệ và xoay)
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    # Log transform để chuẩn hóa biên độ
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    features = np.hstack([[area, aspect_ratio, extent, solidity], hu_log])
    return features.astype(np.float32)


def classify_defects(
    img_aligned: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    padding: int = 5,
) -> List[str]:
    """
    Phân loại từng đốm lỗi trong danh sách bounding boxes.
    Nếu có model huấn luyện (model_classifier.pkl) -> Dự đoán bằng mô hình.
    Nếu chưa có model -> Ước lượng dựa trên đặc trưng hình học hoặc gắn nhãn mặc định.

    Trả về:
        Danh sách nhãn lỗi tương ứng cho từng box: ['Open_circuit', 'Short', ...]
    """
    if not boxes:
        return []

    h_img, w_img = img_aligned.shape[:2]
    labels: List[str] = []

    # Kiểm tra xem có file mô hình đã train hay không
    model = None
    if MODEL_PATH.is_file():
        try:
            import pickle
            with open(MODEL_PATH, "rb") as fh:
                model = pickle.load(fh)
        except Exception:
            model = None

    for i, box in enumerate(boxes, start=1):
        x, y, w, h = box[:4]

        # Cắt patch kèm padding
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(w_img, x + w + padding), min(h_img, y + h + padding)
        patch = img_aligned[y1:y2, x1:x2]

        if model is not None:
            try:
                feats = extract_defect_features(patch).reshape(1, -1)
                pred_label = model.predict(feats)[0]
                labels.append(str(pred_label))
                continue
            except Exception:
                pass

        # Phân loại theo luật hình học (Rule-based Fallback khi chưa train model):
        area = w * h
        aspect_ratio = float(w) / h if h > 0 else 1.0

        if 0.8 <= aspect_ratio <= 1.25 and area < 400:
            labels.append("Missing_hole")
        elif aspect_ratio > 3.0 or aspect_ratio < 0.33:
            labels.append("Open_circuit")
        elif area > 1000:
            labels.append("Short")
        else:
            # Gán nhãn khuyết tật tổng quát
            labels.append(f"Defect #{i}")

    return labels

