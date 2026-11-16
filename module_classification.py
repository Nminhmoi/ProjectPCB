"""Phân loại khuyết tật PCB bằng ML truyền thống (SVM / Random Forest, `cv2.ml`).

Hợp đồng dùng chung giữa phần phát hiện và `gui_app.py`:
`CLASS_NAMES`, `FEATURE_NAMES`, `extract_features`, `load_model`, `save_model`,
`classify`. Huấn luyện/đánh giá nằm ở `experiments/exp_classify.py`.

Chưa có `model_classifier.pkl` thì `classify` trả `"chưa phân loại"` cho mọi box.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]

DEFECT_CLASSES: List[str] = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper",
]

BACKGROUND_CLASS = "background"
CLASS_NAMES: List[str] = DEFECT_CLASSES + [BACKGROUND_CLASS]

LABEL_UNKNOWN = "chưa phân loại"
LABEL_BACKGROUND = "không phải lỗi"

# Thứ tự phải khớp tuyệt đối với cột của model đã huấn luyện.
FEATURE_NAMES: List[str] = [
    "area", "perimeter", "aspect", "extent", "solidity", "circularity",
    "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7", "mean_diff",
]

DEFAULT_MODEL_PATH = Path("model_classifier.pkl")


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Đảm bảo ảnh 1 kênh uint8."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _largest_contour(crop: np.ndarray) -> Optional[np.ndarray]:
    """Contour ngoài lớn nhất trong vùng crop."""
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return contour if cv2.contourArea(contour) > 0 else None


def _features_one(gray: np.ndarray, mask: np.ndarray, box: Box) -> np.ndarray:
    """Vector đặc trưng của blob lớn nhất trong 1 bbox."""
    empty = np.zeros(len(FEATURE_NAMES), np.float32)
    height, width = mask.shape[:2]
    x, y, w, h = (int(v) for v in box)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return empty

    crop = np.ascontiguousarray(mask[y0:y1, x0:x1])
    contour = _largest_contour(crop)
    if contour is None:
        return empty

    # Diện tích tính bằng số pixel thuộc blob để extent/solidity nằm gọn trong [0, 1].
    blob = np.zeros(crop.shape, np.uint8)
    cv2.drawContours(blob, [contour], -1, 255, -1)
    area = float(cv2.countNonZero(blob))
    if area <= 0.0:
        return empty

    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0.0:
        return empty

    (_, _), (side_a, side_b), _ = cv2.minAreaRect(contour)
    aspect = max(side_a, side_b) / max(min(side_a, side_b), 1e-6)

    _, _, box_w, box_h = cv2.boundingRect(blob)
    extent = area / float(box_w * box_h) if box_w * box_h else 0.0

    hull_blob = np.zeros(crop.shape, np.uint8)
    cv2.fillConvexPoly(hull_blob, cv2.convexHull(contour), 255)
    hull_area = float(cv2.countNonZero(hull_blob))
    solidity = area / hull_area if hull_area > 0.0 else 0.0

    circularity = 4.0 * np.pi * area / (perimeter ** 2)

    hu = cv2.HuMoments(cv2.moments(contour)).ravel()
    hu_log = np.zeros(7, dtype=np.float64)
    nonzero = np.abs(hu) > 0.0
    hu_log[nonzero] = -np.sign(hu[nonzero]) * np.log10(np.abs(hu[nonzero]))

    pixels = gray[y0:y1, x0:x1][blob > 0]
    mean_diff = float(pixels.mean()) if pixels.size else 0.0

    row = np.concatenate((
        np.array([area, perimeter, aspect, extent, solidity, circularity]),
        hu_log,
        np.array([mean_diff]),
    )).astype(np.float32)
    return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(gray: np.ndarray, mask: np.ndarray,
                     boxes: Sequence[Box]) -> np.ndarray:
    """Trả (N, len(FEATURE_NAMES)) float32 cho từng bbox."""
    boxes = list(boxes or [])
    if not boxes:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32)

    gray = _to_gray(np.asarray(gray))
    mask = _to_gray(np.asarray(mask))
    if mask.shape[:2] != gray.shape[:2]:
        raise ValueError(
            f"gray {gray.shape[:2]} và mask {mask.shape[:2]} phải cùng kích thước."
        )
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    return np.vstack([_features_one(gray, mask, box) for box in boxes])


def load_model(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Nạp model pickle; trả `None` nếu chưa có file."""
    model_path = Path(path) if path is not None else DEFAULT_MODEL_PATH
    if not model_path.is_file():
        return None
    with model_path.open("rb") as fh:
        model = pickle.load(fh)
    if not isinstance(model, dict) or "model" not in model:
        raise ValueError(
            f"{model_path} không đúng định dạng model (thiếu khoá 'model')."
        )
    return model


def save_model(model: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Lưu model ra pickle; trả đường dẫn đã ghi."""
    model_path = Path(path) if path is not None else DEFAULT_MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return model_path


def classify(gray: np.ndarray, mask: np.ndarray, boxes: Sequence[Box],
             model: Optional[Dict[str, Any]] = None) -> List[str]:
    """Gán nhãn cho từng bbox; chưa có model thì trả `LABEL_UNKNOWN`."""
    boxes = list(boxes or [])
    if not boxes:
        return []

    model = model if model is not None else load_model()
    if model is None:
        return [LABEL_UNKNOWN] * len(boxes)

    features = extract_features(gray, mask, boxes)
    scaler = model.get("scaler")
    if scaler is not None:
        std = np.asarray(scaler["std"], dtype=np.float64)
        std = np.where(std < 1e-8, 1.0, std)
        features = (np.asarray(features, dtype=np.float64)
                    - np.asarray(scaler["mean"], dtype=np.float64)) / std

    output = model["model"].predict(np.asarray(features, dtype=np.float32))
    predictions = output[1] if isinstance(output, tuple) else output
    predictions = np.asarray(predictions).ravel()

    classes = list(model.get("classes", CLASS_NAMES))
    labels: List[str] = []
    for value in predictions:
        index = int(round(float(value)))
        if not 0 <= index < len(classes):
            labels.append(LABEL_UNKNOWN)
            continue
        name = classes[index]
        labels.append(LABEL_BACKGROUND if name == BACKGROUND_CLASS else name)
    return labels
