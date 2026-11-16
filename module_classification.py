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


def extract_features(gray: np.ndarray, mask: np.ndarray,
                     boxes: Sequence[Box]) -> np.ndarray:
    """Trả (N, len(FEATURE_NAMES)) float32 cho từng bbox."""
    raise NotImplementedError(
        "extract_features chưa hoàn thiện (việc P1 #3 trong TASKS_NOW.md). "
        "Việc P0 #1 chỉ chốt hợp đồng: tên lớp, tên đặc trưng, load_model, classify."
    )


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
