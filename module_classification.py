"""Phân loại khuyết tật PCB bằng Machine Learning truyền thống (SVM / Random Forest).

Module này là **hợp đồng dùng chung** giữa phần phát hiện (đốm lỗi từ
`module_detection.extract_bounding_boxes`) và phần giao diện (`gui_app.py`).

Phạm vi:
    - Định nghĩa tên lớp, tên đặc trưng, nhãn hiển thị.
    - Trích đặc trưng hình thái cho từng blob (`extract_features`).
    - Nạp/lưu model đã huấn luyện (`load_model`, `save_model`).
    - Suy luận nhãn cho danh sách bbox (`classify`).

KHÔNG thuộc phạm vi module này: huấn luyện và đánh giá. Việc đó nằm ở
`experiments/build_classifier_dataset.py` (sinh dataset) và
`experiments/exp_classify.py` (train, confusion matrix, lưu model).

Ví dụ dùng trong GUI khi model đã có:

    from module_classification import classify
    labels = classify(gray, mask, boxes)      # ["Spur", "không phải lỗi", ...]

Khi chưa huấn luyện xong (hoặc không tìm thấy `model_classifier.pkl`),
`classify` trả về `"chưa phân loại"` cho mọi bbox — GUI vẫn chạy được.

Dùng OpenCV ML (`cv2.ml.SVM`, `cv2.ml.RTrees`), không cần scikit-learn.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[int, int, int, int]

# --- Tên lớp -------------------------------------------------------------
# 6 nhóm khuyết tật theo IPC-A-600, đúng thứ tự `class_name` trong
# data/PKU-Market-PCB/pairs_manifest.csv.
DEFECT_CLASSES: List[str] = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper",
]

# Lớp thứ 7: blob phát hiện nhưng không khớp nhãn GT nào (báo động giả).
# Nhờ lớp này, model có thể học cách loại đốm nhiễu thay vì chỉ gán nhãn.
BACKGROUND_CLASS = "background"

CLASS_NAMES: List[str] = DEFECT_CLASSES + [BACKGROUND_CLASS]

# Nhãn hiển thị cho người dùng cuối (GUI / ảnh kết quả).
LABEL_UNKNOWN = "chưa phân loại"      # chưa có model
LABEL_BACKGROUND = "không phải lỗi"   # model đoán là báo động giả

# --- Tên đặc trưng -------------------------------------------------------
# Thứ tự này phải khớp tuyệt đối với vector do `extract_features` trả về,
# vì model đã huấn luyện phụ thuộc vào đúng thứ tự cột.
FEATURE_NAMES: List[str] = [
    "area",         # diện tích blob (pixel)
    "perimeter",    # chu vi contour
    "aspect",       # cạnh dài / cạnh ngắn của minAreaRect (>= 1)
    "extent",       # area / (w*h) của bounding box
    "solidity",     # area / diện tích convex hull
    "circularity",  # 4*pi*area / perimeter^2
    "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7",  # log-scale Hu moments
    "mean_diff",    # độ sáng trung bình trong blob (bản đồ vi sai)
]

DEFAULT_MODEL_PATH = Path("model_classifier.pkl")


def extract_features(gray: np.ndarray, mask: np.ndarray,
                     boxes: Sequence[Box]) -> np.ndarray:
    """Trích vector đặc trưng hình thái cho từng bbox.

    Tham số:
        gray  : ảnh xám của ảnh test đã căn chỉnh (uint8, shape HxW).
        mask  : mask nhị phân của bản đồ vi sai (uint8, 0/255, cùng shape).
        boxes : danh sách (x, y, w, h) từ `extract_bounding_boxes`.

    Trả về:
        np.ndarray shape (N, len(FEATURE_NAMES)), dtype float32, không có NaN.

    Ghi chú triển khai (việc P1 #3 trong TASKS_NOW.md):
        - Tách contour con trong từng bbox bằng `cv2.findContours` trên `mask`.
        - Hu moments phải log-scale: ``-sign(h) * log10(|h|)`` (Hu thô lệch
          nhiều bậc độ lớn, không chuẩn hoá thì SVM gần như bỏ qua).
        - `mean_diff` lấy trung bình `gray` trong contour (hoặc trong bbox).
    """
    raise NotImplementedError(
        "extract_features chưa hoàn thiện (việc P1 #3 trong TASKS_NOW.md). "
        "Việc P0 #1 chỉ chốt hợp đồng: tên lớp, tên đặc trưng, load_model, classify."
    )


def load_model(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Nạp model đã huấn luyện; trả `None` nếu chưa có file (GUI vẫn chạy).

    Định dạng file (pickle) — do `experiments/exp_classify.py` ghi ra::

        {
            "model":   cv2.ml.SVM / cv2.ml.RTrees (có .predict),
            "scaler":  {"mean": np.ndarray, "std": np.ndarray} hoặc None,
            "classes": ["Missing_hole", ..., "background"],
            "feature_names": FEATURE_NAMES,
        }
    """
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
    """Lưu model ra pickle; trả về đường dẫn đã ghi."""
    model_path = Path(path) if path is not None else DEFAULT_MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return model_path


def classify(gray: np.ndarray, mask: np.ndarray, boxes: Sequence[Box],
             model: Optional[Dict[str, Any]] = None) -> List[str]:
    """Gán nhãn cho từng bbox; không có model thì trả `LABEL_UNKNOWN`.

    Chữ ký này là hợp đồng với `gui_app.py`: chỉ cần gray + mask + boxes.
    Lớp `background` được đổi thành nhãn thân thiện `"không phải lỗi"`.
    """
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

    estimator = model["model"]
    output = estimator.predict(np.asarray(features, dtype=np.float32))
    # cv2.ml trả về tuple (retval, results); một số wrapper chỉ trả mảng.
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
