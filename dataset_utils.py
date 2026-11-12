"""Tiện ích đọc manifest cặp ảnh PKU-Market-PCB.

Dữ liệu dùng cấu trúc gốc `raw/PCB_USED`, `raw/images`, `raw/Annotations`
kèm `pairs_manifest.csv` và `labels/`; không còn cặp `_temp`/`_test`.

Mọi module (main.py, experiments/*.py) nên dùng chung các hàm ở đây để
tránh mỗi nơi tự quét ảnh theo một kiểu.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def _resolve_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_manifest(manifest_path: Path) -> List[Dict[str, object]]:
    """Đọc manifest và chuyển các cột đường dẫn thành Path tuyệt đối."""
    manifest_path = Path(manifest_path)
    base = manifest_path.parent

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    pairs: List[Dict[str, object]] = []
    for row in rows:
        pair = dict(row)
        pair["class_id"] = int(pair["class_id"]) if pair.get("class_id") else None
        pair["num_objects"] = (
            int(pair["num_objects"]) if pair.get("num_objects") else 0
        )
        pair["width"] = int(pair["width"]) if pair.get("width") else None
        pair["height"] = int(pair["height"]) if pair.get("height") else None

        pair["test_path"] = _resolve_path(base, pair.get("test_path"))
        pair["template_path"] = _resolve_path(base, pair.get("template_path"))
        pair["annotation_path"] = _resolve_path(base, pair.get("annotation_path"))
        pair["label_path"] = _resolve_path(base, pair.get("label_path"))

        if pair["test_path"] is None or pair["template_path"] is None:
            continue
        pairs.append(pair)

    pairs.sort(key=lambda item: str(item["pair_id"]).lower())
    return pairs


def find_pairs(data_dir: str = "data") -> List[Dict[str, object]]:
    """Tìm `pairs_manifest.csv` trong `data_dir` và trả về danh sách cặp ảnh."""
    root = Path(data_dir)
    if not root.exists():
        return []

    if root.is_file() and root.name == "pairs_manifest.csv":
        return load_manifest(root)

    candidates = [
        root / "pairs_manifest.csv",
        root / "PKU-Market-PCB" / "pairs_manifest.csv",
    ]
    candidates.extend(sorted(root.rglob("pairs_manifest.csv")))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return load_manifest(candidate)

    return []


def load_image(path: Path) -> np.ndarray:
    """Đọc ảnh BGR, hỗ trợ đường dẫn Unicode trên Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return image
