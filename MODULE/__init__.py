"""Package MODULE: Chứa toàn bộ các module xử lý cốt lõi của hệ thống kiểm tra PCB.

Bao gồm:
- module_preprocess: Tiền xử lý, lọc nhiễu, căn chỉnh ảnh (ORB/SIFT/Homography), trừ vi sai.
- module_detection: Phát hiện đường viền (contours), trích xuất bounding boxes và cắt đốm lỗi.
- module_visualization / module_visualize: Trực quan hóa điểm đặc trưng, bàn cờ checkerboard, alpha overlay, đóng khung lỗi.
- module_classification: Trích xuất đặc trưng hình thái đốm lỗi và phân loại khuyết tật (ML).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc dự án và thư mục MODULE luôn có trong sys.path
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent

for _p in (str(_PROJECT_ROOT), str(_MODULE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from . import module_preprocess
from . import module_detection
from . import module_visualization
from . import module_visualize
from . import module_classification

__all__ = [
    "module_preprocess",
    "module_detection",
    "module_visualization",
    "module_visualize",
    "module_classification",
]

