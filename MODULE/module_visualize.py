"""Alias module chuyển tiếp tới module_visualization.py.

Đảm bảo tương thích với cả hai cách đặt tên:
- module_visualize.py (theo TASKS_NOW.md)
- module_visualization.py (module chuẩn trong dự án)
"""

try:
    from .module_visualization import (
        draw_feature_matches,
        draw_checkerboard,
        draw_alpha_overlay,
        draw_homography_verification,
        draw_results,
    )
except ImportError:
    from MODULE.module_visualization import (
        draw_feature_matches,
        draw_checkerboard,
        draw_alpha_overlay,
        draw_homography_verification,
        draw_results,
    )

__all__ = [
    "draw_feature_matches",
    "draw_checkerboard",
    "draw_alpha_overlay",
    "draw_homography_verification",
    "draw_results",
]
