"""CLI đánh giá định lượng hệ thống phát hiện lỗi PCB (evaluate_metrics.py).

Bọc logic từ `evaluation.py` để tính toán:
- Tỷ lệ phát hiện (Recall), độ chính xác (Precision), điểm F1.
- Tiêu chí chạm nhau có dung sai 5px (`touch_tol5`).
- Tiêu chuẩn IoU 1-1 (IoU >= 0.3 và IoU >= 0.5).
- Tỷ lệ báo động giả / ảnh (False blobs / img).
- Bảng tổng hợp toàn diện (Overall) và bảng phân tích theo từng lớp lỗi (Per-class).

Chạy:
    python evaluate_metrics.py --limit 3               # Test nhanh 3 cặp theo DoD
    python evaluate_metrics.py --per-class 2           # Mỗi lớp lỗi lấy 2 cặp
    python evaluate_metrics.py --class-name Short      # Đánh giá riêng lớp Short
    python evaluate_metrics.py --data-dir data         # Đánh giá full 693 cặp
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Đảm bảo đường dẫn gốc và thư mục MODULE luôn có trong sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "MODULE")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np

from dataset_utils import find_pairs, load_image
from evaluation import IOU_THRESHOLDS, detection_counts, prf_from_counts, read_labels
from MODULE.module_detection import extract_bounding_boxes
from MODULE.module_preprocess import (
    DEFAULT_DETECTOR,
    DEFAULT_MAX_FEATURES,
    DEFAULT_MIN_AREA,
    DEFAULT_MIN_NCC,
    DEFAULT_OPEN_KSIZE,
    DEFAULT_THRESHOLD_FLOOR,
    AlignmentError,
    align_images_robust,
    detect_difference,
    warp_valid_mask,
)

CLASS_NAMES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper",
]


# =====================================================================
# 1. BỘ SINH ẢNH GIẢ LẬP KHI CHƯA CÓ THƯ MỤC RAW/ 1.9GB
# =====================================================================
def _create_synthetic_pair(
    pair: dict, gt_boxes: List[Tuple[int, int, int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Tạo cặp ảnh PCB mô phỏng (template và test) khi máy chưa tải bộ 1.9GB raw/.

    Vẽ các vết khuyết tật tại đúng các tọa độ Ground Truth để kiểm thử
    toàn bộ chuỗi tính toán IoU, Precision, Recall và định dạng bảng.
    """
    w = int(pair.get("width") or 800)
    h = int(pair.get("height") or 600)

    # Co giãn kích thước về mức nhẹ nhàng nếu ảnh gốc quá lớn mà chạy giả lập
    scale = min(1.0, 1000.0 / float(w))
    eff_w = max(400, int(w * scale))
    eff_h = max(300, int(h * scale))

    # Canvas bo mạch màu xanh lá đậm
    template = np.full((eff_h, eff_w, 3), (35, 100, 35), dtype=np.uint8)
    rng = np.random.default_rng(int(pair.get("class_id") or 1) * 100 + 42)

    # Vẽ các pad linh kiện và đường mạch nền để tạo điểm đặc trưng
    for _ in range(80):
        px = int(rng.integers(30, eff_w - 30))
        py = int(rng.integers(30, eff_h - 30))
        psize = int(rng.integers(8, 20))
        color = (
            int(rng.integers(0, 50)),
            int(rng.integers(180, 255)),
            int(rng.integers(200, 255)),
        )
        cv2.rectangle(template, (px, py), (px + psize, py + psize), color, -1)
        cv2.circle(
            template,
            (px + psize // 2, py + psize // 2),
            max(2, psize // 4),
            (0, 0, 0),
            -1,
        )

    for i in range(8):
        y_line = int((i + 1) * eff_h / 9)
        cv2.line(
            template, (20, y_line), (eff_w - 20, y_line), (0, 210, 240), 2
        )

    # Ảnh test được tạo từ template kèm các đốm lỗi ở vị trí GT (màu trắng tương phản cao để trừ vi sai)
    test = template.copy()
    for x, y, bw, bh in gt_boxes:
        sx = int(x * scale)
        sy = int(y * scale)
        sbw = max(6, int(bw * scale))
        sbh = max(6, int(bh * scale))
        cv2.rectangle(test, (sx, sy), (sx + sbw, sy + sbh), (255, 255, 255), -1)

    # Thêm nhiễu nhẹ và độ lệch nhỏ để mô phỏng camera
    M = cv2.getRotationMatrix2D((eff_w // 2, eff_h // 2), 0.3, 1.0)
    M[0, 2] += 2.0
    M[1, 2] += 1.0
    test = cv2.warpAffine(test, M, (eff_w, eff_h), borderMode=cv2.BORDER_REPLICATE)

    return template, test


# =====================================================================
# 2. XỬ LÝ 1 CẶP ẢNH VÀ ĐO ĐẠC METRICS
# =====================================================================
def evaluate_single_pair(
    pair: dict,
    detector: str = DEFAULT_DETECTOR,
    max_features: int = DEFAULT_MAX_FEATURES,
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
    force_synthetic: bool = False,
) -> Dict[str, object]:
    """Chạy pipeline và so sánh bounding box dự đoán với Ground Truth."""
    t0 = time.perf_counter()

    # 1. Đọc Ground Truth
    gt_boxes = read_labels(pair.get("label_path"))

    # 2. Kiểm tra xem ảnh gốc có tồn tại hay không
    test_path = Path(str(pair.get("test_path", "")))
    temp_path = Path(str(pair.get("template_path", "")))
    is_real = test_path.is_file() and temp_path.is_file() and not force_synthetic

    if is_real:
        img_test = load_image(test_path)
        img_temp = load_image(temp_path)
    else:
        # Nếu chưa có ảnh gốc raw/ 1.9GB -> Tự động dùng ảnh mô phỏng có đốm lỗi chuẩn
        img_temp, img_test = _create_synthetic_pair(pair, gt_boxes)

    # 3. Căn chỉnh ảnh (Alignment)
    try:
        aligned, stats = align_images_robust(
            img_temp, img_test,
            max_features=max_features,
            detector=detector,
            min_ncc=DEFAULT_MIN_NCC,
        )
    except AlignmentError:
        aligned = img_test
        stats = {"success": False, "inliers": 0, "ncc": 0.0}

    # 4. Trừ vi sai & Nhị phân hóa
    valid_mask = None
    if stats.get("success") and stats.get("H") is not None:
        valid_mask = warp_valid_mask(
            img_test, stats["H"], img_temp.shape[:2], erode_border=3
        )

    diff_mask = detect_difference(
        img_temp, aligned,
        threshold_floor=DEFAULT_THRESHOLD_FLOOR,
        open_ksize=DEFAULT_OPEN_KSIZE,
        valid=valid_mask,
    )

    # 5. Trích xuất bounding boxes
    pred_boxes = extract_bounding_boxes(diff_mask, min_area=DEFAULT_MIN_AREA)

    # Nếu dùng ảnh giả lập co giãn, scale lại tọa độ pred_boxes về hệ gốc của GT
    if not is_real and len(gt_boxes) > 0 and len(pred_boxes) > 0:
        w = int(pair.get("width") or 800)
        scale = min(1.0, 1000.0 / float(w))
        if scale < 1.0:
            pred_boxes = [
                (
                    int(bx / scale),
                    int(by / scale),
                    int(bw / scale),
                    int(bh / scale),
                )
                for bx, by, bw, bh in pred_boxes
            ]

    # 6. So khớp với GT qua evaluation.py
    counts = detection_counts(pred_boxes, gt_boxes, iou_thresholds=iou_thresholds)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    row: Dict[str, object] = {
        "pair_id": pair.get("pair_id", ""),
        "class_name": pair.get("class_name", ""),
        "is_synthetic": not is_real,
        "elapsed_ms": elapsed_ms,
        "align_success": stats.get("success", False),
        "inliers": stats.get("inliers", 0),
        "ncc": stats.get("ncc", 0.0),
    }
    row.update(counts)
    return row


# =====================================================================
# 3. ĐỊNH DẠNG BẢNG BÁO CÁO (MARKDOWN & CLI ALIGNED)
# =====================================================================
def str_width(s: object) -> int:
    """Tính độ rộng hiển thị (display width) theo ký tự monospace, hỗ trợ tiếng Việt có dấu."""
    normalized = unicodedata.normalize("NFC", str(s))
    w = 0
    for ch in normalized:
        ea = unicodedata.east_asian_width(ch)
        w += 2 if ea in ("W", "F") else 1
    return w


def pad_cell(s: object, width: int, align: str = "center") -> str:
    """Canh lề nội dung ô theo chiều rộng hiển thị monospace."""
    text = str(s)
    sw = str_width(text)
    diff = max(0, width - sw)
    if align == "left":
        return text + " " * diff
    elif align == "right":
        return " " * diff + text
    else:  # center
        left = diff // 2
        right = diff - left
        return " " * left + text + " " * right


def render_table(
    headers: List[str],
    aligns: List[str],
    rows: List[List[object]],
    style: str = "markdown",
) -> str:
    """Vẽ bảng với các cột được căn thẳng hàng tuyệt đối.

    Hỗ trợ hai phong cách:
    - 'markdown': Bảng chuẩn GFM Markdown, các dấu gạch đứng | thẳng tắp.
    - 'box': Bảng viền Unicode Box Drawing thanh mảnh, hiện đại cho CLI.
    """
    n_cols = len(headers)
    widths = [str_width(h) for h in headers]
    for row in rows:
        for i in range(min(n_cols, len(row))):
            widths[i] = max(widths[i], str_width(row[i]))
    widths = [max(w, 5) for w in widths]

    lines: List[str] = []
    if style == "box":
        top = "┌─" + "─┬─".join("─" * w for w in widths) + "─┐"
        mid = "├─" + "─┼─".join("─" * w for w in widths) + "─┤"
        bot = "└─" + "─┴─".join("─" * w for w in widths) + "─┘"
        hdr = "│ " + " │ ".join(pad_cell(h, widths[i], "center") for i, h in enumerate(headers)) + " │"
        lines.extend([top, hdr, mid])
        for row in rows:
            clean_cells = [str(row[i]).replace("**", "") if i < len(row) else "" for i in range(n_cols)]
            r = "│ " + " │ ".join(pad_cell(clean_cells[i], widths[i], aligns[i]) for i in range(n_cols)) + " │"
            lines.append(r)
        lines.append(bot)
    else:  # markdown
        hdr = "| " + " | ".join(pad_cell(h, widths[i], "center") for i, h in enumerate(headers)) + " |"
        lines.append(hdr)
        for row in rows:
            r = "| " + " | ".join(pad_cell(str(row[i]), widths[i], aligns[i]) for i in range(n_cols)) + " |"
            lines.append(r)

    return "\n".join(lines)


def _fmt_pct(val: Optional[float]) -> str:
    if val is None or math.isnan(val):
        return "–"
    return f"{val * 100.0:.1f}%"


def _fmt_num(val: Optional[float], decimals: int = 4) -> str:
    if val is None or math.isnan(val):
        return "–"
    return f"{val:.{decimals}f}"


def format_report(
    rows: List[Dict[str, object]],
    detector: str = DEFAULT_DETECTOR,
    max_features: int = DEFAULT_MAX_FEATURES,
    total_time_s: float = 0.0,
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
    style: str = "markdown",
) -> str:
    """Tạo văn bản báo cáo gồm 2 bảng chuẩn: Bảng tổng hợp & Bảng theo lớp lỗi."""
    lines: List[str] = []
    bar_width = 110

    # Tiêu đề báo cáo
    lines.append("=" * bar_width)
    lines.append(" BÁO CÁO ĐÁNH GIÁ ĐỊNH LƯỢNG PHÁT HIỆN LỖI PCB (EVALUATION METRICS)")
    lines.append("=" * bar_width)
    lines.append(f" • Bộ đặc trưng căn chỉnh : {detector} (max_features = {max_features})")
    lines.append(f" • Tổng số cặp đã xử lý  : {len(rows)} cặp")
    if rows:
        synth_count = sum(1 for r in rows if r.get("is_synthetic"))
        if synth_count > 0:
            lines.append(f" • Chế độ dữ liệu         : {len(rows) - synth_count} ảnh thật, {synth_count} ảnh mô phỏng GT")
    lines.append(
        f" • Tổng thời gian chạy   : {total_time_s:.2f} s (trung bình: {total_time_s / max(1, len(rows)) * 1000.0:.1f} ms/cặp)"
    )
    lines.append("-" * bar_width)

    # 1. BẢNG 1: TỔNG THỂ THEO TIÊU CHÍ GHÉP (OVERALL BENCHMARK)
    lines.append("\nBẢNG 1: TỔNG HỢP THEO TIÊU CHÍ GHÉP (OVERALL BENCHMARK)\n")
    overall = prf_from_counts(rows, iou_thresholds=iou_thresholds)

    h1 = [
        "Tiêu chí đánh giá (Rule)",
        "Số GT",
        "Số Pred",
        "TP",
        "FP",
        "FN",
        "Recall",
        "Precision",
        "F1-Score",
        "Báo động giả / ảnh",
    ]
    a1 = ["left", "center", "center", "center", "center", "center", "center", "center", "center", "center"]

    r1: List[List[object]] = []
    n_gt = overall["n_gt"]
    n_pred = overall["n_pred"]
    tp_touch = sum(int(r.get("gt_hit", 0)) for r in rows)
    fp_touch = sum(int(r.get("false_blobs", 0)) for r in rows)
    fn_touch = max(0, n_gt - tp_touch)
    rec_touch = overall.get("recall", float("nan"))
    prec_touch = overall.get("precision", float("nan"))
    f1_touch = overall.get("f1", float("nan"))
    f_blobs = overall.get("false_blobs_per_img", float("nan"))

    r1.append([
        "Chạm nhau (Touch tol=5px)",
        str(n_gt),
        str(n_pred),
        str(tp_touch),
        str(fp_touch),
        str(fn_touch),
        _fmt_num(rec_touch),
        _fmt_num(prec_touch),
        f"{_fmt_num(f1_touch)}" if style == "markdown" else _fmt_num(f1_touch),
        _fmt_num(f_blobs, 2),
    ])

    for t in iou_thresholds:
        suffix = f"iou{int(round(t * 100))}"
        tp_iou = sum(int(r.get(f"tp_{suffix}", 0)) for r in rows)
        fp_iou = max(0, n_pred - tp_iou)
        fn_iou = max(0, n_gt - tp_iou)
        r_ = overall.get(f"recall_{suffix}", float("nan"))
        p_ = overall.get(f"precision_{suffix}", float("nan"))
        f_ = overall.get(f"f1_{suffix}", float("nan"))
        r1.append([
            f"Chuẩn IoU >= {t:.2f}",
            str(n_gt),
            str(n_pred),
            str(tp_iou),
            str(fp_iou),
            str(fn_iou),
            _fmt_num(r_),
            _fmt_num(p_),
            f"{_fmt_num(f_)}" if style == "markdown" else _fmt_num(f_),
            _fmt_num(fp_iou / max(1, len(rows)), 2),
        ])

    lines.append(render_table(h1, a1, r1, style=style))

    # 2. BẢNG 2: PHÂN TÍCH CHI TIẾT THEO TỪNG LỚP KHUYẾT TẬT
    lines.append("\nBẢNG 2: ĐÁNH GIÁ CHI TIẾT THEO TỪNG LỚP KHUYẾT TẬT (PER-CLASS PERFORMANCE)\n")
    h2 = [
        "Lớp khuyết tật (Defect Class)",
        "Số cặp",
        "Số GT",
        "Số Pred",
        "Recall (Touch)",
        "Precision (Touch)",
        "F1 (Touch)",
    ]
    for t in iou_thresholds:
        h2.append(f"F1 (IoU@{t:.1f})")

    a2 = ["left", "center", "center", "center", "center", "center", "center"] + ["center"] * len(iou_thresholds)

    all_classes = sorted({str(r.get("class_name")) for r in rows if r.get("class_name")})
    if not all_classes:
        all_classes = CLASS_NAMES

    r2: List[List[object]] = []
    for cls in all_classes:
        cls_rows = [r for r in rows if str(r.get("class_name")) == cls]
        if not cls_rows:
            empty_row = [cls, "0", "0", "0", "–", "–", "–"] + ["–"] * len(iou_thresholds)
            r2.append(empty_row)
            continue

        c_prf = prf_from_counts(cls_rows, iou_thresholds=iou_thresholds)
        row_vals: List[object] = [
            cls,
            str(len(cls_rows)),
            str(c_prf["n_gt"]),
            str(c_prf["n_pred"]),
            _fmt_num(c_prf.get("recall")),
            _fmt_num(c_prf.get("precision")),
            _fmt_num(c_prf.get("f1")),
        ]
        for t in iou_thresholds:
            sfx = f"iou{int(round(t * 100))}"
            row_vals.append(_fmt_num(c_prf.get(f"f1_{sfx}")))
        r2.append(row_vals)

    # Dòng tổng cộng (Micro-average)
    def _b(v: str) -> str:
        return f"{v}" if style == "markdown" else v

    tot_vals: List[object] = [
        _b("TỔNG CỘNG (MICRO)"),
        _b(str(len(rows))),
        _b(str(overall["n_gt"])),
        _b(str(overall["n_pred"])),
        _b(_fmt_num(overall.get("recall"))),
        _b(_fmt_num(overall.get("precision"))),
        _b(_fmt_num(overall.get("f1"))),
    ]
    for t in iou_thresholds:
        sfx = f"iou{int(round(t * 100))}"
        tot_vals.append(_b(_fmt_num(overall.get(f"f1_{sfx}"))))
    r2.append(tot_vals)

    lines.append(render_table(h2, a2, r2, style=style))

    # 3. ĐỐI CHIẾU KPI
    lines.append("\n" + "-" * bar_width)
    lines.append(" KPI ĐỐI CHIẾU DỰ ÁN (theo EXPERIMENT_REVIEW.md mục 5):")
    kpi_rec = overall.get("recall", float("nan"))
    kpi_f1 = overall.get("f1", float("nan"))
    kpi_fb = overall.get("false_blobs_per_img", float("nan"))

    rec_pass = "ĐẠT" if not math.isnan(kpi_rec) and kpi_rec >= 0.97 else "CHƯA ĐẠT"
    f1_pass = "ĐẠT" if not math.isnan(kpi_f1) and kpi_f1 >= 0.975 else "CHƯA ĐẠT"
    fb_pass = "ĐẠT" if not math.isnan(kpi_fb) and kpi_fb <= 0.05 else "CHƯA ĐẠT"

    lines.append(f"   • Recall (Touch)   : {_fmt_num(kpi_rec)} (Mục tiêu >= 0.9700) -> [{rec_pass}]")
    lines.append(f"   • F1-Score (Touch) : {_fmt_num(kpi_f1)} (Mục tiêu >= 0.9750) -> [{f1_pass}]")
    lines.append(f"   • Báo động giả/ảnh : {_fmt_num(kpi_fb, 2)}   (Mục tiêu <= 0.05)   -> [{fb_pass}]")
    lines.append("=" * bar_width)

    return "\n".join(lines)


# =====================================================================
# 4. HÀM CHÍNH & XỬ LÝ DÒNG LỆNH (CLI)
# =====================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá định lượng phát hiện lỗi PCB: IoU, Precision, Recall, F1 trên 693 cặp."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Thư mục dataset chứa pairs_manifest.csv (mặc định: 'data').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Giới hạn số cặp chạy thử nghiệm (ví dụ --limit 3; 0 = chạy tất cả).",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=0,
        help="Giới hạn số cặp trên mỗi lớp khuyết tật (0 = không giới hạn).",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        help="Lọc chỉ đánh giá một lớp lỗi cụ thể (ví dụ: Missing_hole, Short).",
    )
    parser.add_argument(
        "--detector",
        default=DEFAULT_DETECTOR,
        help=f"Bộ đặc trưng căn chỉnh ảnh (mặc định: {DEFAULT_DETECTOR}).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        help=f"Số lượng đặc trưng tối đa (mặc định: {DEFAULT_MAX_FEATURES}).",
    )
    parser.add_argument(
        "--iou-thresh",
        nargs="+",
        type=float,
        default=list(IOU_THRESHOLDS),
        help="Các ngưỡng IoU để đánh giá (mặc định: 0.3 0.5).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Bắt buộc sử dụng dữ liệu mô phỏng để kiểm thử nhanh khi không có ảnh raw.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Đường dẫn lưu bảng kết quả tổng hợp ra file CSV.",
    )
    parser.add_argument(
        "--per-pair-csv",
        type=str,
        default=None,
        help="Đường dẫn lưu chi tiết metrics từng cặp ảnh ra file CSV.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "box"],
        default="markdown",
        help="Định dạng hiển thị bảng: 'markdown' (mặc định, căn lề thẳng hàng) hoặc 'box' (khung viền Unicode CLI).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Chạy ẩn log chi tiết từng cặp, chỉ in bảng tổng kết cuối cùng.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    # 1. Tìm và nạp danh sách các cặp ảnh
    try:
        all_pairs = find_pairs(args.data_dir)
    except Exception as exc:
        print(f"[LỖI] Không nạp được manifest từ '{args.data_dir}': {exc}")
        sys.exit(1)

    if not all_pairs:
        print(f"[LỖI] Không tìm thấy cặp ảnh nào trong '{args.data_dir}'.")
        sys.exit(1)

    # 2. Lọc theo lớp lỗi (nếu có)
    if args.class_name:
        selected_pairs = [
            p for p in all_pairs if str(p.get("class_name", "")).lower() == args.class_name.lower()
        ]
        if not selected_pairs:
            print(f"[LỖI] Không tìm thấy cặp ảnh nào thuộc lớp '{args.class_name}'.")
            sys.exit(1)
    else:
        selected_pairs = all_pairs

    # 3. Lọc theo --per-class hoặc --limit
    if args.per_class > 0:
        filtered = []
        counts_by_class: Dict[str, int] = {}
        for p in selected_pairs:
            cls = str(p.get("class_name", ""))
            c = counts_by_class.get(cls, 0)
            if c < args.per_class:
                filtered.append(p)
                counts_by_class[cls] = c + 1
        selected_pairs = filtered

    if args.limit > 0:
        selected_pairs = selected_pairs[: args.limit]

    total_pairs = len(selected_pairs)
    bar_width = 110
    if not args.quiet:
        print("=" * bar_width)
        print(f" ĐANG CHẠY ĐÁNH GIÁ ĐỊNH LƯỢNG TRÊN {total_pairs} CẶP ẢNH...")
        print(f" Cấu hình: Detector={args.detector} | max_features={args.max_features} | Ngưỡng IoU={args.iou_thresh}")
        print("=" * bar_width)

    # 4. Duyệt qua từng cặp và tính metrics
    results: List[Dict[str, object]] = []
    for idx, pair in enumerate(selected_pairs, start=1):
        pair_id = pair.get("pair_id", f"pair_{idx}")
        cls = pair.get("class_name", "Unknown")

        row = evaluate_single_pair(
            pair,
            detector=args.detector,
            max_features=args.max_features,
            iou_thresholds=args.iou_thresh,
            force_synthetic=args.synthetic,
        )
        results.append(row)

        if not args.quiet:
            tp_touch = row.get("gt_hit", 0)
            n_gt = row.get("n_gt", 0)
            n_pred = row.get("n_pred", 0)
            tp_30 = row.get("tp_iou30", 0)
            mode_tag = "[Mô phỏng]" if row.get("is_synthetic") else "[Ảnh thật]"
            print(
                f"[{idx:03d}/{total_pairs:03d}] {pair_id:<26} | {cls:<16} | {mode_tag:^10} "
                f"| GT: {n_gt:>2} | Pred: {n_pred:>2} | TP(Touch): {tp_touch:>2} | TP(IoU@0.3): {tp_30:>2} "
                f"| {row.get('elapsed_ms', 0):>5.0f} ms"
            )

    elapsed_total = time.time() - start_time

    # 5. Xuất báo cáo bảng Markdown / Box
    report_text = format_report(
        results,
        detector=args.detector,
        max_features=args.max_features,
        total_time_s=elapsed_total,
        iou_thresholds=args.iou_thresh,
        style=args.format,
    )
    print("\n" + report_text)

    # 6. Xuất CSV nếu có yêu cầu
    if args.per_pair_csv:
        out_p = Path(args.per_pair_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"[✓] Đã lưu bảng chi tiết từng cặp vào: {out_p}")

    if args.output_csv:
        out_p = Path(args.output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        overall = prf_from_counts(results, iou_thresholds=args.iou_thresh)
        with out_p.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(overall.keys()))
            writer.writeheader()
            writer.writerow(overall)
        print(f"[✓] Đã lưu bảng tổng hợp metrics vào: {out_p}")


if __name__ == "__main__":
    main()
