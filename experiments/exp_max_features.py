"""
Khảo sát tham số max_features (nfeatures của ORB) trong bước căn chỉnh ảnh.

    Quét max_features từ 500 -> 10.000 với ĐÚNG quy trình của
    module_preprocess.align_images (ORB -> BFMatcher Hamming cross-check ->
    giữ 80% cặp tốt nhất -> RANSAC 3 px -> các kiểm tra độ tin cậy).

    Vì ảnh lỗi PKU-Market-PCB vốn trùng khít ảnh template (lỗi được vẽ lên
    chính ảnh template), script nắn lệch ảnh test bằng một homography NGẪU NHIÊN
    ĐÃ BIẾT (mặc định ±3°, ±40 px, ±3%) rồi yêu cầu ORB nắn ngược lại. Nhờ đó
    đo được sai số thật của phép căn chỉnh chứ không chỉ sai số nội bộ RANSAC.
    Cùng một cặp ảnh luôn bị nắn lệch giống hệt nhau ở mọi giá trị max_features.

   Script đo các chỉ số định lượng (cho mỗi giá trị max_features):
    - kp_template, kp_test : số keypoint ORB thực sự trích được
    - matches              : số cặp khớp sau cross-check
    - inliers, inlier_ratio: số / tỷ lệ cặp khớp được RANSAC giữ lại
    - reprojection_error   : sai số tái chiếu trung vị của inliers (px)
    - corner_error         : sai số trung bình tại 4 góc ảnh so với phép biến
                             đổi thật (px) - chỉ số chính về độ chính xác
    - success              : vượt qua mọi kiểm tra của align_images hay không
    - time_detect/match/ransac/total_ms : thời gian từng bước (ms, trung vị)
    - recall, precision, f1: chất lượng phát hiện lỗi cuối cùng so với nhãn GT
                             (bbox phát hiện "chạm" bbox GT; kèm F1@IoU0.3/0.5)
    - ncc                  : điểm kiểm chứng căn chỉnh bằng điểm ảnh (P0.1). Khảo
                             sát chạy với kiểm tra NCC TẮT để đo chất lượng ORB
                             thô; cột ncc_reject cho biết cặp nào sẽ bị từ chối.

   Tiêu chí đề xuất (P0.2): giá trị NHỎ NHẤT có tỷ lệ thành công cao nhất,
   corner error p95 <= 1 px và F1 không kém mức tốt nhất quá 0.01. Dùng p95
   thay cho max vì vài cặp ngoại lai (lệch tới 47 px ở MỌI mức) làm max vô nghĩa;
   max vẫn được in ra như cảnh báo.

Cách chạy:
    py -3.14 experiments/exp_max_features.py --limit 30
    py -3.14 experiments/exp_max_features.py --values 500 1000 2000 5000 10000
    py -3.14 experiments/exp_max_features.py --per-class 5 --repeats 3

Chạy full 693 cặp (~40 phút với --repeats 1):
    py -3.14 -u experiments/exp_max_features.py `
        --values 500 1000 2000 3000 5000 10000 --repeats 1 --output-dir experiments

Đầu ra:
    experiments/max_features_results.csv
    experiments/charts/max_features_study.png
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Cho phép chạy trực tiếp từ thư mục experiments/: thêm project root vào sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_utils import find_pairs, load_image  # noqa: E402
from module_detection import extract_bounding_boxes  # noqa: E402
from evaluation import detection_counts as count_detections, prf_from_counts  # noqa: E402
from module_preprocess import (  # noqa: E402
    DEFAULT_MAX_FEATURES,
    DEFAULT_MIN_NCC,
    AlignmentError,
    align_images_with_stats,
    detect_difference,
)

import matplotlib

# Notebook (ipykernel) cần backend inline để hiển thị ảnh; CLI dùng Agg để lưu file.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_VALUES = [500, 1000, 2000, 3000, 5000, 7500, 10000]
CURRENT_VALUE = DEFAULT_MAX_FEATURES  # giá trị đang dùng trong module_preprocess.align_images


# 1. Căn chỉnh ORB có đo đạc — dùng chung module_preprocess
def align_orb_measured(
    gray_template: np.ndarray,
    gray_test: np.ndarray,
    max_features: int,
    keep_percentage: float = 0.8,
    test_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Gọi `align_images_with_stats` của module chính và đổi tên khóa thời gian.

    Không còn bản sao logic: mọi thay đổi trong `align_images` tự động áp dụng
    cho thí nghiệm. `time_ransac_ms` tương ứng `time_homography_ms`.
    """
    try:
        _, stats = align_images_with_stats(
            gray_template,
            gray_test,
            max_features=max_features,
            keep_percentage=keep_percentage,
            detector="ORB",
            min_ncc=None,          # khảo sát chất lượng ORB thô (P0.2)
            test_mask=test_mask,
        )
    except AlignmentError as exc:
        stats = exc.stats

    return {
        "kp_template": stats.get("kp_template", 0),
        "kp_test": stats.get("kp_test", 0),
        "matches": stats.get("matches", 0),
        "inliers": stats.get("inliers", 0),
        "inlier_ratio": stats.get("inlier_ratio", 0.0),
        "reprojection_error": stats.get("reprojection_error", float("nan")),
        "time_detect_ms": stats.get("time_detect_ms", 0.0),
        "time_match_ms": stats.get("time_match_ms", 0.0),
        "time_ransac_ms": stats.get("time_homography_ms", 0.0),
        "time_total_ms": stats.get("time_total_ms", 0.0),
        "success": bool(stats.get("success")),
        "ncc": float(stats.get("ncc", float("nan"))),
        "H": stats.get("H"),
        "error": stats.get("error", ""),
    }


# 2. Tiện ích: nắn lệch giả lập, sai số góc, so khớp bbox
def random_homography(
    width: int, height: int, rng: np.random.Generator,
    max_angle: float, max_shift: float, max_scale: float,
) -> np.ndarray:
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


def read_labels(label_path: Optional[Path]) -> List[Tuple[int, int, int, int]]:
    if not label_path or not Path(label_path).is_file():
        return []
    boxes = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 4:
            boxes.append(tuple(int(float(v)) for v in parts[:4]))
    return boxes


def _touch(a, b, tol: int = 5) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + tol < bx or bx + bw + tol < ax
                or ay + ah + tol < by or by + bh + tol < ay)


def match_boxes(pred, gt) -> Tuple[int, int]:
    """Trả về (số GT được phát hiện, số bbox dự đoán đúng)."""
    gt_hit = sum(1 for g in gt if any(_touch(p, g) for p in pred))
    pred_hit = sum(1 for p in pred if any(_touch(p, g) for g in gt))
    return gt_hit, pred_hit


def detection_counts(
    template: np.ndarray, img_test: np.ndarray, H: Optional[np.ndarray],
    gt_boxes: List[Tuple[int, int, int, int]],
) -> Dict[str, int]:
    """Chạy phần sau của pipeline (trừ vi sai + bbox) trên ẢNH GỐC.

    `img_test` là ảnh test chưa bị nắn lệch/nội suy; `H` là ánh xạ
    img_test -> template. Nhờ vậy recall/F1 đo đúng chất lượng phát hiện,
    không bị phép warp giả lập làm mờ lỗi siêu nhỏ.
    """
    h, w = template.shape[:2]
    if H is None:
        # Như main.py: căn chỉnh thất bại -> dùng ảnh chưa căn chỉnh
        aligned = img_test
        valid = None
    else:
        aligned = cv2.warpPerspective(img_test, H, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Bỏ vùng đen do phép căn chỉnh đưa vào (không có trong dữ liệu thật)
        valid = cv2.warpPerspective(np.full(img_test.shape[:2], 255, np.uint8), H,
                                    (w, h), flags=cv2.INTER_NEAREST)
        valid = cv2.erode(valid, np.ones((31, 31), np.uint8))
    diff = detect_difference(template, aligned)
    if valid is not None:
        diff = cv2.bitwise_and(diff, valid)
    boxes = extract_bounding_boxes(diff, min_area=20)
    return count_detections(boxes, gt_boxes)


# 3. Đánh giá 1 cặp ảnh với mọi giá trị max_features
def evaluate_pair(pair: Dict[str, object], values: List[int],
                  args: argparse.Namespace) -> List[Dict[str, object]]:
    template = load_image(Path(pair["template_path"]))
    test = load_image(Path(pair["test_path"]))
    h, w = test.shape[:2]

    seed = zlib.crc32(f"{pair['pair_id']}:{args.seed}".encode())
    H_true = random_homography(w, h, np.random.default_rng(seed),
                               args.warp_angle, args.warp_shift, args.warp_scale)
    warped = cv2.warpPerspective(test, H_true, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    H_true_inv = np.linalg.inv(H_true)

    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H_true, (w, h),
                                      flags=cv2.INTER_NEAREST)
    gt_boxes = read_labels(pair.get("label_path"))

    rows = []
    for value in values:
        runs = [align_orb_measured(gray_template, gray_warped, value, test_mask=warped_mask)
                for _ in range(max(1, args.repeats))]
        res = runs[0]
        for key in ("time_detect_ms", "time_match_ms", "time_ransac_ms", "time_total_ms"):
            res[key] = statistics.median(float(r[key]) for r in runs)

        row: Dict[str, object] = {
            "pair": pair["pair_id"], "class_name": pair.get("class_name", ""),
            "max_features": value,
            **{k: v for k, v in res.items() if k != "H"},
            "corner_error": (corner_error(res["H"], H_true_inv, w, h)
                             if res["success"] else float("nan")),
            "ncc_reject": bool(res["success"] and not (res["ncc"] >= DEFAULT_MIN_NCC)),
        }
        if not args.no_detection:
            # Đo phát hiện trên ẢNH GỐC, không dùng ảnh đã bị nội suy warp.
            # H ước lượng trên ảnh warp (warped -> template); ghép với H_true
            # (test -> warped) để được ánh xạ test -> template: H_detect.
            H_detect = (res["H"] @ H_true) if res["success"] else None
            row.update(detection_counts(template, test, H_detect, gt_boxes))
        rows.append(row)
    return rows


# 4. Tổng hợp, in bảng, vẽ biểu đồ
def _finite(values) -> List[float]:
    return [float(v) for v in values if v is not None and not math.isnan(float(v))]


def _mean(values) -> float:
    vals = _finite(values)
    return statistics.fmean(vals) if vals else float("nan")


def _median(values) -> float:
    vals = _finite(values)
    return statistics.median(vals) if vals else float("nan")


def _pct(values, q: float) -> float:
    vals = _finite(values)
    return float(np.percentile(vals, q)) if vals else float("nan")


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[int(row["max_features"])].append(row)

    summary = []
    for value in sorted(groups):
        items = groups[value]
        ok = [r for r in items if r["success"]]
        entry: Dict[str, object] = {
            "max_features": value,
            "pairs": len(items),
            "success_rate": len(ok) / len(items),
            "kp_test": _mean(r["kp_test"] for r in items),
            "matches": _mean(r["matches"] for r in items),
            "inliers": _mean(r["inliers"] for r in ok),
            "inlier_ratio": _mean(r["inlier_ratio"] for r in ok),
            "reprojection_error": _median(r["reprojection_error"] for r in ok),
            "corner_error_median": _median(r["corner_error"] for r in ok),
            "corner_error_p25": _pct((r["corner_error"] for r in ok), 25),
            "corner_error_p75": _pct((r["corner_error"] for r in ok), 75),
            "corner_error_p95": _pct((r["corner_error"] for r in ok), 95),
            "corner_error_max": max(_finite(r["corner_error"] for r in ok), default=float("nan")),
            "pairs_over_1px": sum(1 for r in ok if float(r["corner_error"]) > 1.0),
            "pairs_over_5px": sum(1 for r in ok if float(r["corner_error"]) > 5.0),
            "ncc_median": _median(r.get("ncc") for r in ok),
            "ncc_rejects": sum(1 for r in ok if r.get("ncc_reject")),
            "time_detect_ms": _mean(r["time_detect_ms"] for r in items),
            "time_match_ms": _mean(r["time_match_ms"] for r in items),
            "time_ransac_ms": _mean(r["time_ransac_ms"] for r in items),
            "time_total_ms": _mean(r["time_total_ms"] for r in items),
            "time_total_median_ms": _median(r["time_total_ms"] for r in items),
            "time_total_p95_ms": _pct((r["time_total_ms"] for r in items), 95),
            # Điều kiện đo (P2.2)
            "cond_detector": "ORB",
            "cond_synthetic_warp": "on",
            "cond_ncc_check": "off",
            "cond_match_rule": "touch_tol5 + IoU",
        }
        if "n_gt" in items[0]:
            entry.update(prf_from_counts(items))
        summary.append(entry)
    return summary


def recommend(summary: List[Dict[str, object]], max_corner_px: float,
              f1_tolerance: float = 0.01) -> Optional[int]:
    """Giá trị max_features NHỎ NHẤT "đủ tốt", tức là đồng thời:
      - tỷ lệ căn chỉnh thành công bằng mức cao nhất trong khảo sát,
      - sai số góc p95 <= max_corner_px (P0.2). Không dùng max: vài cặp ngoại
        lai lệch tới ~47 px ở MỌI mức max_features (lỗi kiểm tra homography,
        xử lý bằng NCC ở P0.1), nên max không phân biệt được các mức. Bước trừ
        vi sai đã chừa dung sai ±2 px nên p95 < 1 px là đủ,
      - F1 phát hiện lỗi không kém mức tốt nhất quá f1_tolerance.
    Chênh lệch dưới 1 px giữa các giá trị là nhiễu ngẫu nhiên, không nên dùng
    để chọn tham số.
    """
    if not summary:
        return None
    best_success = max(float(s["success_rate"]) for s in summary)
    f1_values = [float(s.get("f1", float("nan"))) for s in summary]
    best_f1 = max((v for v in f1_values if not math.isnan(v)), default=float("nan"))
    for s in summary:  # đã sắp xếp tăng dần
        if float(s["success_rate"]) < best_success - 1e-9:
            continue
        p95 = float(s["corner_error_p95"])
        if math.isnan(p95) or p95 > max_corner_px:
            continue
        f1 = float(s.get("f1", float("nan")))
        if not math.isnan(best_f1) and (math.isnan(f1) or f1 < best_f1 - f1_tolerance):
            continue
        return int(s["max_features"])
    return None


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})


def print_summary(summary: List[Dict[str, object]], recommended: Optional[int]) -> None:
    def f(v, fmt):
        return format(v, fmt) if not (isinstance(v, float) and math.isnan(v)) else "-".rjust(int(fmt.split(".")[0]))

    header = (f"{'max_feat':>8} |{'OK %':>6} |{'kp':>6} |{'inliers':>8} |{'reproj':>7} |"
              f"{'corner':>7} |{'c.p95':>7} |{'c.max':>7} |{'>1px':>5} |{'NCC tc':>6} |"
              f"{'ms med':>7} |{'ms p95':>7} |{'F1':>6} |{'F1@.5':>6}")
    print("\n" + header)
    print("-" * len(header))
    for s in summary:
        mark = ""
        if s["max_features"] == CURRENT_VALUE:
            mark += "  <- đang dùng"
        if s["max_features"] == recommended:
            mark += "  <- đề xuất"
        print(f"{s['max_features']:>8} |{f(s['success_rate'] * 100, '6.1f')} |"
              f"{f(s['kp_test'], '6.0f')} |{f(s['inliers'], '8.0f')} |"
              f"{f(s['reprojection_error'], '7.3f')} |"
              f"{f(s['corner_error_median'], '7.3f')} |{f(s['corner_error_p95'], '7.3f')} |"
              f"{f(s['corner_error_max'], '7.2f')} |{s['pairs_over_1px']:>5} |{s['ncc_rejects']:>6} |"
              f"{f(s['time_total_median_ms'], '7.0f')} |{f(s['time_total_p95_ms'], '7.0f')} |"
              f"{f(s.get('f1', float('nan')), '6.3f')} |{f(s.get('f1_iou50', float('nan')), '6.3f')}{mark}")
    print("-" * len(header))
    print("corner / c.p95 / c.max: sai số góc trung vị / p95 / xấu nhất (px) trên cặp căn thành công;\n"
          ">1px: số cặp lệch quá 1 px; NCC tc: số cặp sẽ bị kiểm tra NCC (P0.1) từ chối;\n"
          "ms med / ms p95: thời gian căn chỉnh trung vị / p95; F1@.5: F1 ghép 1-1 theo IoU >= 0.5.")
    worst = max((float(s["corner_error_max"]) for s in summary
                 if not math.isnan(float(s["corner_error_max"]))), default=float("nan"))
    if worst > 5:
        n5 = min(int(s["pairs_over_5px"]) for s in summary)
        print(f"[CẢNH BÁO] Có cặp lệch tới {worst:.1f} px; ít nhất {n5} cặp lệch > 5 px ở MỌI mức "
              "-> lỗi kiểm tra homography, không phải thiếu feature (xem P0.1 / exp_alignment_check.py).")


def plot_study(summary: List[Dict[str, object]], output_path: Path,
               recommended: Optional[int], n_pairs: int) -> None:
    x = [int(s["max_features"]) for s in summary]

    def col(key):
        return [float(s.get(key, float("nan"))) for s in summary]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # (a) thời gian, xếp chồng theo bước
    ax = axes[0, 0]
    det, mat, ran = col("time_detect_ms"), col("time_match_ms"), col("time_ransac_ms")
    ax.stackplot(x, det, mat, ran, labels=["Dò ORB", "Ghép cặp", "RANSAC"],
                 colors=["#4C72B0", "#DD8452", "#55A868"], alpha=0.85)
    ax.set_title("Thời gian căn chỉnh (ms)")
    ax.legend(loc="upper left", fontsize=8)

    # (b) số keypoint / matches / inliers
    ax = axes[0, 1]
    ax.plot(x, col("kp_test"), "o-", label="Keypoint (ảnh test)")
    ax.plot(x, col("matches"), "s-", label="Cặp khớp (cross-check)")
    ax.plot(x, col("inliers"), "^-", label="Inliers RANSAC")
    ax.set_title("Số điểm đặc trưng")
    ax.legend(fontsize=8)

    # (c) tỷ lệ inliers
    ax = axes[0, 2]
    ax.plot(x, [v * 100 for v in col("inlier_ratio")], "o-", color="#8172B2")
    ax.set_title("Tỷ lệ inliers (%)")

    # (d) sai số so với biến đổi thật
    ax = axes[1, 0]
    med, p25, p75 = col("corner_error_median"), col("corner_error_p25"), col("corner_error_p75")
    ax.fill_between(x, p25, p75, color="#C44E52", alpha=0.2, label="Khoảng tứ phân vị")
    ax.plot(x, med, "o-", color="#C44E52", label="Trung vị")
    ax.set_title("Sai số so với biến đổi thật (px) — thấp hơn tốt hơn")
    ax.legend(fontsize=8)

    # (e) tỷ lệ thành công
    ax = axes[1, 1]
    ax.plot(x, [v * 100 for v in col("success_rate")], "o-", color="#937860")
    ax.set_ylim(0, 105)
    ax.set_title("Tỷ lệ căn chỉnh thành công (%)")

    # (f) F1 phát hiện lỗi (nếu có) hoặc sai số tái chiếu
    ax = axes[1, 2]
    if not all(math.isnan(v) for v in col("f1")):
        ax.plot(x, col("recall"), "o-", label="Recall")
        ax.plot(x, col("precision"), "s-", label="Precision")
        ax.plot(x, col("f1"), "^-", label="F1", linewidth=2)
        ax.set_ylim(0, 1.05)
        ax.set_title("Phát hiện lỗi so với nhãn GT")
        ax.legend(fontsize=8)
    else:
        ax.plot(x, col("reprojection_error"), "o-")
        ax.set_title("Sai số tái chiếu RANSAC (px)")

    for ax in axes.ravel():
        ax.set_xlabel("max_features")
        ax.grid(linestyle="--", alpha=0.35)
        ax.axvline(CURRENT_VALUE, color="black", linestyle="--", linewidth=1)
        if recommended and recommended != CURRENT_VALUE:
            ax.axvline(recommended, color="green", linestyle=":", linewidth=1.5)

    note = f"đường đứt đen = {CURRENT_VALUE} (đang dùng)"
    if recommended and recommended != CURRENT_VALUE:
        note += f", chấm xanh = {recommended} (đề xuất)"
    fig.suptitle(f"Khảo sát max_features của ORB trên {n_pairs} cặp ảnh PCB ({note})", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# 5. Entry point
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Khảo sát max_features của ORB trên các cặp ảnh PCB.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Thư mục chứa pairs_manifest.csv (mặc định: data).")
    parser.add_argument("--values", type=int, nargs="+", default=DEFAULT_VALUES,
                        help="Các giá trị max_features cần thử (mặc định: 500 ... 10000).")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ chạy N cặp đầu tiên (0 = tất cả).")
    parser.add_argument("--per-class", type=int, default=0,
                        help="Lấy N cặp mỗi lớp lỗi (0 = tất cả). Gợi ý: 5.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Số lần lặp đo thời gian, lấy trung vị (mặc định: 3).")
    parser.add_argument("--warp-angle", type=float, default=3.0, help="Góc xoay giả lập tối đa (độ).")
    parser.add_argument("--warp-shift", type=float, default=40.0, help="Tịnh tiến giả lập tối đa (px).")
    parser.add_argument("--warp-scale", type=float, default=0.03, help="Co giãn giả lập tối đa (0.03 = ±3%%).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-corner-px", "--max-corner-p95", dest="max_corner_px",
                        type=float, default=1.0,
                        help="Sai số góc xấu nhất chấp nhận được khi đề xuất (px, mặc định 1.0).")
    parser.add_argument("--no-detection", action="store_true",
                        help="Bỏ phần đo recall/precision (chạy nhanh hơn).")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"),
                        help="Thư mục lưu CSV và biểu đồ (mặc định: experiments).")
    return parser.parse_args(argv)


def main(argv=None) -> List[Dict[str, object]]:
    args = parse_args(argv)
    values = sorted(set(args.values))
    if any(v <= 0 for v in values):
        raise SystemExit("[LỖI] max_features phải > 0.")

    pairs = find_pairs(str(args.data_dir))
    if not pairs:
        raise SystemExit(
            f"\n[LỖI] Không tìm thấy cặp ảnh nào trong '{args.data_dir}'.\n"
            "       Hãy chạy experiments/build_pairs.py để sinh pairs_manifest.csv trước khi chạy.\n"
        )
    if args.per_class > 0:
        counter: Dict[str, int] = defaultdict(int)
        picked = []
        for pair in pairs:
            if counter[pair["class_name"]] < args.per_class:
                picked.append(pair)
                counter[pair["class_name"]] += 1
        pairs = picked
    if args.limit > 0:
        pairs = pairs[: args.limit]

    print(f"Tìm thấy {len(pairs)} cặp ảnh trong '{args.data_dir}'.")
    print(f"Giá trị max_features: {values}")
    print(f"Nắn lệch giả lập: ±{args.warp_angle}°, ±{args.warp_shift}px, ±{args.warp_scale * 100:.0f}%")

    rows: List[Dict[str, object]] = []
    done = 0
    for index, pair in enumerate(pairs, start=1):
        try:
            pair_rows = evaluate_pair(pair, values, args)
        except (ValueError, cv2.error) as exc:
            print(f"[{index}/{len(pairs)}] Bỏ qua {pair['pair_id']}: {exc}")
            continue
        rows.extend(pair_rows)
        done += 1
        brief = " ".join(f"{r['max_features']}:{'ok' if r['success'] else 'x'}" for r in pair_rows)
        print(f"[{index}/{len(pairs)}] {pair['pair_id']:<26} {brief}")

    if not rows:
        raise SystemExit("Không có ảnh nào được xử lý thành công.")

    summary = summarize(rows)
    recommended = recommend(summary, args.max_corner_px)
    print_summary(summary, recommended)

    output_dir: Path = args.output_dir
    csv_path = output_dir / "max_features_results.csv"
    chart_path = output_dir / "charts" / "max_features_study.png"
    write_csv(rows, csv_path)
    write_csv(summary, output_dir / "max_features_summary.csv")
    plot_study(summary, chart_path, recommended, done)

    if recommended is not None:
        rec = next(s for s in summary if s["max_features"] == recommended)
        cur = next((s for s in summary if s["max_features"] == CURRENT_VALUE), None)
        print(f"\nĐề xuất: max_features = {recommended} — giá trị nhỏ nhất vừa căn chỉnh "
              f"thành công ở mức cao nhất, vừa có sai số góc p95 "
              f"{float(rec['corner_error_p95']):.3f} px (<= {args.max_corner_px} px), "
              f"vừa giữ F1 ngang mức tốt nhất (thời gian trung vị "
              f"~{float(rec['time_total_median_ms']):.0f} ms, p95 ~{float(rec['time_total_p95_ms']):.0f} ms).")
        if cur is not None and recommended != CURRENT_VALUE:
            print(f"Giá trị đang dùng {CURRENT_VALUE}: thời gian ~{float(cur['time_total_ms']):.0f} ms, "
                  f"sai số góc {float(cur['corner_error_median']):.3f} px.")
    else:
        print(f"\nKhông giá trị nào đạt đủ tiêu chí (sai số góc p95 <= {args.max_corner_px} px). "
              "Xem biểu đồ để chọn thủ công.")
    print(f"\nCSV chỉ số     : {csv_path}")
    print(f"CSV tổng hợp   : {output_dir / 'max_features_summary.csv'}")
    print(f"Biểu đồ        : {chart_path}")
    print("\nGợi ý: dùng biểu đồ để điền Mục 2.2 của PROGRESS_REPORT.md (max_features_study.png).")
    return summary


if __name__ == "__main__":
    main()
