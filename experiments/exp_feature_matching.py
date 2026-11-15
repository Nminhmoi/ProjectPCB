"""
So sánh 3 bộ đặc trưng căn chỉnh: ORB vs SIFT vs AKAZE.

Script dùng CHUNG `module_preprocess.py` + `module_detection.py` của project
chính; mỗi detector chỉ khác tham số `detector=` truyền vào
`align_images_with_stats`, nên mọi bước còn lại (ghép cặp, RANSAC, kiểm tra
homography, trừ vi sai) giống hệt nhau.

Chạy trên CÙNG một tập cặp ảnh và đo:

  Căn chỉnh (alignment)
    - kp_test, matches, inliers, inlier_ratio
    - reprojection_error : sai số tái chiếu trung vị của inliers (px)
    - corner_error       : sai số so với phép biến đổi THẬT (px). Ảnh test được
                           nắn lệch bằng một homography ngẫu nhiên đã biết
                           (xoay/tịnh tiến/co giãn), detector phải nắn ngược lại.
                           Ảnh lỗi PKU-Market-PCB vốn trùng khít template nên
                           đây là "đáp án đúng" để chấm độ chính xác.
    - time_detect_ms, time_match_ms, time_total_ms, success_rate

  Phát hiện lỗi (so với nhãn Ground Truth trong labels/)
    - recall    : tỷ lệ bbox GT có ít nhất 1 bbox phát hiện chạm vào
    - precision : tỷ lệ bbox phát hiện chạm vào ít nhất 1 bbox GT
    - f1
    (Dùng tiêu chí "chạm nhau" thay vì IoU vì pipeline khoanh lõi vi sai nên
     bbox nhỏ hơn nhãn GT nhiều.)

Cách chạy (từ thư mục gốc project):
    py -3.14 experiments/exp_feature_matching.py --per-class 5
    py -3.14 experiments/exp_feature_matching.py --limit 20 --visualize 2
    py -3.14 experiments/exp_feature_matching.py --rotation
    py -3.14 experiments/exp_feature_matching.py --detectors ORB SIFT

Chạy full 693 cặp:
    py -3.14 -u experiments/exp_feature_matching.py --output-dir experiments

AKAZE cần opencv-contrib-python (OpenCV 5.x đặt AKAZE ở cv2.xfeatures2d).
Nếu thiếu, script tự bỏ qua detector đó và in cảnh báo; xem requirements.txt.

Đầu ra (mặc định experiments/results/detector_comparison/):
    per_pair.csv, summary.csv, comparison_chart.png,
    vis/<pair>_<DETECTOR>_matches.jpg, vis/<pair>_<DETECTOR>_checkerboard.jpg
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataset_utils import find_pairs, load_image, load_manifest  # noqa: E402
from module_detection import extract_bounding_boxes  # noqa: E402
from module_preprocess import (  # noqa: E402
    DETECTORS,
    DEFAULT_MAX_FEATURES,
    AlignmentError,
    align_images_with_stats,
    detect_difference,
    is_detector_available,
)

COLORS = {"ORB": "#4C72B0", "SIFT": "#DD8452", "AKAZE": "#55A868"}
SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# 1. Kiểm tra detector khả dụng
# ---------------------------------------------------------------------
def select_detectors(names: List[str]) -> List[str]:
    """Lọc các detector chạy được với OpenCV hiện tại (AKAZE cần contrib)."""
    selected = []
    for name in names:
        name = name.upper()
        if name not in DETECTORS:
            raise SystemExit(f"[LỖI] Detector không hợp lệ: {name}. Chọn trong {DETECTORS}.")
        if not is_detector_available(name):
            print(f"[CẢNH BÁO] Bỏ qua {name}: OpenCV hiện tại không có detector này "
                  f"(AKAZE cần: pip install opencv-contrib-python).")
            continue
        selected.append(name)
    if not selected:
        raise SystemExit("[LỖI] Không có detector nào chạy được.")
    return selected


# ---------------------------------------------------------------------
# 2. Tiện ích hình học / nhãn
# ---------------------------------------------------------------------
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
            x, y, w, h = (int(float(v)) for v in parts[:4])
            boxes.append((x, y, w, h))
    return boxes


def _touch(a, b, tol: int) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + tol < bx or bx + bw + tol < ax
        or ay + ah + tol < by or by + bh + tol < ay
    )


def match_boxes(pred, gt, tol: int = 5) -> Tuple[int, int]:
    """Trả về (số GT được phát hiện, số bbox dự đoán đúng)."""
    gt_hit = sum(1 for g in gt if any(_touch(p, g, tol) for p in pred))
    pred_hit = sum(1 for p in pred if any(_touch(p, g, tol) for g in gt))
    return gt_hit, pred_hit


# ---------------------------------------------------------------------
# 3. Trực quan hóa: đường nối cặp đặc trưng + ảnh bàn cờ
# ---------------------------------------------------------------------
def _fit_width(image: np.ndarray, max_width: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_width / float(w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image, scale


def draw_inlier_matches(
    img_test: np.ndarray, img_template: np.ndarray, stats: dict,
    max_lines: int = 150, max_width: int = 1100,
) -> np.ndarray:
    left, s1 = _fit_width(img_test, max_width)
    right, s2 = _fit_width(img_template, max_width)
    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, left.shape[1] + right.shape[1], 3), np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1]:] = right

    pts_test = stats.get("inlier_pts_test")
    pts_tmpl = stats.get("inlier_pts_template")
    if pts_test is not None and len(pts_test):
        idx = np.linspace(0, len(pts_test) - 1, min(max_lines, len(pts_test))).astype(int)
        rng = np.random.default_rng(0)
        for i in idx:
            color = tuple(int(c) for c in rng.integers(60, 256, size=3))
            p1 = (int(pts_test[i][0] * s1), int(pts_test[i][1] * s1))
            p2 = (int(pts_tmpl[i][0] * s2) + left.shape[1], int(pts_tmpl[i][1] * s2))
            cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, color, -1)
            cv2.circle(canvas, p2, 3, color, -1)

    text = (f"{stats['detector']}: {stats['inliers']} inliers "
            f"({stats['inlier_ratio'] * 100:.1f}%), reproj "
            f"{stats['reprojection_error']:.2f}px, {stats['time_total_ms']:.0f}ms")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def checkerboard(img_template: np.ndarray, img_aligned: np.ndarray, tiles: int = 8) -> np.ndarray:
    """Ghép xen kẽ ô cờ: đường mạch liền mạch qua ranh giới ô = căn chuẩn."""
    h, w = img_template.shape[:2]
    ys = (np.arange(h) * tiles // h)[:, None]
    xs = (np.arange(w) * tiles // w)[None, :]
    mask = ((ys + xs) % 2 == 0)
    out = img_aligned.copy()
    out[mask] = img_template[mask]
    step_x, step_y = w / tiles, h / tiles
    for i in range(1, tiles):
        cv2.line(out, (int(i * step_x), 0), (int(i * step_x), h), (0, 0, 255), 1)
        cv2.line(out, (0, int(i * step_y)), (w, int(i * step_y)), (0, 0, 255), 1)
    return out


# ---------------------------------------------------------------------
# 4. Hàm so sánh chính
# ---------------------------------------------------------------------
def evaluate_pair(
    pair: dict, detectors: List[str],
    args: argparse.Namespace, vis_dir: Optional[Path],
) -> List[Dict[str, object]]:
    """Chạy cả 3 detector trên 1 cặp ảnh, trả về 1 dòng kết quả / detector."""
    template = load_image(pair["template_path"])
    test = load_image(pair["test_path"])
    test_original = test
    kind = pair.get("kind") or "original"
    gt_boxes = read_labels(pair.get("label_path")) if kind == "original" else []

    H_true = None
    H_true_inv = None
    if kind == "original" and args.synthetic_warp:
        seed = zlib.crc32(f"{pair['pair_id']}:{args.seed}".encode())
        rng = np.random.default_rng(seed)
        h, w = test.shape[:2]
        H_true = random_homography(w, h, rng, args.warp_angle, args.warp_shift, args.warp_scale)
        test = cv2.warpPerspective(test, H_true, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        H_true_inv = np.linalg.inv(H_true)

    rows = []
    for name in detectors:
        row: Dict[str, object] = {"pair_id": pair["pair_id"], "kind": kind,
                                  "class_name": pair.get("class_name", ""), "detector": name}
        t0 = time.perf_counter()
        try:
            aligned, stats = align_images_with_stats(
                template, test, max_features=args.max_features, detector=name
            )
        except AlignmentError as exc:
            aligned, stats = test, exc.stats

        for key in ("success", "kp_template", "kp_test", "matches", "good_matches",
                    "inliers", "inlier_ratio", "reprojection_error",
                    "time_detect_ms", "time_match_ms", "time_homography_ms",
                    "time_warp_ms", "time_total_ms", "error"):
            row[key] = stats.get(key)
        if not stats["success"]:
            # Thời gian tổng khi thất bại = thời gian đã chạy tới lúc dừng
            row["time_total_ms"] = (stats.get("time_detect_ms", 0) + stats.get("time_match_ms", 0)
                                    + stats.get("time_homography_ms", 0))

        row["corner_error"] = float("nan")
        if stats["success"] and H_true_inv is not None:
            h, w = test.shape[:2]
            row["corner_error"] = corner_error(stats["H"], H_true_inv, w, h)

        # Phát hiện lỗi (chỉ với cặp có nhãn)
        if kind == "original":
            # Đo trên ẢNH GỐC, không dùng ảnh đã bị nội suy warp.
            # H ước lượng trên ảnh warp (warped -> template); nếu có warp giả lập
            # thì ghép H_true (test -> warped) để được ánh xạ test -> template.
            if stats["success"]:
                H_detect = stats["H"] if H_true is None else stats["H"] @ H_true
                aligned_detect = cv2.warpPerspective(
                    test_original, H_detect,
                    (template.shape[1], template.shape[0]),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                valid = cv2.warpPerspective(
                    np.full(test_original.shape[:2], 255, np.uint8), H_detect,
                    (template.shape[1], template.shape[0]), flags=cv2.INTER_NEAREST)
                valid = cv2.erode(valid, np.ones((31, 31), np.uint8))
                diff = detect_difference(template, aligned_detect)
                diff = cv2.bitwise_and(diff, valid)
            else:
                aligned_detect = test_original
                if aligned_detect.shape[:2] != template.shape[:2]:
                    aligned_detect = cv2.resize(
                        aligned_detect, (template.shape[1], template.shape[0]))
                diff = detect_difference(template, aligned_detect)
            boxes = extract_bounding_boxes(diff, min_area=args.min_area)
            gt_hit, pred_hit = match_boxes(boxes, gt_boxes)
            row.update({"n_gt": len(gt_boxes), "n_pred": len(boxes),
                        "gt_hit": gt_hit, "pred_hit": pred_hit})
        row["time_pipeline_ms"] = (time.perf_counter() - t0) * 1000.0

        if vis_dir is not None and stats["success"]:
            vis_dir.mkdir(parents=True, exist_ok=True)
            base = vis_dir / f"{pair['pair_id']}_{name}"
            cv2.imwrite(f"{base}_matches.jpg", draw_inlier_matches(test, template, stats))
            board, _ = _fit_width(checkerboard(template, aligned), 1600)
            cv2.imwrite(f"{base}_checkerboard.jpg", board)
        rows.append(row)
    return rows


def _mean(values) -> float:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.fmean(vals) if vals else float("nan")


def _median(values) -> float:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.median(vals) if vals else float("nan")


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["kind"]), str(row["detector"]))].append(row)

    summary = []
    for kind in sorted({k for k, _ in groups}):
        for name in DETECTORS:
            items = groups.get((kind, name))
            if not items:
                continue
            ok = [r for r in items if r["success"]]
            entry = {
                "kind": kind,
                "detector": name,
                "pairs": len(items),
                "success_rate": len(ok) / len(items),
                "kp_test": _mean(r["kp_test"] for r in items),
                "matches": _mean(r["matches"] for r in items),
                "inliers": _mean(r["inliers"] for r in ok),
                "inlier_ratio": _mean(r["inlier_ratio"] for r in ok),
                "reprojection_error": _mean(r["reprojection_error"] for r in ok),
                "corner_error_mean": _mean(r["corner_error"] for r in ok),
                "corner_error_median": _median(r["corner_error"] for r in ok),
                "time_detect_ms": _mean(r["time_detect_ms"] for r in items),
                "time_match_ms": _mean(r["time_match_ms"] for r in items),
                "time_align_ms": _mean(r["time_total_ms"] for r in items),
                "time_pipeline_ms": _mean(r["time_pipeline_ms"] for r in items),
            }
            if kind == "original":
                n_gt = sum(int(r.get("n_gt", 0)) for r in items)
                n_pred = sum(int(r.get("n_pred", 0)) for r in items)
                gt_hit = sum(int(r.get("gt_hit", 0)) for r in items)
                pred_hit = sum(int(r.get("pred_hit", 0)) for r in items)
                recall = gt_hit / n_gt if n_gt else float("nan")
                precision = pred_hit / n_pred if n_pred else float("nan")
                f1 = (2 * precision * recall / (precision + recall)
                      if precision + recall > 0 else 0.0)
                entry.update({"recall": recall, "precision": precision, "f1": f1})
            summary.append(entry)
    return summary


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


def print_summary(summary: List[Dict[str, object]], synthetic_warp: bool = True) -> None:
    for kind in sorted({s["kind"] for s in summary}):
        items = [s for s in summary if s["kind"] == kind]
        if kind == "original":
            title = ("ẢNH LỖI GỐC (có nắn lệch giả lập)" if synthetic_warp
                     else "ẢNH LỖI GỐC (không nắn lệch)")
        else:
            title = "ẢNH XOAY (pairs_rotation_manifest)"
        print(f"\n=== {title} — {items[0]['pairs']} cặp ===")
        header = (f"{'Detector':<8}|{'OK %':>7}|{'Keypts':>8}|{'Inliers':>8}|{'Inl %':>7}|"
                  f"{'Reproj':>8}|{'Corner':>8}|{'Align ms':>9}|{'Recall':>7}|{'Prec':>7}|{'F1':>6}")
        print(header)
        print("-" * len(header))
        for s in items:
            def f(v, fmt):
                if isinstance(v, (int, float)) and not math.isnan(v):
                    return format(v, fmt)
                return "-".rjust(int(fmt.split(".")[0]))
            print(f"{s['detector']:<8}|{f(s['success_rate'] * 100, '7.1f')}|"
                  f"{f(s['kp_test'], '8.0f')}|{f(s['inliers'], '8.0f')}|"
                  f"{f(s['inlier_ratio'] * 100, '7.1f')}|{f(s['reprojection_error'], '8.3f')}|"
                  f"{f(s['corner_error_mean'], '8.3f')}|{f(s['time_align_ms'], '9.0f')}|"
                  f"{f(s.get('recall', float('nan')), '7.3f')}|"
                  f"{f(s.get('precision', float('nan')), '7.3f')}|"
                  f"{f(s.get('f1', float('nan')), '6.3f')}")
        print("(Reproj/Corner tính bằng px, chỉ trên cặp căn thành công. "
              "Corner = sai số so với biến đổi thật.)")


def plot_summary(summary: List[Dict[str, object]], path: Path) -> None:
    items = [s for s in summary if s["kind"] == "original"] or summary
    names = [s["detector"] for s in items]
    colors = [COLORS.get(n, "gray") for n in names]
    panels = [
        ("time_align_ms", "Thời gian căn chỉnh (ms) — thấp hơn tốt hơn"),
        ("inliers", "Số inliers RANSAC — cao hơn tốt hơn"),
        ("inlier_ratio", "Tỷ lệ inliers — cao hơn tốt hơn"),
        ("reprojection_error", "Sai số tái chiếu (px) — thấp hơn tốt hơn"),
        ("corner_error_mean", "Sai số so với biến đổi thật (px) — thấp hơn tốt hơn"),
        ("f1", "F1 phát hiện lỗi — cao hơn tốt hơn"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (key, title) in zip(axes.ravel(), panels):
        values = [float(s.get(key, float("nan"))) for s in items]
        plot_vals = [0.0 if math.isnan(v) else v for v in values]
        bars = ax.bar(names, plot_vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, v in zip(bars, values):
            label = "-" if math.isnan(v) else (f"{v:.3f}" if v < 10 else f"{v:.0f}")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label,
                    ha="center", va="bottom", fontsize=9)
    n_pairs = items[0]["pairs"] if items else 0
    fig.suptitle(f"So sánh {' vs '.join(names)} trên {n_pairs} cặp ảnh PCB", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_detectors(args: argparse.Namespace) -> List[Dict[str, object]]:
    """Hàm so sánh các detector: chạy, in bảng, ghi CSV + biểu đồ. Trả về summary."""
    names = select_detectors(list(args.detectors))
    data_dir = args.data_dir or (PROJECT_ROOT / "data")
    pairs = find_pairs(str(data_dir))
    if not pairs:
        raise SystemExit(f"[LỖI] Không tìm thấy pairs_manifest.csv trong '{data_dir}'.")

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

    if args.rotation:
        rot_manifest = None
        for candidate in [Path(data_dir) / "pairs_rotation_manifest.csv",
                          Path(data_dir) / "PKU-Market-PCB" / "pairs_rotation_manifest.csv",
                          *Path(data_dir).rglob("pairs_rotation_manifest.csv")]:
            if candidate.is_file():
                rot_manifest = candidate
                break
        if rot_manifest is None:
            print("[CẢNH BÁO] Không thấy pairs_rotation_manifest.csv, bỏ qua --rotation.")
        else:
            wanted = {p["pair_id"] for p in pairs}
            rot_pairs = [p for p in load_manifest(rot_manifest)
                         if p["pair_id"].removesuffix("_rot") in wanted]
            pairs = pairs + rot_pairs

    print(f"Detector  : {', '.join(names)}")
    print(f"Dữ liệu    : {data_dir}  —  {len(pairs)} cặp")
    print(f"max_features = {args.max_features}; nắn lệch giả lập: "
          f"{'bật' if args.synthetic_warp else 'tắt'}"
          + (f" (±{args.warp_angle}°, ±{args.warp_shift}px, ±{args.warp_scale * 100:.0f}%)"
             if args.synthetic_warp else ""))

    out_dir: Path = args.output_dir
    rows: List[Dict[str, object]] = []
    for index, pair in enumerate(pairs, start=1):
        vis_dir = out_dir / "vis" if index <= args.visualize else None
        try:
            pair_rows = evaluate_pair(pair, names, args, vis_dir)
        except Exception as exc:  # ảnh hỏng / thiếu file
            print(f"[{index}/{len(pairs)}] {pair['pair_id']}: BỎ QUA ({exc})")
            continue
        rows.extend(pair_rows)
        brief = "  ".join(
            f"{r['detector']}:{'ok' if r['success'] else 'FAIL'} {float(r['time_total_ms']):.0f}ms"
            for r in pair_rows)
        print(f"[{index}/{len(pairs)}] {pair['pair_id']:<28} {brief}")

    if not rows:
        raise SystemExit("Không có cặp ảnh nào chạy được (kiểm tra thư mục raw/).")

    summary = summarize(rows)
    write_csv(rows, out_dir / "per_pair.csv")
    write_csv(summary, out_dir / "summary.csv")
    plot_summary(summary, out_dir / "comparison_chart.png")
    print_summary(summary, synthetic_warp=args.synthetic_warp)
    if args.synthetic_warp:
        print("\n[GHI CHÚ] Căn chỉnh đo trên ảnh ĐÃ nắn lệch (có corner_error);"
              "\n         phát hiện lỗi đo trên ẢNH GỐC đã căn chỉnh lại, nên recall/precision/F1"
              "\n         không bị phép nội suy warp làm mờ lỗi siêu nhỏ.")
    print(f"\nCSV từng cặp : {out_dir / 'per_pair.csv'}")
    print(f"CSV tổng hợp : {out_dir / 'summary.csv'}")
    print(f"Biểu đồ      : {out_dir / 'comparison_chart.png'}")
    if args.visualize:
        print(f"Trực quan    : {out_dir / 'vis'}")
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="So sánh ORB vs SIFT vs AKAZE trên PCB.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Thư mục chứa pairs_manifest.csv (mặc định: data/ của project).")
    parser.add_argument("--detectors", nargs="+", default=list(DETECTORS),
                        help=f"Chọn detector để so sánh (mặc định: {' '.join(DETECTORS)}).")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ chạy N cặp đầu (0 = tất cả).")
    parser.add_argument("--per-class", type=int, default=0,
                        help="Lấy N cặp mỗi lớp lỗi (0 = tất cả). Gợi ý chạy nhanh: 5.")
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES,
                        help=f"Số đặc trưng tối đa (mặc định: {DEFAULT_MAX_FEATURES}).")
    parser.add_argument("--min-area", type=int, default=20,
                        help="Diện tích tối thiểu của bbox phát hiện (mặc định: 20).")
    parser.add_argument("--no-synthetic-warp", dest="synthetic_warp", action="store_false",
                        help="Không nắn lệch ảnh test (khi đó không đo được corner_error).")
    parser.add_argument("--warp-angle", type=float, default=3.0, help="Góc xoay tối đa (độ).")
    parser.add_argument("--warp-shift", type=float, default=40.0, help="Tịnh tiến tối đa (px).")
    parser.add_argument("--warp-scale", type=float, default=0.03, help="Co giãn tối đa (0.03 = ±3%%).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotation", action="store_true",
                        help="Chạy thêm ảnh trong pairs_rotation_manifest.csv (chỉ đo căn chỉnh).")
    parser.add_argument("--visualize", type=int, default=1,
                        help="Lưu ảnh drawMatches + bàn cờ cho N cặp đầu (0 = tắt).")
    parser.add_argument("--output-dir", type=Path,
                        default=SCRIPT_DIR / "results" / "detector_comparison")
    return parser.parse_args(argv)


if __name__ == "__main__":
    compare_detectors(parse_args())
