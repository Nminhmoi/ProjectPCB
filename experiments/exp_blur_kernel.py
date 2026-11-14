"""
Khảo sát kích thước kernel Gaussian blur trong bước trừ vi sai (detect_difference).

    Lý giải vì sao pipeline dùng Gaussian 3x3 thay vì 5x5, 6x6 hay 7x7.
    So sánh các kernel: 1x1 (không làm mờ), 3x3, 5x5, 6x6, 7x7.

   Phần A — Lý thuyết (không cần dữ liệu):
    - sigma          : OpenCV tự tính khi sigma=0:
                       sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    - noise_left     : % độ lệch chuẩn nhiễu trắng còn lại sau khi làm mờ
                       (càng thấp càng khử nhiễu mạnh)
    - line_keep_Wpx  : % độ tương phản còn lại của một nét mảnh rộng W px
                       (vết cắn chuột, gai đồng, đứt mạch thường chỉ vài px)
    - center_shift   : độ lệch tâm (px) của ảnh sau khi lọc. Kernel CHẴN (6x6)
                       không có pixel trung tâm nên làm ảnh trôi 0.5 px, và
                       cv2.GaussianBlur từ chối kernel chẵn.

   Phần B — Thực nghiệm trên cặp ảnh PKU-Market-PCB (pipeline thật):
    Căn chỉnh bằng module_preprocess.align_images, rồi chạy lại detect_difference
    với từng kernel (mọi bước khác giữ nguyên) ở 2 điều kiện:
      - clean : ảnh gốc của dataset
      - noise : ảnh test được cộng nhiễu Gaussian (mặc định sigma = 6) mô phỏng
                nhiễu cảm biến camera. Ảnh lỗi PKU được vẽ lên chính ảnh template
                nên gần như không có nhiễu — điều kiện này mới cho thấy vai trò
                khử nhiễu của blur.
    Chỉ số:
      - recall, precision, f1 : so với nhãn GT (bbox phát hiện "chạm" bbox GT)
      - false_blobs           : số vùng lỗi giả (không chạm nhãn GT nào)
      - snr                   : tín hiệu lỗi / nhiễu nền trên bản đồ vi sai
                                (trước khi lấy ngưỡng)
      - recall theo lớp lỗi   : lớp nét mảnh (Mouse_bite, Spur, Open_circuit)
                                nhạy với blur lớn nhất
      - time_blur_ms          : thời gian làm mờ 2 ảnh

Cách chạy:
    py -3.14 experiments/exp_blur_kernel.py --limit 30
    py -3.14 experiments/exp_blur_kernel.py --per-class 5 --noise-sigma 8
    py -3.14 experiments/exp_blur_kernel.py --theory-only

Đầu ra:
    experiments/blur_kernel_results.csv
    experiments/charts/blur_comparison.png
    experiments/charts/blur_visual.png
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

# Cho phép chạy trực tiếp từ thư mục experiments/: thêm project root vào sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_utils import find_pairs, load_image  # noqa: E402
from module_detection import extract_bounding_boxes  # noqa: E402
from module_preprocess import _gray_clahe, align_images, detect_difference  # noqa: E402

import matplotlib

# Notebook (ipykernel) cần backend inline để hiển thị ảnh; CLI dùng Agg để lưu file.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_KERNELS = [1, 3, 5, 6, 7]
CURRENT_KERNEL = 3
THIN_CLASSES = ("Mouse_bite", "Spur", "Open_circuit")
LINE_WIDTHS = (1, 2, 3, 5)
COLORS = {1: "#8C8C8C", 3: "#4C72B0", 5: "#DD8452", 6: "#C44E52", 7: "#55A868"}


# 1. Làm mờ với kernel k (xử lý riêng 1x1 và kernel chẵn)
def gaussian_sigma(k: int) -> float:
    """sigma OpenCV tự chọn khi truyền sigma=0 (công thức của getGaussianKernel)."""
    return 0.3 * ((k - 1) * 0.5 - 1) + 0.8


def kernel_label(k: int) -> str:
    return "1x1 (không blur)" if k == 1 else f"{k}x{k}"


def try_opencv_gaussian(k: int) -> str:
    """Thử cv2.GaussianBlur với kernel k, trả về '' nếu được, ngược lại là thông báo lỗi."""
    try:
        cv2.GaussianBlur(np.zeros((16, 16), np.uint8), (k, k), 0)
        return ""
    except cv2.error as exc:
        text = " ".join(str(exc).split())
        start = text.find("Assertion failed")
        return text[start:] if start >= 0 else text


def blur(gray: np.ndarray, k: int) -> np.ndarray:
    """Gaussian k x k, sigma tự động như pipeline.

    - k = 1 : không làm mờ.
    - k lẻ  : cv2.GaussianBlur (đúng như module_preprocess).
    - k chẵn: cv2.GaussianBlur báo lỗi, nên dùng sepFilter2D với cùng kernel
              Gaussian để vẫn đo được. Kernel chẵn không có tâm -> ảnh lệch 0.5 px.
    """
    if k == 1:
        return gray.copy()
    if k % 2 == 1:
        return cv2.GaussianBlur(gray, (k, k), 0)
    kernel = cv2.getGaussianKernel(k, 0)
    return cv2.sepFilter2D(gray, -1, kernel, kernel, borderType=cv2.BORDER_REFLECT_101)


def detect_difference_k(img_template: np.ndarray, img_test_aligned: np.ndarray,
                        k: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Bản sao detect_difference, chỉ thay kernel Gaussian 3x3 bằng k x k.

    Trả về (mask nhị phân cuối cùng, bản đồ vi sai trước ngưỡng, thời gian blur ms).
    """
    gray_template = _gray_clahe(img_template)
    gray_test = _gray_clahe(img_test_aligned)

    t0 = time.perf_counter()
    blur_template = blur(gray_template, k)
    blur_test = blur(gray_test, k)
    t_blur = (time.perf_counter() - t0) * 1000.0

    kernel = np.ones((5, 5), np.uint8)
    template_eroded = cv2.erode(blur_template, kernel)
    template_dilated = cv2.dilate(blur_template, kernel)
    test_brighter = cv2.subtract(blur_test, template_dilated)
    test_darker = cv2.subtract(template_eroded, blur_test)
    diff = cv2.max(test_brighter, test_darker)

    otsu_val, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, diff_binary = cv2.threshold(diff, max(int(otsu_val), 30), 255, cv2.THRESH_BINARY)

    border = 8
    diff_binary[:border, :] = 0
    diff_binary[-border:, :] = 0
    diff_binary[:, :border] = 0
    diff_binary[:, -border:] = 0

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, clean_kernel)
    return mask, diff, t_blur


# 2. Phần A — chỉ số lý thuyết
def theory_table(kernels: List[int]) -> List[Dict[str, object]]:
    """Tính sigma, khử nhiễu, giữ nét mảnh và độ lệch tâm cho từng kernel."""
    rows = []
    for k in kernels:
        if k == 1:
            k1d = np.array([1.0])
        else:
            k1d = cv2.getGaussianKernel(k, 0).ravel()
        k2d = np.outer(k1d, k1d)
        row: Dict[str, object] = {
            "kernel": k,
            "label": kernel_label(k),
            "sigma": 0.0 if k == 1 else gaussian_sigma(k),
            "noise_left_pct": float(np.sqrt(np.sum(k2d ** 2))) * 100.0,
            "opencv_error": try_opencv_gaussian(k) if k > 1 else "",
        }
        # Giữ tương phản của nét mảnh rộng W px (nét dọc -> chỉ cần lọc 1 chiều)
        for width in LINE_WIDTHS:
            profile = np.zeros(64)
            profile[32 - width // 2: 32 - width // 2 + width] = 1.0
            row[f"line_keep_{width}px_pct"] = float(np.convolve(profile, k1d, mode="same").max()) * 100.0
        # Độ lệch tâm = trọng tâm kernel - điểm neo (anchor = k // 2 của OpenCV).
        # Kernel lẻ: trọng tâm trùng anchor -> 0. Kernel chẵn: trọng tâm ở (k-1)/2
        # nhưng anchor ở k/2 -> mỗi pixel kết quả lấy trung bình lệch 0.5 px.
        centroid = float(np.sum(np.arange(len(k1d)) * k1d))
        shift = abs(centroid - len(k1d) // 2)
        row["shift_x_px"] = row["shift_y_px"] = shift
        row["center_shift_px"] = shift
        rows.append(row)
    return rows


def print_theory(rows: List[Dict[str, object]]) -> None:
    print("\n=== PHẦN A — LÝ THUYẾT ===")
    header = (f"{'Kernel':<17}|{'sigma':>6} |{'nhiễu còn':>10} |"
              + "".join(f"{f'nét {w}px':>9} |" for w in LINE_WIDTHS)
              + f"{'lệch tâm':>9} | cv2.GaussianBlur")
    print(header)
    print("-" * (len(header) + 6))
    for r in rows:
        status = "OK" if not r["opencv_error"] else "LỖI (kernel phải lẻ)"
        if r["kernel"] == 1:
            status = "-"
        print(f"{r['label']:<17}|{r['sigma']:>6.2f} |{r['noise_left_pct']:>9.1f}% |"
              + "".join(f"{r[f'line_keep_{w}px_pct']:>8.1f}% |" for w in LINE_WIDTHS)
              + f"{r['center_shift_px']:>7.2f}px | {status}")
    print("nhiễu còn: % độ lệch chuẩn nhiễu trắng còn lại; nét Wpx: % tương phản còn lại của "
          "nét mảnh rộng W px; lệch tâm: ảnh bị trôi theo mỗi chiều sau khi lọc.")
    for r in rows:
        if r["opencv_error"]:
            print(f"cv2.GaussianBlur({r['kernel']}x{r['kernel']}) -> {r['opencv_error']}")


# 3. Phần B — đánh giá trên cặp ảnh
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


def snr(diff: np.ndarray, gt_boxes: List[Tuple[int, int, int, int]], margin: int = 15) -> float:
    """Tín hiệu (trung bình 10% giá trị vi sai cao nhất trong bbox GT) chia cho
    nhiễu nền (phân vị 99.9 của vi sai ngoài bbox GT, đã nới margin px)."""
    if not gt_boxes:
        return float("nan")
    h, w = diff.shape
    inside = np.zeros((h, w), bool)
    outside = np.ones((h, w), bool)
    for x, y, bw, bh in gt_boxes:
        inside[max(0, y):y + bh, max(0, x):x + bw] = True
        outside[max(0, y - margin):y + bh + margin, max(0, x - margin):x + bw + margin] = False
    border = 8
    outside[:border, :] = outside[-border:, :] = False
    outside[:, :border] = outside[:, -border:] = False
    sig_vals = diff[inside]
    bg_vals = diff[outside]
    if sig_vals.size == 0 or bg_vals.size == 0:
        return float("nan")
    top = np.sort(sig_vals)[-max(1, sig_vals.size // 10):]
    return float(top.mean()) / (float(np.percentile(bg_vals, 99.9)) + 1.0)


def evaluate_pair(pair: Dict[str, object], kernels: List[int], args: argparse.Namespace,
                  check_consistency: bool) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    template = load_image(Path(pair["template_path"]))
    test = load_image(Path(pair["test_path"]))
    try:
        aligned = align_images(template, test)
        aligned_ok = True
    except Exception:  # như main.py: căn chỉnh thất bại -> dùng ảnh chưa căn chỉnh
        aligned, aligned_ok = test, False
    if aligned.shape[:2] != template.shape[:2]:
        aligned = cv2.resize(aligned, (template.shape[1], template.shape[0]))

    gt_boxes = read_labels(pair.get("label_path"))
    conditions = {"clean": aligned}
    if args.noise_sigma > 0:
        rng = np.random.default_rng(zlib.crc32(f"{pair['pair_id']}:{args.seed}".encode()))
        noise = rng.normal(0.0, args.noise_sigma, aligned.shape)
        conditions["noise"] = np.clip(aligned.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if check_consistency:
        ref = detect_difference(template, aligned)
        mine, _, _ = detect_difference_k(template, aligned, CURRENT_KERNEL)
        print(f"    Kiểm tra đối chiếu với module_preprocess.detect_difference (3x3): "
              f"{'KHỚP' if np.array_equal(ref, mine) else 'KHÁC (kiểm tra lại code!)'}")

    rows = []
    visual: Dict[str, object] = {"pair": pair["pair_id"], "aligned": aligned,
                                 "noisy": conditions.get("noise"), "gt": gt_boxes, "maps": {}}
    for cond_name, image in conditions.items():
        for k in kernels:
            times, result = [], None
            for _ in range(max(1, args.repeats)):
                result = detect_difference_k(template, image, k)
                times.append(result[2])
            mask, diff, _ = result
            boxes = extract_bounding_boxes(mask, min_area=20)
            gt_hit = sum(1 for g in gt_boxes if any(_touch(p, g) for p in boxes))
            pred_hit = sum(1 for p in boxes if any(_touch(p, g) for g in gt_boxes))
            rows.append({
                "pair": pair["pair_id"], "class_name": pair.get("class_name", ""),
                "condition": cond_name, "kernel": k, "aligned_ok": aligned_ok,
                "n_gt": len(gt_boxes), "n_pred": len(boxes), "gt_hit": gt_hit,
                "pred_hit": pred_hit, "false_blobs": len(boxes) - pred_hit,
                "snr": snr(diff, gt_boxes), "time_blur_ms": statistics.median(times),
            })
            visual["maps"][(cond_name, k)] = (diff, mask)
    return rows, visual


# 4. Tổng hợp
def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition"]), int(row["kernel"]))].append(row)

    summary = []
    for (cond, k) in sorted(groups, key=lambda x: (x[0] != "clean", x[1])):
        items = groups[(cond, k)]
        n_gt = sum(int(r["n_gt"]) for r in items)
        n_pred = sum(int(r["n_pred"]) for r in items)
        recall = _ratio(sum(int(r["gt_hit"]) for r in items), n_gt)
        precision = _ratio(sum(int(r["pred_hit"]) for r in items), n_pred)
        f1 = (2 * precision * recall / (precision + recall)
              if not math.isnan(precision) and not math.isnan(recall) and precision + recall > 0 else 0.0)
        thin = [r for r in items if r["class_name"] in THIN_CLASSES]
        snrs = [float(r["snr"]) for r in items if not math.isnan(float(r["snr"]))]
        entry: Dict[str, object] = {
            "condition": cond, "kernel": k, "pairs": len(items),
            "recall": recall, "precision": precision, "f1": f1,
            "recall_thin": _ratio(sum(int(r["gt_hit"]) for r in thin), sum(int(r["n_gt"]) for r in thin)),
            "false_blobs_per_img": statistics.fmean(int(r["false_blobs"]) for r in items),
            "snr": statistics.median(snrs) if snrs else float("nan"),
            "time_blur_ms": statistics.fmean(float(r["time_blur_ms"]) for r in items),
        }
        for cls in sorted({str(r["class_name"]) for r in items}):
            cls_items = [r for r in items if r["class_name"] == cls]
            entry[f"recall_{cls}"] = _ratio(sum(int(r["gt_hit"]) for r in cls_items),
                                            sum(int(r["n_gt"]) for r in cls_items))
        summary.append(entry)
    return summary


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


def print_summary(summary: List[Dict[str, object]], n_pairs: int, noise_sigma: float) -> None:
    def f(v, fmt):
        return format(v, fmt) if not (isinstance(v, float) and math.isnan(v)) else "-".rjust(int(fmt.split(".")[0]))

    for cond in ("clean", "noise"):
        items = [s for s in summary if s["condition"] == cond]
        if not items:
            continue
        title = "ẢNH GỐC" if cond == "clean" else f"ẢNH CÓ NHIỄU (sigma = {noise_sigma})"
        print(f"\n=== PHẦN B — {title} — {n_pairs} cặp ===")
        header = (f"{'Kernel':<17}|{'Recall':>7} |{'Prec':>6} |{'F1':>6} |{'Recall nét mảnh':>16} |"
                  f"{'Lỗi giả/ảnh':>12} |{'SNR':>6} |{'Blur ms':>8}")
        print(header)
        print("-" * len(header))
        for s in items:
            mark = "  <- đang dùng" if s["kernel"] == CURRENT_KERNEL else ""
            print(f"{kernel_label(int(s['kernel'])):<17}|{f(s['recall'], '7.3f')} |"
                  f"{f(s['precision'], '6.3f')} |{f(s['f1'], '6.3f')} |{f(s['recall_thin'], '16.3f')} |"
                  f"{f(s['false_blobs_per_img'], '12.2f')} |{f(s['snr'], '6.2f')} |"
                  f"{f(s['time_blur_ms'], '8.2f')}{mark}")
    print("Recall nét mảnh: chỉ tính lớp " + ", ".join(THIN_CLASSES) + ".")


def conclusions(theory: List[Dict[str, object]], summary: List[Dict[str, object]]) -> List[str]:
    """Sinh kết luận TỪ SỐ LIỆU (không viết sẵn), để dán vào báo cáo."""
    lines: List[str] = []
    th = {int(r["kernel"]): r for r in theory}

    if 6 in th:
        r6 = th[6]
        msg = "cv2.GaussianBlur từ chối" if r6["opencv_error"] else "cv2.GaussianBlur chấp nhận"
        lines.append(
            f"6x6: {msg} kernel chẵn (yêu cầu ksize lẻ). Kernel chẵn không có pixel trung tâm "
            f"nên không đối xứng quanh pixel đang xét: đo được ảnh bị trôi "
            f"{r6['shift_x_px']:.2f} px theo mỗi chiều x và y, "
            f"toạ độ bbox lỗi lệch theo. Mức làm mờ "
            f"(sigma {r6['sigma']:.2f}) nằm giữa 5x5 và 7x7 nên không đem lại lợi ích riêng."
        )
    if 3 in th and 5 in th:
        lines.append(
            f"Lý thuyết 3x3 vs 5x5: 3x3 giữ {th[3]['line_keep_2px_pct']:.0f}% tương phản nét rộng 2 px "
            f"(5x5: {th[5]['line_keep_2px_pct']:.0f}%"
            + (f", 7x7: {th[7]['line_keep_2px_pct']:.0f}%" if 7 in th else "")
            + f") trong khi vẫn giảm nhiễu trắng còn {th[3]['noise_left_pct']:.0f}% "
            f"(5x5: {th[5]['noise_left_pct']:.0f}%)."
        )

    for cond in ("clean", "noise"):
        items = {int(s["kernel"]): s for s in summary if s["condition"] == cond}
        if CURRENT_KERNEL not in items:
            continue
        cur = items[CURRENT_KERNEL]
        best_f1 = max(round(float(s["f1"]), 3) for s in items.values())
        tied = sorted(k for k in items if round(float(items[k]["f1"]), 3) == best_f1)
        tag = "ảnh gốc" if cond == "clean" else "ảnh có nhiễu"
        if tied == [CURRENT_KERNEL]:
            lines.append(f"Thực nghiệm ({tag}): 3x3 có F1 cao nhất ({float(cur['f1']):.3f}).")
        elif CURRENT_KERNEL in tied:
            others = ", ".join(kernel_label(k) for k in tied if k != CURRENT_KERNEL)
            lines.append(
                f"Thực nghiệm ({tag}): 3x3 đạt F1 cao nhất ({best_f1:.3f}), ngang {others}. "
                f"Khi ngang nhau, 3x3 được ưu tiên vì giữ nét mảnh tốt hơn và nhanh hơn."
            )
        else:
            names = ", ".join(kernel_label(k) for k in tied)
            lines.append(
                f"Thực nghiệm ({tag}): F1 cao nhất là {names} ({best_f1:.3f}), "
                f"3x3 đạt {float(cur['f1']):.3f} — dữ liệu này KHÔNG ủng hộ 3x3, cần xem lại."
            )
        if 1 in items and float(items[1]["f1"]) < float(cur["f1"]) - 1e-9:
            lines.append(
                f"  Không blur (1x1): F1 {float(items[1]['f1']):.3f} < 3x3 — cần làm mờ để "
                f"giảm nhiễu/viền răng cưa trước khi trừ ảnh."
            )
        bigger = [k for k in items if k > CURRENT_KERNEL]
        if bigger:
            parts = ", ".join(f"{kernel_label(k)}: {float(items[k]['recall_thin']):.3f}" for k in sorted(bigger))
            lines.append(f"  Recall lớp nét mảnh — 3x3: {float(cur['recall_thin']):.3f}; {parts}.")
        if 1 in items:
            lines.append(
                f"  Lỗi giả/ảnh — không blur: {float(items[1]['false_blobs_per_img']):.2f}, "
                f"3x3: {float(cur['false_blobs_per_img']):.2f}."
            )
        snr_parts = ", ".join(f"{kernel_label(k).split(' ')[0]}: {float(items[k]['snr']):.2f}"
                              for k in sorted(items))
        lines.append(f"  SNR bản đồ vi sai — {snr_parts} (blur lớn tăng SNR trên vùng lỗi lớn "
                     f"nhưng làm mờ lỗi nhỏ; xem recall nét mảnh).")
    return lines


# 5. Biểu đồ
def plot_comparison(theory: List[Dict[str, object]], summary: List[Dict[str, object]],
                    output_path: Path, n_pairs: int) -> None:
    has_data = bool(summary)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9)) if has_data else plt.subplots(1, 3, figsize=(17, 4.8))
    axes = np.atleast_2d(axes)

    # (a) giữ nét mảnh theo bề rộng
    ax = axes[0, 0]
    for r in theory:
        k = int(r["kernel"])
        ax.plot(LINE_WIDTHS, [r[f"line_keep_{w}px_pct"] for w in LINE_WIDTHS], "o-",
                color=COLORS.get(k, "black"), label=kernel_label(k),
                linewidth=2.5 if k == CURRENT_KERNEL else 1.3)
    ax.set_xlabel("Bề rộng nét lỗi (px)")
    ax.set_title("Tương phản nét mảnh còn lại (%) — cao hơn tốt hơn")
    ax.legend(fontsize=8)

    # (b) đánh đổi: khử nhiễu vs giữ nét 2px
    ax = axes[0, 1]
    for r in theory:
        k = int(r["kernel"])
        ax.scatter(r["noise_left_pct"], r["line_keep_2px_pct"], s=140, color=COLORS.get(k, "black"),
                   edgecolor="black", zorder=3)
        ax.annotate(kernel_label(k), (r["noise_left_pct"], r["line_keep_2px_pct"]),
                    textcoords="offset points", xytext=(8, -14 if k % 2 == 0 else 6), fontsize=9)
    ax.set_xlabel("Nhiễu còn lại (%) — thấp hơn = khử nhiễu mạnh hơn")
    ax.set_ylabel("Nét 2 px còn lại (%)")
    ax.set_title("Đánh đổi khử nhiễu ↔ giữ chi tiết")

    # (c) độ lệch tâm
    ax = axes[0, 2]
    labels = [kernel_label(int(r["kernel"])).split(" ")[0] for r in theory]
    shifts = [float(r["center_shift_px"]) for r in theory]
    bars = ax.bar(labels, shifts, color=[COLORS.get(int(r["kernel"]), "gray") for r in theory],
                  edgecolor="black")
    for bar, v in zip(bars, shifts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.2f}", ha="center", va="bottom")
    ax.set_title("Độ lệch tâm mỗi chiều (px) — kernel chẵn lệch 0.5 px")

    if has_data:
        kernels = sorted({int(s["kernel"]) for s in summary})
        conds = [c for c in ("clean", "noise") if any(s["condition"] == c for s in summary)]
        cond_names = {"clean": "Ảnh gốc", "noise": "Có nhiễu"}
        width = 0.8 / len(conds)
        x = np.arange(len(kernels))

        def grouped(ax, key, title):
            for i, cond in enumerate(conds):
                vals = []
                for k in kernels:
                    s = next((s for s in summary if s["condition"] == cond and s["kernel"] == k), None)
                    v = float(s[key]) if s else float("nan")
                    vals.append(0.0 if math.isnan(v) else v)
                bars = ax.bar(x + (i - (len(conds) - 1) / 2) * width, vals, width,
                              label=cond_names[cond], alpha=0.9 if cond == "clean" else 0.6,
                              color=[COLORS.get(k, "gray") for k in kernels], edgecolor="black",
                              hatch="" if cond == "clean" else "//")
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([kernel_label(k).split(" ")[0] for k in kernels])
            ax.set_title(title)
            ax.legend(fontsize=8)

        grouped(axes[1, 0], "f1", "F1 phát hiện lỗi — cao hơn tốt hơn")
        grouped(axes[1, 1], "recall_thin", "Recall lớp nét mảnh — cao hơn tốt hơn")
        grouped(axes[1, 2], "false_blobs_per_img", "Số lỗi giả / ảnh — thấp hơn tốt hơn")

    for ax in axes.ravel():
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    title = "Khảo sát kernel Gaussian blur trong detect_difference"
    if has_data:
        title += f" ({n_pairs} cặp ảnh PCB)"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_visual(visual: Dict[str, object], kernels: List[int], output_path: Path,
                pad: int = 40) -> None:
    """Cắt quanh 1 lỗi GT: ảnh test và bản đồ vi sai / mask theo từng kernel."""
    gt = visual["gt"]
    if not gt:
        return
    x, y, w, h = max(gt, key=lambda b: -b[2] * b[3])  # lỗi nhỏ nhất -> khó nhất
    aligned = visual["aligned"]
    H, W = aligned.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    conds = [c for c in ("clean", "noise") if (c, kernels[0]) in visual["maps"]]

    rows = 2 * len(conds)
    fig, axes = plt.subplots(rows, len(kernels) + 1, figsize=(2.6 * (len(kernels) + 1), 2.6 * rows))
    axes = np.atleast_2d(axes)
    for ci, cond in enumerate(conds):
        src = visual["aligned"] if cond == "clean" else visual["noisy"]
        crop = cv2.cvtColor(src[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        for ri in (0, 1):
            ax = axes[2 * ci + ri, 0]
            ax.imshow(crop)
            ax.add_patch(plt.Rectangle((x - x0, y - y0), w, h, fill=False, color="lime", linewidth=1.5))
            ax.set_title(("Ảnh test" if cond == "clean" else "Ảnh test + nhiễu")
                         + (" | vi sai" if ri == 0 else " | mask"), fontsize=8)
        for kj, k in enumerate(kernels, start=1):
            diff, mask = visual["maps"][(cond, k)]
            axes[2 * ci, kj].imshow(diff[y0:y1, x0:x1], cmap="inferno", vmin=0, vmax=max(1, int(diff.max())))
            axes[2 * ci, kj].set_title(f"{kernel_label(k).split(' ')[0]} — vi sai", fontsize=8)
            axes[2 * ci + 1, kj].imshow(mask[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
            axes[2 * ci + 1, kj].set_title(f"{kernel_label(k).split(' ')[0]} — mask", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{visual['pair']}: lỗi nhỏ nhất (khung xanh = nhãn GT)", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# 6. Entry point
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Khảo sát kernel Gaussian blur trên các cặp ảnh PCB.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Thư mục chứa pairs_manifest.csv (mặc định: data).")
    parser.add_argument("--kernels", type=int, nargs="+", default=DEFAULT_KERNELS,
                        help="Các kích thước kernel cần thử (mặc định: 1 3 5 6 7).")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ chạy N cặp đầu tiên (0 = tất cả).")
    parser.add_argument("--per-class", type=int, default=0,
                        help="Lấy N cặp mỗi lớp lỗi (0 = tất cả). Gợi ý: 5.")
    parser.add_argument("--noise-sigma", type=float, default=6.0,
                        help="Độ lệch chuẩn nhiễu Gaussian cho điều kiện 'noise' (0 = bỏ điều kiện này).")
    parser.add_argument("--repeats", type=int, default=1, help="Số lần lặp đo thời gian (mặc định: 1).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theory-only", action="store_true", help="Chỉ chạy phần lý thuyết (không cần ảnh).")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"),
                        help="Thư mục lưu CSV và biểu đồ (mặc định: experiments).")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    kernels = sorted(set(args.kernels))
    if any(k < 1 for k in kernels):
        raise SystemExit("[LỖI] Kích thước kernel phải >= 1.")
    if CURRENT_KERNEL not in kernels:
        kernels = sorted(kernels + [CURRENT_KERNEL])

    output_dir: Path = args.output_dir
    chart_path = output_dir / "charts" / "blur_comparison.png"
    visual_path = output_dir / "charts" / "blur_visual.png"

    pairs: List[Dict[str, object]] = []
    if not args.theory_only:
        pairs = find_pairs(str(args.data_dir))
        if not pairs:
            raise SystemExit(
                f"\n[LỖI] Không tìm thấy cặp ảnh nào trong '{args.data_dir}'.\n"
                "       Hãy chạy experiments/build_pairs.py để sinh pairs_manifest.csv, "
                "hoặc thêm --theory-only để chỉ chạy phần lý thuyết.\n"
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

    theory = theory_table(kernels)
    print_theory(theory)

    rows: List[Dict[str, object]] = []
    visual = None
    done = 0
    if pairs:
        print(f"\nTìm thấy {len(pairs)} cặp ảnh trong '{args.data_dir}'. Kernel: {kernels}")
        for index, pair in enumerate(pairs, start=1):
            try:
                pair_rows, pair_visual = evaluate_pair(pair, kernels, args, check_consistency=(done == 0))
            except (ValueError, cv2.error) as exc:
                print(f"[{index}/{len(pairs)}] Bỏ qua {pair['pair_id']}: {exc}")
                continue
            rows.extend(pair_rows)
            done += 1
            # Giữ cặp có lỗi nhỏ nhất làm ảnh minh hoạ
            if pair_visual["gt"]:
                smallest = min(b[2] * b[3] for b in pair_visual["gt"])
                if visual is None or smallest < visual["_smallest"]:
                    pair_visual["_smallest"] = smallest
                    visual = pair_visual
            print(f"[{index}/{len(pairs)}] Đã xử lý {pair['pair_id']}")

    summary = summarize(rows) if rows else []
    if summary:
        print_summary(summary, done, args.noise_sigma)

    print("\n=== KẾT LUẬN (sinh từ số liệu) ===")
    for line in conclusions(theory, summary):
        print("- " + line if not line.startswith("  ") else "  " + line.strip())

    plot_comparison(theory, summary, chart_path, done)
    if rows:
        write_csv(rows, output_dir / "blur_kernel_results.csv")
        print(f"\nCSV chỉ số     : {output_dir / 'blur_kernel_results.csv'}")
    if visual is not None:
        plot_visual(visual, kernels, visual_path)
        print(f"Ảnh so sánh    : {visual_path}")
    print(f"Biểu đồ        : {chart_path}")
    print("\nGợi ý: dùng bảng và biểu đồ để điền Mục 2.4 của PROGRESS_REPORT.md (blur_comparison.png).")


if __name__ == "__main__":
    main()
