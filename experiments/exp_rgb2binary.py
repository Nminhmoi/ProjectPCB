"""
Khảo sát phương pháp chuyển ảnh màu (RGB/BGR) sang ảnh nhị phân.

    So sánh 3 nhóm phương pháp:
      1. Baseline      : Gray -> Gaussian 3x3 -> Otsu
      2. Không gian màu: Lab (kênh b* và a*) -> Gaussian 3x3 -> Otsu
      3. Thích nghi    : Gray -> CLAHE -> Gaussian 3x3 -> Otsu

   Script đo các chỉ số định lượng:
    - time_ms          : thời gian xử lý trung vị (ms)
    - otsu_eta         : độ tách lớp Otsu eta = sigma_between^2 / sigma_total^2
                         (càng gần 1 thì ngưỡng Otsu càng tách tốt 2 đỉnh)
    - foreground_ratio : tỷ lệ pixel được coi là "đồng" (%)
    - edge_retention   : tỷ lệ biên cạnh Canny của ảnh xám được mask nhị phân
                         giữ lại (bảo toàn cấu trúc đường mạch)
    - illum_iou        : IoU giữa mask gốc và mask khi ảnh bị chiếu sáng không
                         đều (mô phỏng đèn công nghiệp) - càng cao càng bền sáng

Cách chạy:
    py -3.14 experiments/exp_rgb2binary.py

Đầu ra:
    experiments/rgb2binary_results.csv
    experiments/charts/rgb2binary_metrics.png
    experiments/charts/rgb2binary_visual.png
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Cho phép chạy trực tiếp từ thư mục experiments/: thêm project root vào sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_utils import find_pairs, load_image  # noqa: E402

import matplotlib

# Notebook (ipykernel) cần backend inline để hiển thị ảnh; CLI dùng Agg để lưu file.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Kiểu hàm: nhận ảnh BGR, trả về (mask nhị phân uint8, kênh dùng để tính Otsu).
Binarizer = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


# 1. Các phương pháp nhị phân hóa
def _otsu(blurred: np.ndarray) -> np.ndarray:
    """Ngưỡng Otsu, coi pixel sáng hơn ngưỡng là foreground (đồng)."""
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask


def binarize_gray(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline: RGB -> Grayscale -> Gaussian 3x3 -> Otsu."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return _otsu(blur), blur


def binarize_lab_b(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Tách kênh b* (vàng - xanh lam) của Lab.

    Đồng/thiếc vàng có b* cao, lớp phủ xanh (solder mask) có b* thấp
    => ngưỡng Otsu tách được đường mạch khỏi nền.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    channel = lab[:, :, 2]
    blur = cv2.GaussianBlur(channel, (3, 3), 0)
    return _otsu(blur), blur


def binarize_lab_a(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Tách kênh a* (đỏ - xanh lá) của Lab.

    Đồng đỏ/vàng có a* cao, lớp phủ xanh có a* thấp.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    channel = lab[:, :, 1]
    blur = cv2.GaussianBlur(channel, (3, 3), 0)
    return _otsu(blur), blur


def binarize_clahe(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gray -> CLAHE (cân bằng sáng cục bộ) -> Gaussian 3x3 -> Otsu."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    blur = cv2.GaussianBlur(equalized, (3, 3), 0)
    return _otsu(blur), blur


METHODS: Dict[str, Binarizer] = {
    "gray_otsu": binarize_gray,
    "lab_b_otsu": binarize_lab_b,
    "lab_a_otsu": binarize_lab_a,
    "clahe_otsu": binarize_clahe,
}


# 2. Chỉ số đánh giá
def otsu_separability(channel: np.ndarray) -> float:
    """Tính eta = sigma_between^2 / sigma_total^2 từ histogram.

    eta ~ 1: histogram lưỡng đỉnh rõ, ngưỡng Otsu đáng tin.
    eta ~ 0: histogram phẳng/một đỉnh, ngưỡng Otsu kém tin cậy.
    """
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()
    total = float(channel.size)
    if total == 0:
        return 0.0

    levels = np.arange(256, dtype=np.float64)
    sum_total = float(np.dot(levels, hist))
    sum_all_sq = float(np.dot(levels**2, hist))
    mean_total = sum_total / total
    var_total = sum_all_sq / total - mean_total**2
    if var_total <= 1e-12:
        return 0.0

    weight_bg = 0.0
    sum_bg = 0.0
    best_eta = 0.0
    for threshold in range(256):
        weight_bg += hist[threshold]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += levels[threshold] * hist[threshold]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        eta = var_between / (total**2 * var_total)
        if eta > best_eta:
            best_eta = eta
    return float(best_eta)


def edge_retention(mask: np.ndarray, gray: np.ndarray) -> float:
    """Bao nhiêu % biên cạnh của ảnh xám được mask nhị phân giữ lại.

    Dùng Canny trên ảnh xám làm biên tham chiếu, Canny trên mask làm biên
    nhị phân. Nới biên mask 1 pixel để bù sai lệch làm tròn tọa độ.
    """
    edges_gray = cv2.Canny(gray, 50, 150)
    edges_mask = cv2.Canny(mask, 50, 150)
    total_edges = int(np.count_nonzero(edges_gray))
    if total_edges == 0:
        return 0.0
    dilated_mask_edges = cv2.dilate(
        edges_mask, np.ones((3, 3), np.uint8), iterations=1
    )
    covered = int(np.count_nonzero(edges_gray & dilated_mask_edges))
    return covered / total_edges


def apply_uneven_lighting(img_bgr: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Mô phỏng ánh sáng công nghiệp không đều bằng gradient ngang."""
    height, width = img_bgr.shape[:2]
    gradient = np.linspace(
        1.0 - strength, 1.0 + strength, width, dtype=np.float32
    )
    shaded = img_bgr.astype(np.float32) * gradient[None, :, None]
    return np.clip(shaded, 0, 255).astype(np.uint8)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int(np.count_nonzero(cv2.bitwise_and(mask_a, mask_b)))
    union = int(np.count_nonzero(cv2.bitwise_or(mask_a, mask_b)))
    return inter / union if union else 0.0


def evaluate_pair(
    img_bgr: np.ndarray,
    method: Binarizer,
    repeats: int = 3,
    shade_strength: float = 0.25,
) -> Dict[str, float]:
    """Chạy 1 phương pháp trên 1 ảnh và trả về toàn bộ chỉ số."""
    timings: List[float] = []
    mask = None
    channel = None
    for _ in range(repeats):
        start = time.perf_counter()
        mask, channel = method(img_bgr)
        timings.append((time.perf_counter() - start) * 1000.0)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    shaded = apply_uneven_lighting(img_bgr, strength=shade_strength)
    mask_shaded, _ = method(shaded)

    return {
        "time_ms": statistics.median(timings),
        "otsu_eta": otsu_separability(channel),
        "foreground_ratio": float(np.count_nonzero(mask)) / mask.size,
        "edge_retention": edge_retention(mask, gray),
        "illum_iou": mask_iou(mask, mask_shaded),
    }

# 3. Xuất CSV, biểu đồ, ảnh so sánh trực quan
def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair",
        "method",
        "time_ms",
        "otsu_eta",
        "foreground_ratio",
        "edge_retention",
        "illum_iou",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: List[Dict[str, object]],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    present = [name for name in METHODS if any(r["method"] == name for r in rows)]
    for method in present or list(METHODS):
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {}
        for metric in (
            "time_ms",
            "otsu_eta",
            "foreground_ratio",
            "edge_retention",
            "illum_iou",
        ):
            values = [float(row[metric]) for row in method_rows]
            mean = statistics.fmean(values) if values else 0.0
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            summary[method][metric] = (mean, std)
    return summary


def plot_metrics(
    summary: Dict[str, Dict[str, Tuple[float, float]]], output_path: Path
) -> None:
    metrics = [
        ("time_ms", "Thời gian (ms) - thấp hơn tốt hơn"),
        ("otsu_eta", "Độ tách lớp Otsu eta - cao hơn tốt hơn"),
        ("edge_retention", "Bảo toàn biên cạnh - cao hơn tốt hơn"),
        ("illum_iou", "IoU khi lệch sáng - cao hơn tốt hơn"),
    ]
    method_names = list(summary.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        means = [summary[name][metric][0] for name in method_names]
        stds = [summary[name][metric][1] for name in method_names]
        bars = axis.bar(
            method_names,
            means,
            yerr=stds,
            capsize=4,
            color=colors[: len(method_names)],
            edgecolor="black",
            linewidth=0.6,
        )
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.tick_params(axis="x", rotation=15)
        for bar, mean in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        "So sánh các phương pháp chuyển ảnh màu PCB sang nhị phân", fontsize=13
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_visual_comparison(
    pairs: List[Dict[str, object]],
    output_path: Path,
    max_pairs: int = 3,
    max_width: int = 520,
    method_names: List[str] | None = None,
) -> None:
    selected = pairs[:max_pairs]
    if not selected:
        return

    method_names = method_names or list(METHODS.keys())
    n_rows = len(selected)
    n_cols = 1 + len(method_names)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 3.2 * n_rows), squeeze=False
    )

    for row_index, pair in enumerate(selected):
        test_path = Path(pair["test_path"])
        img = load_image(test_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        def to_display(image: np.ndarray) -> np.ndarray:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            if width > max_width:
                scale = max_width / width
                image = cv2.resize(
                    image,
                    (max_width, int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            return image

        axes[row_index][0].imshow(to_display(img))
        axes[row_index][0].set_title(f"{test_path.name}\n(ảnh gốc)")
        axes[row_index][0].axis("off")

        for col_index, method_name in enumerate(method_names, start=1):
            mask, _ = METHODS[method_name](img)
            axes[row_index][col_index].imshow(
                to_display(mask), cmap="gray", vmin=0, vmax=255
            )
            axes[row_index][col_index].set_title(method_name)
            axes[row_index][col_index].axis("off")

    fig.suptitle(
        "Trực quan hóa mask nhị phân theo từng phương pháp", fontsize=13
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def print_summary(
    summary: Dict[str, Dict[str, Tuple[float, float]]], pair_count: int
) -> None:
    print(f"\nĐã khảo sát {pair_count} cặp ảnh.")
    print("-" * 92)
    print(
        f"{'Phương pháp':<14} | {'Time (ms)':>10} | {'Otsu eta':>9} | "
        f"{'Foreground %':>13} | {'Edge ret.':>10} | {'Illum IoU':>10}"
    )
    print("-" * 92)
    for method, metrics in summary.items():
        time_mean = metrics["time_ms"][0]
        eta_mean = metrics["otsu_eta"][0]
        fg_mean = metrics["foreground_ratio"][0] * 100.0
        edge_mean = metrics["edge_retention"][0]
        illum_mean = metrics["illum_iou"][0]
        print(
            f"{method:<14} | {time_mean:>10.3f} | {eta_mean:>9.4f} | "
            f"{fg_mean:>12.2f}% | {edge_mean:>10.4f} | {illum_mean:>10.4f}"
        )
    print("-" * 92)


# 5. Entry point
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Khảo sát RGB -> Binary trên các cặp ảnh PCB."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Thư mục chứa pairs_manifest.csv (mặc định: data).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Chỉ chạy N cặp đầu tiên (0 = chạy tất cả).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Số lần lặp đo thời gian cho mỗi ảnh (mặc định: 3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments"),
        help="Thư mục lưu CSV và biểu đồ (mặc định: experiments).",
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Không xuất ảnh so sánh trực quan.",
    )
    parser.add_argument(
        "--method",
        choices=list(METHODS) + ["all"],
        default="all",
        help="Chỉ chạy một phương pháp (mặc định: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pairs = find_pairs(str(args.data_dir))
    if not pairs:
        raise SystemExit(
            f"\n[LỖI] Không tìm thấy cặp ảnh nào trong '{args.data_dir}'.\n"
            "       Hãy chạy experiments/build_pairs.py để sinh pairs_manifest.csv "
            "trước khi chạy.\n"
        )

    if args.limit > 0:
        pairs = pairs[: args.limit]

    print(f"Tìm thấy {len(pairs)} cặp ảnh trong '{args.data_dir}'.")

    methods = METHODS
    if args.method != "all":
        methods = {args.method: METHODS[args.method]}

    rows: List[Dict[str, object]] = []
    processed_pairs: List[Dict[str, object]] = []

    for index, pair in enumerate(pairs, start=1):
        test_path = Path(pair["test_path"])
        temp_path = Path(pair["template_path"])
        pair_name = str(pair["pair_id"])

        try:
            test_image = load_image(test_path)
            # Nạp cả ảnh template để phát hiện sớm file lỗi/không cùng kích thước.
            template_image = load_image(temp_path)
        except ValueError as exc:
            print(f"[{index}/{len(pairs)}] Bỏ qua: {exc}")
            continue

        if test_image.shape[:2] != template_image.shape[:2]:
            print(
                f"[{index}/{len(pairs)}] Cảnh báo: {test_path.name} và "
                f"{temp_path.name} khác kích thước, vẫn đo trên ảnh test."
            )

        for method_name, method in methods.items():
            metrics = evaluate_pair(
                test_image, method, repeats=max(1, args.repeats)
            )
            rows.append(
                {
                    "pair": pair_name,
                    "method": method_name,
                    **{key: round(value, 6) for key, value in metrics.items()},
                }
            )

        processed_pairs.append(pair)
        print(f"[{index}/{len(pairs)}] Đã xử lý {pair_name}")

    if not rows:
        raise SystemExit("Không có ảnh nào được xử lý thành công.")

    output_dir: Path = args.output_dir
    csv_path = output_dir / "rgb2binary_results.csv"
    write_csv(rows, csv_path)

    summary = summarize(rows)
    print_summary(summary, len(processed_pairs))

    charts_dir = output_dir / "charts"
    metrics_chart = charts_dir / "rgb2binary_metrics.png"
    plot_metrics(summary, metrics_chart)

    visual_chart = charts_dir / "rgb2binary_visual.png"
    if not args.no_visual:
        plot_visual_comparison(
            processed_pairs, visual_chart, method_names=list(methods.keys())
        )

    print(f"\nCSV chỉ số     : {csv_path}")
    print(f"Biểu đồ        : {metrics_chart}")
    if not args.no_visual:
        print(f"Ảnh so sánh    : {visual_chart}")
    print(
        "\nGợi ý: dùng bảng CSV để điền Mục 3 của PROGRESS_REPORT.md, "
        "kết luận phương pháp nào có eta / edge_retention / illum_iou cao "
        "nhưng thời gian chấp nhận được."
    )


if __name__ == "__main__":
    main()
