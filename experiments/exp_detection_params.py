"""
P0.3 + P1.1 + P1.2 — Dò tham số bước nhị phân hóa và đo F1 chặt hơn bằng IoU.

    Bối cảnh (EXPERIMENT_REVIEW.md): F1 ~0.97 đã bão hòa ở bước căn chỉnh; lỗi
    còn lại nằm ở bước trừ vi sai -> nhị phân hóa:
      - 3 cặp bỏ sót dù căn chỉnh tốt: 01_spur_04, 04_spurious_copper_05, 01_mouse_bite_19
      - Spur / Spurious_copper có recall thấp nhất (~0.92)
      - Nghi vấn: sàn ngưỡng max(Otsu, 30) + opening 3x3 + min_area 20 xóa nét mảnh.

    Script tính bản đồ vi sai MỘT lần cho mỗi cặp rồi thử cả lưới tham số:
      --floors       ngưỡng sàn (mặc định 15 20 25 30)
      --opens        kernel opening, 1 = bỏ opening (mặc định 1 2 3)
      --hysteresis   0 = ngưỡng đơn; 0.5 = hysteresis (giữ vùng >= 0.5*ngưỡng nếu
                     nối với vùng vượt ngưỡng) (mặc định 0 0.5)
      --min-areas    diện tích đốm tối thiểu (mặc định 10 20)
    Cấu hình gốc của pipeline: floor 30, open 3, hysteresis 0, min_area 20.

    Hai điều kiện đo (--conditions):
      clean    : ảnh test gốc của PKU, căn chỉnh như main.py (align_images_robust).
      resample : ảnh test bị nắn lệch giả lập rồi căn ngược lại -> ảnh bị NỘI SUY
                 như ảnh camera thật. Đây là điều kiện phát hiện bị ảnh hưởng mạnh
                 (recall có thể tụt rất sâu), nên tham số phải tốt ở CẢ HAI điều kiện.

    Chỉ số cho mỗi cấu hình:
      - recall / precision / F1 theo tiêu chí cũ ("chạm nhau", dung sai 5 px)
      - F1@IoU0.3 và F1@IoU0.5 (ghép 1-1) - P1.2
      - số đốm giả / ảnh, recall theo từng lớp lỗi, recall 3 cặp trọng điểm
      - đạt / không đạt các chỉ tiêu KPI (mục 5 của EXPERIMENT_REVIEW.md)
    Chẩn đoán (cấu hình gốc): mỗi lỗi GT bị bỏ sót được quy về BƯỚC làm mất nó:
      "ngưỡng"   : không pixel nào trong bbox GT vượt ngưỡng
      "opening"  : có pixel vượt ngưỡng nhưng opening xóa hết
      "min_area" : còn đốm sau opening nhưng nhỏ hơn min_area
    và --inspect lưu ảnh cắt (template | test | vi sai | mask gốc | mask tốt nhất).

Cách chạy:
    py -3.14 experiments/exp_detection_params.py --pairs 01_spur_04 04_spurious_copper_05 01_mouse_bite_19
    py -3.14 experiments/exp_detection_params.py --per-class 10
    py -3.14 experiments/exp_detection_params.py --per-class 10 --conditions clean

Chạy full 693 cặp (~1.5–3 giờ; 693 × 2 điều kiện × 49 cấu hình ≈ 67.9k lần Otsu +
    morphology + contours). Nên chạy `--per-class` trước để chọn lưới tham số gọn hơn:
    py -3.14 -u experiments/exp_detection_params.py --output-dir experiments

    Chạy dài có thể bị gián đoạn: script tự ghi checkpoint sau mỗi --checkpoint-every
    cặp (mặc định 20) vào <output-dir>/detection_params_state.pkl. Nếu dừng giữa
    chừng, chạy lại với --resume để tiếp tục (bỏ qua cặp đã xong):
    py -3.14 -u experiments/exp_detection_params.py --output-dir experiments --resume
    Checkpoint bị bỏ qua nếu cấu hình/điều kiện/danh sách cặp khác lần chạy trước.

    Ghi chú: cả hai điều kiện đều dùng mask "vùng hợp lệ" (pixel thật sau warp, erode
    VALID_ERODE px) đưa vào bước TÍNH NGƯỠNG, không phải mask sau khi nhị phân hóa.
    Lý do: dải biên đen do warp chi phối Otsu và xóa mất lỗi mảnh (CODE_REVIEW.md 2.3).

Đầu ra:
    experiments/detection_params_summary.csv   (mỗi điều kiện x cấu hình)
    experiments/detection_params_per_pair.csv  (mỗi cặp x điều kiện x cấu hình)
    experiments/detection_missed_diagnosis.csv (mỗi lỗi bị sót ở cấu hình gốc)
    experiments/charts/detection_params.png
    experiments/inspect/<pair>_<điều kiện>.jpg (với --inspect)
    experiments/detection_params_state.pkl (checkpoint khi đang chạy; tự xóa khi xong)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataset_utils import find_pairs, load_image  # noqa: E402
from evaluation import (  # noqa: E402
    boxes_touch,
    detection_counts,
    prf_from_counts,
    read_labels,
    synthetic_warp,
)
from MODULE.module_detection import extract_bounding_boxes  # noqa: E402
from MODULE.module_preprocess import (  # noqa: E402
    DEFAULT_HYSTERESIS_LOW,
    DEFAULT_MIN_AREA,
    DEFAULT_OPEN_KSIZE,
    DEFAULT_THRESHOLD_FLOOR,
    AlignmentError,
    align_images_robust,
    binarize_difference,
    difference_map,
)

FOCUS_PAIRS = ["01_spur_04", "04_spurious_copper_05", "01_mouse_bite_19"]
WEAK_CLASSES = ("Spur", "Spurious_copper")
# Bỏ thêm vài pixel sát biên sau warp (nơi nội suy dễ sai) khi tạo mask hợp lệ.
VALID_ERODE = 3
BASELINE = (DEFAULT_THRESHOLD_FLOOR, DEFAULT_OPEN_KSIZE,
            float(DEFAULT_HYSTERESIS_LOW or 0.0), DEFAULT_MIN_AREA)
KPI = {"recall": 0.97, "f1": 0.975, "weak_recall": 0.96, "false_blobs": 0.05}
Config = Tuple[int, int, float, int]  # (floor, open, hysteresis, min_area)


def config_name(cfg: Config) -> str:
    floor, open_k, hyst, min_area = cfg
    return (f"floor{floor}_open{open_k}_" + (f"hyst{hyst:g}" if hyst else "single")
            + f"_area{min_area}")


# ---------------------------------------------------------------------
# Chuẩn bị bản đồ vi sai cho 1 cặp ở 1 điều kiện
# ---------------------------------------------------------------------
def prepare(pair: Dict[str, object], condition: str, args: argparse.Namespace):
    """Trả về (template, aligned, diff, valid_mask hoặc None, align_info)."""
    template = load_image(Path(pair["template_path"]))
    test = load_image(Path(pair["test_path"]))
    h, w = template.shape[:2]
    test_mask = None
    if condition == "resample":
        test, test_mask, _ = synthetic_warp(test, str(pair["pair_id"]), args.seed,
                                            args.warp_angle, args.warp_shift, args.warp_scale)
    try:
        aligned, stats = align_images_robust(template, test, test_mask=test_mask)
        info = {"align_ok": True, "align_detector": stats["detector"],
                "align_ncc": stats.get("ncc", float("nan")),
                "align_verified": stats.get("verified", True)}
        H = stats["H"]
    except AlignmentError as exc:
        aligned = test if test.shape[:2] == (h, w) else cv2.resize(test, (w, h))
        info = {"align_ok": False, "align_detector": "", "align_ncc": float("nan"),
                "align_verified": False, "align_error": str(exc)}
        H = None

    valid = None
    if H is not None:
        # Mask pixel thật sau warp; đưa vào bước tính ngưỡng (không phải mask
        # sau khi nhị phân hóa) để dải biên đen không đẩy Otsu lên.
        src = test_mask if test_mask is not None else np.full(test.shape[:2], 255, np.uint8)
        valid = cv2.warpPerspective(src, H, (w, h), flags=cv2.INTER_NEAREST)
        k = 2 * VALID_ERODE + 1
        valid = cv2.erode(valid, np.ones((k, k), np.uint8))
    diff, _ = difference_map(template, aligned)
    return template, aligned, diff, valid, info


def detect(diff: np.ndarray, valid: Optional[np.ndarray], cfg: Config):
    floor, open_k, hyst, min_area = cfg
    mask = binarize_difference(diff, threshold_floor=floor, open_ksize=open_k,
                               hysteresis_low=hyst if hyst else None, valid=valid)
    return mask, extract_bounding_boxes(mask, min_area=min_area)


# ---------------------------------------------------------------------
# Chẩn đoán lỗi bị sót: mất ở bước nào?
# ---------------------------------------------------------------------
def diagnose_missed(diff: np.ndarray, valid: Optional[np.ndarray], gt_boxes, cfg: Config,
                    boxes) -> List[Dict[str, object]]:
    floor, open_k, hyst, min_area = cfg
    # Otsu phải lấy mẫu trong cùng vùng với binarize_difference (bỏ dải biên đen).
    sample = diff[valid > 0] if valid is not None else diff
    if sample.size:
        otsu, _ = cv2.threshold(np.ascontiguousarray(sample).reshape(-1, 1), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        otsu = 0
    thr = max(int(otsu), floor)
    raw = binarize_difference(diff, threshold_floor=floor, open_ksize=1,
                              hysteresis_low=hyst if hyst else None, valid=valid)
    opened = binarize_difference(diff, threshold_floor=floor, open_ksize=open_k,
                                 hysteresis_low=hyst if hyst else None, valid=valid)
    out = []
    for g in gt_boxes:
        if any(boxes_touch(p, g) for p in boxes):
            continue
        x, y, w, h = g
        sl = (slice(max(0, y), y + h), slice(max(0, x), x + w))
        roi = diff[sl]
        n_raw = int(np.count_nonzero(raw[sl]))
        n_open = int(np.count_nonzero(opened[sl]))
        if n_raw == 0:
            stage = "ngưỡng"
        elif n_open == 0:
            stage = "opening"
        else:
            stage = "min_area"
        largest = 0
        if n_open:
            n, _, st, _ = cv2.connectedComponentsWithStats(opened[sl], connectivity=8)
            largest = int(st[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
        out.append({"gt_x": x, "gt_y": y, "gt_w": w, "gt_h": h, "stage": stage,
                    "diff_max": int(roi.max()) if roi.size else 0,
                    "diff_p99": float(np.percentile(roi, 99)) if roi.size else 0.0,
                    "threshold": thr, "otsu": int(otsu),
                    "px_above_thr": n_raw, "px_after_open": n_open,
                    "largest_blob": largest})
    return out


# ---------------------------------------------------------------------
# Ảnh kiểm tra trực quan
# ---------------------------------------------------------------------
def save_inspect(path: Path, template, aligned, diff, gt_boxes, masks: Dict[str, np.ndarray],
                 boxes_by: Dict[str, list], pad: int = 30, max_items: int = 6) -> None:
    tiles = []
    for g in gt_boxes[:max_items]:
        x, y, w, h = g
        H, W = template.shape[:2]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)

        def crop(img, color=False):
            c = img[y0:y1, x0:x1]
            if c.ndim == 2:
                c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
            c = c.copy()
            cv2.rectangle(c, (x - x0, y - y0), (x - x0 + w, y - y0 + h), (0, 255, 0), 1)
            return c

        diff_vis = cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_INFERNO)
        cells = [("template", crop(template)), ("test", crop(aligned)), ("vi sai", crop(diff_vis))]
        for name, m in masks.items():
            c = crop(m)
            hit = any(boxes_touch(p, g) for p in boxes_by[name])
            cv2.putText(c, "OK" if hit else "SOT", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 0) if hit else (0, 0, 255), 1, cv2.LINE_AA)
            cells.append((name, c))
        size = (160, 160)
        row = []
        for name, c in cells:
            c = cv2.resize(c, size, interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(c, (0, size[1] - 16), (size[0], size[1]), (0, 0, 0), -1)
            cv2.putText(c, name[:22], (3, size[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
            row.append(c)
        tiles.append(np.hstack(row))
    if tiles:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.vstack(tiles))


# ---------------------------------------------------------------------
# Tổng hợp
# ---------------------------------------------------------------------
def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        groups[(str(r["condition"]), str(r["config"]))].append(r)
    classes = sorted({str(r["class_name"]) for r in rows})
    out = []
    for (cond, name), items in sorted(groups.items()):
        first = items[0]
        entry: Dict[str, object] = {
            "condition": cond, "config": name, "floor": first["floor"], "open": first["open"],
            "hysteresis": first["hysteresis"], "min_area": first["min_area"],
            "is_baseline": first["is_baseline"], "pairs": len(items),
        }
        entry.update(prf_from_counts(items))
        for cls in classes:
            entry[f"recall_{cls}"] = prf_from_counts(
                [r for r in items if r["class_name"] == cls])["recall"]
        weak = [r for r in items if r["class_name"] in WEAK_CLASSES]
        entry["recall_weak"] = prf_from_counts(weak)["recall"] if weak else float("nan")
        focus = [r for r in items if r["pair"] in FOCUS_PAIRS]
        entry["recall_focus"] = prf_from_counts(focus)["recall"] if focus else float("nan")
        entry["pairs_all_missed"] = sum(1 for r in items if int(r["n_gt"]) and int(r["gt_hit"]) == 0)
        out.append(entry)

    # So với cấu hình gốc cùng điều kiện
    base = {s["condition"]: s for s in out if s["is_baseline"]}
    for s in out:
        b = base.get(s["condition"])
        if not b:
            continue
        s["delta_f1"] = float(s["f1"]) - float(b["f1"])
        s["delta_precision"] = float(s["precision"]) - float(b["precision"])
        bf = float(b["false_blobs_per_img"])
        s["false_blobs_change_pct"] = ((float(s["false_blobs_per_img"]) - bf) / bf * 100
                                       if bf > 0 else (0.0 if float(s["false_blobs_per_img"]) == 0 else float("inf")))
        weak_ok = math.isnan(float(s["recall_weak"])) or float(s["recall_weak"]) >= KPI["weak_recall"]
        s["kpi_pass"] = bool(float(s["recall"]) >= KPI["recall"] and float(s["f1"]) >= KPI["f1"]
                             and weak_ok and float(s["false_blobs_per_img"]) <= KPI["false_blobs"])
    return out


def pick_best(summary: List[Dict[str, object]], conditions: Sequence[str]) -> Optional[str]:
    """Cấu hình tốt nhất: precision KHÔNG giảm so với gốc ở mọi điều kiện, rồi tối đa
    F1 trung bình qua các điều kiện (nguyên tắc ở mục 6 EXPERIMENT_REVIEW.md)."""
    by_cfg: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for s in summary:
        by_cfg[s["config"]][s["condition"]] = s
    def distance(s: Dict[str, object]) -> float:
        # Mức thay đổi so với cấu hình gốc: hòa F1 thì chọn cấu hình thay đổi ít nhất
        return (abs(int(s["floor"]) - BASELINE[0]) / 5 + abs(int(s["open"]) - BASELINE[1])
                + (float(s["hysteresis"]) != BASELINE[2]) + abs(int(s["min_area"]) - BASELINE[3]) / 10)

    best, best_key = None, None
    for name, per_cond in by_cfg.items():
        if any(c not in per_cond for c in conditions):
            continue
        if any(float(per_cond[c].get("delta_precision", 0.0)) < -1e-9 for c in conditions):
            continue
        score = sum(float(per_cond[c]["f1"]) for c in conditions) / len(conditions)
        any_s = per_cond[conditions[0]]
        key = (round(score, 4), -distance(any_s))
        if best_key is None or key > best_key:
            best, best_key = name, key
    return best


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
            writer.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})


def fmt(v, spec: str) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "-".rjust(int(spec.split(".")[0]) if spec[0].isdigit() else 1)
    return format(v, spec)


def print_table(summary: List[Dict[str, object]], cond: str, best: Optional[str], top: int) -> None:
    items = [s for s in summary if s["condition"] == cond]
    if not items:
        return
    base = next((s for s in items if s["is_baseline"]), None)
    ranked = sorted(items, key=lambda s: -float(s["f1"]))
    shown = ranked[:top]
    for extra in (base, next((s for s in items if s["config"] == best), None)):
        if extra is not None and extra not in shown:
            shown.append(extra)
    title = {"clean": "ẢNH GỐC (căn chỉnh như main.py)",
             "resample": "ẢNH ĐÃ NỘI SUY (nắn lệch giả lập rồi căn lại)"}.get(cond, cond)
    print(f"\n=== {title} — {items[0]['pairs']} cặp, {len(items)} cấu hình (top {top} theo F1) ===")
    header = (f"{'Cấu hình':<32}|{'Recall':>7} |{'Prec':>6} |{'F1':>6} |{'F1@.3':>6} |{'F1@.5':>6} |"
              f"{'Giả/ảnh':>8} |{'Spur+SpCu':>9} |{'3 cặp':>6} |{'Sót hết':>7} | KPI")
    print(header)
    print("-" * len(header))
    for s in shown:
        mark = " <- gốc" if s["is_baseline"] else ""
        if s["config"] == best:
            mark += " <- đề xuất"
        print(f"{s['config']:<32}|{fmt(s['recall'], '7.4f')} |{fmt(s['precision'], '6.3f')} |"
              f"{fmt(s['f1'], '6.4f')} |{fmt(s.get('f1_iou30', float('nan')), '6.3f')} |"
              f"{fmt(s.get('f1_iou50', float('nan')), '6.3f')} |"
              f"{fmt(s['false_blobs_per_img'], '8.3f')} |{fmt(s['recall_weak'], '9.3f')} |"
              f"{fmt(s['recall_focus'], '6.2f')} |{s['pairs_all_missed']:>7} | "
              f"{'ĐẠT' if s.get('kpi_pass') else '-'}{mark}")


def plot(summary: List[Dict[str, object]], path: Path, best: Optional[str]) -> None:
    conds = sorted({s["condition"] for s in summary})
    fig, axes = plt.subplots(1, len(conds), figsize=(8 * len(conds), 6), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        items = [s for s in summary if s["condition"] == cond]
        x = [float(s["false_blobs_per_img"]) for s in items]
        y = [float(s["recall"]) for s in items]
        c = [float(s["f1"]) for s in items]
        sc = ax.scatter(x, y, c=c, cmap="viridis", s=40, edgecolor="black", linewidth=0.3)
        for s in items:
            if s["is_baseline"] or s["config"] == best:
                ax.annotate("gốc" if s["is_baseline"] else "đề xuất",
                            (float(s["false_blobs_per_img"]), float(s["recall"])),
                            textcoords="offset points", xytext=(6, 6), fontsize=9,
                            color="red" if s["is_baseline"] else "green")
        ax.axhline(KPI["recall"], color="gray", linestyle=":", linewidth=1)
        ax.axvline(KPI["false_blobs"], color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Đốm giả / ảnh (thấp hơn tốt hơn)")
        ax.set_ylabel("Recall (cao hơn tốt hơn)")
        ax.set_title({"clean": "Ảnh gốc", "resample": "Ảnh đã nội suy"}.get(cond, cond)
                     + " — mỗi điểm là 1 cấu hình (màu = F1)")
        ax.grid(linestyle="--", alpha=0.35)
        fig.colorbar(sc, ax=ax, label="F1")
    fig.suptitle("Dò tham số nhị phân hóa (đường chấm = chỉ tiêu KPI)", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dò tham số nhị phân hóa + F1@IoU (P0.3, P1.1, P1.2).")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--per-class", type=int, default=0)
    p.add_argument("--pairs", nargs="+", default=None, help="Chỉ chạy các pair_id này.")
    p.add_argument("--conditions", nargs="+", default=["clean", "resample"],
                   choices=["clean", "resample"])
    p.add_argument("--floors", type=int, nargs="+", default=[15, 20, 25, 30])
    p.add_argument("--opens", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--hysteresis", type=float, nargs="+", default=[0.0, 0.5])
    p.add_argument("--min-areas", type=int, nargs="+", default=[10, 20])
    p.add_argument("--inspect", nargs="*", default=None,
                   help="Lưu ảnh chẩn đoán cho các pair_id (không ghi gì = 3 cặp trọng điểm).")
    p.add_argument("--top", type=int, default=8, help="Số cấu hình in ra mỗi bảng.")
    p.add_argument("--no-per-pair", action="store_true", help="Không ghi CSV từng cặp (file lớn).")
    p.add_argument("--warp-angle", type=float, default=3.0)
    p.add_argument("--warp-shift", type=float, default=40.0)
    p.add_argument("--warp-scale", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("experiments"))
    p.add_argument("--resume", action="store_true",
                   help="Tiếp tục từ checkpoint, bỏ qua các cặp đã chạy xong.")
    p.add_argument("--checkpoint-every", type=int, default=20,
                   help="Ghi checkpoint sau mỗi N cặp (0 = tắt, mặc định 20).")
    p.add_argument("--state-file", type=Path, default=None,
                   help="File checkpoint (mặc định: <output-dir>/detection_params_state.pkl).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    configs: List[Config] = sorted(set(itertools.product(
        args.floors, args.opens, [float(h) for h in args.hysteresis], args.min_areas)))
    if BASELINE not in configs:
        configs.append(BASELINE)
    inspect = set(FOCUS_PAIRS if args.inspect == [] else (args.inspect or []))

    pairs = find_pairs(str(args.data_dir))
    if args.pairs:
        wanted = set(args.pairs)
        pairs = [p for p in pairs if p["pair_id"] in wanted]
    if args.per_class > 0:
        counter: Dict[str, int] = defaultdict(int)
        picked = []
        for p in pairs:
            if counter[p["class_name"]] < args.per_class or p["pair_id"] in FOCUS_PAIRS:
                picked.append(p)
                counter[p["class_name"]] += 1
        pairs = picked
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(f"[LỖI] Không có cặp ảnh nào trong '{args.data_dir}'.")
    missing = [p for p in pairs if not Path(p["template_path"]).is_file()
               or not Path(p["test_path"]).is_file()]
    if missing:
        raise SystemExit(f"[LỖI] Thiếu ảnh gốc cho {len(missing)}/{len(pairs)} cặp, ví dụ "
                         f"{missing[0]['test_path']}. Chép dataset vào data/PKU-Market-PCB/raw/.")

    print(f"{len(pairs)} cặp x {len(args.conditions)} điều kiện x {len(configs)} cấu hình "
          f"(gốc: {config_name(BASELINE)})")

    state_path = args.state_file or (args.output_dir / "detection_params_state.pkl")
    signature = {
        "data_dir": str(args.data_dir),
        "conditions": list(args.conditions),
        "configs": [config_name(c) for c in configs],
        "warp": [args.warp_angle, args.warp_shift, args.warp_scale, args.seed],
        "pairs": [p["pair_id"] for p in pairs],
    }

    rows: List[Dict[str, object]] = []
    diag: List[Dict[str, object]] = []
    done = set()
    if args.resume and state_path.is_file():
        try:
            with state_path.open("rb") as fh:
                state = pickle.load(fh)
        except (OSError, pickle.UnpicklingError, EOFError) as exc:
            print(f"[CẢNH BÁO] Không đọc được checkpoint {state_path}: {exc}. Chạy lại từ đầu.")
            state = None
        if state is not None:
            if state.get("signature") == signature:
                rows = state["rows"]
                diag = state["diag"]
                done = set(state["done"])
                print(f"[RESUME] {state_path.name}: đã xong {len(done)}/{len(pairs)} cặp, "
                      f"tiếp tục {len(pairs) - len(done)} cặp còn lại.")
            else:
                print("[CẢNH BÁO] Checkpoint thuộc cấu hình khác (data/conditions/configs/"
                      "warp/pairs) -> bỏ qua, chạy lại từ đầu.")
    elif args.resume:
        print(f"[RESUME] Chưa có checkpoint {state_path} -> chạy từ đầu.")

    def save_state() -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_name(state_path.name + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump({"signature": signature, "rows": rows, "diag": diag,
                         "done": sorted(done)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(state_path)

    pending = [p for p in pairs if p["pair_id"] not in done]
    base = len(pairs) - len(pending)
    for i, pair in enumerate(pending, start=1):
        gt = read_labels(pair.get("label_path"))
        notes = []
        for cond in args.conditions:
            try:
                template, aligned, diff, valid, info = prepare(pair, cond, args)
            except (ValueError, OSError, cv2.error) as exc:
                notes.append(f"{cond}: lỗi {exc}")
                continue
            masks, boxes_by = {}, {}
            for cfg in configs:
                mask, boxes = detect(diff, valid, cfg)
                counts = detection_counts(boxes, gt)
                rows.append({"pair": pair["pair_id"], "class_name": pair.get("class_name", ""),
                             "condition": cond, "config": config_name(cfg),
                             "floor": cfg[0], "open": cfg[1], "hysteresis": cfg[2],
                             "min_area": cfg[3], "is_baseline": cfg == BASELINE,
                             **{k: v for k, v in info.items() if k != "align_error"}, **counts})
                if cfg == BASELINE:
                    for d in diagnose_missed(diff, valid, gt, cfg, boxes):
                        diag.append({"pair": pair["pair_id"], "class_name": pair.get("class_name", ""),
                                     "condition": cond, **d})
                    masks[f"gốc {config_name(cfg)}"] = mask
                    boxes_by[f"gốc {config_name(cfg)}"] = boxes
            base_row = next(r for r in reversed(rows) if r["condition"] == cond and r["is_baseline"]
                            and r["pair"] == pair["pair_id"])
            notes.append(f"{cond}: {base_row['gt_hit']}/{base_row['n_gt']} GT, "
                         f"{base_row['false_blobs']} giả"
                         + ("" if info["align_ok"] else " [căn chỉnh hỏng]"))
            if pair["pair_id"] in inspect:
                # cấu hình có recall cao nhất trên cặp này (ưu tiên ít đốm giả)
                cand = [r for r in rows if r["pair"] == pair["pair_id"] and r["condition"] == cond]
                top = max(cand, key=lambda r: (int(r["gt_hit"]), -int(r["false_blobs"])))
                cfg_top = (int(top["floor"]), int(top["open"]), float(top["hysteresis"]), int(top["min_area"]))
                m_top, b_top = detect(diff, valid, cfg_top)
                masks[f"tốt nhất {config_name(cfg_top)}"] = m_top
                boxes_by[f"tốt nhất {config_name(cfg_top)}"] = b_top
                save_inspect(args.output_dir / "inspect" / f"{pair['pair_id']}_{cond}.jpg",
                             template, aligned, diff, gt, masks, boxes_by)
        print(f"[{base + i}/{len(pairs)}] {pair['pair_id']:<26} " + " | ".join(notes))
        done.add(pair["pair_id"])
        if args.checkpoint_every > 0 and i % args.checkpoint_every == 0:
            save_state()
            print(f"    [checkpoint] {len(done)}/{len(pairs)} cặp -> {state_path.name}")

    if not rows:
        raise SystemExit("Không xử lý được cặp nào.")

    summary = summarize(rows)
    best = pick_best(summary, args.conditions)
    for cond in args.conditions:
        print_table(summary, cond, best, args.top)

    print("\n=== CHẨN ĐOÁN LỖI BỊ SÓT Ở CẤU HÌNH GỐC (bị mất ở bước nào) ===")
    for cond in args.conditions:
        d = [x for x in diag if x["condition"] == cond]
        if not d:
            print(f"{cond}: không có lỗi nào bị sót.")
            continue
        stages = Counter(x["stage"] for x in d)
        by_cls = Counter(x["class_name"] for x in d)
        print(f"{cond}: {len(d)} lỗi bị sót — " + ", ".join(f"{k}: {v}" for k, v in stages.most_common())
              + " | theo lớp: " + ", ".join(f"{k} {v}" for k, v in by_cls.most_common()))
        near = [x for x in d if x["stage"] == "ngưỡng"]
        if near:
            gaps = [x["threshold"] - x["diff_max"] for x in near]
            print(f"   'ngưỡng': vi sai lớn nhất trong bbox thấp hơn ngưỡng trung vị "
                  f"{float(np.median(gaps)):.0f} mức xám (ngưỡng trung vị {float(np.median([x['threshold'] for x in near])):.0f})")

    print("\n=== KẾT LUẬN ===")
    by = {(s["condition"], s["config"]): s for s in summary}
    if best is None:
        print("Không cấu hình nào giữ được precision >= cấu hình gốc ở mọi điều kiện.")
    else:
        for cond in args.conditions:
            b = by[(cond, config_name(BASELINE))]
            n = by[(cond, best)]
            print(f"{cond:<9}: gốc F1 {float(b['f1']):.4f} (recall {float(b['recall']):.4f}, "
                  f"Spur+SpCu {fmt(float(b['recall_weak']), '.3f')}, giả {float(b['false_blobs_per_img']):.3f}/ảnh)"
                  f"  ->  {best}: F1 {float(n['f1']):.4f} (recall {float(n['recall']):.4f}, "
                  f"Spur+SpCu {fmt(float(n['recall_weak']), '.3f')}, giả {float(n['false_blobs_per_img']):.3f}/ảnh)")
        if best == config_name(BASELINE):
            print("-> Cấu hình gốc vẫn là tốt nhất trong lưới đã thử.")
        else:
            floor, open_k, hyst, area = next((s["floor"], s["open"], s["hysteresis"], s["min_area"])
                                             for s in summary if s["config"] == best)
            print("-> Để áp dụng, sửa trong module_preprocess.py: "
                  f"DEFAULT_THRESHOLD_FLOOR = {floor}, DEFAULT_OPEN_KSIZE = {open_k}, "
                  f"DEFAULT_HYSTERESIS_LOW = {hyst if hyst else None}, DEFAULT_MIN_AREA = {area}")
            print("   (chỉ áp dụng sau khi đã chạy đủ 693 cặp và cấu hình này vẫn đứng đầu).")
    f50 = [float(s.get("f1_iou50", float("nan"))) for s in summary if s["is_baseline"]]
    if f50 and all(v < 0.5 for v in f50 if not math.isnan(v)):
        print("[P1.2] F1@IoU0.5 của cấu hình gốc rất thấp: bbox pipeline chỉ khoanh LÕI vi sai "
              "nên nhỏ hơn nhãn GT nhiều. Khi báo cáo phải ghi rõ tiêu chí ghép bbox.")

    out = args.output_dir
    write_csv(summary, out / "detection_params_summary.csv")
    if not args.no_per_pair:
        write_csv(rows, out / "detection_params_per_pair.csv")
    write_csv(diag, out / "detection_missed_diagnosis.csv")
    plot(summary, out / "charts" / "detection_params.png", best)
    if state_path.is_file():
        state_path.unlink()  # chạy xong -> không cần checkpoint nữa
    print(f"\nCSV tổng hợp  : {out / 'detection_params_summary.csv'}")
    if not args.no_per_pair:
        print(f"CSV từng cặp  : {out / 'detection_params_per_pair.csv'}")
    print(f"CSV chẩn đoán : {out / 'detection_missed_diagnosis.csv'}")
    print(f"Biểu đồ       : {out / 'charts' / 'detection_params.png'}")
    if inspect:
        print(f"Ảnh chẩn đoán : {out / 'inspect'}")


if __name__ == "__main__":
    main()
