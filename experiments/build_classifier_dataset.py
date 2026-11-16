"""Sinh dataset phân loại từ 693 cặp: mỗi blob phát hiện -> 1 dòng đặc trưng + nhãn.

Nhãn lấy từ `class_name` của cặp khi blob khớp một GT box (tâm blob nằm trong GT
hoặc IoU >= --min-iou); blob không khớp bị gán `background`.

    python experiments/build_classifier_dataset.py --limit 3
    python experiments/build_classifier_dataset.py            # full 693 cặp
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_utils import find_pairs
from evaluation import read_labels
from main import process_pair
from module_classification import (BACKGROUND_CLASS, CLASS_NAMES,
                                   FEATURE_NAMES, extract_features)

Box = Tuple[int, int, int, int]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "classifier_dataset.csv"


def center_inside(box: Box, gt: Box, tol: float = 2.0) -> bool:
    x, y, w, h = box
    gx, gy, gw, gh = gt
    cx, cy = x + w / 2.0, y + h / 2.0
    return (gx - tol <= cx <= gx + gw + tol) and (gy - tol <= cy <= gy + gh + tol)


def match_box(box: Box, gt_boxes, min_iou: float):
    """Trả (index GT, IoU) tốt nhất; ưu tiên tâm blob nằm trong GT."""
    best_idx, best_iou, best_inside = -1, 0.0, False
    for idx, gt in enumerate(gt_boxes):
        value = iou_value(box, gt)
        inside = center_inside(box, gt)
        if (inside, value) > (best_inside, best_iou):
            best_idx, best_iou, best_inside = idx, value, inside
    if best_idx >= 0 and (best_inside or best_iou >= min_iou):
        return best_idx, best_iou
    return -1, best_iou


def iou_value(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N cặp đầu.")
    parser.add_argument("--per-class", type=int, default=None,
                        help="Chỉ chạy N cặp mỗi lớp (để test nhanh).")
    parser.add_argument("--min-iou", type=float, default=0.1,
                        help="Ngưỡng IoU tối thiểu để coi blob khớp GT.")
    parser.add_argument("--include-failed-align", action="store_true",
                        help="Giữ cả cặp căn chỉnh thất bại (mặc định: bỏ).")
    parser.add_argument("--save-outputs", action="store_true",
                        help="Ghi crops/ + results/ như main.py (chậm, tốn đĩa).")
    return parser.parse_args()


def select_pairs(pairs, args):
    if args.per_class:
        count, kept = Counter(), []
        for pair in pairs:
            name = str(pair["class_name"])
            if count[name] >= args.per_class:
                continue
            count[name] += 1
            kept.append(pair)
        pairs = kept
    if args.limit:
        pairs = pairs[: args.limit]
    return pairs


def main() -> int:
    args = parse_args()
    pairs = select_pairs(find_pairs(args.data_dir), args)
    if not pairs:
        print("[LỖI] Không tìm thấy cặp nào trong manifest.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = (["pair_id", "class_name", "label", "matched", "iou", "x", "y", "w", "h"]
               + FEATURE_NAMES)

    label_counts = Counter()
    n_box = n_matched = n_skipped = 0
    start = time.time()

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)

        for order, pair in enumerate(pairs, start=1):
            pair_id = str(pair["pair_id"])
            class_name = str(pair["class_name"])
            extra = {}
            boxes, _, _, _, stats = process_pair(
                pair, extra_out=extra, save_outputs=args.save_outputs)

            if not stats.get("success") and not args.include_failed_align:
                n_skipped += 1
                continue

            gt_boxes = read_labels(pair.get("label_path"))
            features = extract_features(extra["gray"], extra["mask"], boxes)

            for box, row in zip(boxes, features):
                idx, value = match_box(box, gt_boxes, args.min_iou)
                label = class_name if idx >= 0 else BACKGROUND_CLASS
                n_box += 1
                n_matched += int(idx >= 0)
                label_counts[label] += 1
                x, y, w, h = (int(v) for v in box)
                writer.writerow([pair_id, class_name, label, int(idx >= 0),
                                 f"{value:.4f}", x, y, w, h]
                                + [f"{v:.6g}" for v in row])

            if order % 25 == 0 or order == len(pairs):
                rate = n_matched / n_box if n_box else 0.0
                print(f"[{order}/{len(pairs)}] {pair_id}: {len(boxes)} blob, "
                      f"khớp GT {rate * 100:.1f}%, "
                      f"{time.time() - start:.0f}s", flush=True)

    total = sum(label_counts.values())
    print(f"\nDataset: {args.output}")
    print(f"Cặp dùng được   : {len(pairs) - n_skipped}/{len(pairs)} (bỏ {n_skipped} cặp căn chỉnh lỗi)")
    print(f"Blob            : {n_box}")
    print(f"Khớp GT         : {n_matched} ({n_matched / n_box * 100:.1f}%)" if n_box else "Blob            : 0")
    print(f"Báo động giả    : {label_counts[BACKGROUND_CLASS]} "
          f"({label_counts[BACKGROUND_CLASS] / total * 100:.1f}%)" if total else "")
    print("Phân bố nhãn    :")
    for name in CLASS_NAMES:
        if label_counts[name]:
            print(f"  {name:<16} {label_counts[name]:6d}  ({label_counts[name] / total * 100:5.1f}%)")
    print(f"Thời gian       : {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
