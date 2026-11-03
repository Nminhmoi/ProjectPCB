#!/usr/bin/env python3
"""Đánh giá định lượng pipeline trên DeepPCB.

Cấu trúc dữ liệu (theo repo https://github.com/tangsanli5201/DeepPCB):
    PCBData/group00041/group0004110/00041000_temp.jpg
    PCBData/group00041/group0004110/00041000_test.jpg
    PCBData/group00041/group0004110/00041000.txt

Mỗi dòng annotation: x1,y1,x2,y2,type
    type: 1=open, 2=short, 3=mousebite, 4=spur, 5=copper, 6=pin-hole

Metric: precision / recall / F1 @ IoU >= 0.33 (giống benchmark DeepPCB),
ở mức định vị (bỏ qua nhãn lớp vì pipeline hiện chưa phân loại).

Ví dụ:
    python tools/evaluate_deeppcb.py --data "D:\\DeepPCB\\PCBData" --limit 50
    python tools/evaluate_deeppcb.py --data "D:\\DeepPCB\\PCBData" --save-dir results/eval
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module_detection import extract_bounding_boxes  # noqa: E402
from module_preprocess import align_images, detect_difference  # noqa: E402

TYPE_NAMES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}


def parse_annotation(path: Path) -> list[tuple[float, float, float, float, int]]:
    """Đọc file annotation DeepPCB, trả về list (x1, y1, x2, y2, type)."""
    boxes = []
    for raw in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in parts[:4])
        # DeepPCB dùng 1..6; nếu file chỉ có 4 cột thì coi là không rõ lớp
        label = int(float(parts[4])) if len(parts) >= 5 else 0
        boxes.append((x1, y1, x2, y2, label))
    return boxes


def find_annotation(test_path: Path) -> Path | None:
    base = test_path.name[: -len("_test.jpg")] if test_path.name.endswith("_test.jpg") else test_path.stem
    candidates = [
        test_path.with_name(f"{base}.txt"),
        test_path.with_name(f"{base}_test.txt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def xywh_to_xyxy(box) -> tuple[float, float, float, float]:
    x, y, w, h = box[:4]
    return float(x), float(y), float(x + w), float(y + h)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(
    predictions: list[tuple[float, float, float, float]],
    ground_truths: list[tuple[float, float, float, float, int]],
    iou_threshold: float,
) -> tuple[list[int], set[int], set[int]]:
    """Ghép 1-1 tham lam theo IoU giảm dần.

    Trả về (matched_gt_indexes, matched_pred_indexes, unmatched_pred_indexes).
    """
    pairs = []
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truths):
            value = iou(pred, gt[:4])
            if value >= iou_threshold:
                pairs.append((value, pi, gi))
    pairs.sort(reverse=True)

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    ordered_gt: list[int] = []
    for _, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        ordered_gt.append(gi)

    unmatched_pred = set(range(len(predictions))) - matched_pred
    return ordered_gt, matched_pred, unmatched_pred


def draw_overlay(img, predictions, matched_pred, ground_truths, matched_gt):
    canvas = img.copy()
    for gi, gt in enumerate(ground_truths):
        x1, y1, x2, y2, label = gt
        color = (0, 255, 0) if gi in matched_gt else (255, 0, 0)  # xanh = bắt được, xanh dương = bỏ sót
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        name = TYPE_NAMES.get(label, str(label))
        cv2.putText(canvas, f"GT:{name}", (int(x1), max(10, int(y1) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    for pi, pred in enumerate(predictions):
        x1, y1, x2, y2 = (int(v) for v in pred)
        color = (0, 0, 255) if pi in matched_pred else (0, 165, 255)  # đỏ = TP, cam = FP
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
    return canvas


def evaluate(data_dir: Path, iou_threshold: float, min_area: int, limit: int | None, save_dir: Path | None):
    test_paths = sorted(data_dir.rglob("*_test.jpg"))
    if limit:
        test_paths = test_paths[:limit]
    if not test_paths:
        raise SystemExit(f"Không tìm thấy ảnh *_test.jpg trong: {data_dir}")

    total_gt = total_pred = total_tp = total_fp = total_fn = 0
    per_type_total = {k: 0 for k in TYPE_NAMES}
    per_type_hit = {k: 0 for k in TYPE_NAMES}
    failed_images: list[tuple[Path, str]] = []
    times: list[float] = []

    for index, test_path in enumerate(test_paths, start=1):
        ann_path = find_annotation(test_path)
        if ann_path is None:
            print(f"[SKIP] Không có annotation cho {test_path.name}")
            continue
        temp_path = test_path.with_name(test_path.name.replace("_test.jpg", "_temp.jpg"))
        if not temp_path.exists():
            print(f"[SKIP] Không có ảnh template cho {test_path.name}")
            continue

        img_test = cv2.imread(str(test_path), cv2.IMREAD_COLOR)
        img_temp = cv2.imread(str(temp_path), cv2.IMREAD_COLOR)
        ground_truths = parse_annotation(ann_path)

        start = time.perf_counter()
        try:
            aligned = align_images(img_temp, img_test)
            diff_mask = detect_difference(img_temp, aligned)
            raw_boxes = extract_bounding_boxes(diff_mask, min_area=min_area)
            predictions = [xywh_to_xyxy(b) for b in raw_boxes]
        except Exception as exc:  # noqa: BLE001 - ghi nhận lỗi từng ảnh, không dừng cả batch
            failed_images.append((test_path, repr(exc)))
            predictions = []
            aligned = img_test
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        matched_gt, matched_pred, unmatched_pred = match_boxes(predictions, ground_truths, iou_threshold)

        tp = len(matched_gt)
        fp = len(unmatched_pred)
        fn = len(ground_truths) - tp
        total_gt += len(ground_truths)
        total_pred += len(predictions)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        for gi, gt in enumerate(ground_truths):
            label = gt[4]
            if label in per_type_total:
                per_type_total[label] += 1
                if gi in matched_gt:
                    per_type_hit[label] += 1

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            overlay = draw_overlay(aligned, predictions, matched_pred, ground_truths, set(matched_gt))
            cv2.imwrite(str(save_dir / f"{test_path.stem}_eval.jpg"), overlay)

        if index % 25 == 0:
            print(f"... đã xử lý {index}/{len(test_paths)} ảnh")

    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("\n" + "=" * 62)
    print("KẾT QUẢ ĐÁNH GIÁ DeepPCB")
    print("=" * 62)
    print(f"Ảnh đánh giá            : {len(times)}")
    print(f"Ảnh lỗi pipeline        : {len(failed_images)}")
    print(f"Ground truth            : {total_gt}")
    print(f"Dự đoán                 : {total_pred}")
    print(f"TP / FP / FN            : {total_tp} / {total_fp} / {total_fn}")
    print(f"IoU threshold           : {iou_threshold}")
    print(f"Precision               : {precision:.4f}")
    print(f"Recall                  : {recall:.4f}")
    print(f"F1-score                : {f1:.4f}")
    if times:
        print(f"Thời gian / ảnh (TB)    : {1000 * sum(times) / len(times):.1f} ms")

    print("\nRecall theo loại lỗi (chỉ tính ảnh có GT):")
    for label, name in TYPE_NAMES.items():
        total = per_type_total[label]
        hit = per_type_hit[label]
        rate = hit / total if total else float("nan")
        print(f"  {name:<10}: {hit:>5}/{total:<5}  recall={rate:.3f}" if total else f"  {name:<10}: không có GT")

    if failed_images:
        print("\nDanh sách ảnh lỗi:")
        for path, error in failed_images[:20]:
            print(f"  {path.name}: {error}")
        if len(failed_images) > 20:
            print(f"  ... và {len(failed_images) - 20} ảnh khác")

    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Đánh giá pipeline PCB trên DeepPCB")
    parser.add_argument("--data", required=True, type=Path, help="Thư mục gốc PCBData (tìm đệ quy *_test.jpg)")
    parser.add_argument("--iou", type=float, default=0.33, help="Ngưỡng IoU để tính TP (mặc định 0.33)")
    parser.add_argument("--min-area", type=int, default=20, help="min_area truyền vào extract_bounding_boxes")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N ảnh đầu (để tune nhanh)")
    parser.add_argument("--save-dir", type=Path, default=None, help="Thư mục lưu ảnh overlay TP/FP/FN")
    args = parser.parse_args()

    evaluate(args.data, args.iou, args.min_area, args.limit, args.save_dir)


if __name__ == "__main__":
    main()
