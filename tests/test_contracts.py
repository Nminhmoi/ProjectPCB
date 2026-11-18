"""Kiem thu hop dong API + toan ven du lieu (viec #11, TASKS_NOW.md muc 6).

Chi assert nhung gi chac chan dung, khong phu thuoc du lieu that. Chay ~2 giay:

    python tests/test_contracts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_utils import find_pairs  # noqa: E402
from evaluation import iou, prf_from_counts  # noqa: E402
from module_detection import extract_bounding_boxes  # noqa: E402
from module_preprocess import binarize_difference  # noqa: E402

PASSED = 0


def check(name: str, fn) -> None:
    global PASSED
    try:
        fn()
    except AssertionError as exc:
        print(f"FAIL  {name}: {exc}")
        raise SystemExit(1)
    print(f"PASS  {name}")
    PASSED += 1


def test_binarize_shape_dtype():
    rng = np.random.default_rng(0)
    diff = rng.integers(0, 256, size=(60, 80), dtype=np.uint8)
    mask = binarize_difference(diff, border=0)
    assert mask.shape == diff.shape, f"shape {mask.shape} != {diff.shape}"
    assert mask.dtype == np.uint8, f"dtype {mask.dtype}"
    values = set(np.unique(mask).tolist())
    assert values <= {0, 255}, f"gia tri la {sorted(values)}"


def test_binarize_respects_valid():
    diff = np.full((200, 200), 255, np.uint8)   # ngoài vùng hợp lệ: sáng hết
    diff[50:150, 50:150] = 0
    diff[80:120, 80:120] = 200                  # blob thật nằm trong vùng hợp lệ
    valid = np.zeros((200, 200), np.uint8)
    valid[50:150, 50:150] = 255
    mask = binarize_difference(diff, border=0, valid=valid)
    assert mask[valid == 0].max(initial=0) == 0, "con pixel ngoai vung hop le"
    assert mask[80:120, 80:120].any(), "blob trong vung hop le bi xoa mat"


def test_iou():
    box = (10, 10, 20, 20)
    assert abs(iou(box, box) - 1.0) < 1e-9, iou(box, box)
    assert iou(box, (100, 100, 20, 20)) == 0.0, iou(box, (100, 100, 20, 20))


def test_prf_from_counts():
    rows = [{"n_gt": 10, "gt_hit": 8, "n_pred": 9, "pred_hit": 8, "false_blobs": 1}]
    out = prf_from_counts(rows)
    assert abs(out["recall"] - 0.8) < 1e-9, out["recall"]
    assert abs(out["precision"] - 8 / 9) < 1e-9, out["precision"]
    assert abs(out["f1"] - 2 * 0.8 * (8 / 9) / (0.8 + 8 / 9)) < 1e-9, out["f1"]
    assert abs(out["false_blobs_per_img"] - 1.0) < 1e-9, out["false_blobs_per_img"]


def test_extract_bounding_boxes_min_area():
    mask = np.zeros((40, 40), np.uint8)
    mask[5:8, 5:8] = 255          # blob 3×3 = 9 px
    kept = extract_bounding_boxes(mask, min_area=5)
    dropped = extract_bounding_boxes(mask, min_area=10)
    assert len(kept) == 1, f"min_area=5 phai giu blob, nhan duoc {len(kept)}"
    assert len(dropped) == 0, f"min_area=10 phai loai blob 9 px, nhan duoc {len(dropped)}"


def test_find_pairs_manifest():
    pairs = find_pairs(str(PROJECT_ROOT / "data"))
    if not pairs:
        print("      SKIP: chua co data/PKU-Market-PCB/pairs_manifest.csv")
        raise SkipTest
    missing = [p["pair_id"] for p in pairs if not Path(str(p["template_path"])).is_file()]
    if missing:
        print(f"      SKIP: thieu anh goc cho {len(missing)}/{len(pairs)} cap")
        raise SkipTest
    assert len(pairs) == 693, f"manifest co {len(pairs)} cap, cho doi 693"


class SkipTest(Exception):
    pass


def main() -> int:
    cases = [
        ("binarize_difference: shape/dtype/gia tri", test_binarize_shape_dtype),
        ("binarize_difference: ton trong mask valid", test_binarize_respects_valid),
        ("iou: box voi chinh no va box roi nhau", test_iou),
        ("prf_from_counts: so hoc", test_prf_from_counts),
        ("extract_bounding_boxes: min_area", test_extract_bounding_boxes_min_area),
        ("dataset_utils.find_pairs: manifest", test_find_pairs_manifest),
    ]
    skipped = 0
    for name, fn in cases:
        try:
            check(name, fn)
        except SkipTest:
            print(f"SKIP  {name}")
            skipped += 1
    print(f"\n{PASSED} PASS, {skipped} SKIP, 0 FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
