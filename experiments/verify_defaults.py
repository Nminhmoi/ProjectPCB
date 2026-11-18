"""Kiem chung `DEFAULT_*` trong module_preprocess bang cach chay lai dung cau hinh do.

Chay 693 cap x 2 dieu kien voi cau hinh lay tu `DEFAULT_*` hien tai, roi so tung
chi so voi `experiments/detection_params_summary.csv` da luu. Sai lech phai = 0
(exit 0); lech bat ky chi so nao -> in ro va exit 1.

    python experiments/verify_defaults.py --limit 2      # thu code
    python experiments/verify_defaults.py                # full, ~10 phut
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from dataset_utils import find_pairs  # noqa: E402
from evaluation import detection_counts, read_labels  # noqa: E402
from exp_detection_params import (BASELINE, config_name, detect, prepare,  # noqa: E402
                                  summarize)
from module_preprocess import (DEFAULT_HYSTERESIS_LOW, DEFAULT_MIN_AREA,  # noqa: E402
                               DEFAULT_OPEN_KSIZE, DEFAULT_THRESHOLD_FLOOR)

# Dung sai cho phép: CSV làm tròn 5 chữ số thập phân.
TOL = 2e-5
FLOAT_FIELDS = ("recall", "precision", "f1", "false_blobs_per_img",
                "recall_weak", "f1_iou30", "f1_iou50")
INT_FIELDS = ("pairs", "n_gt", "n_pred", "pairs_all_missed")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kiem chung DEFAULT_* bang luoi tham so da luu.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--summary-csv", type=Path,
                   default=SCRIPT_DIR / "detection_params_summary.csv")
    p.add_argument("--conditions", nargs="+", default=["clean", "resample"],
                   choices=["clean", "resample"])
    p.add_argument("--limit", type=int, default=0, help="Chi chay N cap dau (thu code).")
    p.add_argument("--per-class", type=int, default=0)
    p.add_argument("--warp-angle", type=float, default=3.0)
    p.add_argument("--warp-shift", type=float, default=40.0)
    p.add_argument("--warp-scale", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def load_expected(path: Path, name: str):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["config"] == name]
    return {r["condition"]: r for r in rows}


def compare(expected: dict, actual: dict):
    """Tra ve danh sach (field, expected, actual, lech) cho cac chi so khong khop."""
    bad = []
    for field in FLOAT_FIELDS:
        if field not in expected or field not in actual:
            continue
        want, got = float(expected[field]), float(actual[field])
        if abs(want - got) > TOL:
            bad.append((field, want, got, got - want))
    for field in INT_FIELDS:
        if field not in expected:
            continue
        want, got = int(float(expected[field])), int(actual[field])
        if want != got:
            bad.append((field, want, got, got - want))
    return bad


def pick_pairs(pairs, args):
    if args.per_class > 0:
        kept, count = [], {}
        for pair in pairs:
            name = str(pair["class_name"])
            if count.get(name, 0) < args.per_class:
                kept.append(pair)
                count[name] = count.get(name, 0) + 1
        pairs = kept
    if args.limit > 0:
        pairs = pairs[: args.limit]
    return pairs


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = (DEFAULT_THRESHOLD_FLOOR, DEFAULT_OPEN_KSIZE,
           float(DEFAULT_HYSTERESIS_LOW or 0.0), DEFAULT_MIN_AREA)
    name = config_name(cfg)

    expected = load_expected(args.summary_csv, name)
    if not expected:
        print(f"[LOI] Khong thay cau hinh '{name}' trong {args.summary_csv}.")
        print("      DEFAULT_* da doi ra ngoai luoi da chay -> phai chay lai exp_detection_params.")
        return 1
    if tuple(BASELINE) != cfg:
        print(f"[LOI] BASELINE cua exp_detection_params = {BASELINE}, DEFAULT_* = {cfg}. "
              f"Chay lai script sau khi dong bo module_preprocess.")
        return 1

    pairs = pick_pairs(find_pairs(str(args.data_dir)), args)
    if not pairs:
        print(f"[LOI] Khong co cap anh nao trong '{args.data_dir}'.")
        return 1
    full = not args.limit and not args.per_class

    print(f"Cau hinh kiem chung : {name} (floor={cfg[0]}, open={cfg[1]}, "
          f"hysteresis={cfg[2]:g}, min_area={cfg[3]})")
    print(f"Du lieu             : {len(pairs)} cap x {len(args.conditions)} dieu kien")
    print(f"Doi chieu voi       : {args.summary_csv}")

    ns = argparse.Namespace(seed=args.seed, warp_angle=args.warp_angle,
                            warp_shift=args.warp_shift, warp_scale=args.warp_scale)
    start = time.time()
    mismatches = 0

    for cond in args.conditions:
        rows = []
        for order, pair in enumerate(pairs, start=1):
            template, aligned, diff, valid, info = prepare(pair, cond, ns)
            mask, boxes = detect(diff, valid, cfg)
            counters = detection_counts(boxes, read_labels(pair.get("label_path")))
            rows.append({"pair": pair["pair_id"], "class_name": pair.get("class_name", ""),
                         "condition": cond, "config": name, "floor": cfg[0], "open": cfg[1],
                         "hysteresis": cfg[2], "min_area": cfg[3],
                         "is_baseline": cfg == BASELINE,
                         **{k: v for k, v in info.items() if k != "align_error"},
                         **counters})
            if order % 50 == 0 or order == len(pairs):
                print(f"  [{cond}] {order}/{len(pairs)} cap, {time.time() - start:.0f}s", flush=True)

        actual = summarize(rows)
        if len(actual) != 1:
            print(f"[LOI] summarize tra {len(actual)} dong cho dieu kien '{cond}'.")
            return 1
        actual = actual[0]
        bad = compare(expected[cond], actual)
        if not full:
            print(f"\n=== {cond}: CHI THU CODE ({len(pairs)} cap) - khong so sanh voi CSV 693 cap ===")
            print(f"  F1={actual['f1']:.5f} precision={actual['precision']:.5f} "
                  f"recall={actual['recall']:.5f} da chay duoc.")
            continue
        status = "KHOP" if not bad else f"LECH {len(bad)} chi so"
        print(f"\n=== {cond}: {status} ===")
        header = f"  {'chi so':<20}{'CSV':>14}{'chay lai':>14}{'lech':>12}"
        print(header)
        for field in ("pairs", "n_gt", "n_pred", "recall", "precision", "f1",
                      "false_blobs_per_img", "pairs_all_missed", "recall_weak") + FLOAT_FIELDS[-2:]:
            want = expected[cond].get(field)
            got = actual.get(field)
            if want is None or got is None:
                continue
            if field in INT_FIELDS:
                mark = "" if int(float(want)) == int(got) else "  <-- LECH"
                print(f"  {field:<20}{int(float(want)):>14}{int(got):>14}{'':>12}{mark}")
            else:
                delta = float(got) - float(want)
                mark = "" if abs(delta) <= TOL else "  <-- LECH"
                print(f"  {field:<20}{float(want):>14.5f}{float(got):>14.5f}{delta:>12.5f}{mark}")
        for field, want, got, delta in bad:
            print(f"  [LECH] {field}: CSV={want} -> chay lai={got} (lech {delta:+g})")
        mismatches += len(bad)

    print(f"\nTong thoi gian: {time.time() - start:.0f}s")
    if not full:
        print(f"KET QUA: PASS (chi thu code, {len(pairs)} cap) - chay khong tham so de doi chieu day du.")
        return 0
    if mismatches:
        print(f"KET QUA: FAIL - {mismatches} chi so khong khop.")
        return 1
    print(f"KET QUA: PASS - {name} tai lap dung so trong CSV o ca "
          f"{len(args.conditions)} dieu kien.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
