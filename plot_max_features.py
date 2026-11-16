"""
Vẽ lại experiments/charts/max_features_study.png từ CSV (việc #9), KHÔNG chạy lại thí nghiệm.

Dùng đúng `summarize`, `recommend` và `plot_study` của exp_max_features.py; khi có CSV
mới chỉ cần chạy lại file này.

Khác biệt duy nhất: ô "Thời gian căn chỉnh" mặc định vẽ TRUNG VỊ từng bước (--time-stat
median) thay vì trung bình, để khớp cột "Thời gian (ms)" của bảng §2.2. Lần chạy 15/11 có
vài cặp đo lúc máy bận (tới ~453 s/cặp) làm trung bình ở mức 2000 vọt lên ~777 ms trong khi
trung vị là 107 ms. Dùng --time-stat mean để vẽ như bản cũ.

Cách chạy:
    py -3.14 experiments/plot_max_features.py
    py -3.14 experiments/plot_max_features.py --values 500 1000 2000 3000 5000 10000
    py -3.14 experiments/plot_max_features.py --csv experiments/max_features_results.csv --output experiments/charts/max_features_study.png

In kèm bảng tóm tắt và giá trị được đề xuất theo tiêu chí p95 của exp_max_features.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent
for p in (str(PROJECT_ROOT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp_max_features as emf  # noqa: E402
from report_tables import load_rows  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vẽ lại biểu đồ khảo sát max_features từ CSV.")
    p.add_argument("--csv", type=Path, default=EXP_DIR / "max_features_results.csv")
    p.add_argument("--output", type=Path, default=EXP_DIR / "charts" / "max_features_study.png")
    p.add_argument("--values", type=int, nargs="+", default=None,
                   help="Chỉ vẽ các mức này (mặc định: mọi mức có trong CSV).")
    p.add_argument("--time-stat", choices=["median", "mean"], default="median",
                   help="Thống kê thời gian cho ô thời gian (mặc định: median, khớp bảng §2.2).")
    p.add_argument("--max-corner-px", type=float, default=1.0,
                   help="Ngưỡng corner p95 để đánh dấu giá trị đề xuất (như exp_max_features).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.csv.is_file():
        print(f"[LỖI] Không thấy {args.csv}. Chạy experiments/exp_max_features.py trước.")
        return 1
    rows = load_rows(args.csv, keep_text=("pair", "class_name", "error"))
    if args.values:
        wanted = set(args.values)
        rows = [r for r in rows if int(r["max_features"]) in wanted]
    if not rows:
        print("[LỖI] Không có dòng nào khớp --values.")
        return 1

    summary = emf.summarize(rows)
    if args.time_stat == "median":
        # plot_study vẽ time_detect/match/ransac_ms (trung bình) -> thay bằng trung vị từng bước
        for s in summary:
            items = [r for r in rows if int(r["max_features"]) == int(s["max_features"])]
            for key in ("time_detect_ms", "time_match_ms", "time_ransac_ms", "time_total_ms"):
                vals = [float(r[key]) for r in items if r.get(key) == r.get(key)]
                if vals:
                    s[key] = statistics.median(vals)
    recommended = emf.recommend(summary, args.max_corner_px)
    n_pairs = len({r["pair"] for r in rows})
    emf.print_summary(summary, recommended)
    emf.plot_study(summary, args.output, recommended, n_pairs)
    print(f"\n{n_pairs} cặp, {len(summary)} mức max_features — nguồn: {args.csv}")
    print(f"Đề xuất (corner p95 <= {args.max_corner_px} px): {recommended}")
    print(f"Ô thời gian vẽ theo: {'trung vị' if args.time_stat == 'median' else 'trung bình'} từng bước")
    print(f"Biểu đồ: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
