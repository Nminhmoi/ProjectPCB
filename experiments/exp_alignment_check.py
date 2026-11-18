"""
P0.1 — Chốt ngưỡng kiểm chứng căn chỉnh bằng điểm ảnh (NCC).

    Vấn đề (EXPERIMENT_REVIEW.md, mục 4 P0.1): có cặp có corner error tới 47 px
    vẫn lọt qua mọi kiểm tra hình học (inliers, inlier ratio, reprojection, det),
    vì các kiểm tra đó chỉ nhìn tập inlier của chính RANSAC. module_preprocess
    giờ tính thêm NCC giữa template và ảnh test đã nắn (độ phân giải thấp, có
    mask vùng hợp lệ) và từ chối khi NCC < DEFAULT_MIN_NCC; main.py khi đó có thể
    thử detector dự phòng nếu được bật bằng --fallback (mặc định: không).

    Lưu ý (CODE_REVIEW.md mục 2.2): 3 cặp từng bị coi là "ngoại lai" (06_spur_08,
    04_spurious_copper_11, 06_spurious_copper_04) thực ra chỉ bị DỊCH CHUYỂN SẴN
    giữa template và test. corner_error ở đây đã bù dịch chuyển đó bằng H_pre nên
    không còn báo sai.

    Script này đo trên toàn bộ cặp ảnh để CHỐT ngưỡng:
      1. Nắn lệch ảnh test bằng homography ngẫu nhiên đã biết (cùng seed với
         exp_feature_matching / exp_max_features).
      2. Căn chỉnh bằng ORB với kiểm tra NCC TẮT -> lấy H, corner error (sai số
         thật), NCC, NCC gradient, MAE.
      3. Với cặp có NCC thấp, chạy thêm detector dự phòng (mặc định SIFT) để
         mô phỏng align_images_robust ở mọi ngưỡng.
      4. Quét ngưỡng 0.90 → 0.995 và báo cáo cho từng ngưỡng:
           - false_reject   : cặp TỐT (corner <= 1 px) bị NCC từ chối
           - caught / missed: cặp NGOẠI LAI (corner > 5 px) bị chặn / lọt
           - fallback_rate  : tỷ lệ cặp phải chạy detector dự phòng
           - final_bad      : số cặp còn lệch > 5 px SAU khi dùng dự phòng
           - extra_ms       : thời gian dự phòng cộng thêm, trung bình mỗi cặp
      5. Đề xuất ngưỡng: nằm giữa NCC cao nhất của cặp ngoại lai và NCC thấp
         nhất của cặp tốt (nếu hai nhóm tách được).

Cách chạy:
    py -3.14 experiments/exp_alignment_check.py --per-class 5
    py -3.14 experiments/exp_alignment_check.py --pairs 06_spur_08 04_spurious_copper_11 06_spurious_copper_04
    py -3.14 experiments/exp_alignment_check.py --no-synthetic-warp     # điều kiện thật (H ~ I)

Chạy full 693 cặp (~15–25 phút, phụ thuộc số cặp cần SIFT):
    py -3.14 -u experiments/exp_alignment_check.py --output-dir experiments

Đầu ra:
    experiments/alignment_check.csv          (từng cặp)
    experiments/alignment_check_thresholds.csv (từng ngưỡng)
    experiments/charts/alignment_check.png
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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
    corner_error,
    estimate_content_homography,
    synthetic_warp,
)
from MODULE.module_preprocess import (  # noqa: E402
    DEFAULT_MAX_FEATURES,
    DEFAULT_MIN_NCC,
    AlignmentError,
    align_images_with_stats,
    is_detector_available,
)

GOOD_PX = 1.0      # cặp "tốt": corner <= 1 px
BAD_PX = 5.0       # cặp "ngoại lai": corner > 5 px
DEFAULT_THRESHOLDS = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]


def run_align(template, test, test_mask, detector: str, max_features: int) -> Dict[str, object]:
    """Căn chỉnh với kiểm tra NCC TẮT, trả về số liệu cần thiết."""
    try:
        _, stats = align_images_with_stats(template, test, max_features=max_features,
                                           detector=detector, min_ncc=None,
                                           test_mask=test_mask)
    except AlignmentError as exc:
        stats = exc.stats
    return stats


def evaluate_pair(pair: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    template = load_image(Path(pair["template_path"]))
    test = load_image(Path(pair["test_path"]))
    h, w = test.shape[:2]

    if args.synthetic_warp:
        test_in, mask, H_true = synthetic_warp(test, str(pair["pair_id"]), args.seed,
                                               args.warp_angle, args.warp_shift, args.warp_scale)
        H_true_inv = np.linalg.inv(H_true)
    else:
        # Ảnh lỗi PKU trùng khít template -> phép biến đổi thật là đơn vị
        test_in, mask, H_true_inv = test, None, np.eye(3)

    # Mốc so sánh bù dịch chuyển CÓ SẴN giữa template và test (một số cặp PKU
    # lệch 7/28/47 px, xem CODE_REVIEW.md mục 2.2). Không có bước này, H đúng
    # vẫn bị chấm là "căn chỉnh lỗi".
    H_pre = estimate_content_homography(template, test)

    row: Dict[str, object] = {"pair": pair["pair_id"], "class_name": pair.get("class_name", "")}
    for tag, det in (("main", args.detector), ("fb", args.fallback)):
        if tag == "fb":
            # Chỉ chạy dự phòng khi cần cho ít nhất một ngưỡng trong lưới
            need = (not row["main_geom_ok"]) or not (row["main_ncc"] >= max(args.thresholds))
            if not det or not need:
                break
        stats = run_align(template, test_in, mask, det, args.max_features)
        H = stats.get("H_candidate")
        geom_ok = H is not None
        row.update({
            f"{tag}_detector": det,
            f"{tag}_geom_ok": geom_ok,
            f"{tag}_error": stats.get("error", "") if not geom_ok else "",
            f"{tag}_inliers": stats.get("inliers", 0),
            f"{tag}_reprojection": stats.get("reprojection_error", float("nan")),
            f"{tag}_corner": corner_error(H, H_true_inv, w, h, H_pre=H_pre) if geom_ok else float("nan"),
            f"{tag}_ncc": float(stats.get("ncc", float("nan"))) if geom_ok else float("nan"),
            f"{tag}_ncc_grad": float(stats.get("ncc_grad", float("nan"))) if geom_ok else float("nan"),
            f"{tag}_mae": float(stats.get("mae", float("nan"))) if geom_ok else float("nan"),
            f"{tag}_time_ms": float(stats.get("time_total_ms", 0.0)),
            f"{tag}_verify_ms": float(stats.get("time_verify_ms", 0.0)),
        })
        if tag == "main" and not geom_ok:
            row["main_ncc"] = float("nan")
    return row


def _nan_to(v: float, default: float) -> float:
    return default if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def simulate(rows: List[Dict[str, object]], threshold: float, metric: str) -> Dict[str, object]:
    """Mô phỏng align_images_robust với ngưỡng `threshold` trên chỉ số `metric`."""
    good = bad = false_reject = caught = missed = fallback = final_bad = unverified = 0
    extra_ms: List[float] = []
    for r in rows:
        corner = _nan_to(r.get("main_corner"), float("inf"))
        score = _nan_to(r.get(f"main_{metric}"), -1.0)
        accepted_main = bool(r.get("main_geom_ok")) and score >= threshold
        if corner <= GOOD_PX:
            good += 1
            false_reject += not accepted_main
        if corner > BAD_PX:
            bad += 1
            caught += not accepted_main
            missed += accepted_main

        if accepted_main:
            final = corner
            extra_ms.append(0.0)
        else:
            fallback += 1
            fb_ok = bool(r.get("fb_geom_ok"))
            fb_score = _nan_to(r.get(f"fb_{metric}"), -1.0)
            extra_ms.append(_nan_to(r.get("fb_time_ms"), 0.0))
            if fb_ok and fb_score >= threshold:
                final = _nan_to(r.get("fb_corner"), float("inf"))
            else:
                unverified += 1
                # Không ai vượt: chọn H có điểm cao hơn (như align_images_robust)
                cands = [(score, corner)] if r.get("main_geom_ok") else []
                if fb_ok:
                    cands.append((fb_score, _nan_to(r.get("fb_corner"), float("inf"))))
                final = (max(cands, key=lambda c: (c[0], -c[1]))[1]
                         if cands else float("inf"))
        final_bad += final > BAD_PX
    n = len(rows)
    return {
        "metric": metric, "threshold": threshold, "pairs": n,
        "good_pairs": good, "bad_pairs": bad,
        "false_reject": false_reject,
        "false_reject_rate": false_reject / good if good else float("nan"),
        "caught": caught, "missed": missed,
        "fallback_rate": fallback / n if n else float("nan"),
        "unverified": unverified,
        "final_bad": final_bad,
        "extra_ms_mean": statistics.fmean(extra_ms) if extra_ms else 0.0,
    }


def suggest(rows: List[Dict[str, object]], metric: str) -> Dict[str, object]:
    """Ngưỡng tách nhóm tốt / ngoại lai (nếu tách được).

    Cặp hỏng hình học (không có H) chắc chắn bị chặn ở bước kiểm tra hình học, nên
    không thể ràng buộc ngưỡng NCC; chúng được đếm riêng ở `bad_geom_fail` để không
    gây cảm giác "tách được" nhưng vẫn còn cặp lọt.
    """
    good = [float(r[f"main_{metric}"]) for r in rows
            if r.get("main_geom_ok") and _nan_to(r.get("main_corner"), 1e9) <= GOOD_PX
            and not math.isnan(float(r[f"main_{metric}"]))]
    bad = [float(r[f"main_{metric}"]) for r in rows
           if r.get("main_geom_ok") and _nan_to(r.get("main_corner"), 0) > BAD_PX
           and not math.isnan(float(r[f"main_{metric}"]))]
    bad_geom_fail = sum(1 for r in rows if not r.get("main_geom_ok"))
    out: Dict[str, object] = {
        "metric": metric,
        "good_min": min(good) if good else float("nan"),
        "good_p01": float(np.percentile(good, 1)) if good else float("nan"),
        "good_median": statistics.median(good) if good else float("nan"),
        "bad_max": max(bad) if bad else float("nan"),
        "n_good": len(good), "n_bad": len(bad), "bad_geom_fail": bad_geom_fail,
    }
    if good and bad and max(bad) < min(good):
        out["separable"] = True
        out["suggested"] = round((max(bad) + min(good)) / 2, 3)
    elif good and bad:
        out["separable"] = False
        out["suggested"] = float("nan")
    else:
        out["separable"] = None
        out["suggested"] = float("nan")
    return out


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


def plot(rows: List[Dict[str, object]], sims: List[Dict[str, object]], path: Path,
         chosen: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, metric in zip(axes[:2], ("ncc", "ncc_grad")):
        xs = [_nan_to(r.get("main_corner"), np.nan) for r in rows if r.get("main_geom_ok")]
        ys = [_nan_to(r.get(f"main_{metric}"), np.nan) for r in rows if r.get("main_geom_ok")]
        colors = ["#C44E52" if x > BAD_PX else ("#DD8452" if x > GOOD_PX else "#4C72B0") for x in xs]
        ax.scatter(np.maximum(xs, 1e-3), ys, c=colors, s=16, edgecolor="none", alpha=0.8)
        ax.set_xscale("log")
        ax.axvline(GOOD_PX, color="gray", linestyle=":", linewidth=1)
        ax.axvline(BAD_PX, color="gray", linestyle=":", linewidth=1)
        if metric == "ncc":
            ax.axhline(chosen, color="black", linestyle="--", linewidth=1,
                       label=f"ngưỡng {chosen}")
            ax.legend(fontsize=8, loc="lower left")
        ax.set_xlabel("Corner error thật (px, log)")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} theo sai số căn chỉnh thật\n"
                     "(xanh ≤1 px, cam 1–5 px, đỏ >5 px)")
        ax.grid(linestyle="--", alpha=0.35)

    ax = axes[2]
    s_ncc = [s for s in sims if s["metric"] == "ncc"]
    th = [s["threshold"] for s in s_ncc]
    ax.plot(th, [s["false_reject"] for s in s_ncc], "o-", label="Cặp tốt bị từ chối")
    ax.plot(th, [s["missed"] for s in s_ncc], "s-", label="Ngoại lai lọt qua")
    ax.plot(th, [s["final_bad"] for s in s_ncc], "^-", label="Còn lệch >5 px sau dự phòng")
    ax2 = ax.twinx()
    ax2.plot(th, [s["fallback_rate"] * 100 for s in s_ncc], "d--", color="gray",
             label="% cặp chạy dự phòng")
    ax2.set_ylabel("% cặp chạy dự phòng")
    ax.axvline(chosen, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Ngưỡng NCC")
    ax.set_ylabel("Số cặp")
    ax.set_title("Đánh đổi theo ngưỡng NCC")
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=8, loc="upper left")
    ax.grid(linestyle="--", alpha=0.35)

    fig.suptitle(f"P0.1 — Kiểm chứng căn chỉnh bằng NCC trên {len(rows)} cặp", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chốt ngưỡng kiểm chứng căn chỉnh bằng NCC (P0.1).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-class", type=int, default=0)
    parser.add_argument("--pairs", nargs="+", default=None, help="Chỉ chạy các pair_id này.")
    parser.add_argument("--detector", default="ORB")
    parser.add_argument("--fallback", default="SIFT",
                        help="Detector dự phòng để mô phỏng (\"\" = không dùng).")
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--no-synthetic-warp", dest="synthetic_warp", action="store_false",
                        help="Không nắn lệch: đo NCC trong điều kiện thật (H ~ đơn vị).")
    parser.add_argument("--warp-angle", type=float, default=3.0)
    parser.add_argument("--warp-shift", type=float, default=40.0)
    parser.add_argument("--warp-scale", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.fallback and not is_detector_available(args.fallback):
        print(f"[CẢNH BÁO] Không có detector dự phòng {args.fallback}; bỏ phần mô phỏng dự phòng.")
        args.fallback = ""

    pairs = find_pairs(str(args.data_dir))
    if args.pairs:
        wanted = set(args.pairs)
        pairs = [p for p in pairs if p["pair_id"] in wanted]
    if args.per_class > 0:
        counter: Dict[str, int] = defaultdict(int)
        picked = []
        for p in pairs:
            if counter[p["class_name"]] < args.per_class:
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

    print(f"{len(pairs)} cặp | detector {args.detector}/{args.max_features}, dự phòng "
          f"{args.fallback or 'không'} | nắn lệch: "
          + (f"±{args.warp_angle}°, ±{args.warp_shift}px, ±{args.warp_scale * 100:.0f}%"
             if args.synthetic_warp else "tắt"))

    rows: List[Dict[str, object]] = []
    for i, pair in enumerate(pairs, start=1):
        try:
            row = evaluate_pair(pair, args)
        except (ValueError, OSError, cv2.error) as exc:
            print(f"[{i}/{len(pairs)}] Bỏ qua {pair['pair_id']}: {exc}")
            continue
        rows.append(row)
        flag = ""
        if _nan_to(row.get("main_corner"), 1e9) > BAD_PX:
            flag = "  <- NGOẠI LAI"
        fb = (f" | {args.fallback}: corner {row['fb_corner']:.2f}, NCC {row['fb_ncc']:.3f}"
              if "fb_corner" in row else "")
        print(f"[{i}/{len(pairs)}] {pair['pair_id']:<26} corner {_nan_to(row.get('main_corner'), np.nan):7.2f} px"
              f"  NCC {_nan_to(row.get('main_ncc'), np.nan):.3f}  grad {_nan_to(row.get('main_ncc_grad'), np.nan):.3f}"
              f"{fb}{flag}")

    if not rows:
        raise SystemExit("Không xử lý được cặp nào.")

    sims = [simulate(rows, t, m) for m in ("ncc", "ncc_grad") for t in sorted(args.thresholds)]
    sugg = {m: suggest(rows, m) for m in ("ncc", "ncc_grad")}

    print("\n=== PHÂN BỐ ===")
    for m, s in sugg.items():
        extra = (f" | {s['bad_geom_fail']} cặp hỏng hình học (đã bị chặn, không cần NCC)"
                 if s["bad_geom_fail"] else "")
        print(f"{m:<8}: cặp tốt (≤{GOOD_PX:g}px, n={s['n_good']}) min {s['good_min']:.4f}, "
              f"p1 {s['good_p01']:.4f}, median {s['good_median']:.4f} | "
              f"ngoại lai (>{BAD_PX:g}px, n={s['n_bad'] + s['bad_geom_fail']}) "
              f"max {s['bad_max']:.4f}{extra}")

    print("\n=== MÔ PHỎNG align_images_robust THEO NGƯỠNG (chỉ số NCC) ===")
    header = (f"{'Ngưỡng':>7} |{'Tốt bị từ chối':>15} |{'Ngoại lai chặn/lọt':>19} |"
              f"{'% dự phòng':>11} |{'Chưa kiểm chứng':>16} |{'Lệch>5px/hỏng':>14} |{'+ms/cặp':>8}")
    print(header)
    print("-" * len(header))
    for s in sims:
        if s["metric"] != "ncc":
            continue
        mark = "  <- mặc định hiện tại" if abs(s["threshold"] - DEFAULT_MIN_NCC) < 1e-9 else ""
        print(f"{s['threshold']:>7.3f} |{s['false_reject']:>8} ({s['false_reject_rate'] * 100 if s['good_pairs'] else 0:4.1f}%) |"
              f"{s['caught']:>9} / {s['missed']:<7} |{s['fallback_rate'] * 100:>10.1f}% |"
              f"{s['unverified']:>16} |{s['final_bad']:>14} |{s['extra_ms_mean']:>8.0f}{mark}")

    s = sugg["ncc"]
    chosen = DEFAULT_MIN_NCC
    print("\n=== ĐỀ XUẤT ===")
    if s["separable"]:
        chosen = float(s["suggested"])
        print(f"NCC tách được 2 nhóm: ngoại lai ≤ {s['bad_max']:.4f} < {s['good_min']:.4f} ≤ cặp tốt.")
        print(f"-> Ngưỡng đề xuất DEFAULT_MIN_NCC = {chosen} (điểm giữa). "
              f"Hiện tại: {DEFAULT_MIN_NCC}.")
    elif s["separable"] is False:
        best = min((x for x in sims if x["metric"] == "ncc"),
                   key=lambda x: (x["final_bad"], x["false_reject"], -x["threshold"]))
        chosen = best["threshold"]
        print(f"NCC KHÔNG tách hoàn toàn (ngoại lai max {s['bad_max']:.4f} ≥ cặp tốt min "
              f"{s['good_min']:.4f}). Ngưỡng ít cặp lệch >5 px nhất sau dự phòng: {chosen} "
              f"(còn {best['final_bad']} cặp, {best['false_reject']} cặp tốt bị từ chối).")
        g = sugg["ncc_grad"]
        if g["separable"]:
            print(f"Gợi ý: NCC gradient tách được (ngoại lai ≤ {g['bad_max']:.4f} < {g['good_min']:.4f}); "
                  "cân nhắc đổi chỉ số trong module_preprocess.")
    else:
        print("Chưa có cặp ngoại lai (>5 px) trong tập đã chạy -> chưa chốt được ngưỡng; "
              "hãy chạy đủ 693 cặp. Lưu ý: 3 cặp từng bị coi là ngoại lai "
              "(06_spur_08, 04_spurious_copper_11, 06_spurious_copper_04) đã được xác định "
              "là dịch chuyển có sẵn giữa template và test, không phải căn chỉnh sai "
              "(CODE_REVIEW.md mục 2.2).")

    out = args.output_dir
    write_csv(rows, out / "alignment_check.csv")
    write_csv(sims, out / "alignment_check_thresholds.csv")
    plot(rows, sims, out / "charts" / "alignment_check.png", chosen)
    print(f"\nCSV từng cặp   : {out / 'alignment_check.csv'}")
    print(f"CSV theo ngưỡng: {out / 'alignment_check_thresholds.csv'}")
    print(f"Biểu đồ        : {out / 'charts' / 'alignment_check.png'}")


if __name__ == "__main__":
    main()
