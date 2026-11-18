"""
Việc #8 — Kiểm nhánh dự phòng SIFT của align_images_robust trên vài cặp ảnh.

Câu hỏi cần trả lời cho báo cáo: KHI NÀO nên bật `--fallback SIFT`?
    Mặc định hiện tại (CODE_REVIEW.md §6.4): DEFAULT_MIN_NCC = 0.90 chỉ là chốt chặn
    thảm hoạ, DEFAULT_FALLBACKS = () (không dùng SIFT).

Với mỗi cặp, script:
    1. Căn chỉnh bằng ORB và bằng SIFT (kiểm chứng NCC tắt) -> NCC, NCC gradient,
       thời gian, độ chênh giữa hai H (px tại 4 góc ảnh), kết quả phát hiện lỗi.
    2. Chạy align_images_robust đúng như main.py ở 2 chế độ:
         mặc định : min_ncc = DEFAULT_MIN_NCC, không dự phòng
         ép SIFT  : min_ncc = 0.999, fallback SIFT   (như `main.py --min-ncc 0.999 --fallback SIFT`)
       -> detector thực sự được dùng, có vượt kiểm chứng không, tổng thời gian.
    3. Ghi bảng + kết luận ra experiments/sift_fallback_note.md (dán thẳng vào báo cáo),
       kèm số liệu 693 cặp ORB vs SIFT từ experiments/summary.csv nếu có.

Nhãn GT nằm theo toạ độ ảnh test; ảnh sau căn chỉnh nằm theo toạ độ template. Với cặp
có độ lệch sẵn (vd 06_spur_08 lệch ~47 px) bbox GT được chiếu qua H trước khi so khớp.

Cách chạy (chỉ vài cặp, không tranh CPU với job đang chạy):
    py -3.14 experiments/check_sift_fallback.py
    py -3.14 experiments/check_sift_fallback.py --pairs 01_missing_hole_01 06_spur_08 04_spurious_copper_05
    py -3.14 experiments/check_sift_fallback.py --forced-ncc 0.9995

Đầu ra:
    experiments/sift_fallback_check.csv
    experiments/sift_fallback_note.md
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_utils import find_pairs, load_image  # noqa: E402
from evaluation import corner_error, detection_counts, read_labels  # noqa: E402
from module_detection import extract_bounding_boxes  # noqa: E402
from module_preprocess import (  # noqa: E402
    DEFAULT_MAX_FEATURES,
    DEFAULT_MIN_AREA,
    DEFAULT_MIN_NCC,
    AlignmentError,
    align_images_robust,
    align_images_with_stats,
    detect_difference,
    is_detector_available,
    warp_valid_mask,
)

# Cặp bình thường / cặp lệch sẵn 47 px / cặp lỗi mảnh từng bị bỏ sót
DEFAULT_PAIRS = ["01_missing_hole_01", "06_spur_08", "04_spurious_copper_05"]


def project_boxes(boxes, H: np.ndarray, shape) -> List[Tuple[int, int, int, int]]:
    """Chiếu bbox (toạ độ ảnh test) sang toạ độ template qua H, lấy hình chữ nhật bao."""
    h, w = shape[:2]
    out = []
    for x, y, bw, bh in boxes:
        pts = np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]).reshape(-1, 1, 2)
        p = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        x0, y0 = np.floor(p.min(axis=0)).astype(int)
        x1, y1 = np.ceil(p.max(axis=0)).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            out.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    return out


def detect_with_h(template, test, H, gt_test) -> Dict[str, int]:
    """Phát hiện lỗi như main.py với H cho trước, so với GT đã chiếu sang toạ độ template."""
    h, w = template.shape[:2]
    aligned = cv2.warpPerspective(test, H, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = warp_valid_mask(test, H, template.shape[:2], erode_border=3)
    mask = detect_difference(template, aligned, valid=valid)
    boxes = extract_bounding_boxes(mask, min_area=DEFAULT_MIN_AREA)
    return detection_counts(boxes, project_boxes(gt_test, H, template.shape))


def run_detector(template, test, name: str, max_features: int) -> Dict[str, object]:
    t0 = time.perf_counter()
    try:
        _, st = align_images_with_stats(template, test, max_features=max_features,
                                        detector=name, min_ncc=None)
    except AlignmentError as exc:
        st = exc.stats
    st["wall_ms"] = (time.perf_counter() - t0) * 1000.0
    return st


def run_robust(template, test, min_ncc, fallbacks, max_features) -> Dict[str, object]:
    t0 = time.perf_counter()
    try:
        _, st = align_images_robust(template, test, max_features=max_features,
                                    fallbacks=fallbacks, min_ncc=min_ncc)
    except AlignmentError as exc:
        st = dict(exc.stats)
        st["success"] = False
    st["wall_ms"] = (time.perf_counter() - t0) * 1000.0
    return st


def _f(v, spec=".3f") -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "–"
    return "–" if math.isnan(v) else format(v, spec)


def load_global_summary(path: Path) -> Optional[Dict[str, Dict[str, float]]]:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig") as handle:
        rows = {r["detector"]: r for r in csv.DictReader(handle) if r.get("kind") == "original"}
    return rows if {"ORB", "SIFT"} <= set(rows) else None


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kiểm nhánh dự phòng SIFT (việc #8).")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--forced-ncc", type=float, default=0.999,
                   help="Ngưỡng NCC để ép chạy nhánh dự phòng (mặc định 0.999).")
    p.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    p.add_argument("--output-dir", type=Path, default=EXP_DIR)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not is_detector_available("SIFT"):
        print("[LỖI] OpenCV này không có SIFT.")
        return 1
    by_id = {p["pair_id"]: p for p in find_pairs(str(args.data_dir))}
    pairs = [by_id[i] for i in args.pairs if i in by_id]
    for i in args.pairs:
        if i not in by_id:
            print(f"[BỎ QUA] Không có cặp {i} trong manifest.")
    if not pairs:
        print("[LỖI] Không còn cặp nào để chạy.")
        return 1

    # Khởi động: lần gọi OpenCV đầu tiên tốn thêm vài giây (nạp thư viện, cấp phát bộ nhớ)
    # -> chạy thử ORB + SIFT một lần trên cặp đầu và bỏ kết quả, để không làm sai thời gian đo.
    try:
        warm_t = load_image(Path(pairs[0]["template_path"]))
        warm_s = load_image(Path(pairs[0]["test_path"]))
        for name in ("ORB", "SIFT"):
            run_detector(warm_t, warm_s, name, args.max_features)
    except (ValueError, OSError):
        pass

    rows: List[Dict[str, object]] = []
    for pair in pairs:
        try:
            template = load_image(Path(pair["template_path"]))
            test = load_image(Path(pair["test_path"]))
        except (ValueError, OSError) as exc:
            print(f"[BỎ QUA] {pair['pair_id']}: {exc}")
            continue
        gt = read_labels(pair.get("label_path"))
        h, w = test.shape[:2]
        orb = run_detector(template, test, "ORB", args.max_features)
        sift = run_detector(template, test, "SIFT", args.max_features)
        H_orb, H_sift = orb.get("H_candidate"), sift.get("H_candidate")

        row: Dict[str, object] = {"pair": pair["pair_id"], "class_name": pair.get("class_name", ""),
                                  "n_gt": len(gt)}
        for tag, st, H in (("orb", orb, H_orb), ("sift", sift, H_sift)):
            row[f"{tag}_ncc"] = st.get("ncc", float("nan"))
            row[f"{tag}_ncc_grad"] = st.get("ncc_grad", float("nan"))
            row[f"{tag}_time_ms"] = st.get("time_total_ms", float("nan"))
            row[f"{tag}_verify_ms"] = st.get("time_verify_ms", float("nan"))
            row[f"{tag}_shift_px"] = corner_error(H, np.eye(3), w, h) if H is not None else float("nan")
            if H is not None:
                c = detect_with_h(template, test, H, gt)
                row[f"{tag}_gt_hit"], row[f"{tag}_n_pred"] = c["gt_hit"], c["n_pred"]
                row[f"{tag}_false_blobs"] = c["false_blobs"]
        row["h_diff_px"] = (corner_error(H_orb, H_sift, w, h)
                            if H_orb is not None and H_sift is not None else float("nan"))

        default = run_robust(template, test, DEFAULT_MIN_NCC, (), args.max_features)
        forced = run_robust(template, test, args.forced_ncc, (("SIFT", args.max_features),),
                            args.max_features)
        for tag, st in (("default", default), ("forced", forced)):
            row[f"{tag}_detector"] = st.get("detector", "") if st.get("success") else "THẤT BẠI"
            row[f"{tag}_verified"] = bool(st.get("verified", False))
            row[f"{tag}_time_ms"] = st.get("time_all_attempts_ms", st.get("time_total_ms", float("nan")))
            row[f"{tag}_attempts"] = " -> ".join(st.get("attempts", []))
        rows.append(row)
        print(f"{pair['pair_id']:<24} ORB NCC {_f(row['orb_ncc'], '.4f')} ({_f(row['orb_time_ms'], '.0f')} ms) | "
              f"SIFT NCC {_f(row['sift_ncc'], '.4f')} ({_f(row['sift_time_ms'], '.0f')} ms) | "
              f"H lệch nhau {_f(row['h_diff_px'], '.2f')} px | ép SIFT -> {row['forced_detector']}"
              f"{'' if row['forced_verified'] else ' (chưa kiểm chứng)'}, {_f(row['forced_time_ms'], '.0f')} ms")

    if not rows:
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "sift_fallback_check.csv"
    fields: List[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # Ghi chú markdown
    n = len(rows)
    extra = [float(r["forced_time_ms"]) - float(r["default_time_ms"]) for r in rows]
    same_det = sum(1 for r in rows if r.get("orb_gt_hit") == r.get("sift_gt_hit"))
    max_hdiff = max((float(r["h_diff_px"]) for r in rows if not math.isnan(float(r["h_diff_px"]))),
                    default=float("nan"))
    min_orb_ncc = min(float(r["orb_ncc"]) for r in rows)
    ran_fallback = sum(1 for r in rows if "SIFT" in str(r["forced_attempts"]))
    triggered = sum(1 for r in rows if r["forced_detector"] == "SIFT")
    unverified = sum(1 for r in rows if not r["forced_verified"])

    lines = [
        f"### Kiểm nhánh dự phòng SIFT (`--min-ncc {args.forced_ncc} --fallback SIFT`)",
        "",
        f"<sub>Sinh bởi `experiments/check_sift_fallback.py` trên {n} cặp; "
        f"max_features = {args.max_features}; mặc định pipeline: `DEFAULT_MIN_NCC = {DEFAULT_MIN_NCC:.2f}`, "
        "không dự phòng.</sub>",
        "",
        "| Cặp | NCC ORB | NCC SIFT | ORB (ms) | SIFT (ms) | H ORB vs SIFT (px) | Lệch sẵn (px) | GT ORB | GT SIFT | Ép SIFT → dùng | Thời gian mặc định → ép (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        used = r["forced_detector"] + ("" if r["forced_verified"] else " *")
        lines.append(
            f"| `{r['pair']}` | {_f(r['orb_ncc'], '.4f')} | {_f(r['sift_ncc'], '.4f')} | "
            f"{_f(r['orb_time_ms'], '.0f')} | {_f(r['sift_time_ms'], '.0f')} | {_f(r['h_diff_px'], '.2f')} | "
            f"{_f(r['orb_shift_px'], '.1f')} | {r.get('orb_gt_hit', '–')}/{r['n_gt']} | "
            f"{r.get('sift_gt_hit', '–')}/{r['n_gt']} | {used} | "
            f"{_f(r['default_time_ms'], '.0f')} → {_f(r['forced_time_ms'], '.0f')} |")
    lines += ["", "<sub>\\* không phương án nào đạt ngưỡng ép; `align_images_robust` dùng H có NCC cao nhất.</sub>", ""]

    g = load_global_summary(EXP_DIR / "summary.csv")
    lines.append("**Kết quả:**")
    lines.append(
        f"- Ép ngưỡng {args.forced_ncc}: nhánh SIFT được chạy ở {ran_fallback}/{n} cặp "
        f"(kết quả cuối dùng SIFT ở {triggered} cặp; {unverified} cặp không phương án nào đạt ngưỡng), "
        f"thời gian tăng trung bình **{np.mean(extra):+.0f} ms/cặp** (tối đa {max(extra):+.0f} ms).")
    lines.append(
        f"- H của ORB và SIFT lệch nhau tối đa **{_f(max_hdiff, '.2f')} px** tại 4 góc ảnh; "
        f"số lỗi GT phát hiện được **giống nhau ở {same_det}/{n} cặp**. "
        f"NCC của ORB thấp nhất {min_orb_ncc:.4f} — "
        + (f"cao hơn nhiều ngưỡng mặc định {DEFAULT_MIN_NCC:.2f}." if min_orb_ncc >= DEFAULT_MIN_NCC
           else f"**dưới** ngưỡng mặc định {DEFAULT_MIN_NCC:.2f}, nhánh dự phòng sẽ có ích ở cặp này."))
    if g:
        o, s = g["ORB"], g["SIFT"]
        lines.append(
            f"- Trên 693 cặp (`experiments/summary.csv`): SIFT căn chính xác hơn (corner trung vị "
            f"{float(s['corner_error_median']):.3f} so với {float(o['corner_error_median']):.3f} px) nhưng chậm "
            f"{float(s['time_align_ms']) / float(o['time_align_ms']):.1f} lần "
            f"({float(s['time_align_ms']):.0f} so với {float(o['time_align_ms']):.0f} ms) và F1 chỉ hơn "
            f"{float(s['f1']) - float(o['f1']):+.4f} vì bước trừ vi sai đã chừa dung sai ±2 px.")
    effect = ("kết quả phát hiện không đổi" if same_det == n
              else f"kết quả phát hiện chỉ đổi ở {n - same_det}/{n} cặp (xem bảng)")
    lines += [
        "",
        "**Kết luận — khi nào bật `--fallback SIFT`:**",
        f"- **Không bật mặc định**: trên PKU, ORB luôn cho NCC ≥ 0.995 (CODE_REVIEW.md §2.1), nhánh dự "
        f"phòng không bao giờ chạy với ngưỡng {DEFAULT_MIN_NCC:.2f}; ép chạy thì tốn thêm thời gian mà {effect}.",
        "- **Nên bật** khi ORB thực sự hỏng: căn chỉnh thất bại hình học, hoặc NCC < 0.90 — ảnh camera "
        "thật ít chi tiết (vùng phủ sơn lớn), mờ, lệch sáng mạnh, hoặc biến dạng lớn. Khi đó SIFT chỉ chạy "
        "cho số ít cặp bị từ chối nên chi phí trung bình nhỏ.",
        "- **Chế độ độ chính xác cao (offline)**: dùng thẳng `--detector SIFT` thay vì dự phòng, "
        "nếu cần sai số căn chỉnh < 0.1 px và chấp nhận ~1.5 s/cặp.",
    ]
    note = args.output_dir / "sift_fallback_note.md"
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nCSV     : {out_csv}\nGhi chú : {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
