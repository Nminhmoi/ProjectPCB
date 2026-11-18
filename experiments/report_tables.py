"""
Sinh lại và đối chiếu các bảng số liệu của PROGRESS_REPORT.md từ CSV (việc #9).

Mỗi con số trong bảng §2.1 / §2.2 / §2.4 phải truy được về một file CSV trong
experiments/. Script này tính lại đúng các cột của báo cáo, bằng CHÍNH các hàm
tổng hợp của script thí nghiệm (không tự định nghĩa lại chỉ số):

    §2.1  ORB / SIFT / AKAZE   <- experiments/per_pair.csv, experiments/akaze/per_pair.csv
                                  (đầu ra của exp_feature_matching.py)
    §2.2  max_features         <- experiments/max_features_results.csv
                                  (exp_max_features.summarize)
    §2.4  Gaussian blur        <- experiments/blur_kernel_results.csv
                                  (exp_blur_kernel.summarize)

Quy ước (ghi rõ để người đọc báo cáo không bị nhầm):
    - p95 dùng numpy.percentile mặc định (nội suy tuyến tính) — giống exp_max_features.
    - §2.1: thời gian = TRUNG BÌNH, reprojection = trung bình các trung vị từng cặp
      (như summary.csv); §2.2: thời gian và reprojection = TRUNG VỊ (như
      exp_max_features). Hai bảng đo ở hai lần chạy khác nhau, đừng so thời gian chéo.
    - Recall / Precision / F1 = micro (cộng dồn TP/GT/pred trên mọi cặp), tiêu chí
      bbox "chạm nhau" dung sai 5 px.

Cách dùng:
    py -3.14 experiments/report_tables.py                 # in 3 bảng markdown
    py -3.14 experiments/report_tables.py --check         # đối chiếu với PROGRESS_REPORT.md
    py -3.14 experiments/report_tables.py --write         # ghi đè bảng trong PROGRESS_REPORT.md
    py -3.14 experiments/report_tables.py --sections 2.2 --check

`--check` trả exit code 1 nếu có ô lệch -> dùng được như cổng kiểm tra trước khi nộp.
Không chạy pipeline, không cần ảnh, chỉ đọc CSV (vài giây).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent
for p in (str(PROJECT_ROOT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

REPORT = (PROJECT_ROOT / "TASK" / "PROGRESS_REPORT.md") if (PROJECT_ROOT / "TASK" / "PROGRESS_REPORT.md").exists() else (PROJECT_ROOT / "PROGRESS_REPORT.md")
DETECTOR_ORDER = ["ORB", "SIFT", "AKAZE"]
CURRENT_DETECTOR = "ORB"
REPORT_MAX_FEATURES = [500, 1000, 2000, 3000, 5000, 10000]


# ---------------------------------------------------------------------
# Đọc CSV, ép kiểu
# ---------------------------------------------------------------------
_BOOL = {"true": True, "false": False}


def _convert(value: str):
    if value is None:
        return None
    v = value.strip()
    if v == "":
        return float("nan")
    if v.lower() in _BOOL:
        return _BOOL[v.lower()]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def load_rows(path: Path, keep_text: Sequence[str] = ()) -> List[Dict[str, object]]:
    """Đọc CSV, chuyển số/bool; các cột trong keep_text giữ nguyên chuỗi."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append({k: (v if k in keep_text else _convert(v)) for k, v in raw.items()})
    return rows


def _finite(values) -> List[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(f):
            out.append(f)
    return out


# ---------------------------------------------------------------------
# Bảng: cấu trúc chung
# ---------------------------------------------------------------------
class Table:
    """Một bảng markdown: header + các dòng; mỗi ô là chuỗi đã định dạng."""

    def __init__(self, section: str, title: str, header: List[str], align: List[str],
                 rows: List[List[str]], keys: List[str], source: str):
        self.section, self.title = section, title
        self.header, self.align, self.rows, self.keys = header, align, rows, keys
        self.source = source

    def markdown(self) -> str:
        lines = ["| " + " | ".join(self.header) + " |",
                 "| " + " | ".join(self.align) + " |"]
        lines += ["| " + " | ".join(r) + " |" for r in self.rows]
        return "\n".join(lines)


def bold_best(rows: List[List[float]], directions: List[Optional[str]],
              decimals: Optional[List[int]] = None) -> List[List[bool]]:
    """Đánh dấu ô tốt nhất theo cột: 'min' / 'max' / None (không đánh dấu).

    So sánh theo giá trị đã làm tròn như khi hiển thị, để hai ô in ra giống nhau
    thì cùng được (hoặc cùng không được) in đậm.
    """
    marks = [[False] * len(r) for r in rows]
    for j, d in enumerate(directions):
        if d is None:
            continue
        nd = decimals[j] if decimals else 6
        vals = [round(r[j], nd) if isinstance(r[j], (int, float)) and not math.isnan(r[j]) else r[j]
                for r in rows]
        finite = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
        if len(finite) < 2:
            continue
        best = min(finite) if d == "min" else max(finite)
        if all(abs(v - best) < 1e-12 for v in finite):
            continue
        for i, v in enumerate(vals):
            if isinstance(v, (int, float)) and not math.isnan(v) and abs(v - best) < 1e-12:
                marks[i][j] = True
    return marks


def fmt(v, spec: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return format(v, spec)


def _b(text: str, on: bool) -> str:
    return f"**{text}**" if on else text


# ---------------------------------------------------------------------
# §2.1 — ORB vs SIFT vs AKAZE
# ---------------------------------------------------------------------
def table_21(exp_dir: Path) -> Optional[Table]:
    files = sorted(exp_dir.rglob("per_pair.csv"))
    rows = []
    for f in files:
        rows += [r for r in load_rows(f, keep_text=("pair_id", "detector", "kind", "class_name", "error"))
                 if r.get("kind") == "original"]
    if not rows:
        return None
    detectors = [d for d in DETECTOR_ORDER if any(r["detector"] == d for r in rows)]
    detectors += sorted({str(r["detector"]) for r in rows} - set(detectors))

    values, labels = [], []
    for det in detectors:
        items = [r for r in rows if r["detector"] == det]
        ok = [r for r in items if r.get("success") is True]
        ce = _finite(r.get("corner_error") for r in ok)
        n_gt = sum(int(r["n_gt"]) for r in items)
        n_pred = sum(int(r["n_pred"]) for r in items)
        rec = sum(int(r["gt_hit"]) for r in items) / n_gt if n_gt else float("nan")
        prec = sum(int(r["pred_hit"]) for r in items) / n_pred if n_pred else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else float("nan")
        values.append([
            float(np.mean(_finite(r["time_total_ms"] for r in items))),
            float(np.mean(_finite(r["kp_test"] for r in items))),
            float(np.mean(_finite(r["matches"] for r in items))),
            float(np.mean(_finite(r["inlier_ratio"] for r in ok))) * 100,
            float(np.mean(_finite(r["reprojection_error"] for r in ok))),
            float(np.median(ce)) if ce else float("nan"),
            float(np.percentile(ce, 95)) if ce else float("nan"),
            rec, prec, f1,
            float(sum(1 for v in ce if v > 1.0)),
            float(len(items)),
        ])
        labels.append(det)

    dirs = ["min", None, None, "max", "min", "min", "min", "max", "max", "max", None, None]
    marks = bold_best(values, dirs, [1, 0, 0, 1, 3, 3, 3, 4, 4, 4, 0, 0])
    out_rows = []
    for det, v, m in zip(labels, values, marks):
        out_rows.append([
            _b(det, det == CURRENT_DETECTOR),
            _b(fmt(v[0], ".1f"), m[0]), fmt(v[1], ".0f"), fmt(v[2], ".0f"),
            _b(fmt(v[3], ".1f"), m[3]), _b(fmt(v[4], ".3f"), m[4]),
            _b(fmt(v[5], ".3f"), m[5]) + " / " + _b(fmt(v[6], ".3f"), m[6]),
            _b(fmt(v[7], ".4f"), m[7]), _b(fmt(v[8], ".4f"), m[8]), _b(fmt(v[9], ".4f"), m[9]),
        ])
    header = ["Bộ dò đặc trưng", "Thời gian căn chỉnh (ms)", "Keypoints", "Matches",
              "Inliers RANSAC (%)", "Reprojection Error (px)", "Corner Error trung vị / p95 (px)",
              "Recall", "Precision", "F1"]
    align = [":---"] + [":---:"] * (len(header) - 1)
    t = Table("2.1", "Đối sánh bộ dò đặc trưng", header, align, out_rows, labels,
              ", ".join(str(f.relative_to(PROJECT_ROOT)).replace("\\", "/") for f in files))
    t.extra = {lab: {"pairs_over_1px": int(v[10]), "pairs": int(v[11])} for lab, v in zip(labels, values)}
    return t


# ---------------------------------------------------------------------
# §2.2 — max_features
# ---------------------------------------------------------------------
def table_22(exp_dir: Path, values_filter: Sequence[int]) -> Optional[Table]:
    path = exp_dir / "max_features_results.csv"
    if not path.is_file():
        return None
    import exp_max_features as emf  # dùng đúng hàm tổng hợp của script thí nghiệm

    rows = load_rows(path, keep_text=("pair", "class_name", "error"))
    summary = [s for s in emf.summarize(rows)
               if not values_filter or int(s["max_features"]) in values_filter]
    vals = [[float(s["kp_test"]), float(s["matches"]), float(s["inliers"]),
             float(s["reprojection_error"]), float(s["corner_error_median"]),
             float(s["corner_error_p95"]), float(s["pairs_over_1px"]),
             float(s["time_total_median_ms"]), float(s.get("f1", float("nan")))] for s in summary]
    dirs = [None, None, None, None, "min", "min", "min", "min", "max"]
    marks = bold_best(vals, dirs, [0, 0, 0, 3, 3, 3, 0, 1, 4])
    out_rows, keys = [], []
    for s, v, m in zip(summary, vals, marks):
        mf = int(s["max_features"])
        label = f"**{mf} (đang dùng)**" if mf == emf.CURRENT_VALUE else str(mf)
        out_rows.append([label, fmt(v[0], ".0f"), fmt(v[1], ".0f"), fmt(v[2], ".0f"),
                         fmt(v[3], ".3f"), _b(fmt(v[4], ".3f"), m[4]), _b(fmt(v[5], ".3f"), m[5]),
                         _b(fmt(v[6], ".0f"), m[6]), _b(fmt(v[7], ".1f"), m[7]),
                         _b(fmt(v[8], ".4f"), m[8])])
        keys.append(str(mf))
    header = ["`max_features`", "Keypoints", "Matches", "Inliers", "Reprojection (px)",
              "Corner trung vị (px)", "Corner p95 (px)", "Số cặp > 1 px", "Thời gian (ms)", "F1"]
    align = [":---:"] * len(header)
    t = Table("2.2", "Khảo sát max_features", header, align, out_rows, keys,
              str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    t.summary = summary
    return t


# ---------------------------------------------------------------------
# §2.4 — Gaussian blur
# ---------------------------------------------------------------------
def table_24(exp_dir: Path) -> Optional[Table]:
    path = exp_dir / "blur_kernel_results.csv"
    if not path.is_file():
        return None
    import exp_blur_kernel as ebk  # dùng đúng hàm tổng hợp của script thí nghiệm

    rows = load_rows(path, keep_text=("pair", "class_name", "condition"))
    summary = ebk.summarize(rows)
    clean = {int(s["kernel"]): s for s in summary if s["condition"] == "clean"}
    noise = {int(s["kernel"]): s for s in summary if s["condition"] == "noise"}
    kernels = sorted(clean)
    vals = []
    for k in kernels:
        c, n = clean[k], noise.get(k, {})
        vals.append([float(c["recall"]), float(c["f1"]), float(c["recall_thin"]),
                     float(n.get("f1", float("nan"))), float(n.get("snr", float("nan"))),
                     float(c["time_blur_ms"]),
                     float(c["recall_std"]), float(c["f1_std"]), float(n.get("f1_std", float("nan")))])
    out_rows, keys = [], []
    t_marks = bold_best(vals, [None, None, None, None, None, "min", None, None, None],
                        [3, 3, 3, 3, 2, 2, 3, 3, 3])
    for k, v, m in zip(kernels, vals, t_marks):
        label = ("1x1 (không blur)" if k == 1 else f"{k}x{k}")
        if k == getattr(ebk, "CURRENT_KERNEL", 3):
            label = f"**{label} (đang dùng)**"
        out_rows.append([label, f"{fmt(v[0], '.3f')}±{fmt(v[6], '.3f')}",
                         f"{fmt(v[1], '.3f')}±{fmt(v[7], '.3f')}", fmt(v[2], ".3f"),
                         f"{fmt(v[3], '.3f')}±{fmt(v[8], '.3f')}", fmt(v[4], ".2f"),
                         _b(fmt(v[5], ".2f"), m[5])])
        keys.append(str(k))
    header = ["Kernel", "Recall±sd (gốc)", "F1±sd (gốc)", "Recall nét mảnh", "F1±sd (nhiễu)",
              "SNR (nhiễu)", "Thời gian blur (ms)"]
    align = [":---:"] * len(header)
    return Table("2.4", "Khảo sát kernel Gaussian blur", header, align, out_rows, keys,
                 str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))


# ---------------------------------------------------------------------
# Đối chiếu / ghi đè PROGRESS_REPORT.md
# ---------------------------------------------------------------------
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _row_key(first_cell: str) -> str:
    text = re.sub(r"[*`]", "", first_cell).strip()
    m = re.match(r"(\d+)x\d+", text)
    if m:
        return m.group(1)
    m = re.match(r"(\d+)", text)
    if m:
        return m.group(1)
    return text.split()[0] if text else text


def find_table(lines: List[str], section: str) -> Optional[Tuple[int, int]]:
    """(dòng bắt đầu, dòng kết thúc + 1) của bảng đầu tiên sau heading '### <section>.'"""
    head = re.compile(rf"^#+\s*{re.escape(section)}\.?\s")
    start = None
    for i, line in enumerate(lines):
        if head.match(line):
            start = i
            break
    if start is None:
        return None
    for i in range(start + 1, len(lines)):
        if re.match(r"^#+\s", lines[i]):
            return None
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            return i, j
    return None


def _cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _tolerance(text: str) -> float:
    m = re.search(r"\.(\d+)", text)
    return 0.5 * 10 ** (-len(m.group(1))) + 1e-9 if m else 0.5 + 1e-9


def compare(table: Table, lines: List[str]) -> List[str]:
    loc = find_table(lines, table.section)
    if loc is None:
        return [f"§{table.section}: không tìm thấy bảng trong {REPORT.name}"]
    report_rows = {_row_key(_cells(l)[0]): _cells(l) for l in lines[loc[0] + 2:loc[1]]}
    issues = []
    for key, row in zip(table.keys, table.rows):
        rep = report_rows.get(key)
        if rep is None:
            issues.append(f"§{table.section} dòng '{key}': có trong CSV nhưng không có trong báo cáo")
            continue
        for j in range(1, min(len(row), len(rep))):
            mine = [float(x) for x in _NUM.findall(row[j].replace("**", ""))]
            theirs_txt = _NUM.findall(rep[j].replace("**", ""))
            theirs = [float(x) for x in theirs_txt]
            if len(mine) != len(theirs):
                issues.append(f"§{table.section} '{key}' cột '{table.header[j]}': báo cáo "
                              f"'{rep[j]}' / CSV '{row[j]}'")
                continue
            for a, b, txt in zip(mine, theirs, theirs_txt):
                if abs(a - b) > _tolerance(txt):
                    issues.append(f"§{table.section} '{key}' cột '{table.header[j]}': báo cáo "
                                  f"{rep[j].replace('**', '')} ≠ CSV {row[j].replace('**', '')}")
                    break
    for key in report_rows:
        if key not in table.keys:
            issues.append(f"§{table.section} dòng '{key}': có trong báo cáo nhưng không có trong CSV")
    return issues


def write_tables(tables: List[Table]) -> List[str]:
    lines = REPORT.read_text(encoding="utf-8").splitlines()
    done = []
    # thay từ cuối file lên để chỉ số dòng không lệch
    for t in sorted(tables, key=lambda t: -(find_table(lines, t.section) or (0, 0))[0]):
        loc = find_table(lines, t.section)
        if loc is None:
            continue
        lines[loc[0]:loc[1]] = t.markdown().splitlines()
        done.append(t.section)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return done


# ---------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sinh lại / đối chiếu bảng §2.1, §2.2, §2.4 từ CSV.")
    p.add_argument("--sections", nargs="+", default=["2.1", "2.2", "2.4"],
                   choices=["2.1", "2.2", "2.4"])
    p.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    p.add_argument("--check", action="store_true", help="Đối chiếu với PROGRESS_REPORT.md.")
    p.add_argument("--write", action="store_true", help="Ghi đè bảng trong PROGRESS_REPORT.md.")
    p.add_argument("--values", type=int, nargs="+", default=REPORT_MAX_FEATURES,
                   help="Các mức max_features đưa vào bảng §2.2.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    builders: Dict[str, Callable[[], Optional[Table]]] = {
        "2.1": lambda: table_21(args.exp_dir),
        "2.2": lambda: table_22(args.exp_dir, args.values),
        "2.4": lambda: table_24(args.exp_dir),
    }
    tables = []
    for sec in args.sections:
        t = builders[sec]()
        if t is None:
            print(f"[BỎ QUA] §{sec}: không có CSV nguồn trong {args.exp_dir}")
            continue
        tables.append(t)
        print(f"\n### §{t.section} — {t.title}\n_Nguồn: {t.source}_\n")
        print(t.markdown())
        if t.section == "2.1":
            print("\nSố cặp corner > 1 px: " + ", ".join(
                f"{k} {v['pairs_over_1px']}/{v['pairs']}" for k, v in t.extra.items()))

    status = 0
    if args.check:
        lines = REPORT.read_text(encoding="utf-8").splitlines()
        issues = [i for t in tables for i in compare(t, lines)]
        print("\n=== ĐỐI CHIẾU VỚI PROGRESS_REPORT.md ===")
        if issues:
            for i in issues:
                print(" - " + i)
            print(f"=> {len(issues)} ô lệch.")
            status = 1
        else:
            print("=> Khớp toàn bộ.")
    if args.write:
        done = write_tables(tables)
        print(f"\nĐã ghi đè bảng §{', §'.join(done)} trong {REPORT.name}.")
    return status


if __name__ == "__main__":
    sys.exit(main())
