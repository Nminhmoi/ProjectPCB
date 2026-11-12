"""
Sinh manifest cặp ảnh PKU-Market-PCB.

Bộ PKU-Market-PCB gốc lưu mỗi bo mạch như sau:
    raw/PCB_USED/<board>.JPG                      <- ảnh bo mạch sạch (template)
    raw/images/<Defect>/<board>_<defect>_<n>.jpg  <- ảnh cùng bo mạch có lỗi (test)
    raw/Annotations/<Defect>/<...>.xml            <- nhãn bounding box
    raw/rotation/<Defect>_rotation/<...>.jpg      <- ảnh test đã xoay (benchmark align)

Script giữ nguyên cấu trúc gốc, không tạo/copy ảnh. Công việc:
    1. Ghép template + test theo mã bo mạch.
    2. Parse XML, kiểm tra bbox và tên lớp.
    3. Ghi data/PKU-Market-PCB/pairs_manifest.csv (đường dẫn tương đối).
    4. Ghi data/PKU-Market-PCB/labels/<pair_id>.txt  (x y w h class_id).
    5. Ghi data/PKU-Market-PCB/classes.csv.

Cấu trúc sau khi chạy:
    data/PKU-Market-PCB/
    ├── raw/                       <- giữ nguyên dataset gốc
    ├── labels/                    <- nhãn chuyển từ XML sang txt
    ├── pairs_manifest.csv
    ├── pairs_rotation_manifest.csv (chỉ khi --include-rotation)
    └── classes.csv

Cách chạy:
    py -3.14 experiments/build_pairs.py
    py -3.14 experiments/build_pairs.py --per-class 20
    py -3.14 experiments/build_pairs.py --include-rotation
"""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Thứ tự lớp khớp 6 lỗi trong README; dùng thống nhất cho toàn bộ đồ án.
CLASS_IDS: Dict[str, int] = {
    "Open_circuit": 1,
    "Short": 2,
    "Mouse_bite": 3,
    "Spur": 4,
    "Spurious_copper": 5,
    "Missing_hole": 6,
}

DEFAULT_DEST = Path("data") / "PKU-Market-PCB"
DEFAULT_SOURCE = DEFAULT_DEST / "raw"

MANIFEST_FIELDS = [
    "pair_id",
    "class_name",
    "class_id",
    "board_id",
    "kind",
    "test_path",
    "template_path",
    "annotation_path",
    "label_path",
    "num_objects",
    "width",
    "height",
]


def read_size(path: Path) -> Optional[Tuple[int, int]]:
    """Đọc nhanh kích thước ảnh (width, height), không cần giải mã toàn ảnh."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def parse_xml(
    xml_path: Path, expected_class: str, width: int, height: int
) -> List[Tuple[int, int, int, int]]:
    """Đọc XML, trả về list bbox (x, y, w, h) đã kiểm tra hợp lệ."""
    root = ET.parse(xml_path).getroot()
    boxes: List[Tuple[int, int, int, int]] = []

    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name.lower() != expected_class.lower():
            raise ValueError(
                f"{xml_path.name}: lớp '{name}' không khớp thư mục "
                f"'{expected_class}'"
            )

        bndbox = obj.find("bndbox")
        if bndbox is None:
            raise ValueError(f"{xml_path.name}: object thiếu <bndbox>")

        x1 = int(float(bndbox.findtext("xmin")))
        y1 = int(float(bndbox.findtext("ymin")))
        x2 = int(float(bndbox.findtext("xmax")))
        y2 = int(float(bndbox.findtext("ymax")))

        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"{xml_path.name}: bbox ({x1},{y1})-({x2},{y2}) nằm ngoài "
                f"ảnh {width}x{height}"
            )
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    if not boxes:
        raise ValueError(f"{xml_path.name}: không có object nào")
    return boxes


def collect_original(
    source: Path, per_class: int
) -> Tuple[List[Dict[str, object]], List[str], Counter]:
    """Ghép các cặp gốc (template/test) và đọc nhãn XML."""
    template_dir = source / "PCB_USED"
    images_dir = source / "images"
    annotations_dir = source / "Annotations"

    if not template_dir.exists() or not images_dir.exists():
        raise SystemExit(
            f"[LỖI] Không thấy '{template_dir}' hoặc '{images_dir}'.\n"
            "       Kiểm tra --source (mặc định: data/PKU-Market-PCB/raw)."
        )

    templates: Dict[str, Path] = {}
    for path in template_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            templates[path.stem.lower()] = path

    template_sizes: Dict[str, Tuple[int, int]] = {}
    for board, path in templates.items():
        size = read_size(path)
        if size is not None:
            template_sizes[board] = size

    pairs: List[Dict[str, object]] = []
    warnings: List[str] = []
    class_counts: Counter = Counter()
    per_class_counter: Counter = Counter()

    for class_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        class_name = class_dir.name
        if class_name not in CLASS_IDS:
            warnings.append(f"Bỏ qua lớp lạ '{class_name}' (chưa có class_id)")
            continue

        for test_path in sorted(class_dir.iterdir()):
            if not test_path.is_file() or test_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if per_class and per_class_counter[class_name] >= per_class:
                continue

            board = test_path.name[:2].lower()
            template = templates.get(board)
            if template is None:
                warnings.append(f"{test_path.name}: không có template bo '{board}'")
                continue

            test_size = read_size(test_path)
            if test_size is None:
                warnings.append(f"{test_path.name}: không đọc được ảnh")
                continue

            expected = template_sizes.get(board)
            if expected is not None and test_size != expected:
                warnings.append(
                    f"{test_path.name}: kích thước {test_size} khác "
                    f"template {expected}"
                )
                continue

            xml_path = annotations_dir / class_name / f"{test_path.stem}.xml"
            if not xml_path.exists():
                warnings.append(f"{test_path.name}: thiếu XML {xml_path.name}")
                continue

            try:
                boxes = parse_xml(
                    xml_path, class_name, test_size[0], test_size[1]
                )
            except (ET.ParseError, ValueError) as exc:
                warnings.append(f"{test_path.name}: {exc}")
                continue

            pairs.append(
                {
                    "pair_id": test_path.stem,
                    "class_name": class_name,
                    "board_id": board,
                    "kind": "original",
                    "template": template,
                    "test": test_path,
                    "annotation": xml_path,
                    "boxes": boxes,
                    "width": test_size[0],
                    "height": test_size[1],
                }
            )
            class_counts[class_name] += 1
            per_class_counter[class_name] += 1

    pairs.sort(key=lambda item: str(item["pair_id"]).lower())
    return pairs, warnings, class_counts


def collect_rotation(source: Path) -> List[Dict[str, object]]:
    """Ảnh test đã xoay, dùng cho benchmark căn chỉnh (không có nhãn XML)."""
    rotation_dir = source / "rotation"
    template_dir = source / "PCB_USED"
    if not rotation_dir.exists() or not template_dir.exists():
        return []

    templates: Dict[str, Path] = {}
    for path in template_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            templates[path.stem.lower()] = path

    pairs: List[Dict[str, object]] = []
    for class_dir in sorted(p for p in rotation_dir.iterdir() if p.is_dir()):
        class_name = class_dir.name.replace("_rotation", "")
        if class_name not in CLASS_IDS:
            continue
        for test_path in sorted(class_dir.iterdir()):
            if not test_path.is_file() or test_path.suffix.lower() not in IMAGE_EXTS:
                continue
            board = test_path.name[:2].lower()
            template = templates.get(board)
            size = read_size(test_path)
            if template is None or size is None:
                continue
            pairs.append(
                {
                    "pair_id": f"{test_path.stem}_rot",
                    "class_name": class_name,
                    "board_id": board,
                    "kind": "rotation",
                    "template": template,
                    "test": test_path,
                    "annotation": None,
                    "boxes": [],
                    "width": size[0],
                    "height": size[1],
                }
            )

    pairs.sort(key=lambda item: str(item["pair_id"]).lower())
    return pairs


def relative_to_dest(path: Optional[Path], dest: Path) -> str:
    if path is None:
        return ""
    return Path(path).resolve().relative_to(dest.resolve()).as_posix()


def write_manifest(
    pairs: List[Dict[str, object]],
    dest: Path,
    manifest_path: Path,
    write_labels: bool,
) -> int:
    labels_dir = dest / "labels"
    if write_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for item in pairs:
        pair_id = str(item["pair_id"])
        class_name = str(item["class_name"])
        class_id = CLASS_IDS[class_name]
        boxes = item["boxes"]  # type: ignore[assignment]

        label_path: Optional[Path] = None
        if write_labels and boxes:
            label_path = labels_dir / f"{pair_id}.txt"
            lines = [f"{x} {y} {w} {h} {class_id}" for x, y, w, h in boxes]
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        rows.append(
            {
                "pair_id": pair_id,
                "class_name": class_name,
                "class_id": class_id,
                "board_id": item["board_id"],
                "kind": item["kind"],
                "test_path": relative_to_dest(item["test"], dest),  # type: ignore[arg-type]
                "template_path": relative_to_dest(item["template"], dest),  # type: ignore[arg-type]
                "annotation_path": relative_to_dest(item["annotation"], dest),  # type: ignore[arg-type]
                "label_path": relative_to_dest(label_path, dest),
                "num_objects": len(boxes),  # type: ignore[arg-type]
                "width": item["width"],
                "height": item["height"],
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_classes(dest: Path) -> Path:
    path = dest / "classes.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name"])
        for name, class_id in sorted(CLASS_IDS.items(), key=lambda kv: kv[1]):
            writer.writerow([class_id, name])
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sinh manifest + nhãn cho PKU-Market-PCB."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Thư mục dataset gốc chứa PCB_USED/, images/, Annotations/ "
        "(mặc định: data/PKU-Market-PCB/raw).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Thư mục đích chứa manifest/labels (mặc định: data/PKU-Market-PCB).",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=0,
        help="Chỉ lấy N ảnh mỗi lớp lỗi (0 = tất cả).",
    )
    parser.add_argument(
        "--include-rotation",
        action="store_true",
        help="Sinh thêm pairs_rotation_manifest.csv cho benchmark căn chỉnh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source: Path = args.source
    dest: Path = args.dest

    # Cho phép truyền thẳng data/PKU-Market-PCB thay vì .../raw.
    if not (source / "images").exists() and (source / "raw" / "images").exists():
        source = source / "raw"

    # Manifest ghi đường dẫn tương đối theo --dest, nên source phải nằm trong dest.
    try:
        source.resolve().relative_to(dest.resolve())
    except ValueError:
        raise SystemExit(
            f"[LỖI] --source '{source}' phải nằm trong --dest '{dest}' "
            "để đường dẫn trong manifest là tương đối."
        )

    print(f"Nguồn dữ liệu : {source}")
    print(f"Đích manifest : {dest}")

    pairs, warnings, class_counts = collect_original(
        source, per_class=max(0, args.per_class)
    )
    if not pairs:
        raise SystemExit("[LỖI] Không ghép được cặp ảnh nào.")

    manifest_path = dest / "pairs_manifest.csv"
    count = write_manifest(pairs, dest, manifest_path, write_labels=True)
    classes_path = write_classes(dest)

    print("-" * 60)
    for class_name, class_count in sorted(class_counts.items()):
        print(f"{class_name:<20} : {class_count:>3} cặp")
    print("-" * 60)
    print(f"Tổng cộng            : {count} cặp")
    print(f"Nhãn                 : {dest / 'labels'}")
    print(f"Manifest             : {manifest_path}")
    print(f"Bảng lớp             : {classes_path}")

    if args.include_rotation:
        rotation_pairs = collect_rotation(source)
        rotation_manifest = dest / "pairs_rotation_manifest.csv"
        rotation_count = write_manifest(
            rotation_pairs, dest, rotation_manifest, write_labels=False
        )
        print(f"Manifest rotation    : {rotation_manifest} ({rotation_count} ảnh)")

    if warnings:
        print(f"\nBỏ qua / cảnh báo    : {len(warnings)} mục")
        for reason in warnings[:10]:
            print(f"  - {reason}")
        if len(warnings) > 10:
            print(f"  ... và {len(warnings) - 10} mục khác")

    print(
        "\nBước tiếp theo:\n"
        "  py -3.14 main.py --pair 01_short_01\n"
        "  py -3.14 experiments/exp_rgb2binary.py --limit 5\n"
    )


if __name__ == "__main__":
    main()
