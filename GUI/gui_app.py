"""GUI Demo - He thong kiem tra chat luong & phat hien loi bo mach PCB.

Chay:
    pip install PySide6
    python gui_app.py                      # mo giao dien
    python gui_app.py --data-dir data      # chi dinh thu muc dataset

Luong su dung:
    1. Chon anh mau (Template) + anh test  (hoac chon mot cap trong dataset).
    2. Bam "KIEM TRA".
    3. Xem ket qua o 6 tab: dau vao -> khop dac trung -> can chinh ->
       vi sai/nhi phan -> khung loi -> cac dom loi (crops).

Giao dien chi goi lai cac ham co san cua pipeline (module_preprocess,
module_detection, dataset_utils, evaluation), khong sua logic xu ly.
Phan loai loi (SVM/RF) la tuy chon: neu co `module_classification.py` voi ham
`classify_defects(img_aligned, boxes) -> list[str]` thi GUI se hien nhan loi.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

try:
    from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QThread, Signal
    from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetrics, QIcon, QImage, QPainter, QPalette, QPen, QPixmap)
    from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                                   QComboBox, QDoubleSpinBox, QFileDialog,
                                   QFormLayout, QFrame, QGraphicsPixmapItem,
                                   QGraphicsRectItem, QGraphicsScene,
                                   QGraphicsView, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                                   QListView, QListWidget, QListWidgetItem,
                                   QMainWindow, QMessageBox, QProgressBar,
                                   QPushButton, QScrollArea, QSizePolicy, QSlider,
                                   QSpinBox, QSplitter, QTableWidget,
                                   QTableWidgetItem, QTabWidget, QTextBrowser,
                                   QVBoxLayout, QWidget)
except ImportError:
    print("\n" + "=" * 65)
    print("[LỖI] Không tìm thấy thư viện đồ họa 'PySide6'.")
    print("Vui lòng cài đặt bằng lệnh:")
    print("    pip install PySide6")
    print("=" * 65 + "\n")
    sys.exit(1)

# Đảm bảo import được các module ở thư mục gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_utils import find_pairs, load_image
from evaluation import detection_counts, read_labels
from module_detection import extract_bounding_boxes
from module_preprocess import (DEFAULT_DETECTOR, DEFAULT_FALLBACKS,
                               DEFAULT_MAX_FEATURES, DEFAULT_MIN_AREA,
                               DEFAULT_MIN_NCC, DEFAULT_OPEN_KSIZE,
                               DEFAULT_THRESHOLD_FLOOR, DETECTORS,
                               AlignmentError, align_images_robust,
                               binarize_difference, difference_map,
                               is_detector_available, warp_valid_mask)

# Phân loại là tuỳ chọn: thiếu module/hàm thì bỏ qua bước này.
try:  # pragma: no cover - phụ thuộc tiến độ của nhóm
    from module_classification import classify_defects as _classify_defects
except Exception:  # noqa: BLE001
    _classify_defects = None

RED = (0, 0, 255)      # BGR
GREEN = (0, 200, 0)
YELLOW = (0, 215, 255)
IMG_FILTER = "Ảnh (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.JPG *.PNG)"

# Lõi pipeline (không phụ thuộc Qt, dùng được cho test)
def fit_width(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_width / float(w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return image, scale


def draw_matches(img_test, img_template, stats, max_lines=150, max_width=1100):
    """Anh noi cap dac trung (inlier): test ben trai, template ben phai."""
    left, s1 = fit_width(img_test, max_width)
    right, s2 = fit_width(img_template, max_width)
    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, left.shape[1] + right.shape[1], 3), np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1]:] = right

    pts_t = stats.get("inlier_pts_test")
    pts_m = stats.get("inlier_pts_template")
    if pts_t is not None and len(pts_t):
        idx = np.linspace(0, len(pts_t) - 1, min(max_lines, len(pts_t))).astype(int)
        for i in idx:
            p1 = (int(pts_t[i][0] * s1), int(pts_t[i][1] * s1))
            p2 = (int(pts_m[i][0] * s2) + left.shape[1], int(pts_m[i][1] * s2))
            cv2.line(canvas, p1, p2, (0, 230, 0), 1, cv2.LINE_AA)   # inlier = xanh lá
            cv2.circle(canvas, p1, 3, (0, 230, 0), -1)
            cv2.circle(canvas, p2, 3, (0, 230, 0), -1)

    text = (f"{stats['detector']}: {stats['inliers']} inliers "
            f"({stats['inlier_ratio'] * 100:.1f}%), reproj "
            f"{stats['reprojection_error']:.2f}px")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def checkerboard(img_template, img_aligned, tiles=8):
    """Ghep xen ke o co: duong mach lien mach qua ranh gioi o = can chuan."""
    h, w = img_template.shape[:2]
    ys = (np.arange(h) * tiles // h)[:, None]
    xs = (np.arange(w) * tiles // w)[None, :]
    mask = ((ys + xs) % 2 == 0)
    out = img_aligned.copy()
    out[mask] = img_template[mask]
    sx, sy = w / tiles, h / tiles
    for i in range(1, tiles):
        cv2.line(out, (int(i * sx), 0), (int(i * sx), h), RED, 1)
        cv2.line(out, (0, int(i * sy)), (w, int(i * sy)), RED, 1)
    return out


def annotate(img, boxes, labels=None, gt_boxes=None, thickness=2):
    """Ve khung do cho loi phat hien (kem nhan), khung xanh la cho Ground Truth."""
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in (gt_boxes or []):
        cv2.rectangle(out, (x, y), (x + w, y + h), GREEN, thickness)
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(out, (x, y), (x + w, y + h), RED, thickness)
        label = f"#{i}" + (f" {labels[i - 1]}" if labels else "")
        (tw, th), base = cv2.getTextSize(label, font, fs, ft)
        ly = max(y - 6, th + 6)
        cv2.rectangle(out, (x, ly - th - 4), (x + tw + 6, ly + base - 2), RED, -1)
        cv2.putText(out, label, (x + 3, ly - 3), font, fs, (255, 255, 255), ft,
                    cv2.LINE_AA)
    return out


def crop_patches(img, boxes, padding=5):
    """Cat cac dom loi trong bo nho (cung quy tac voi crop_defect_patches)."""
    h, w = img.shape[:2]
    patches = []
    for (x, y, bw, bh) in boxes:
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(w, x + bw + padding), min(h, y + bh + padding)
        patches.append(img[y1:y2, x1:x2].copy())
    return patches


def run_inspection(template, test, params, gt_boxes=None, progress=None):
    """Chay toan bo pipeline tren 1 cap anh, tra ve dict ket qua + anh truc quan."""
    say = progress or (lambda _m: None)
    t0 = time.perf_counter()
    warnings = []
    res = {"template": template, "test": test, "gt_boxes": list(gt_boxes or [])}

    # Bước 1-2: căn chỉnh ảnh (detector + RANSAC Homography), kiểm chứng bằng NCC
    say("Căn chỉnh ảnh (đặc trưng + Homography)...")
    fallbacks = [(d, params["max_features"]) for d in params["fallbacks"]]
    min_ncc = params["min_ncc"] if params["verify"] else None
    t1 = time.perf_counter()
    try:
        aligned, stats = align_images_robust(
            template, test, max_features=params["max_features"],
            detector=params["detector"], fallbacks=fallbacks, min_ncc=min_ncc)
    except AlignmentError as exc:
        warnings.append(f"Căn chỉnh thất bại: {exc} — dùng ảnh chưa căn chỉnh.")
        stats = exc.stats
        aligned = test
        if aligned.shape[:2] != template.shape[:2]:
            aligned = cv2.resize(test, (template.shape[1], template.shape[0]))
            warnings.append("Kích thước ảnh test khác template nên đã resize.")
    stats["wall_align_ms"] = (time.perf_counter() - t1) * 1000.0
    aligned_ok = bool(stats.get("success"))
    if aligned_ok and not stats.get("verified", True):
        warnings.append("Không phương án nào vượt kiểm chứng NCC — nên kiểm tra lại ảnh.")
    res["aligned"] = aligned
    res["stats"] = stats
    res["aligned_ok"] = aligned_ok

    # Bước 3: vẽ khớp đặc trưng + kiểm tra độ khít
    say("Dựng hình khớp đặc trưng / bàn cờ...")
    if aligned_ok and stats.get("inlier_pts_test") is not None:
        res["matches_img"] = draw_matches(test, template, stats)
    else:
        res["matches_img"] = None
    res["checker_img"] = checkerboard(template, aligned)

    # Bước 4-5: trừ vi sai, nhị phân hoá Otsu + morphology
    say("Trừ vi sai và nhị phân hóa...")
    t2 = time.perf_counter()
    valid = None
    if aligned_ok and stats.get("H") is not None:
        valid = warp_valid_mask(test, stats["H"], template.shape[:2], erode_border=3)
    diff, _ = difference_map(template, aligned, 3)
    mask = binarize_difference(
        diff, threshold_floor=params["threshold_floor"],
        open_ksize=params["open_ksize"], hysteresis_low=None, valid=valid)
    res["diff"], res["mask"] = diff, mask

    # Bước 6: contour → bounding box
    say("Định vị đốm lỗi...")
    boxes = extract_bounding_boxes(mask, min_area=params["min_area"])
    res["boxes"] = boxes
    res["patches"] = crop_patches(aligned, boxes, padding=5)
    stats["wall_detect_ms"] = (time.perf_counter() - t2) * 1000.0

    # Bước 7: phân loại (tuỳ chọn)
    labels = None
    if _classify_defects is not None and boxes:
        say("Phân loại khuyết tật...")
        try:
            labels = list(_classify_defects(aligned, boxes))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Phân loại lỗi thất bại: {exc}")
    res["labels"] = labels

    # Bước 8: ảnh kết quả + đánh giá với Ground Truth (nếu có)
    res["result_img"] = annotate(aligned, boxes, labels)
    res["result_gt_img"] = annotate(aligned, boxes, labels, gt_boxes=res["gt_boxes"])
    res["counts"] = detection_counts(boxes, res["gt_boxes"]) if res["gt_boxes"] else None

    res["warnings"] = warnings
    res["total_s"] = time.perf_counter() - t0
    say("Hoàn tất.")
    return res

# Luồng chạy nền
class InspectWorker(QThread):
    progress = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, template_path, test_path, params, gt_boxes, parent=None):
        super().__init__(parent)
        self.template_path, self.test_path = template_path, test_path
        self.params, self.gt_boxes = params, gt_boxes

    def run(self):
        try:
            self.progress.emit("Đọc ảnh...")
            template = load_image(Path(self.template_path))
            test = load_image(Path(self.test_path))
            res = run_inspection(template, test, self.params, self.gt_boxes,
                                 progress=self.progress.emit)
            self.done.emit(res)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

# Widget hiển thị
def bgr_to_pixmap(img: np.ndarray) -> QPixmap:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    bytes_per_line = int(img.strides[0])
    qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


class ImageViewer(QGraphicsView):
    """Xem anh: lan chuot de zoom, keo de di chuyen, double-click de vua khung."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = None
        self._hl = None
        self._fit = True
        self.title = title
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform
                            | QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(24, 24, 27)))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._placeholder("Chưa có dữ liệu")

    def _placeholder(self, text):
        self._scene.clear()
        self._item = self._hl = None
        t = self._scene.addText(text, QFont("Segoe UI", 12))
        t.setDefaultTextColor(QColor(113, 113, 122))
        self._scene.setSceneRect(t.boundingRect())
        self.resetTransform()

    def set_message(self, text):
        self._placeholder(text)

    def set_image(self, img):
        if img is None:
            self._placeholder("Không có dữ liệu cho bước này")
            return
        self._scene.clear()
        self._hl = None
        self._item = QGraphicsPixmapItem(bgr_to_pixmap(img))
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)
        self._scene.setSceneRect(self._item.boundingRect())
        self.fit()

    def fit(self):
        if self._item is not None:
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
            self._fit = True

    def highlight(self, box, margin=60):
        """Zoom vao mot bounding box (x, y, w, h) va to vien vang."""
        if self._item is None:
            return
        if self._hl is not None:
            self._scene.removeItem(self._hl)
        x, y, w, h = box
        self._hl = QGraphicsRectItem(QRectF(x - 3, y - 3, w + 6, h + 6))
        pen = QPen(QColor(250, 204, 21), 3)
        pen.setCosmetic(True)
        self._hl.setPen(pen)
        self._scene.addItem(self._hl)
        self.fitInView(QRectF(x - margin, y - margin, w + 2 * margin, h + 2 * margin),
                       Qt.AspectRatioMode.KeepAspectRatio)
        self._fit = False

    def wheelEvent(self, event):
        if self._item is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self._fit = False

    def mouseDoubleClickEvent(self, event):
        if self._hl is not None:
            self._scene.removeItem(self._hl)
            self._hl = None
        self.fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit:
            self.fit()


class TitledViewer(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.caption = QLabel(title)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setStyleSheet("color:#a1a1aa; font-weight:600; padding:4px;")
        self.viewer = ImageViewer()
        lay.addWidget(self.caption)
        lay.addWidget(self.viewer, 1)


class PathPicker(QWidget):
    changed = Signal()

    def __init__(self, button_text, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        row = QHBoxLayout()
        self.btn = QPushButton(button_text)
        self.btn.clicked.connect(self._pick)
        row.addWidget(self.btn)
        lay.addLayout(row)
        self.thumb = QLabel("— chưa chọn —")
        self.thumb.setFixedSize(300, 90)
        self.thumb.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet("background:#18181b; border:1px solid #3f3f46; border-radius:4px; color:#71717a; font-size:11px;")
        self.label = QLabel("— chưa chọn —")
        self.label.setStyleSheet("color:#71717a;")
        self.label.setWordWrap(False)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.thumb, 0, Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.label)
        self.path = ""

    def _pick(self):
        start = str(Path(self.path).parent) if self.path else ""
        p, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", start, IMG_FILTER)
        if p:
            self.set_path(p)
            self.changed.emit()

    def set_path(self, path: str):
        self.path = str(path)
        full_name = Path(path).name
        fm = QFontMetrics(self.label.font())
        elided_name = fm.elidedText(full_name, Qt.TextElideMode.ElideMiddle, 280)
        self.label.setText(elided_name)
        self.label.setStyleSheet("color:#e4e4e7;")
        self.label.setToolTip(str(path))
        try:
            img = load_image(Path(path))
            small, _ = fit_width(img, 600)
            pm = bgr_to_pixmap(small).scaled(
                self.thumb.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.thumb.setPixmap(pm)
        except Exception:  # noqa: BLE001
            self.thumb.clear()
            self.thumb.setText("Không đọc được ảnh")

    def clear(self):
        self.path = ""
        self.label.setText("— chưa chọn —")
        self.label.setToolTip("")
        self.label.setStyleSheet("color:#71717a;")
        self.thumb.clear()
        self.thumb.setText("— chưa chọn —")

# Cửa sổ chính
class MainWindow(QMainWindow):
    def __init__(self, data_dir="data"):
        super().__init__()
        self.setWindowTitle("PCB Defect Inspector — Demo phát hiện lỗi bo mạch")
        self.resize(1480, 900)
        self.pairs: list[dict] = []
        self.gt_boxes: list = []
        self.result: dict | None = None
        self.worker: InspectWorker | None = None
        self._build_ui()
        self.load_dataset(data_dir, silent=True)

    # Giao diện
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)
        split.addWidget(self._build_sidebar())

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self._build_tabs())
        right.addWidget(self._build_bottom())
        right.setStretchFactor(0, 5)
        right.setStretchFactor(1, 2)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([340, 1140])

        self.statusBar().showMessage("Sẵn sàng. Chọn ảnh mẫu + ảnh test rồi bấm KIỂM TRA.")

    def _build_sidebar(self):
        box = QWidget()
        box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        box.setMaximumWidth(340)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(2, 6, 2, 6)
        lay.setSpacing(6)

        title = QLabel("PCB Defect Inspector")
        title.setStyleSheet("font-size:18px; font-weight:700; color:#f4f4f5;")
        sub = QLabel("Classical CV · ORB/SIFT · Homography · Differencing")
        sub.setStyleSheet("color:#71717a; font-size:11px;")
        lay.addWidget(title)
        lay.addWidget(sub)

        # chọn nhanh từ dataset
        g_ds = QGroupBox("Chọn nhanh từ dataset")
        f = QVBoxLayout(g_ds)
        f.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        self.cb_class = QComboBox()
        self.cb_class.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cb_class.setMinimumContentsLength(10)
        self.cb_class.currentIndexChanged.connect(self._filter_pairs)
        self.btn_data = QPushButton("Thư mục…")
        self.btn_data.clicked.connect(self._pick_data_dir)
        row.addWidget(self.cb_class, 1)
        row.addWidget(self.btn_data)
        f.addLayout(row)
        self.cb_pair = QComboBox()
        self.cb_pair.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cb_pair.setMinimumContentsLength(10)
        self.cb_pair.activated.connect(self._pair_selected)
        f.addWidget(self.cb_pair)
        self.lbl_ds = QLabel("")
        self.lbl_ds.setStyleSheet("color:#71717a; font-size:11px;")
        self.lbl_ds.setWordWrap(False)
        f.addWidget(self.lbl_ds)
        lay.addWidget(g_ds)

        # chọn ảnh thủ công
        g_in = QGroupBox("Ảnh đầu vào")
        v = QVBoxLayout(g_in)
        v.setContentsMargins(0, 4, 0, 4)
        self.pick_template = PathPicker("📂  Chọn ảnh MẪU (Template)")
        self.pick_test = PathPicker("📂  Chọn ảnh TEST (cần kiểm tra)")
        self.pick_template.changed.connect(self._manual_changed)
        self.pick_test.changed.connect(self._manual_changed)
        v.addWidget(self.pick_template)
        v.addWidget(self.pick_test)
        lay.addWidget(g_in)

        # tham số
        g_p = QGroupBox("Tham số (mặc định = đã chốt)")
        form = QFormLayout(g_p)
        form.setContentsMargins(6, 6, 6, 6)
        self.cb_det = QComboBox()
        self.cb_det.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cb_det.setMinimumContentsLength(10)
        for d in DETECTORS:
            ok = is_detector_available(d)
            self.cb_det.addItem(d if ok else f"{d} (thiếu opencv-contrib)", d)
        self.cb_det.setCurrentIndex(DETECTORS.index(DEFAULT_DETECTOR))
        self.sp_feat = QSpinBox()
        self.sp_feat.setRange(200, 20000)
        self.sp_feat.setSingleStep(500)
        self.sp_feat.setValue(DEFAULT_MAX_FEATURES)
        self.chk_fb = QCheckBox("Dự phòng SIFT khi ORB bị từ chối")
        self.chk_fb.setChecked(bool(DEFAULT_FALLBACKS))
        self.chk_ncc = QCheckBox("Kiểm chứng căn chỉnh bằng NCC")
        self.chk_ncc.setChecked(True)
        self.sp_ncc = QDoubleSpinBox()
        self.sp_ncc.setRange(0.5, 1.0)
        self.sp_ncc.setSingleStep(0.01)
        self.sp_ncc.setDecimals(3)
        self.sp_ncc.setValue(DEFAULT_MIN_NCC)
        self.sp_floor = QSpinBox()
        self.sp_floor.setRange(1, 200)
        self.sp_floor.setValue(DEFAULT_THRESHOLD_FLOOR)
        self.sp_open = QSpinBox()
        self.sp_open.setRange(1, 15)
        self.sp_open.setValue(DEFAULT_OPEN_KSIZE)
        self.sp_area = QSpinBox()
        self.sp_area.setRange(1, 5000)
        self.sp_area.setValue(DEFAULT_MIN_AREA)
        form.addRow("Bộ đặc trưng", self.cb_det)
        form.addRow("max_features", self.sp_feat)
        form.addRow(self.chk_fb)
        form.addRow(self.chk_ncc)
        form.addRow("Ngưỡng NCC", self.sp_ncc)
        form.addRow("Ngưỡng sàn", self.sp_floor)
        form.addRow("Opening kernel", self.sp_open)
        form.addRow("Diện tích tối thiểu", self.sp_area)
        lay.addWidget(g_p)

        # nút bấm
        self.btn_run = QPushButton("▶  KIỂM TRA")
        self.btn_run.setObjectName("primary")
        self.btn_run.setMinimumHeight(44)
        self.btn_run.clicked.connect(self.run)
        self.btn_save = QPushButton("💾  Lưu kết quả…")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_results)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        lay.addWidget(self.btn_run)
        lay.addWidget(self.progress)
        lay.addWidget(self.btn_save)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(box)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedWidth(340)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _build_tabs(self):
        self.tabs = QTabWidget()

        # tab ảnh đầu vào
        sp = QSplitter(Qt.Orientation.Horizontal)
        self.v_template = TitledViewer("Ảnh MẪU (Template)")
        self.v_test = TitledViewer("Ảnh TEST")
        sp.addWidget(self.v_template)
        sp.addWidget(self.v_test)
        self.tabs.addTab(sp, "1 · Ảnh đầu vào")

        # tab khớp đặc trưng
        self.v_matches = TitledViewer("Các cặp đặc trưng khớp đúng (inlier, xanh lá) — test ⟷ mẫu")
        self.tabs.addTab(self.v_matches, "2 · Khớp đặc trưng")

        # tab độ khít sau căn chỉnh
        w3 = QWidget()
        l3 = QVBoxLayout(w3)
        l3.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.cb_blend = QComboBox()
        self.cb_blend.addItems(["Bàn cờ (Checkerboard)", "Phủ mờ (Alpha blend)"])
        self.cb_blend.currentIndexChanged.connect(self._update_blend)
        self.sl_alpha = QSlider(Qt.Orientation.Horizontal)
        self.sl_alpha.setRange(0, 100)
        self.sl_alpha.setValue(50)
        self.sl_alpha.valueChanged.connect(self._update_blend)
        self.lbl_alpha = QLabel("Alpha 0.50")
        bar.addWidget(QLabel("Chế độ:"))
        bar.addWidget(self.cb_blend)
        bar.addSpacing(16)
        bar.addWidget(self.lbl_alpha)
        bar.addWidget(self.sl_alpha, 1)
        l3.addLayout(bar)
        self.v_align = ImageViewer()
        l3.addWidget(self.v_align, 1)
        self.tabs.addTab(w3, "3 · Độ khít sau căn chỉnh")

        # tab vi sai
        sp4 = QSplitter(Qt.Orientation.Horizontal)
        self.v_diff = TitledViewer("Bản đồ vi sai (heatmap)")
        self.v_mask = TitledViewer("Mặt nạ nhị phân (Otsu + Opening)")
        sp4.addWidget(self.v_diff)
        sp4.addWidget(self.v_mask)
        self.tabs.addTab(sp4, "4 · Vi sai && Nhị phân")

        # tab kết quả
        w5 = QWidget()
        l5 = QVBoxLayout(w5)
        l5.setContentsMargins(0, 0, 0, 0)
        bar5 = QHBoxLayout()
        self.chk_gt = QCheckBox("Hiện Ground Truth (khung xanh lá)")
        self.chk_gt.toggled.connect(self._update_result_view)
        hint = QLabel("Khung đỏ = lỗi phát hiện · click dòng trong bảng để zoom · double-click để vừa khung")
        hint.setStyleSheet("color:#71717a;")
        bar5.addWidget(self.chk_gt)
        bar5.addStretch(1)
        bar5.addWidget(hint)
        l5.addLayout(bar5)
        self.v_result = ImageViewer()
        l5.addWidget(self.v_result, 1)
        self.tabs.addTab(w5, "5 · Kết quả (khung lỗi)")

        # tab ảnh cắt
        self.list_crops = QListWidget()
        self.list_crops.setViewMode(QListView.ViewMode.IconMode)
        self.list_crops.setIconSize(QSize(140, 140))
        self.list_crops.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_crops.setMovement(QListView.Movement.Static)
        self.list_crops.setSpacing(10)
        self.list_crops.itemClicked.connect(lambda it: self._focus_defect(it.data(Qt.ItemDataRole.UserRole)))
        self.tabs.addTab(self.list_crops, "6 · Các đốm lỗi (crops)")
        return self.tabs

    def _build_bottom(self):
        sp = QSplitter(Qt.Orientation.Horizontal)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "x", "y", "w", "h", "Loại lỗi / Diện tích"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._table_selected)
        sp.addWidget(self.table)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        self.verdict = QLabel("CHƯA KIỂM TRA")
        self.verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict.setMinimumHeight(44)
        self._set_verdict("CHƯA KIỂM TRA", "#3f3f46")
        pl.addWidget(self.verdict)
        self.info = QTextBrowser()
        self.info.setOpenLinks(False)
        pl.addWidget(self.info, 1)
        sp.addWidget(panel)
        sp.setSizes([520, 620])
        return sp

    # nạp dataset
    def load_dataset(self, data_dir, silent=False):
        try:
            self.pairs = find_pairs(data_dir)
        except Exception as exc:  # noqa: BLE001
            self.pairs = []
            if not silent:
                QMessageBox.warning(self, "Dataset", f"Không đọc được manifest:\n{exc}")
        self.data_dir = data_dir
        self.cb_class.blockSignals(True)
        self.cb_class.clear()
        self.cb_class.addItem("Tất cả loại lỗi")
        for name in sorted({str(p["class_name"]) for p in self.pairs}):
            self.cb_class.addItem(name)
        self.cb_class.blockSignals(False)
        self._filter_pairs()
        if self.pairs:
            fm = QFontMetrics(self.lbl_ds.font())
            elided_dir = fm.elidedText(str(data_dir), Qt.TextElideMode.ElideMiddle, 160)
            self.lbl_ds.setText(f"{len(self.pairs)} cặp · {elided_dir}")
            self.lbl_ds.setToolTip(f"{len(self.pairs)} cặp trong '{data_dir}'. Ảnh gốc phải có trong thư mục raw/.")
        else:
            self.lbl_ds.setText("Không tìm thấy pairs_manifest.csv")
            self.lbl_ds.setToolTip("Không tìm thấy pairs_manifest.csv — hãy chọn ảnh thủ công bên dưới.")

    def _filter_pairs(self):
        cls = self.cb_class.currentText()
        self.cb_pair.clear()
        self.cb_pair.addItem("— chọn một cặp —", None)
        for p in self.pairs:
            if cls == "Tất cả loại lỗi" or str(p["class_name"]) == cls:
                self.cb_pair.addItem(str(p["pair_id"]), p)

    def _pick_data_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục dataset (chứa pairs_manifest.csv)")
        if d:
            self.load_dataset(d)

    def _pair_selected(self, _idx):
        pair = self.cb_pair.currentData()
        if not pair:
            return
        tp, mp = Path(pair["test_path"]), Path(pair["template_path"])
        missing = [str(p) for p in (tp, mp) if not p.is_file()]
        if missing:
            QMessageBox.warning(
                self, "Thiếu ảnh gốc",
                "Repo chỉ chứa manifest + nhãn, KHÔNG chứa ảnh gốc (~1.9 GB).\n"
                "Hãy đặt dataset vào data/PKU-Market-PCB/raw/.\n\nKhông tìm thấy:\n" + "\n".join(missing))
            return
        self.pick_template.set_path(str(mp))
        self.pick_test.set_path(str(tp))
        self.gt_boxes = read_labels(pair.get("label_path"))
        self.chk_gt.setChecked(False)
        self.statusBar().showMessage(
            f"{pair['pair_id']} · lớp {pair['class_name']} · {len(self.gt_boxes)} lỗi Ground Truth")

    def _manual_changed(self):
        # chọn tay thì không còn Ground Truth của cặp dataset
        self.gt_boxes = []

    # chạy pipeline
    def _params(self):
        det = self.cb_det.currentData()
        fb = []
        if self.chk_fb.isChecked():
            fb = [d for d, _ in DEFAULT_FALLBACKS if d != det and is_detector_available(d)]
        return {
            "detector": det, "max_features": self.sp_feat.value(), "fallbacks": fb,
            "verify": self.chk_ncc.isChecked(), "min_ncc": self.sp_ncc.value(),
            "threshold_floor": self.sp_floor.value(), "open_ksize": self.sp_open.value(),
            "min_area": self.sp_area.value(),
        }

    def run(self):
        if not self.pick_template.path or not self.pick_test.path:
            QMessageBox.information(self, "Thiếu ảnh", "Hãy chọn cả ảnh MẪU và ảnh TEST.")
            return
        if not is_detector_available(self.cb_det.currentData()):
            QMessageBox.warning(self, "Detector", "Bộ đặc trưng này cần opencv-contrib-python.")
            return
        self.btn_run.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.progress.setVisible(True)
        self._set_verdict("ĐANG XỬ LÝ…", "#a16207")
        self.worker = InspectWorker(self.pick_template.path, self.pick_test.path,
                                    self._params(), self.gt_boxes, self)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_failed(self, msg):
        self._finish_run()
        self._set_verdict("LỖI", "#b91c1c")
        self.statusBar().showMessage("Xử lý thất bại.")
        QMessageBox.critical(self, "Lỗi pipeline", msg[:1500])

    def _finish_run(self):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)

    def _on_done(self, res):
        self._finish_run()
        self.result = res
        self.btn_save.setEnabled(True)
        n = len(res["boxes"])

        self.v_template.viewer.set_image(res["template"])
        self.v_test.viewer.set_image(res["test"])
        if res["matches_img"] is not None:
            self.v_matches.viewer.set_image(res["matches_img"])
        else:
            self.v_matches.viewer.set_message("Căn chỉnh thất bại nên không có cặp đặc trưng để vẽ")
        # bản đồ vi sai → heatmap, chuẩn hoá theo percentile cho dễ nhìn
        hi = max(float(np.percentile(res["diff"], 99.9)), 1.0)
        norm = np.clip(res["diff"].astype(np.float32) * (255.0 / hi), 0, 255).astype(np.uint8)
        self.v_diff.viewer.set_image(cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO))
        self.v_mask.viewer.set_image(res["mask"])
        # ảnh phủ mờ: thu nhỏ để kéo thanh trượt mượt
        self._blend_t, s = fit_width(res["template"], 1600)
        self._blend_a = cv2.resize(res["aligned"], (self._blend_t.shape[1], self._blend_t.shape[0]),
                                   interpolation=cv2.INTER_AREA)
        self._checker_small, _ = fit_width(res["checker_img"], 1600)
        self._update_blend()
        self._update_result_view()
        self._fill_table(res)
        self._fill_crops(res)
        self._fill_info(res)

        if n == 0:
            self._set_verdict("KHÔNG PHÁT HIỆN LỖI — ĐẠT", "#15803d")
        else:
            self._set_verdict(f"PHÁT HIỆN {n} LỖI — KHÔNG ĐẠT", "#b91c1c")
        self.tabs.setCurrentIndex(4)
        self.statusBar().showMessage(f"Hoàn tất trong {res['total_s']:.2f} s · {n} đốm lỗi.")

    # hiển thị
    def _set_verdict(self, text, color):
        self.verdict.setText(text)
        self.verdict.setStyleSheet(
            f"background:{color}; color:white; font-size:16px; font-weight:700; border-radius:6px;")

    def _update_blend(self):
        if self.result is None:
            return
        mode = self.cb_blend.currentIndex()
        self.sl_alpha.setEnabled(mode == 1)
        a = self.sl_alpha.value() / 100.0
        self.lbl_alpha.setText(f"Alpha {a:.2f}")
        if mode == 0:
            self.v_align.set_image(self._checker_small)
        else:
            self.v_align.set_image(cv2.addWeighted(self._blend_t, 1 - a, self._blend_a, a, 0))

    def _update_result_view(self):
        if self.result is None:
            return
        key = "result_gt_img" if (self.chk_gt.isChecked() and self.result["gt_boxes"]) else "result_img"
        self.v_result.set_image(self.result[key])

    def _fill_table(self, res):
        boxes, labels = res["boxes"], res["labels"]
        self.table.setRowCount(len(boxes))
        for i, (x, y, w, h) in enumerate(boxes):
            last = labels[i] if labels else f"{w * h} px²"
            for c, v in enumerate([i + 1, x, y, w, h, last]):
                it = QTableWidgetItem(str(v))
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(i, c, it)
        self.table.setHorizontalHeaderItem(5, QTableWidgetItem("Loại lỗi" if labels else "Diện tích"))

    def _fill_crops(self, res):
        self.list_crops.clear()
        for i, patch in enumerate(res["patches"]):
            p = patch if patch.size else np.zeros((8, 8, 3), np.uint8)
            h, w = p.shape[:2]
            s = 140.0 / max(h, w)
            big = cv2.resize(p, (max(1, int(w * s)), max(1, int(h * s))),
                             interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
            caption = f"#{i + 1}" + (f" · {res['labels'][i]}" if res["labels"] else "")
            it = QListWidgetItem(QIcon(bgr_to_pixmap(big)), caption)
            it.setData(Qt.ItemDataRole.UserRole, i)
            it.setToolTip(f"{w}×{h}px")
            self.list_crops.addItem(it)

    def _fill_info(self, res):
        s = res["stats"]
        f = lambda v, fmt="{:.2f}": fmt.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "—"
        rows = []
        if res["aligned_ok"]:
            rows += [
                ("Bộ đặc trưng dùng", s.get("detector", "—") + (" (dự phòng)" if s.get("fallback_used") else "")),
                ("Keypoints test / mẫu", f"{s.get('kp_test', 0)} / {s.get('kp_template', 0)}"),
                ("Cặp khớp (good)", f"{s.get('good_matches', 0)}"),
                ("Inliers RANSAC", f"{s.get('inliers', 0)} ({s.get('inlier_ratio', 0) * 100:.0f}%)"),
                ("Reprojection error", f"{f(s.get('reprojection_error'))} px (mục tiêu < 1.5)"),
                ("NCC kiểm chứng", f"{f(s.get('ncc'), '{:.3f}')}"
                                   + ("" if s.get("verified", True) else "  ⚠ chưa vượt ngưỡng")),
                ("Thời gian căn chỉnh", f"{s.get('wall_align_ms', 0):.0f} ms"),
            ]
        else:
            rows.append(("Căn chỉnh", "THẤT BẠI — " + str(s.get("error", "?"))))
        rows += [("Thời gian phát hiện", f"{s.get('wall_detect_ms', 0):.0f} ms"),
                 ("Tổng thời gian", f"{res['total_s']:.2f} s"),
                 ("Số đốm lỗi", f"{len(res['boxes'])}")]
        c = res["counts"]
        if c:
            rec = c["gt_hit"] / c["n_gt"] if c["n_gt"] else float("nan")
            pre = c["pred_hit"] / c["n_pred"] if c["n_pred"] else float("nan")
            rows += [("— Đối chiếu Ground Truth —", ""),
                     ("Số lỗi GT", f"{c['n_gt']}"),
                     ("GT được phát hiện (Recall)", f"{c['gt_hit']}/{c['n_gt']} = {f(rec * 100, '{:.0f}')}%"),
                     ("Đốm trúng GT (Precision)", f"{c['pred_hit']}/{c['n_pred']} = {f(pre * 100, '{:.0f}')}%"),
                     ("TP @ IoU≥0.5", f"{c.get('tp_iou50', 0)}")]
        html = ["<table cellspacing=4>"]
        for k, v in rows:
            if v == "":
                html.append(f"<tr><td colspan=2 style='color:#a1a1aa'><b>{k}</b></td></tr>")
            else:
                html.append(f"<tr><td style='color:#a1a1aa'>{k}</td><td><b>{v}</b></td></tr>")
        html.append("</table>")
        for w in res["warnings"]:
            html.append(f"<p style='color:#facc15'>⚠ {w}</p>")
        self.info.setHtml("".join(html))

    def _table_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self._focus_defect(rows[0].row())

    def _focus_defect(self, idx):
        if self.result is None or idx is None or idx >= len(self.result["boxes"]):
            return
        self.tabs.setCurrentIndex(4)
        self.v_result.highlight(self.result["boxes"][idx])
        self.table.blockSignals(True)
        self.table.selectRow(idx)
        self.table.blockSignals(False)

    # lưu kết quả
    def save_results(self):
        if self.result is None:
            return
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu kết quả")
        if not d:
            return
        name = Path(self.pick_test.path).stem
        out = Path(d) / f"gui_{name}"
        (out / "crops").mkdir(parents=True, exist_ok=True)
        r = self.result

        def write(fn, img):
            ok, buf = cv2.imencode(Path(fn).suffix, img)   # an toàn với đường dẫn Unicode
            if ok:
                buf.tofile(str(out / fn))

        write("result.jpg", r["result_img"])
        if r["gt_boxes"]:
            write("result_with_gt.jpg", r["result_gt_img"])
        if r["matches_img"] is not None:
            write("feature_matches.jpg", r["matches_img"])
        write("checkerboard_alignment.jpg", r["checker_img"])
        write("diff_map.png", r["diff"])
        write("mask.png", r["mask"])
        for i, p in enumerate(r["patches"], start=1):
            if p.size:
                write(f"crops/defect_{i:03d}.png", p)
        with open(out / "defects.csv", "w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.writer(fh)
            wr.writerow(["id", "x", "y", "w", "h", "label"])
            for i, (x, y, w, h) in enumerate(r["boxes"], start=1):
                wr.writerow([i, x, y, w, h, r["labels"][i - 1] if r["labels"] else ""])
        self.statusBar().showMessage(f"Đã lưu kết quả vào {out}")

# Theme tối (Fusion + palette)
def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    c = QColor
    p.setColor(QPalette.ColorRole.Window, c(39, 39, 42))
    p.setColor(QPalette.ColorRole.WindowText, c(228, 228, 231))
    p.setColor(QPalette.ColorRole.Base, c(24, 24, 27))
    p.setColor(QPalette.ColorRole.AlternateBase, c(39, 39, 42))
    p.setColor(QPalette.ColorRole.Text, c(228, 228, 231))
    p.setColor(QPalette.ColorRole.Button, c(63, 63, 70))
    p.setColor(QPalette.ColorRole.ButtonText, c(244, 244, 245))
    p.setColor(QPalette.ColorRole.ToolTipBase, c(24, 24, 27))
    p.setColor(QPalette.ColorRole.ToolTipText, c(228, 228, 231))
    p.setColor(QPalette.ColorRole.Highlight, c(37, 99, 235))
    p.setColor(QPalette.ColorRole.HighlightedText, c(255, 255, 255))
    p.setColor(QPalette.ColorRole.PlaceholderText, c(113, 113, 122))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, c(113, 113, 122))
    app.setPalette(p)
    app.setStyleSheet("""
        QGroupBox { border:1px solid #3f3f46; border-radius:6px; margin-top:12px; padding:10px 8px 8px 8px; font-weight:600; }
        QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px; color:#a1a1aa; }
        QPushButton { padding:6px 10px; border-radius:5px; border:1px solid #52525b; }
        QPushButton:hover { background:#52525b; }
        QPushButton:disabled { color:#71717a; border-color:#3f3f46; }
        QPushButton#primary { background:#2563eb; border:none; font-size:14px; font-weight:700; color:white; }
        QPushButton#primary:hover { background:#1d4ed8; }
        QPushButton#primary:disabled { background:#3f3f46; color:#71717a; }
        QTabBar::tab { padding:8px 14px; background:#27272a; border:1px solid #3f3f46; border-bottom:none; }
        QTabBar::tab:selected { background:#3f3f46; color:white; }
        QTabWidget::pane { border:1px solid #3f3f46; }
        QHeaderView::section { background:#3f3f46; color:#e4e4e7; padding:4px; border:none; }
        QProgressBar { background:#3f3f46; border:none; border-radius:3px; }
        QProgressBar::chunk { background:#2563eb; border-radius:3px; }
        QListWidget, QTextBrowser, QTableWidget { background:#18181b; border:1px solid #3f3f46; }
        QSplitter::handle { background:#3f3f46; }
    """)


def main():
    ap = argparse.ArgumentParser(description="GUI demo phát hiện lỗi PCB")
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    args, qt_args = ap.parse_known_args()
    app = QApplication([sys.argv[0]] + qt_args)
    apply_dark_theme(app)
    win = MainWindow(args.data_dir)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
