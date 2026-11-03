#!/usr/bin/env python3
"""Smoke test nhanh cho pipeline PCB.

Chạy:
    python tools/test_pipeline.py -v

Hai test được đánh dấu ``expectedFailure`` vì chúng phơi ra lỗi đã biết của
pipeline hiện tại (xem README/ghi chú đánh giá):
  * ảnh không lỗi nhưng bị nén JPEG lại => Otsu sinh rất nhiều false positive;
  * gai đồng (spur) rộng 1 px bị ``cv2.contourArea`` lọc mất.

Chúng sẽ tự chuyển thành PASS khi bạn sửa xong, lúc đó hãy bỏ decorator.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module_detection import crop_defect_patches, extract_bounding_boxes  # noqa: E402
from module_preprocess import align_images, detect_difference  # noqa: E402

SAMPLE_TEMP = ROOT / "data" / "92000000_temp.jpg"


def make_trace_canvas() -> np.ndarray:
    """Ảnh tổng hợp 200x200: nền trắng, một thanh đồng đen nằm ngang."""
    canvas = np.full((200, 200), 255, np.uint8)
    cv2.rectangle(canvas, (20, 20), (180, 60), 0, -1)
    return canvas


class PipelineSmokeTest(unittest.TestCase):
    def test_align_identity(self):
        img = cv2.imread(str(SAMPLE_TEMP), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img, "Thiếu data/92000000_temp.jpg")
        aligned = align_images(img, img)
        self.assertEqual(aligned.shape, img.shape)
        self.assertLess(np.mean(cv2.absdiff(img, aligned)), 2.0)

    def test_align_translation(self):
        img = cv2.imread(str(SAMPLE_TEMP), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        shifted = cv2.warpAffine(
            img, np.float32([[1, 0, 6], [0, 1, -4]]), (w, h), borderMode=cv2.BORDER_REPLICATE
        )
        aligned = align_images(img, shifted)
        diff = cv2.absdiff(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
        self.assertLess(np.mean(diff), 10.0)

    def test_detect_synthetic_hole(self):
        template = make_trace_canvas()
        test = template.copy()
        cv2.rectangle(test, (90, 30), (102, 42), 255, -1)  # mất đồng 13x13
        mask = detect_difference(template, test)
        boxes = extract_bounding_boxes(mask, min_area=20)
        self.assertEqual(len(boxes), 1, f"Kỳ vọng đúng 1 box, nhận {boxes}")
        x, y, w, h = boxes[0]
        self.assertLessEqual(abs(x - 90), 3)
        self.assertLessEqual(abs(y - 30), 3)

    def test_identical_images_no_defect(self):
        template = make_trace_canvas()
        boxes = extract_bounding_boxes(detect_difference(template, template), min_area=20)
        self.assertEqual(boxes, [])

    def test_crop_padding_and_clipping(self):
        template = make_trace_canvas()
        with tempfile.TemporaryDirectory() as tmp:
            paths = crop_defect_patches(template, [(20, 20, 10, 10), (195, 195, 5, 5)], tmp, padding=5)
            self.assertEqual(len(paths), 2)
            patch = cv2.imread(paths[0])
            self.assertEqual(patch.shape[:2], (20, 20))  # 10 + 2*5
            edge_patch = cv2.imread(paths[1])
            self.assertEqual(edge_patch.shape[:2], (10, 10))  # bị cắt ở biên ảnh

    def test_invalid_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                crop_defect_patches(make_trace_canvas(), [(1, 1, 5, 5)], tmp, padding=-1)

    @unittest.expectedFailure
    def test_jpeg_noise_does_not_create_boxes(self):
        """KNOWN ISSUE: nén JPEG lại làm Otsu sinh false positive khắp ảnh."""
        img = cv2.imread(str(SAMPLE_TEMP), cv2.IMREAD_COLOR)
        with tempfile.TemporaryDirectory() as tmp:
            noisy_path = Path(tmp) / "recompressed.jpg"
            cv2.imwrite(str(noisy_path), img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            noisy = cv2.imread(str(noisy_path), cv2.IMREAD_COLOR)
        aligned = align_images(img, noisy)
        boxes = extract_bounding_boxes(detect_difference(img, aligned), min_area=20)
        self.assertEqual(boxes, [], f"Có {len(boxes)} false positive trên ảnh không lỗi")

    @unittest.expectedFailure
    def test_thin_spur_is_detected(self):
        """KNOWN ISSUE: spur rộng 1 px có contourArea = 0 nên bị lọc mất."""
        template = make_trace_canvas()
        test = template.copy()
        cv2.line(test, (40, 100), (150, 100), 255, 1)
        boxes = extract_bounding_boxes(detect_difference(template, test), min_area=20)
        self.assertGreaterEqual(len(boxes), 1, "Bỏ sót spur 1 px")


if __name__ == "__main__":
    unittest.main(verbosity=2)
