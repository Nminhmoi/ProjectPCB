import time

import cv2
import numpy as np


# =====================================================================
# BO DO DAC TRUNG: ORB / SIFT / AKAZE
# ---------------------------------------------------------------------
# Ba detector dung chung MOT pipeline:
#   detectAndCompute -> BFMatcher cross-check -> loc theo keep_percentage
#   -> RANSAC homography -> kiem tra reprojection error -> warpPerspective
# Chi khac nhau o khoi tao detector va chuan khoang cach (norm) cua matcher.
# Vi vay chi can mot registry thay vi 3 module rieng.
# =====================================================================
DETECTORS = ("ORB", "SIFT", "AKAZE")
DEFAULT_DETECTOR = "ORB"
# Diem can bang do chinh xac / toc do, chot tu experiments/exp_max_features.py.
DEFAULT_MAX_FEATURES = 2000

# Ten cu, giu de code/nhan xet cu van doc duoc. main.py nen truyen --detector.
DETECTOR_NAME = DEFAULT_DETECTOR

MATCHER_NORMS = {
    "ORB": cv2.NORM_HAMMING,
    "SIFT": cv2.NORM_L2,
    "AKAZE": cv2.NORM_HAMMING,
}


def _normalize_detector(detector: str) -> str:
    name = str(detector).upper()
    if name not in DETECTORS:
        raise ValueError(
            f"Detector khong hop le: {detector!r}. Chon mot trong {DETECTORS}."
        )
    return name


def _create_akaze():
    # OpenCV 4.x: cv2.AKAZE_create; OpenCV 5.x chuyen sang cv2.xfeatures2d
    # (can cai opencv-contrib-python).
    if hasattr(cv2, "AKAZE_create"):
        return cv2.AKAZE_create()
    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "AKAZE_create"):
        return cv2.xfeatures2d.AKAZE_create()
    raise ImportError(
        "Ban OpenCV nay khong co AKAZE. Hay cai: pip install opencv-contrib-python"
    )


def is_detector_available(detector: str) -> bool:
    """Kiem tra detector co chay duoc voi OpenCV hien tai khong."""
    try:
        create_detector(detector)
        return True
    except (ImportError, ValueError):
        return False


def create_detector(detector: str = DEFAULT_DETECTOR):
    """Tao doi tuong detector cua OpenCV theo ten."""
    name = _normalize_detector(detector)
    if name == "ORB":
        return cv2.ORB_create()
    if name == "SIFT":
        return cv2.SIFT_create()
    return _create_akaze()


def detect_and_describe(
        gray: np.ndarray,
        max_features: int,
        detector: str = DEFAULT_DETECTOR):
    """Tra ve (keypoints, descriptors) cua anh xam.

    - ORB / SIFT: gioi han `nfeatures`.
    - AKAZE: khong co tham so nfeatures, nen giu lai `max_features`
      keypoint co response manh nhat de so sanh cong bang.
    """
    name = _normalize_detector(detector)

    if name == "ORB":
        orb = cv2.ORB_create(nfeatures=max_features)
        return orb.detectAndCompute(gray, None)

    if name == "SIFT":
        sift = cv2.SIFT_create(nfeatures=max_features)
        return sift.detectAndCompute(gray, None)

    akaze = _create_akaze()
    keypoints = akaze.detect(gray, None)
    if max_features and len(keypoints) > max_features:
        keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)
        keypoints = keypoints[:max_features]
    return akaze.compute(gray, keypoints)


class AlignmentError(RuntimeError):
    """Loi can chinh, kem theo so lieu da do duoc (de benchmark)."""

    def __init__(self, message: str, stats: dict):
        super().__init__(message)
        self.stats = stats


def _to_gray(image: np.ndarray, name: str) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"{name} co so kenh anh khong hop le!")


def _gray_clahe(image: np.ndarray) -> np.ndarray:
    """Chuyen anh sang gray + can bang sang cuc bo CLAHE.

    Phuong phap thang trong khao sat RGB -> Binary (experiments/exp_rgb2binary.py):
    CLAHE giu bien canh tot nhat (0.356) va on dinh voi anh sang lech (IoU 0.870),
    dong thoi giam hien tuong loa sang tren be mat kim loai cua PCB.
    """
    if image.ndim not in (2, 3):
        raise ValueError("Anh dau vao phai la anh grayscale hoac anh mau")
    gray = _to_gray(image, "Anh")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def align_images_with_stats(
        img_template: np.ndarray,
        img_test: np.ndarray,
        max_features: int = DEFAULT_MAX_FEATURES,
        keep_percentage: float = 0.8,
        detector: str = DEFAULT_DETECTOR):
    """Can chinh anh test theo anh template.

    Tra ve (img_test_aligned, stats). `stats` gom so keypoint, so cap khop,
    inliers, reprojection error, thoi gian tung buoc va ma tran H.
    Neu that bai -> raise AlignmentError (co thuoc tinh .stats).

    `detector` chon bo do dac trung: "ORB" (mac dinh), "SIFT" hoac "AKAZE".
    """
    name = _normalize_detector(detector)
    stats = {
        "detector": name,
        "success": False,
        "kp_template": 0,
        "kp_test": 0,
        "matches": 0,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "reprojection_error": float("nan"),
        "time_detect_ms": 0.0,
        "time_match_ms": 0.0,
        "time_homography_ms": 0.0,
        "time_warp_ms": 0.0,
        "time_total_ms": 0.0,
        "H": None,
        "inlier_pts_test": None,
        "inlier_pts_template": None,
        "error": "",
    }

    def fail(message: str):
        stats["error"] = message
        raise AlignmentError(message, stats)

    # Kiem tra input
    if img_template is None or img_test is None:
        raise ValueError("Anh template va anh test khong hop le")
    if img_template.ndim not in (2, 3) or img_test.ndim not in (2, 3):
        raise ValueError("Anh dau vao phai la anh grayscale hoac anh mau")
    if img_template.size == 0 or img_test.size == 0:
        raise ValueError("Anh dau vao rong")
    if not (0 < keep_percentage <= 1):
        raise ValueError("keep_percentage phai nam trong (0,1].")

    gray_template = _to_gray(img_template, "Template")
    gray_test = _to_gray(img_test, "Test")

    t_start = time.perf_counter()

    # 1. Do dac trung (detector co the khong co san, vd AKAZE thieu contrib)
    try:
        kp_template, des_template = detect_and_describe(
            gray_template, max_features, name
        )
        kp_test, des_test = detect_and_describe(gray_test, max_features, name)
    except ImportError as exc:
        fail(str(exc))
    t_detect = time.perf_counter()
    stats["time_detect_ms"] = (t_detect - t_start) * 1000.0
    stats["kp_template"] = len(kp_template)
    stats["kp_test"] = len(kp_test)

    # Neu anh qua tron/it chi tiet thi khong trich xuat duoc keypoint
    if des_template is None or des_test is None:
        fail("Khong the trich xuat diem dac trung tu anh dau vao.")

    # 2. Ghep cap (cross-check), giu lai diem match tot nhat
    matcher = cv2.BFMatcher(MATCHER_NORMS[name], crossCheck=True)
    matches = matcher.match(des_test, des_template)
    t_match = time.perf_counter()
    stats["time_match_ms"] = (t_match - t_detect) * 1000.0
    stats["matches"] = len(matches)

    # Homography co 8 bac tu do nen can toi thieu 4 diem khong thang hang
    if len(matches) < 4:
        fail(f"Chi co {len(matches)} cap khop, can it nhat 4 de tinh homography")

    matches = sorted(matches, key=lambda x: x.distance)  # distance tang dan
    num_keeps = min(len(matches), max(int(len(matches) * keep_percentage), 4))
    good_matches = matches[:num_keeps]
    stats["good_matches"] = len(good_matches)

    pts_test = np.float32(
        [kp_test[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    pts_template = np.float32(
        [kp_template[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # 3. Tinh Homography bang RANSAC
    H, mask = cv2.findHomography(pts_test, pts_template, cv2.RANSAC, 3.0)
    t_homo = time.perf_counter()
    stats["time_homography_ms"] = (t_homo - t_match) * 1000.0

    if H is None or mask is None:
        fail("Khong tim thay ma tran homography phu hop!")

    inliers = mask.ravel().astype(bool)  # mask RANSAC: 1 = inlier
    num_inliers = int(np.count_nonzero(inliers))
    inlier_ratio = num_inliers / len(good_matches)
    stats["inliers"] = num_inliers
    stats["inlier_ratio"] = inlier_ratio
    # Toa do cac cap inlier (dung de ve duong noi khi truc quan hoa)
    stats["inlier_pts_test"] = pts_test[inliers].reshape(-1, 2)
    stats["inlier_pts_template"] = pts_template[inliers].reshape(-1, 2)

    # Doi chieu cac diem test qua H voi toa do template tuong ung
    if num_inliers > 0:
        projected = cv2.perspectiveTransform(pts_test[inliers], H)
        errors = np.linalg.norm(projected - pts_template[inliers], axis=2)
        # Dung median thay vi mean de tranh vai outlier lam lech ket qua
        stats["reprojection_error"] = float(np.median(errors))

    if num_inliers < 12 or inlier_ratio < 0.15:
        fail(
            f"Homography khong dang tin cay: {num_inliers} inliers, "
            f"ty le {inlier_ratio:.2f}."
        )

    reprojection_error = stats["reprojection_error"]
    if not np.isfinite(reprojection_error) or reprojection_error >= 3.0:
        fail(f"Reprojection error qua lon: {reprojection_error:.2f} px.")

    if not np.all(np.isfinite(H)):
        fail("Homography chua gia tri khong hop le.")
    if abs(H[2, 2]) < 1e-8:
        fail("Homography khong the chuan hoa.")

    H = H / H[2, 2]  # Chuan hoa H[2,2] = 1

    # det cua phan affine 2x2 xap xi ti le dien tich anh
    det = np.linalg.det(H[:2, :2])
    if not (0.5 < det < 2.0):
        fail(f"Homography gay bien dang bat thuong (det = {det:.3f}).")

    # 4. Nan anh test ve he toa do template
    h, w = gray_template.shape[:2]
    img_test_aligned = cv2.warpPerspective(
        img_test, H, (w, h),
        flags=cv2.INTER_LINEAR,  # noi suy tu cac pixel lan can
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    t_end = time.perf_counter()
    stats["time_warp_ms"] = (t_end - t_homo) * 1000.0
    stats["time_total_ms"] = (t_end - t_start) * 1000.0
    stats["H"] = H
    stats["success"] = True

    return img_test_aligned, stats


def align_images(
        img_template: np.ndarray,
        img_test: np.ndarray,
        max_features: int = DEFAULT_MAX_FEATURES,
        keep_percentage: float = 0.8,
        detector: str = DEFAULT_DETECTOR) -> np.ndarray:
    """Giu nguyen giao dien cu: chi tra ve anh da can chinh."""
    aligned, _ = align_images_with_stats(
        img_template, img_test, max_features, keep_percentage, detector
    )
    return aligned


def gaussian_blur(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Gaussian blur dung chung cho pipeline va thi nghiem kernel.

    Kernel phai le (co pixel tam). ksize=1 nghia la khong lam mo.
    """
    if ksize < 1:
        raise ValueError("ksize phai >= 1.")
    if ksize % 2 == 0:
        raise ValueError(f"ksize phai la so le, nhan duoc {ksize}.")
    if ksize == 1:
        return gray.copy()
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def difference_map(
        img_template: np.ndarray,
        img_test_aligned: np.ndarray,
        blur_ksize: int = 3):
    """Buoc lam mo + tru vi sai co dung sai vi tri (±2 px).

    Tra ve (diff, time_blur_ms). `diff` la ban do vi sai CHUA nhi phan,
    dung de do SNR hoac truc quan hoa. Thi nghiem kernel goi ham nay roi
    tu nhi phan bang `binarize_difference`.
    """
    if img_template is None or img_test_aligned is None:
        raise ValueError("Anh dau vao khong hop le.")

    # Chuyen gray + CLAHE (phuong phap thang cua khao sat RGB -> Binary).
    gray_template = _gray_clahe(img_template)
    gray_test = _gray_clahe(img_test_aligned)

    if gray_template.shape != gray_test.shape:
        raise ValueError("Kich thuoc template khong giong nhau")

    h, w = gray_template.shape
    if h <= 16 or w <= 16:
        raise ValueError("Anh qua nho de loai bo vung bien 8 pixel.")

    start = time.perf_counter()
    blur_template = gaussian_blur(gray_template, blur_ksize)
    blur_test = gaussian_blur(gray_test, blur_ksize)
    blur_ms = (time.perf_counter() - start) * 1000.0

    kernel = np.ones((5, 5), np.uint8)
    template_eroded = cv2.erode(blur_template, kernel)
    template_dilated = cv2.dilate(blur_template, kernel)

    test_brighter = cv2.subtract(blur_test, template_dilated)  # loi open, pin-hole,..
    test_darker = cv2.subtract(template_eroded, blur_test)  # loi short, spur,..

    diff = cv2.max(test_brighter, test_darker)
    return diff, blur_ms


def binarize_difference(diff: np.ndarray, border: int = 8) -> np.ndarray:
    """Otsu + nguong san + bo vien + morphology opening."""
    if diff is None or diff.size == 0:
        raise ValueError("Ban do vi sai rong.")

    h, w = diff.shape[:2]
    if h <= 2 * border or w <= 2 * border:
        raise ValueError(f"Anh qua nho de loai bo vung bien {border} pixel.")

    # Nhi phan hoa Otsu + nguong san
    otsu_val, _ = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold_value = max(int(otsu_val), 30)

    _, diff_binary = cv2.threshold(
        diff, threshold_value, 255, cv2.THRESH_BINARY
    )

    # Bo vien anh
    diff_binary[:border, :] = 0
    diff_binary[-border:, :] = 0
    diff_binary[:, :border] = 0
    diff_binary[:, -border:] = 0

    # Khu nhieu bang morphology opening
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    diff_cleaned = cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, clean_kernel)

    return diff_cleaned


def detect_difference(
    img_template: np.ndarray,
    img_test_aligned: np.ndarray,
    blur_ksize: int = 3,
) -> np.ndarray:
    """Tru vi sai template voi anh test da can chinh, tra ve mask nhi phan.

    `blur_ksize` cho phep thi nghiem doi kich thuoc kernel Gaussian (mac dinh 3).
    """
    diff, _ = difference_map(img_template, img_test_aligned, blur_ksize)
    return binarize_difference(diff)
