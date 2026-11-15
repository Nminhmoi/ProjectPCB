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

# --- P0.1: kiem chung homography bang diem anh (NCC) ----------------------
# Sau khi tinh H, so khop template voi anh test da nan o do phan giai thap.
# Can chinh dung tren PKU cho NCC ~0.998; cac cap can lech 7-47 px cho
# NCC 0.80-0.95 (EXPERIMENT_REVIEW.md, muc 4 P0.1). Nguong chot bang
# experiments/exp_alignment_check.py tren 693 cap. None = tat kiem tra.
DEFAULT_MIN_NCC = 0.97
APPEARANCE_WIDTH = 400          # chieu rong (px) anh thu nho de tinh NCC (~6 ms)
APPEARANCE_BLUR = 3             # Gaussian sau khi thu nho: can bang do sac net giua
                                # template (khong noi suy) va anh test (da noi suy).
                                # Do tren anh mau: khop dung 0.995, lech 2 px 0.984,
                                # lech 4 px 0.941, lech 7 px 0.857.
# Thu lan luot khi detector chinh khong vuot kiem tra (align_images_robust).
DEFAULT_FALLBACKS = (("SIFT", DEFAULT_MAX_FEATURES),)

# --- Tham so nhi phan hoa ban do vi sai (P0.3 / P1.1) ---------------------
# Gia tri mac dinh = pipeline da chay 693 cap (F1 0.972). Doi bang
# experiments/exp_detection_params.py truoc khi sua o day.
DEFAULT_THRESHOLD_FLOOR = 30    # nguong san: max(Otsu, floor)
DEFAULT_OPEN_KSIZE = 3          # kernel morphology opening (<=1: bo opening)
DEFAULT_HYSTERESIS_LOW = None   # vd 0.5: giu vung >= 0.5*nguong neu noi voi vung manh
DEFAULT_MIN_AREA = 20           # dien tich toi thieu cua dom loi (module_detection)

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


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 1e-9 else float("nan")


def appearance_score(
        img_template: np.ndarray,
        img_test: np.ndarray,
        H: np.ndarray,
        width: int = APPEARANCE_WIDTH,
        border: int = 3,
        test_mask: "np.ndarray | None" = None) -> dict:
    """Do muc khop diem anh giua template va anh test sau khi nan bang H.

    Tinh o do phan giai thap (chieu rong `width`) cho nhanh (~vai ms):
      - ncc      : he so tuong quan Pearson cua cuong do sang (1 = khop hoan toan)
      - ncc_grad : Pearson cua do lon gradient (Sobel) - nhay hon voi lech vai px
      - mae      : sai khac tuyet doi trung binh (muc xam 0-255)
      - valid_fraction : ty le dien tich template duoc anh test phu (sau khi nan)
    Chi tinh tren vung hop le: bo phan den do warp, co them `border` px.
    `test_mask` (tuy chon, cung kich thuoc anh test, >0 = pixel that): dung khi
    anh test tu no da co vung den (vd thi nghiem nan lech gia lap).
    Khong can nhan GT nen chay duoc tren du lieu that.
    """
    gray_t = _to_gray(img_template, "Template")
    gray_s = _to_gray(img_test, "Test")
    h_t, w_t = gray_t.shape[:2]
    h_s, w_s = gray_s.shape[:2]

    scale_t = min(1.0, width / float(w_t))
    scale_s = min(1.0, width / float(w_s))
    size_t = (max(1, int(round(w_t * scale_t))), max(1, int(round(h_t * scale_t))))
    size_s = (max(1, int(round(w_s * scale_s))), max(1, int(round(h_s * scale_s))))
    small_t = cv2.resize(gray_t, size_t, interpolation=cv2.INTER_AREA)
    small_s = cv2.resize(gray_s, size_s, interpolation=cv2.INTER_AREA)

    # H o do phan giai thap: S_t * H * S_s^-1
    S_t = np.diag([scale_t, scale_t, 1.0])
    S_s_inv = np.diag([1.0 / scale_s, 1.0 / scale_s, 1.0])
    H_small = S_t @ H @ S_s_inv
    warped = cv2.warpPerspective(small_s, H_small, size_t, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    src_valid = np.full(small_s.shape[:2], 255, np.uint8)
    if test_mask is not None:
        src_valid = cv2.resize(test_mask, size_s, interpolation=cv2.INTER_NEAREST)
        src_valid = np.where(src_valid > 0, 255, 0).astype(np.uint8)
    valid = cv2.warpPerspective(src_valid, H_small, size_t,
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
    if border > 0:
        valid = cv2.erode(valid, np.ones((2 * border + 1, 2 * border + 1), np.uint8))
    if APPEARANCE_BLUR and APPEARANCE_BLUR > 1:
        k = (APPEARANCE_BLUR, APPEARANCE_BLUR)
        small_t = cv2.GaussianBlur(small_t, k, 0)
        warped = cv2.GaussianBlur(warped, k, 0)
    mask = valid > 0
    result = {"ncc": float("nan"), "ncc_grad": float("nan"), "mae": float("nan"),
              "valid_fraction": float(np.count_nonzero(mask)) / mask.size}
    if np.count_nonzero(mask) < 500:
        return result

    a = small_t[mask]
    b = warped[mask]
    result["ncc"] = _pearson(a, b)
    result["mae"] = float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))

    def grad_mag(img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    inner = cv2.erode(valid, np.ones((3, 3), np.uint8)) > 0  # bo vien gradient gia
    if np.count_nonzero(inner) >= 500:
        result["ncc_grad"] = _pearson(grad_mag(small_t)[inner], grad_mag(warped)[inner])
    return result


def align_images_with_stats(
        img_template: np.ndarray,
        img_test: np.ndarray,
        max_features: int = DEFAULT_MAX_FEATURES,
        keep_percentage: float = 0.8,
        detector: str = DEFAULT_DETECTOR,
        min_ncc: "float | None" = DEFAULT_MIN_NCC,
        test_mask: "np.ndarray | None" = None):
    """Can chinh anh test theo anh template.

    Tra ve (img_test_aligned, stats). `stats` gom so keypoint, so cap khop,
    inliers, reprojection error, thoi gian tung buoc va ma tran H.
    Neu that bai -> raise AlignmentError (co thuoc tinh .stats).

    `detector` chon bo do dac trung: "ORB" (mac dinh), "SIFT" hoac "AKAZE".
    `min_ncc`  nguong NCC kiem chung homography bang diem anh (P0.1);
               None = bo qua kiem tra (van tinh va ghi NCC vao stats).
    `test_mask` vung pixel that cua anh test (chi dung cho NCC), xem appearance_score.
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
        "time_verify_ms": 0.0,
        "time_total_ms": 0.0,
        "ncc": float("nan"),
        "ncc_grad": float("nan"),
        "mae": float("nan"),
        "valid_fraction": float("nan"),
        "failed_check": "",
        "H": None,
        "H_candidate": None,
        "inlier_pts_test": None,
        "inlier_pts_template": None,
        "error": "",
    }

    def fail(message: str, check: str = "geometry"):
        stats["error"] = message
        stats["failed_check"] = check
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

    # 4. Kiem chung bang diem anh (P0.1): cac kiem tra tren chi dua vao tap
    # inlier nen khong bat duoc H "sai nhung tu nhat quan" (vd khop nham
    # sang mot hoa van lap lai). NCC so sanh truc tiep noi dung anh.
    stats["H_candidate"] = H
    t_verify = time.perf_counter()
    stats.update(appearance_score(gray_template, gray_test, H, test_mask=test_mask))
    stats["time_verify_ms"] = (time.perf_counter() - t_verify) * 1000.0
    if min_ncc is not None:
        ncc = stats["ncc"]
        if not np.isfinite(ncc) or ncc < min_ncc:
            stats["time_total_ms"] = (time.perf_counter() - t_start) * 1000.0
            fail(f"Kiem chung diem anh that bai: NCC {ncc:.3f} < {min_ncc:.3f}.",
                 check="appearance")

    # 5. Nan anh test ve he toa do template
    h, w = gray_template.shape[:2]
    img_test_aligned = cv2.warpPerspective(
        img_test, H, (w, h),
        flags=cv2.INTER_LINEAR,  # noi suy tu cac pixel lan can
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    t_end = time.perf_counter()
    stats["time_warp_ms"] = (t_end - t_homo) * 1000.0 - stats["time_verify_ms"]
    stats["time_total_ms"] = (t_end - t_start) * 1000.0
    stats["H"] = H
    stats["success"] = True

    return img_test_aligned, stats


def align_images(
        img_template: np.ndarray,
        img_test: np.ndarray,
        max_features: int = DEFAULT_MAX_FEATURES,
        keep_percentage: float = 0.8,
        detector: str = DEFAULT_DETECTOR,
        min_ncc: "float | None" = DEFAULT_MIN_NCC) -> np.ndarray:
    """Giu nguyen giao dien cu: chi tra ve anh da can chinh."""
    aligned, _ = align_images_with_stats(
        img_template, img_test, max_features, keep_percentage, detector, min_ncc
    )
    return aligned


def align_images_robust(
        img_template: np.ndarray,
        img_test: np.ndarray,
        max_features: int = DEFAULT_MAX_FEATURES,
        keep_percentage: float = 0.8,
        detector: str = DEFAULT_DETECTOR,
        fallbacks=DEFAULT_FALLBACKS,
        min_ncc: "float | None" = DEFAULT_MIN_NCC,
        test_mask: "np.ndarray | None" = None):
    """Can chinh co phuong an du phong (dung trong main.py).

    1. Thu detector chinh (mac dinh ORB). Vuot moi kiem tra -> dung luon.
    2. Neu that bai, thu lan luot `fallbacks` (mac dinh SIFT: cham hon ~6 lan
       nhung chinh xac hon ~8 lan, chi chay cho so it cap bi tu choi).
    3. Neu khong lan nao vuot kiem tra NCC nhung co H hop le ve hinh hoc,
       chon H co NCC cao nhat va danh dau stats["verified"] = False
       (van tot hon dung anh chua can chinh).
    4. Khong co H nao -> raise AlignmentError.

    stats bo sung: verified, fallback_used, attempts (danh sach ket qua tung lan),
    time_all_attempts_ms.
    """
    attempts = [(detector, max_features)]
    for item in (fallbacks or ()):
        det, mf = (item, max_features) if isinstance(item, str) else item
        if (det.upper(), mf) not in [(d.upper(), m) for d, m in attempts]:
            attempts.append((det, mf))

    log = []
    best = None          # (ncc, stats) cua lan chi hong kiem tra NCC
    last_stats = None
    total_ms = 0.0
    for det, mf in attempts:
        if not is_detector_available(det):
            log.append(f"{det.upper()}:khong co trong OpenCV")
            continue
        try:
            aligned, stats = align_images_with_stats(
                img_template, img_test, mf, keep_percentage, det, min_ncc, test_mask)
            total_ms += stats["time_total_ms"]
            log.append(f"{det.upper()}/{mf}:ok (NCC {stats['ncc']:.3f})")
            stats.update(verified=True, attempts=log,
                         fallback_used=(det.upper(), mf) != (detector.upper(), max_features),
                         time_all_attempts_ms=total_ms)
            return aligned, stats
        except AlignmentError as exc:
            stats = exc.stats
            last_stats = stats
            total_ms += stats.get("time_total_ms", 0.0)
            log.append(f"{det.upper()}/{mf}:{exc}")
            if stats.get("failed_check") == "appearance" and stats.get("H_candidate") is not None:
                ncc = stats.get("ncc", float("nan"))
                score = ncc if np.isfinite(ncc) else -1.0
                if best is None or score > best[0]:
                    best = (score, stats)

    if best is not None:
        stats = dict(best[1])
        H = stats["H_candidate"]
        h, w = img_template.shape[:2]
        aligned = cv2.warpPerspective(img_test, H, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        stats.update(success=True, H=H, verified=False, attempts=log,
                     fallback_used=stats["detector"] != detector.upper(),
                     time_all_attempts_ms=total_ms,
                     error="Khong lan nao vuot kiem tra NCC; dung H co NCC cao nhat.")
        return aligned, stats

    if last_stats is None:
        last_stats = {"success": False, "error": "Khong co detector nao chay duoc."}
    last_stats.update(attempts=log, verified=False, time_all_attempts_ms=total_ms)
    raise AlignmentError("Can chinh that bai o moi phuong an: " + " | ".join(log), last_stats)


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


def binarize_difference(
        diff: np.ndarray,
        border: int = 8,
        threshold_floor: int = DEFAULT_THRESHOLD_FLOOR,
        open_ksize: int = DEFAULT_OPEN_KSIZE,
        hysteresis_low: "float | None" = DEFAULT_HYSTERESIS_LOW) -> np.ndarray:
    """Otsu + nguong san + bo vien + morphology opening.

    threshold_floor : nguong = max(Otsu, threshold_floor)
    open_ksize      : kernel opening khu nhieu; <= 1 thi bo opening
    hysteresis_low  : None = nguong don. Neu la so trong (0, 1), giu them cac
                      pixel >= hysteresis_low * nguong MIEN LA noi lien voi
                      mot pixel vuot nguong (nhu Canny). Giup giu tron net loi
                      manh / tuong phan yeu ma khong them dom nhieu rieng le.
    """
    if diff is None or diff.size == 0:
        raise ValueError("Ban do vi sai rong.")

    h, w = diff.shape[:2]
    if h <= 2 * border or w <= 2 * border:
        raise ValueError(f"Anh qua nho de loai bo vung bien {border} pixel.")

    # Nhi phan hoa Otsu + nguong san
    otsu_val, _ = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold_value = max(int(otsu_val), int(threshold_floor))

    _, diff_binary = cv2.threshold(
        diff, threshold_value, 255, cv2.THRESH_BINARY
    )

    if hysteresis_low is not None and 0 < hysteresis_low < 1:
        low_value = max(1, int(threshold_value * hysteresis_low))
        _, weak = cv2.threshold(diff, low_value, 255, cv2.THRESH_BINARY)
        n_labels, labels = cv2.connectedComponents(weak, connectivity=8)
        if n_labels > 1:
            keep = np.zeros(n_labels, bool)
            keep[np.unique(labels[diff_binary > 0])] = True
            keep[0] = False
            diff_binary = np.where(keep[labels], 255, 0).astype(np.uint8)

    # Bo vien anh
    diff_binary[:border, :] = 0
    diff_binary[-border:, :] = 0
    diff_binary[:, :border] = 0
    diff_binary[:, -border:] = 0

    # Khu nhieu bang morphology opening
    if open_ksize and open_ksize > 1:
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
        return cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, clean_kernel)
    return diff_binary


def detect_difference(
    img_template: np.ndarray,
    img_test_aligned: np.ndarray,
    blur_ksize: int = 3,
    threshold_floor: int = DEFAULT_THRESHOLD_FLOOR,
    open_ksize: int = DEFAULT_OPEN_KSIZE,
    hysteresis_low: "float | None" = DEFAULT_HYSTERESIS_LOW,
) -> np.ndarray:
    """Tru vi sai template voi anh test da can chinh, tra ve mask nhi phan.

    `blur_ksize` cho phep thi nghiem doi kich thuoc kernel Gaussian (mac dinh 3).
    Cac tham so con lai chuyen thang cho `binarize_difference`.
    """
    diff, _ = difference_map(img_template, img_test_aligned, blur_ksize)
    return binarize_difference(diff, threshold_floor=threshold_floor,
                               open_ksize=open_ksize, hysteresis_low=hysteresis_low)
