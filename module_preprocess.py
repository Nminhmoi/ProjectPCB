import time

import cv2
import numpy as np

# Bộ dò đặc trưng: ORB / SIFT / AKAZE.
# Cả ba dùng chung một pipeline (dò → ghép → RANSAC → nắn ảnh), chỉ khác cách
# khởi tạo detector và cách đo khoảng cách của matcher.
DETECTORS = ("ORB", "SIFT", "AKAZE")
DEFAULT_DETECTOR = "ORB"
# Điểm cân bằng giữa chính xác và tốc độ, chốt từ experiments/exp_max_features.py.
DEFAULT_MAX_FEATURES = 2000

# Tên cũ, giữ để code cũ vẫn chạy; nên truyền --detector thay vì dùng biến này.
DETECTOR_NAME = DEFAULT_DETECTOR

# Ngưỡng NCC để kiểm chứng Homography, chặn trường hợp H khớp nhầm vào hoạ tiết lặp.
# Trên 693 cặp PKU: NCC thấp nhất 0.9950, trung vị 0.9998 → 0.90 là biên an toàn.
# None = tắt kiểm tra.
DEFAULT_MIN_NCC = 0.90
APPEARANCE_WIDTH = 400          # chiều rộng ảnh thu nhỏ để tính NCC (px)
APPEARANCE_BLUR = 3             # làm mờ nhẹ để template và ảnh test cùng độ nét

# Mặc định không dùng detector dự phòng: PKU chưa cặp nào bị từ chối, còn SIFT
# tốn thêm ~1.5 s/cặp. Bật khi cần bằng --fallback SIFT.
DEFAULT_FALLBACKS = ()

# Tham số nhị phân hoá bản đồ vi sai, chốt từ lưới 60 cấu hình
# (experiments/detection_defaults.md). Đổi thì chạy lại experiments/verify_defaults.py.
DEFAULT_THRESHOLD_FLOOR = 20    # ngưỡng sàn: max(Otsu, floor)
DEFAULT_OPEN_KSIZE = 2          # kernel opening (<= 1: bỏ opening)
DEFAULT_HYSTERESIS_LOW = None   # vd 0.5: giữ vùng yếu nếu nối với vùng mạnh
DEFAULT_MIN_AREA = 20           # diện tích tối thiểu của đốm lỗi (module_detection)

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
    # OpenCV 5.x: AKAZE nằm ở cv2.xfeatures2d (cần opencv-contrib-python).
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

    Tinh o do phan giai thap (chieu rong `width`) de giam chi phi; do tren 693
    cap PKU la ~38 ms/cap. Cac chi so:
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

    # Đưa H về ảnh thu nhỏ: S_t · H · S_s⁻¹
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

    inner = cv2.erode(valid, np.ones((3, 3), np.uint8)) > 0  # bỏ viền gradient giả
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

    # Kiểm tra đầu vào
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

    # 1. Dò đặc trưng (AKAZE cần opencv-contrib-python)
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

    # Ảnh quá trơn thì không có keypoint
    if des_template is None or des_test is None:
        fail("Khong the trich xuat diem dac trung tu anh dau vao.")

    # 2. Ghép cặp (cross-check), giữ lại các match tốt nhất
    matcher = cv2.BFMatcher(MATCHER_NORMS[name], crossCheck=True)
    matches = matcher.match(des_test, des_template)
    t_match = time.perf_counter()
    stats["time_match_ms"] = (t_match - t_detect) * 1000.0
    stats["matches"] = len(matches)

    # Homography có 8 bậc tự do, cần ít nhất 4 điểm không thẳng hàng
    if len(matches) < 4:
        fail(f"Chi co {len(matches)} cap khop, can it nhat 4 de tinh homography")

    matches = sorted(matches, key=lambda x: x.distance)  # khoảng cách tăng dần
    num_keeps = min(len(matches), max(int(len(matches) * keep_percentage), 4))
    good_matches = matches[:num_keeps]
    stats["good_matches"] = len(good_matches)

    pts_test = np.float32(
        [kp_test[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    pts_template = np.float32(
        [kp_template[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # 3. Ước lượng Homography bằng RANSAC
    H, mask = cv2.findHomography(pts_test, pts_template, cv2.RANSAC, 3.0)
    t_homo = time.perf_counter()
    stats["time_homography_ms"] = (t_homo - t_match) * 1000.0

    if H is None or mask is None:
        fail("Khong tim thay ma tran homography phu hop!")

    inliers = mask.ravel().astype(bool)  # mask RANSAC: 1 = điểm inlier
    num_inliers = int(np.count_nonzero(inliers))
    inlier_ratio = num_inliers / len(good_matches)
    stats["inliers"] = num_inliers
    stats["inlier_ratio"] = inlier_ratio
    # Toạ độ các cặp inlier, dùng để vẽ đường nối khi trực quan hoá
    stats["inlier_pts_test"] = pts_test[inliers].reshape(-1, 2)
    stats["inlier_pts_template"] = pts_template[inliers].reshape(-1, 2)

    # Chiếu điểm test qua H rồi so với điểm template tương ứng
    if num_inliers > 0:
        projected = cv2.perspectiveTransform(pts_test[inliers], H)
        errors = np.linalg.norm(projected - pts_template[inliers], axis=2)
        # Dùng median để vài outlier không kéo lệch kết quả
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

    H = H / H[2, 2]  # chuẩn hoá để H[2,2] = 1

    # định thức khối affine 2×2, xấp xỉ tỉ lệ diện tích ảnh
    det = np.linalg.det(H[:2, :2])
    if not (0.5 < det < 2.0):
        fail(f"Homography gay bien dang bat thuong (det = {det:.3f}).")

    # 4. Kiểm chứng bằng NCC: các kiểm tra trên chỉ dựa vào inlier nên vẫn có thể
    # nhận H sai nhưng "tự nhất quán" (khớp nhầm hoạ tiết lặp lại).
    stats["H_candidate"] = H
    t_verify = time.perf_counter()
    stats.update(appearance_score(gray_template, gray_test, H, test_mask=test_mask))
    t_verify_end = time.perf_counter()
    stats["time_verify_ms"] = (t_verify_end - t_verify) * 1000.0
    if min_ncc is not None:
        ncc = stats["ncc"]
        if not np.isfinite(ncc) or ncc < min_ncc:
            stats["time_total_ms"] = (time.perf_counter() - t_start) * 1000.0
            fail(f"Kiem chung diem anh that bai: NCC {ncc:.3f} < {min_ncc:.3f}.",
                 check="appearance")

    # 5. Nắn ảnh test về hệ toạ độ template
    h, w = gray_template.shape[:2]
    img_test_aligned = cv2.warpPerspective(
        img_test, H, (w, h),
        flags=cv2.INTER_LINEAR,  # nội suy từ các pixel lân cận
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    t_end = time.perf_counter()
    # Chỉ tính riêng bước warp: t_end - t_verify_end. Công thức cũ
    # trừ cả kiểm tra hình học nên có thể ra số âm.
    stats["time_warp_ms"] = max(0.0, (t_end - t_verify_end) * 1000.0)
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
        img_template, img_test,
        max_features=max_features,
        keep_percentage=keep_percentage,
        detector=detector,
        min_ncc=min_ncc,
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
    2. Neu that bai, thu lan luot `fallbacks`. Mac dinh KHONG co du phong; bat
       bang `--fallback SIFT` (cham hon ~6 lan nhung chinh xac hon, chi chay cho
       so it cap bi tu choi).
    3. Neu khong lan nao vuot kiem tra NCC nhung co H hop le ve hinh hoc,
       chon H co NCC cao nhat va danh dau stats["verified"] = False
       (van tot hon dung anh chua can chinh).
    4. Khong co H nao -> raise AlignmentError.

    NCC la chot chan tham hoa (mac dinh 0.90), khong dung de tinh chinh: tren
    693 cap PKU NCC min 0.9950. Xem CODE_REVIEW.md muc 2.1.

    stats bo sung: verified, fallback_used, attempts (danh sach ket qua tung lan),
    time_all_attempts_ms.
    """
    attempts = [(detector, max_features)]
    for item in (fallbacks or ()):
        det, mf = (item, max_features) if isinstance(item, str) else item
        if (det.upper(), mf) not in [(d.upper(), m) for d, m in attempts]:
            attempts.append((det, mf))

    log = []
    best = None          # (ncc, stats) của lần hỏng ở bước kiểm tra NCC
    last_stats = None
    total_ms = 0.0
    for det, mf in attempts:
        if not is_detector_available(det):
            log.append(f"{det.upper()}:khong co trong OpenCV")
            continue
        try:
            aligned, stats = align_images_with_stats(
                img_template, img_test,
                max_features=mf,
                keep_percentage=keep_percentage,
                detector=det,
                min_ncc=min_ncc,
                test_mask=test_mask,
            )
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


def warp_valid_mask(
        img_test: np.ndarray,
        H: np.ndarray,
        out_shape,
        erode_border: int = 0) -> np.ndarray:
    """Mask pixel THAT cua anh test sau khi nan bang H (255 = noi dung that, 0 = vung trong).

    Anh test khi warp se co vung den o mep (do xoay/dich/co gian). Neu khong loai
    vung nay, ban do vi sai co mot dai tuong phan cuc dai o bien; Otsu se bi dai
    nay chi phoi va day nguong len rat cao -> mat het loi manh.

    `out_shape` = (h, w) hoac shape cua template. `erode_border` > 0 de bo them
    vai pixel sat bien (noi noi suy de sai).
    """
    if H is None:
        raise ValueError("Can ma tran homography de tao mask vung hop le.")
    h, w = int(out_shape[0]), int(out_shape[1])
    mask = np.full(img_test.shape[:2], 255, np.uint8)
    mask = cv2.warpPerspective(
        mask, H, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if erode_border and erode_border > 0:
        k = 2 * int(erode_border) + 1
        mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    return mask


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

    # Gray + CLAHE: phương án thắng trong khảo sát RGB → Binary.
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

    test_brighter = cv2.subtract(blur_test, template_dilated)  # lỗi open, pin-hole,...
    test_darker = cv2.subtract(template_eroded, blur_test)  # lỗi short, spur,...

    diff = cv2.max(test_brighter, test_darker)
    return diff, blur_ms


def binarize_difference(
        diff: np.ndarray,
        border: int = 8,
        threshold_floor: int = DEFAULT_THRESHOLD_FLOOR,
        open_ksize: int = DEFAULT_OPEN_KSIZE,
        hysteresis_low: "float | None" = DEFAULT_HYSTERESIS_LOW,
        valid: "np.ndarray | None" = None) -> np.ndarray:
    """Otsu + nguong san + bo vien + morphology opening.

    threshold_floor : nguong = max(Otsu, threshold_floor)
    open_ksize      : kernel opening khu nhieu; <= 1 thi bo opening
    hysteresis_low  : None = nguong don. Neu la so trong (0, 1), giu them cac
                      pixel >= hysteresis_low * nguong MIEN LA noi lien voi
                      mot pixel vuot nguong (nhu Canny). Giup giu tron net loi
                      manh / tuong phan yeu ma khong them dom nhieu rieng le.
    valid           : mask vung hop le (255 = pixel that, xem `warp_valid_mask`).
                      Otsu chi lay mau TRONG vung nay, va moi pixel ngoai vung
                      bi dat 0 truoc khi loc nhieu. Bat buoc khi anh test da bi
                      warp (xoay/dich), neu khong dai bien den se day nguong
                      Otsu len va xoa mat loi manh (CODE_REVIEW.md muc 2.3).
    """
    if diff is None or diff.size == 0:
        raise ValueError("Ban do vi sai rong.")

    h, w = diff.shape[:2]
    if h <= 2 * border or w <= 2 * border:
        raise ValueError(f"Anh qua nho de loai bo vung bien {border} pixel.")

    region = None
    if valid is not None:
        if valid.shape[:2] != diff.shape[:2]:
            raise ValueError(
                f"Mask vung hop le {valid.shape[:2]} khac kich thuoc ban do vi sai {diff.shape[:2]}."
            )
        region = valid > 0
        sample = diff[region]
        if sample.size < 500:
            raise ValueError("Vung hop le qua nho de tinh nguong Otsu.")
    else:
        sample = diff

    # Otsu + ngưỡng sàn; mẫu Otsu chỉ lấy trong vùng hợp lệ.
    otsu_val, _ = cv2.threshold(
        np.ascontiguousarray(sample).reshape(-1, 1), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold_value = max(int(otsu_val), int(threshold_floor))

    _, diff_binary = cv2.threshold(
        diff, threshold_value, 255, cv2.THRESH_BINARY
    )
    if region is not None:
        diff_binary[~region] = 0

    if hysteresis_low is not None and 0 < hysteresis_low < 1:
        low_value = max(1, int(threshold_value * hysteresis_low))
        _, weak = cv2.threshold(diff, low_value, 255, cv2.THRESH_BINARY)
        if region is not None:
            weak[~region] = 0
        n_labels, labels = cv2.connectedComponents(weak, connectivity=8)
        if n_labels > 1:
            keep = np.zeros(n_labels, bool)
            keep[np.unique(labels[diff_binary > 0])] = True
            keep[0] = False
            diff_binary = np.where(keep[labels], 255, 0).astype(np.uint8)

    # Bỏ viền ảnh. border=0 là không bỏ (nếu không, -0 sẽ cắt sạch ảnh).
    if border > 0:
        diff_binary[:border, :] = 0
        diff_binary[-border:, :] = 0
        diff_binary[:, :border] = 0
        diff_binary[:, -border:] = 0

    # Khử nhiễu bằng morphology opening
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
    valid: "np.ndarray | None" = None,
) -> np.ndarray:
    """Tru vi sai template voi anh test da can chinh, tra ve mask nhi phan.

    `blur_ksize` cho phep thi nghiem doi kich thuoc kernel Gaussian (mac dinh 3).
    `valid` la mask vung hop le sau warp (xem `warp_valid_mask`): dung de loai
    dai bien den khoi buoc tinh nguong. Cac tham so con lai chuyen thang cho
    `binarize_difference`.
    """
    diff, _ = difference_map(img_template, img_test_aligned, blur_ksize)
    return binarize_difference(diff, threshold_floor=threshold_floor,
                               open_ksize=open_ksize, hysteresis_low=hysteresis_low,
                               valid=valid)
