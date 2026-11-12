import cv2
import numpy as np


def _gray_clahe(image: np.ndarray) -> np.ndarray:
    """Chuyen anh sang gray + can bang sang cuc bo CLAHE.

    Phuong phap thang trong khao sat RGB -> Binary (experiments/exp_rgb2binary.py):
    CLAHE giu bien canh tot nhat (0.356) va on dinh voi anh sang lech (IoU 0.870),
    dong thoi giam hien tuong loa sang tren be mat kim loai cua PCB.
    """
    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Anh co so kenh mau khong hop le!")
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Anh dau vao phai la anh grayscale hoac anh mau")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


#Can chinh anh test theo anh template
def align_images(
        img_template: np.ndarray,
        img_test: np.ndarray,
        max_features: int = 5000,
        keep_percentage: float = 0.8) -> np.ndarray:

    # Kiem tra input
    if img_template is None or img_test is None:
        raise ValueError("Anh template va anh test khong hop le")

    # anh xam = 2, anh mau = 3
    if img_template.ndim not in (2, 3) or img_test.ndim not in (2, 3):
        raise ValueError("Anh dau vao phai la anh grayscale hoac anh mau")

    if img_template.size == 0 or img_test.size == 0:
        raise ValueError("Anh dau vao rong")

    if not (0 < keep_percentage <= 1):
        raise ValueError("keep_percentage phai nam trong (0,1].")

    # Chuyen ve anh xem de do diem dac trung ORB
    if img_template.ndim == 3:
        if img_template.shape[2] in (3, 4):
            gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Template co so kenh anh khong hop le!")
    else:
        gray_template = img_template

    if img_test.ndim == 3:
        if img_test.shape[2] in (3, 4):
            gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Test co so kenh anh khong hop le!")
    else:
        gray_test = img_test

    # Do dac trung ORB
    orb = cv2.ORB_create(nfeatures=max_features)
    kp_template, des_template = orb.detectAndCompute(gray_template, None)
    kp_test, des_test = orb.detectAndCompute(gray_test, None)

    # Neu anh qua tron/it cach thi khong the trich xuat duoc keypoint
    if des_template is None or des_test is None:
        raise ValueError("ko the trich xuat diem dac trung tu anh dau vao.")

    # Ghep cap diem dac trung, giu lai diem match tot nhat
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_test, des_template)

    # Homography co 8 bac tu do nen can toi thieu 4 diem khong thang hang
    if len(matches) < 4:
        raise RuntimeError(
            f"Chi co {len(matches)} cap khop, can it nhat 4 de tinh homography"
        )

    matches = sorted(matches, key=lambda x: x.distance) # Sap xep distance tang dan

    num_keeps = min(len(matches), max(int(len(matches) * keep_percentage), 4)) # giu 
    good_matches = matches[:num_keeps]

    # Chuyen toa do Keypoint sang mang Numpy
    pts_test = np.float32(
        [kp_test[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    pts_template = np.float32(
        [kp_template[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # Tinh Homorgraphy bang Ransac
    H, mask = cv2.findHomography(pts_test, pts_template, cv2.RANSAC, 3.0)

    if H is None or mask is None:
        raise RuntimeError("Ko tim thay ma tran homography phu hop!")

 
    inliers = mask.ravel().astype(bool)          # mask RANSAC: 1 = inlier
    num_inliers = np.count_nonzero(inliers)
    inlier_ratio = num_inliers / len(good_matches)

    if num_inliers < 12 or inlier_ratio < 0.15:
        raise RuntimeError(
            f"Homography khong dang tin cay: "
            f"{num_inliers} inliers, "
            f"ty le {inlier_ratio:.2f}."
        )

    # Doi chieu cac diem test bang H roi so voi toa do template tuong xung
    projected = cv2.perspectiveTransform(pts_test[inliers], H)
    errors = np.linalg.norm(projected - pts_template[inliers], axis=2)
    reprojection_error = float(np.median(errors)) # Dung median thay vi mean de tranh mot vai outlet lech ket qua

    if not np.isfinite(reprojection_error) or reprojection_error >= 3.0:
        raise RuntimeError(
            f"Reprojection error qua lon: {reprojection_error:.2f} px."
        )

    if not np.all(np.isfinite(H)):
        raise RuntimeError("Homography chua gia tri khong hop le.")

    if abs(H[2, 2]) < 1e-8:
        raise RuntimeError("Homography khong the chuan hoa.")

    H = H / H[2, 2] # Chuan hoa H = 1

    # det cua phan affine 2x2 xap xi ti le dien tich anh
    det = np.linalg.det(H[:2, :2])
    if not (0.5 < det < 2.0):
        raise RuntimeError(
            f"Homography gay bien dang bat thuong (det = {det:.3f})."
        )

    h, w = gray_template.shape[:2]

    img_test_aligned = cv2.warpPerspective(
        img_test, H, (w, h),
        flags=cv2.INTER_LINEAR,           #Tinh gia tri mau dua tren cac pixel lan can
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return img_test_aligned


def detect_difference(
    img_template: np.ndarray,
    img_test_aligned: np.ndarray,
) -> np.ndarray:

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

    #lam mo Gaussian 3x3
    blur_template = cv2.GaussianBlur(gray_template, (3, 3), 0)
    blur_test = cv2.GaussianBlur(gray_test, (3, 3), 0)

    # Tru vi sai co dung sai vi tri (±2 px)
    kernel = np.ones((5, 5), np.uint8)

    template_eroded = cv2.erode(blur_template, kernel)
    template_dilated = cv2.dilate(blur_template, kernel)

    test_brighter = cv2.subtract(blur_test, template_dilated) #kiem tra dc loi open, pin-hole,..
    test_darker = cv2.subtract(template_eroded, blur_test) #kiem tra dc loi short, squr,..

    diff = cv2.max(test_brighter, test_darker)

    # 4. Nhi phan hoa Otsu + nguong san
    otsu_val, _ = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold_value = max(int(otsu_val), 30)

    _, diff_binary = cv2.threshold(
        diff, threshold_value, 255, cv2.THRESH_BINARY
    )

    # Bo vien anh
    border = 8
    diff_binary[:border, :] = 0
    diff_binary[-border:, :] = 0
    diff_binary[:, :border] = 0
    diff_binary[:, -border:] = 0

    # 6. Khu nhieu bang morphology opening
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    diff_cleaned = cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, clean_kernel)

    return diff_cleaned
