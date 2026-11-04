import cv2
import numpy as np

def align_images(
        img_template: np.ndarray, 
        img_test: np.ndarray, 
        max_features: int = 5000, 
        keep_percentage: float = 0.8) -> np.ndarray:

    if img_template is None or img_test is None:
        raise ValueError("Anh template va anh test khong hop le")

    if img_template.ndim not in (2,3) or img_test.ndim not in (2,3):
        raise ValueError("Anh dau vao phai la anh grayscale hoac anh mau")

    if img_template.size == 0 or img_test.size == 0:
        raise ValueError("Anh dau vao rong")

    if not (0<keep_percentage <=1):
        raise ValueError("keep_percentage phai nam trong (0,1].")


    if img_template.ndim == 3:
        if img_template.shape[2] == 3:
            gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
        elif img_template.shape[2] == 4:
            gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Template co so kenh anh khong hop le!")
    else: 
        gray_template = img_template

    if img_test.ndim == 3:
        if img_test.shape[2] == 3:
            gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
        elif img_test.shape[2] == 4:
            gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Test co so kenh anh khong hop le!")
    else: 
        gray_test = img_test

    orb = cv2.ORB_create(nfeatures=max_features)
    kp_template, des_template = orb.detectAndCompute(gray_template, None)
    kp_test, des_test = orb.detectAndCompute(gray_test, None)

    if des_template is None or des_test is None:
        raise ValueError("ko the trich xuat diem dac trung tu anh dau vao.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_test, des_template)

    if len(matches) <4:
        raise RuntimeError(f"Chi co {len(matches)} cap khop, can it nhat 4 de tinh homography")

    matches = sorted(matches, key = lambda x: x.distance)
    num_keeps = min(len(matches), max(int(len(matches) * keep_percentage),4))
    good_matches = matches[:num_keeps]

    pts_test = np.float32([kp_test[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)

    pts_template = np.float32([kp_template[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H,mask = cv2.findHomography(pts_test, pts_template, cv2.RANSAC, 3.0)

    if H is None or mask is None:
        raise RuntimeError("Ko tim thay ma tran homography phu hop!")

    inliers = mask.ravel().astype(bool)

    num_inliers = np.count_nonzero(inliers)
    inlier_ratio = num_inliers/len(good_matches)

    if num_inliers < 12 or inlier_ratio < 0.15:
        raise RuntimeError(f"Homography khong dang tin cay: "
            f"{num_inliers} inliers, "
            f"ty le {inlier_ratio:.2f}.")

    projected = cv2.perspectiveTransform(pts_test[inliers], H)

    errors = np.linalg.norm(projected - pts_template[inliers], axis = 2)
    reprojection_error = float(np.median(errors))

    if not np.isfinite(reprojection_error) or reprojection_error >= 3.0:
        raise RuntimeError(f"Reprojection error qua lon: "
            f"{reprojection_error:.2f} px.")

    if not np.all(np.isfinite(H)):
        raise RuntimeError("Homography chua gia tri khong hop le.")

    if abs(H[2, 2]) < 1e-8:
        raise RuntimeError("Homography khong the chuan hoa.")

    H = H/H[2,2]
    det = np.linalg.det(H[:2,:2])

    if not (0.5<det<2.0):
        raise RuntimeError(f"Homography gay bien dang bat thuong "
            f"(det = {det:.3f}).")
    
    h,w = gray_template.shape[:2]

    img_test_aligned = cv2.warpPerspective(img_test, H, (w,h), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return img_test_aligned

def detect_difference(
    img_template: np.ndarray, 
    img_test_aligned: np.ndarray, 
) -> np.ndarray:

    if img_template is None or img_test_aligned is None:
        raise ValueError("Anh dau vao khong hop le.")

    if img_template.ndim == 3:
        if img_template.shape[2] == 3:
            gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
        elif img_template.shape[2] == 4:
            gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
        else:
                raise ValueError("Template co so kenh anh khong hop le!")
    elif img_template.ndim == 2:
        gray_template = img_template
    else:
        raise ValueError("Template khong phai anh hop le")
    
    
    if img_test_aligned.ndim == 3:
        if img_test_aligned.shape[2] == 3:
            gray_test = cv2.cvtColor(img_test_aligned, cv2.COLOR_BGR2GRAY)
        elif img_test_aligned.shape[2] == 4:
            gray_test = cv2.cvtColor(img_test_aligned, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Test co so kenh anh khong hop le!")
    elif img_test_aligned.ndim == 2:
        gray_test = img_test_aligned
    else:
        raise ValueError("Test khong phai anh hop le")

    if gray_template.shape != gray_test.shape:
        raise ValueError("Kich thuoc template khong giong nhau")

    h,w = gray_template.shape

    if h <= 16 or w <= 16:
        raise ValueError("Anh qua nho de loai bo vung bien 8 pixel.")

    blur_template = cv2.GaussianBlur(gray_template, (3,3), 0)
    blur_test = cv2.GaussianBlur(gray_test, (3,3), 0)

    kernel = np.ones((5,5), np.uint8)

    template_eroded = cv2.erode(blur_template, kernel)
    template_dilated = cv2.dilate(blur_template, kernel)

    missing = cv2.subtract(template_eroded, blur_test)
    extra = cv2.subtract(blur_test ,template_dilated)

    diff = cv2.max(missing, extra)

    otsu_val,_ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    threshold_value = max(int(otsu_val), 30)

    _,diff_binary = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    border = 8

    diff_binary[:border, :] = 0
    diff_binary[-border:, :] = 0
    diff_binary[:,:border] = 0
    diff_binary[:,-border:] = 0

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    diff_cleaned = cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, clean_kernel)

    return diff_cleaned