#Nam Anh: Căn chỉnh ORB & Trừ vi sai ==> Output: binary image đốm lỗi
import cv2
import numpy as np

def align_images(
        img_template: np.ndarray, 
        img_test: np.ndarray, 
        max_features: int = 5000, 
        keep_percentage: float = 0.15) -> np.ndarray:

    gray_template = (
        cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
        if len(img_template.shape) == 3
        else img_template
    )

    gray_test = (
        cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
        if len(img_test.shape) == 3
        else img_test
    )

    orb = cv2.ORB_create(max_features)
    kp_template, des_template = orb.detectAndCompute(gray_template, None)
    kp_test, des_test = orb.detectAndCompute(gray_test, None)

    if des_template is None or des_test is None:
        raise ValueError("ko the trich xuat diem dac trung tu anh dau vao.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_test, des_template)

    if len(matches) <4:
        raise RuntimeError(f"Chi co {len(matches)} cap khop, can it nhat 4 de tinh homography")

    matches = sorted(matches, key = lambda x: x.distance)
    num_keeps = int(len(matches) * keep_percentage)
    good_matches = matches[:max(num_keeps, 4)]

    pts_test = np.zeros((len(good_matches), 2), dtype=np.float32)
    pts_template = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        pts_test[i, :] = kp_test[match.queryIdx].pt
        pts_template[i, :]= kp_template[match.trainIdx].pt

    H,mask = cv2.findHomography(pts_test, pts_template, cv2.RANSAC, 5.0)

    if H is None:
        raise RuntimeError("Ko tim thay ma tran homography phu hop!")

    det = np.linalg.det(H[:2,:2])
    if not (0.5 < det < 2.0):
        raise RuntimeError(f"Ma tran homography bat thuong (det = {det:.3f}), anh bi bien dang")

    h,w = gray_template.shape[:2]
    img_test_aligned = cv2.warpPerspective(img_test, H, (w,h), borderMode=cv2.BORDER_REPLICATE)

    return img_test_aligned

def detect_difference(
    img_template: np.ndarray, 
    img_test_aligned: np.ndarray, 
) -> np.ndarray:

    gray_template = (
        cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
        if len(img_template.shape) == 3
        else img_template
        )
    
    gray_test = (
        cv2.cvtColor(img_test_aligned, cv2.COLOR_BGR2GRAY)
        if len(img_test_aligned.shape) == 3
        else img_test_aligned
        )



    blur_template = cv2.GaussianBlur(gray_template, (3,3), 0)
    blur_test = cv2.GaussianBlur(gray_test, (3,3), 0)

    diff = cv2.absdiff(blur_template, blur_test)

    _, diff_binary = cv2.threshold(diff, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    diff_cleaned = cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, kernel)

    return diff_cleaned


