import cv2
import numpy as np

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

    # Giữ top "keep_percentage" cặp tốt nhất, nhưng luôn >= 4 cặp.
    # min(...) để không vượt quá số match thực tế; int(...) vì index list phải là int.
    num_keeps = min(len(matches), max(int(len(matches) * keep_percentage), 4))
    good_matches = matches[:num_keeps]

    # Chuyển toạ độ keypoint sang mảng NumPy shape (N, 1, 2) cho OpenCV.
    # pts_test[i] tương ứng với pts_template[i] qua cặp match thứ i.
    pts_test = np.float32(
        [kp_test[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    pts_template = np.float32(
        [kp_template[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # ------------------------------------------------------------------
    # 5. Tính Homography bằng RANSAC
    # ------------------------------------------------------------------
    # RANSAC tự tìm tập inlier (cặp điểm khớp đúng) và bỏ outlier.
    # Ngưỡng 3.0 px: một cặp được coi là inlier nếu sai số chiếu < 3 px.
    #   3 px ~ 0.06 mm (48 px/mm), vẫn nhỏ hơn kích thước lỗi nhỏ nhất (~9 px),
    #   nên không che lỗi; đồng thời đủ rộng cho nhiễu scan/JPEG.
    H, mask = cv2.findHomography(pts_test, pts_template, cv2.RANSAC, 3.0)

    if H is None or mask is None:
        raise RuntimeError("Ko tim thay ma tran homography phu hop!")

    # ------------------------------------------------------------------
    # 6. Kiểm tra độ tin cậy của Homography
    # ------------------------------------------------------------------
    inliers = mask.ravel().astype(bool)          # mask RANSAC: 1 = inlier
    num_inliers = np.count_nonzero(inliers)
    inlier_ratio = num_inliers / len(good_matches)

    # Hai điều kiện tối thiểu:
    #   - >= 12 inlier: đủ điểm để H ổn định, không bị vài điểm nhiễu chi phối.
    #   - ratio >= 0.15: chặn trường hợp match quá rác.
    # Lưu ý: PCB lặp pattern nên ORB tạo nhiều match trùng; ratio thấp (~0.18 trên
    # cặp mẫu) vẫn có thể cho H đúng, vì vậy còn kiểm tra reprojection bên dưới.
    if num_inliers < 12 or inlier_ratio < 0.15:
        raise RuntimeError(
            f"Homography khong dang tin cay: "
            f"{num_inliers} inliers, "
            f"ty le {inlier_ratio:.2f}."
        )

    # Tái chiếu các điểm test bằng H rồi so với toạ độ template tương ứng.
    # Dùng median thay vì mean để một vài outlier không kéo lệch kết quả.
    projected = cv2.perspectiveTransform(pts_test[inliers], H)
    errors = np.linalg.norm(projected - pts_template[inliers], axis=2)
    reprojection_error = float(np.median(errors))

    # Median reprojection phải < 3 px. Trên cặp mẫu đo được ~1.27 px.
    if not np.isfinite(reprojection_error) or reprojection_error >= 3.0:
        raise RuntimeError(
            f"Reprojection error qua lon: {reprojection_error:.2f} px."
        )

    # H phải chứa toàn số hữu hạn và H[2,2] != 0 để chuẩn hoá được.
    if not np.all(np.isfinite(H)):
        raise RuntimeError("Homography chua gia tri khong hop le.")

    if abs(H[2, 2]) < 1e-8:
        raise RuntimeError("Homography khong the chuan hoa.")

    # Chuẩn hoá để H[2,2] = 1 (ma trận homography sai khác một hằng số tỉ lệ).
    H = H / H[2, 2]

    # det của phần affine 2x2 xấp xỉ tỉ lệ diện tích ảnh.
    #   det = 1   : không co/giãn
    #   0.5 - 2.0 : cho phép co/giãn trong khoảng ~0.7 - 1.4 lần
    # Ngoài khoảng này thường là H suy biến do match sai.
    det = np.linalg.det(H[:2, :2])
    if not (0.5 < det < 2.0):
        raise RuntimeError(
            f"Homography gay bien dang bat thuong (det = {det:.3f})."
        )

    # ------------------------------------------------------------------
    # 7. Warp ảnh test về hệ toạ độ template
    # ------------------------------------------------------------------
    h, w = gray_template.shape[:2]

    # Kích thước đầu ra = kích thước template để 2 ảnh trừ nhau được.
    # INTER_LINEAR: nội suy song tuyến, mượt hơn INTER_NEAREST.
    # BORDER_CONSTANT = 0: vùng ngoài ảnh thành đen; detect_difference sẽ bỏ 8 px viền.
    img_test_aligned = cv2.warpPerspective(
        img_test, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return img_test_aligned


def detect_difference(
    img_template: np.ndarray,
    img_test_aligned: np.ndarray,
) -> np.ndarray:
    """
    Tạo mask nhị phân các điểm khác biệt giữa template và ảnh test đã căn chỉnh.

    Ý tưởng:
        - Làm mờ nhẹ để giảm nhiễu.
        - So sánh có dung sai vị trí: chỉ coi là lỗi khi ảnh test sáng/tối hơn
          HẲN vùng lân cận của template (tránh viền giả do lệch 1-2 px).
        - Nhị phân hoá bằng Otsu nhưng có ngưỡng sàn.
        - Bỏ viền ảnh và làm sạch bằng morphology opening.

    Tham số:
        img_template     : ảnh mẫu (gốc, chưa warp).
        img_test_aligned : ảnh test đã căn chỉnh, cùng kích thước template.

    Trả về:
        mask uint8 cùng kích thước, giá trị 0 hoặc 255.
    """

    # ------------------------------------------------------------------
    # 1. Kiểm tra đầu vào
    # ------------------------------------------------------------------
    if img_template is None or img_test_aligned is None:
        raise ValueError("Anh dau vao khong hop le.")

    # Chuyển template về xám (chấp nhận ảnh xám, BGR, BGRA).
    if img_template.ndim == 3:
        if img_template.shape[2] in (3, 4):
            gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Template co so kenh anh khong hop le!")
    elif img_template.ndim == 2:
        gray_template = img_template
    else:
        raise ValueError("Template khong phai anh hop le")

    # Chuyển ảnh test đã align về xám.
    if img_test_aligned.ndim == 3:
        if img_test_aligned.shape[2] in (3, 4):
            gray_test = cv2.cvtColor(img_test_aligned, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Test co so kenh anh khong hop le!")
    elif img_test_aligned.ndim == 2:
        gray_test = img_test_aligned
    else:
        raise ValueError("Test khong phai anh hop le")

    # Hai ảnh phải cùng kích thước thì mới trừ điểm ảnh tương ứng được.
    if gray_template.shape != gray_test.shape:
        raise ValueError("Kich thuoc template khong giong nhau")

    h, w = gray_template.shape

    # Cần ít nhất 2*border px mỗi chiều để còn vùng ảnh sau khi bỏ viền 8 px.
    if h <= 16 or w <= 16:
        raise ValueError("Anh qua nho de loai bo vung bien 8 pixel.")

    # ------------------------------------------------------------------
    # 2. Làm mờ Gaussian 3x3
    # ------------------------------------------------------------------
    # Giảm nhiễu JPEG/scan trước khi so sánh.
    # 3x3 là kernel nhỏ nhất vừa đủ: 1x1 còn nhiễu, 5x5+ làm mờ lỗi nhỏ/spur 1 px.
    blur_template = cv2.GaussianBlur(gray_template, (3, 3), 0)
    blur_test = cv2.GaussianBlur(gray_test, (3, 3), 0)

    # ------------------------------------------------------------------
    # 3. Trừ vi sai CÓ DUNG SAI VỊ TRÍ (±2 px)
    # ------------------------------------------------------------------
    # Kernel 5x5 tương ứng bán kính 2 px. Vì sau căn chỉnh vẫn có thể lệch biên
    # ~1 px (p99 đo được = 1 px), so sánh tuyệt đối sẽ tạo viền giả dọc mép mạch.
    #
    # Với ảnh PCB: đồng = đen (0), nền = trắng (255).
    #   - erode(template)  : lấy min lân cận -> vùng đồng "nở" ra
    #                        => vùng chắc chắn là đồng.
    #   - dilate(template) : lấy max lân cận -> vùng nền "nở" ra
    #                        => vùng chắc chắn là nền.
    #
    # test_brighter > 0: test sáng hơn cả vùng nền có thể của template
    #                    => tại đó bị MẤT ĐỒNG (hở mạch, lỗ khoan...).
    # test_darker   > 0: test tối hơn cả vùng đồng có thể của template
    #                    => tại đó có ĐỒNG THỪA (chập mạch, spur...).
    kernel = np.ones((5, 5), np.uint8)

    template_eroded = cv2.erode(blur_template, kernel)
    template_dilated = cv2.dilate(blur_template, kernel)

    test_brighter = cv2.subtract(blur_test, template_dilated)
    test_darker = cv2.subtract(template_eroded, blur_test)

    # Lấy max để giữ cả 2 loại lỗi (mất đồng và thừa đồng).
    diff = cv2.max(test_brighter, test_darker)

    # ------------------------------------------------------------------
    # 4. Nhị phân hoá Otsu + ngưỡng sàn
    # ------------------------------------------------------------------
    # Otsu tự chọn ngưỡng từ histogram của diff. Tuy nhiên nếu ảnh KHÔNG có lỗi,
    # Otsu vẫn chia nhiễu thành 2 lớp -> sinh rất nhiều false positive.
    # Vì vậy lấy max(Otsu, 30):
    #   - Nhiễu JPEG q75 trên ảnh không lỗi: max ~11.
    #   - Độ tương phản lỗi thật: median ~163.
    #   => 30 nằm giữa 2 vùng, có biên an toàn.
    otsu_val, _ = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold_value = max(int(otsu_val), 30)

    _, diff_binary = cv2.threshold(
        diff, threshold_value, 255, cv2.THRESH_BINARY
    )

    # ------------------------------------------------------------------
    # 5. Bỏ viền ảnh
    # ------------------------------------------------------------------
    # Sau warpPerspective, 8 px sát biên bị nội suy/không có dữ liệu thật,
    # dễ tạo viền giả. Đánh đổi: lỗi nằm sát mép sẽ không được phát hiện.
    border = 8
    diff_binary[:border, :] = 0
    diff_binary[-border:, :] = 0
    diff_binary[:, :border] = 0
    diff_binary[:, -border:] = 0

    # ------------------------------------------------------------------
    # 6. Khử nhiễu bằng morphology opening
    # ------------------------------------------------------------------
    # MORPH_OPEN = erode rồi dilate: xoá các đốm trắng nhỏ hơn kernel.
    # 3x3 là nhỏ nhất đủ xoá đốm nhiễu mà vẫn giữ spur 1 px;
    # 5x5 bắt đầu xoá mất spur mảnh (đã kiểm chứng).
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    diff_cleaned = cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, clean_kernel)

    return diff_cleaned
