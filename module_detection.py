import cv2
import os

def extract_bounding_boxes(clean_mask, min_area=20):
    # Tìm đường bao ngoài của các vùng trắng liên thông
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        # Bỏ đốm nhiễu có diện tích dưới min_area
        x, y, w, h = cv2.boundingRect(cnt)
        area = int(cv2.countNonZero(clean_mask[y:y+h, x:x+w]))
        if area < min_area:
            continue
        
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: (b[1], b[0])) # sắp xếp từ trên xuống, cùng hàng thì từ trái sang
    
    return boxes

def crop_defect_patches(img_test, boxes, output_dir="crops", padding = 5):
    if img_test is None or img_test.size == 0:
        raise ValueError("Anh khong hop le.")
    
    if padding < 0:
        raise ValueError("Padding phai >= 0.")
    
    os.makedirs(output_dir, exist_ok=True)
    img_height, img_width = img_test.shape[:2]
    crop_paths = []
    
    for i, box in enumerate(boxes, start=1):
        
        # chỉ lấy 4 giá trị đầu: x, y, w, h
        x, y, w, h = box[:4]
        if w<=0 or h<=0:
            continue
        # nới rộng hộp thêm padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_width, x + w + padding)
        y2 = min(img_height, y + h + padding)
        
        if x1 >= x2 or y1 >= y2:
            continue
        # cắt ảnh: Numpy đánh chỉ số theo [y, x]
        patch = img_test[y1:y2, x1:x2]
        
        if patch.size == 0:
            continue
        
        filename = f"defect_{i:03d}.png"
        output_path = os.path.join(output_dir, filename)
        
        success = cv2.imwrite(output_path, patch)
        if success:
            crop_paths.append(output_path)
        else:
            print(f"Khong the luu anh: {output_path}")
    
    return crop_paths