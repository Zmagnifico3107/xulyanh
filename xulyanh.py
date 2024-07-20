import cv2
import numpy as np

# Đọc ảnh gốc
image_path = r'D:\python\xulyanh\Lane.png'
image = cv2.imread(image_path)

# Kiểm tra xem ảnh có được tải không
if image is None:
    print(f"Error: Could not load image {image_path}")
else:
    # Xác định kích thước của ảnh
    height, width = image.shape[:2]

    # Định nghĩa các đỉnh của vùng ROI hình thang
    vertices = np.array([[(0, height),                # bottom-left
                          (width // 2 - 50, height // 2 + 60),  # top-left
                          (width // 2 + 50, height // 2 + 60),  # top-right
                          (width, height)]], dtype=np.int32)    # bottom-right

    # Tạo mask (mặt nạ) có cùng kích thước với ảnh và điền vùng ROI
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, (255, 255, 255))

    # Áp dụng phép AND bit-wise để giữ lại vùng ROI
    roi_image = cv2.bitwise_and(image, mask)

    # Chuyển ảnh ROI sang ảnh xám
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    # Áp dụng bộ lọc Canny để phát hiện cạnh
    edges = cv2.Canny(gray, 50, 150)

    # Áp dụng Hough Transform để phát hiện các đường thẳng
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)

    # Vẽ các đường thẳng lên ảnh gốc
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Hiển thị ảnh kết quả
    cv2.imshow('Original Image with Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
