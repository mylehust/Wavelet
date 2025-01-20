import pywt
import cv2
import matplotlib.pyplot as plt

# Đọc ảnh đầu vào
image_path = "test.jpg"  # Thay bằng đường dẫn tới ảnh của bạn
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Áp dụng DWT tầng 1 để lấy LL1
coeffs2 = pywt.dwt2(image, 'haar')
LL1, (LH1, HL1, HH1) = coeffs2
LL1_normalized = cv2.normalize(LL1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Áp dụng DWT lần 2 trên LL1 để lấy LL2
coeffs2_LL1 = pywt.dwt2(LL1, 'haar')
LL2, (LH2, HL2, HH2) = coeffs2_LL1

# Hiển thị các thành phần LL1 và LL2
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Thành phần LL1")
plt.imshow(LL1)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Thành phần LL2")
plt.imshow(LL2)
plt.axis('off')

plt.show()
