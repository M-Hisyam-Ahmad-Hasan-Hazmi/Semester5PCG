import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_sample_image(size=(300, 300)):
    
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    
    
    
    cv2.circle(img, (100, 100), 50, (255, 0, 0), -1)
    
    cv2.rectangle(img, (175, 175), (275, 275), (0, 0, 255), -1)
    
    pts = np.array([[150, 50], [50, 150], [250, 150]], np.int32)
    cv2.fillPoly(img, [pts], (0, 255, 0))
    
    return img


def apply_gaussian_blur(color_img, gray_img, kernel_size=(5,5)):
    color_blur = cv2.GaussianBlur(color_img, kernel_size, 0)
    gray_blur = cv2.GaussianBlur(gray_img, kernel_size, 0)
    return color_blur, gray_blur

def apply_sobel_edge(color_img, gray_img):
    
    gray_sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    gray_sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gray_edges = np.sqrt(gray_sobel_x**2 + gray_sobel_y**2)
    gray_edges = np.uint8(np.clip(gray_edges, 0, 255))
    
    
    color_edges = np.zeros_like(color_img)
    for i in range(3):
        sobel_x = cv2.Sobel(color_img[:,:,i], cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(color_img[:,:,i], cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        color_edges[:,:,i] = np.uint8(np.clip(edges, 0, 255))
    
    return color_edges, gray_edges

def apply_high_boost(color_img, gray_img, alpha=1.5):
    blur_gray = cv2.GaussianBlur(gray_img, (5,5), 0)
    high_boost_gray = cv2.addWeighted(gray_img, alpha, blur_gray, (1-alpha), 0)
    
    blur_color = cv2.GaussianBlur(color_img, (5,5), 0)
    high_boost_color = cv2.addWeighted(color_img, alpha, blur_color, (1-alpha), 0)
    
    return high_boost_color, high_boost_gray


color_img = create_sample_image()
gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)


color_blur, gray_blur = apply_gaussian_blur(color_img, gray_img)
color_edges, gray_edges = apply_sobel_edge(color_img, gray_img)
color_boost, gray_boost = apply_high_boost(color_img, gray_img)


plt.figure(figsize=(15, 12))


plt.subplot(4, 2, 1)
plt.imshow(color_img)
plt.title('Original Color')
plt.axis('off')

plt.subplot(4, 2, 2)
plt.imshow(gray_img, cmap='gray')
plt.title('Original Grayscale')
plt.axis('off')


plt.subplot(4, 2, 3)
plt.imshow(color_blur)
plt.title('Color Gaussian Blur')
plt.axis('off')

plt.subplot(4, 2, 4)
plt.imshow(gray_blur, cmap='gray')
plt.title('Grayscale Gaussian Blur')
plt.axis('off')


plt.subplot(4, 2, 5)
plt.imshow(color_edges)
plt.title('Color Edge Detection')
plt.axis('off')

plt.subplot(4, 2, 6)
plt.imshow(gray_edges, cmap='gray')
plt.title('Grayscale Edge Detection')
plt.axis('off')


plt.subplot(4, 2, 7)
plt.imshow(color_boost)
plt.title('Color High-boost')
plt.axis('off')

plt.subplot(4, 2, 8)
plt.imshow(gray_boost, cmap='gray')
plt.title('Grayscale High-boost')
plt.axis('off')

plt.tight_layout()
plt.show()