import imageio as iio
import numpy as np
import matplotlib.pyplot as plt

img = iio.imread("C:/Users/Aspire_Black/Pictures/kuc.jpg")

grayscale = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)

histogram, bins = np.histogram(grayscale, bins=256, range=(0, 256))

total_pixels = np.sum(histogram)

dominant_intensity = np.argmax(histogram)
dominant_pixel_count = histogram[dominant_intensity]

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Sebelum")

plt.subplot(1, 3, 2)
plt.imshow(grayscale, cmap="gray")
plt.title("Sesudah")

plt.subplot(1, 3, 3)
plt.bar(range(256), histogram, width=1, color='gray')
plt.title("Histogram")
plt.xlabel("Intensitas Pixel")
plt.ylabel("Frekuensi")

plt.annotate(f'Intensitas dominan: {dominant_intensity}\nJumlah piksel: {dominant_pixel_count}', 
             xy=(dominant_intensity, dominant_pixel_count), 
             xytext=(dominant_intensity+10, dominant_pixel_count), 
             arrowprops=dict(facecolor='red', shrink=0.05))

plt.tight_layout()
plt.show()

print(f"Jumlah total piksel: {total_pixels}")
print(f"Intensitas dominan: {dominant_intensity} dengan {dominant_pixel_count} piksel")