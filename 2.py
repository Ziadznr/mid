import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Number of bins
B = 32

# Compute histogram
hist, bins = np.histogram(img.flatten(), bins=B, range=[0,256])

# Display image
plt.subplot(1,2,1)
plt.title("Grayscale Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

# Display histogram
plt.subplot(1,2,2)
plt.title("Histogram (32 bins)")
plt.bar(range(B), hist)
plt.xlabel("Bins")
plt.ylabel("Frequency")

plt.show()