import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read images
img1 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE)

# Resize second image if sizes differ
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Subtract images
result = cv2.subtract(img1, img2)

# Display
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Image 1")
plt.imshow(img1, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Image 2")
plt.imshow(img2, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Subtracted Image")
plt.imshow(result, cmap='gray')
plt.axis('off')

plt.show()