import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_matching(source, reference):

    src_hist, _ = np.histogram(source.flatten(), 256, [0,256])
    ref_hist, _ = np.histogram(reference.flatten(), 256, [0,256])

    src_cdf = src_hist.cumsum()
    ref_cdf = ref_hist.cumsum()

    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]

    lookup = np.zeros(256)

    for i in range(256):
        diff = np.abs(ref_cdf - src_cdf[i])
        lookup[i] = np.argmin(diff)

    matched = lookup[source]
    return matched.astype(np.uint8)


# Read images
source = cv2.imread("source.jpg", cv2.IMREAD_GRAYSCALE)
reference = cv2.imread("reference.jpg", cv2.IMREAD_GRAYSCALE)

# Apply histogram matching
matched = histogram_matching(source, reference)

# Display
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Source Image")
plt.imshow(source, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Reference Image")
plt.imshow(reference, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Matched Image")
plt.imshow(matched, cmap='gray')
plt.axis('off')

plt.show()