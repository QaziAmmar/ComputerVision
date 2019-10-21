import cv2
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from skimage.feature import blob_dog, blob_log, blob_doh

# input

img = cv2.imread("/Users/qaziammar/Downloads/test_RBC.jpg", 0)

# gaussian Blur
img = cv2.GaussianBlur(img, (15, 15), 0)

# adaptive threshold
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

plt.imshow(th3, cmap='gray', vmax=255, vmin=0)
plt.show()

# cv2.imshow('Noise Filtered Image', th3)
# cv2.waitKey(0)
# cv2.imwrite('data_5/result.png',th3)
