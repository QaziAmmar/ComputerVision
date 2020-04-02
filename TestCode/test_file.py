# This is a file where we fist test our code then implement it into other file
# Annotation draw testing code.
from custom_classes import path, cv_iml
import re
import os
import cv2
import numpy as np
import copy
from custom_classes import path, cv_iml
#
# folder_path = "10X/"
# dataset_path = path.dataset_path + folder_path
# images_name = path.read_all_files_name_from(dataset_path, ".JPG")
# for image in images_name:
#     img = cv2.imread(dataset_path + image)
#     h, w, c = img.shape
#     crop = img[380: w - 380, 380: h - 380, :]
#     cv2.imwrite(dataset_path + image, crop)

rectangle_points = []
# Convert RGB to gray scale and improve contrast of the image
mean_gray = cv2.imread(path.dataset_path + "IML_dataset/mean_image_pf.png")
image_path = "/Users/qaziammar/Desktop/20200106_153153.jpg"
rgb_resized = cv2.imread(image_path)
mean_gray = cv2.resize(mean_gray, (rgb_resized.shape[1], rgb_resized.shape[0]))


# Convert RGB to gray scale and improve contrast of the image
gray = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
imge_clahe = clahe.apply(gray)

# Subtract the Background (mean) image
mean_subtracted = imge_clahe - mean_gray[:, :, 0]
clone = mean_subtracted.copy()

# Remove the pixels which are very close to the mean. 60 is selected after watching a few images
mean_subtracted[mean_subtracted < 60] = 0

# To separate connected cells, do the Erosion. The kernal parameters are randomly selected.
kernel = np.ones((20, 20), np.uint8)
mean_subtracted_erode = cv2.erode(mean_subtracted, kernel)
kernel = np.ones((7, 7), np.uint8)
closing = cv2.morphologyEx(mean_subtracted_erode, cv2.MORPH_CLOSE, kernel)
_, contours_single_erode, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Select the regions which are still large and strongle collected and apply errosion again only to those regions
mean_subtracted_erode_forLarge = copy.deepcopy(mean_subtracted_erode)
area_c = []
for c in contours_single_erode:
    (x, y, w, h) = cv2.boundingRect(c)
    area_c = w * h
    if area_c < 30000:
        mean_subtracted_erode_forLarge[y:y + h, x:x + w] = 0

kernel = np.ones((20, 20), np.uint8)
mean_subtracted_doubleerode_forLarge = cv2.erode(mean_subtracted_erode_forLarge, kernel)

closing = cv2.morphologyEx(mean_subtracted_doubleerode_forLarge, cv2.MORPH_CLOSE, kernel)
_, contours_double_erode, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Plot results with single errosion
rgb_single_erode = copy.deepcopy(rgb_resized)
for c in contours_single_erode:
    (x, y, w, h) = cv2.boundingRect(c)
    x1 = max(1, x - 15)
    y1 = max(1, y - 15)
    x2 = min(x + w + 15, rgb_resized.shape[1])
    y2 = min(y + h + 15, rgb_resized.shape[0])

    area_c = w * h

    rectangle_points.append({"x": x1, "y": y1, "h": y2 - y1, "w": x2 - x1})
    cv2.rectangle(img=rgb_single_erode, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 225), thickness=5)

# Plot results with double errosion
rgb_double_erode = copy.deepcopy(rgb_resized)

for c in contours_single_erode:
    (x, y, w, h) = cv2.boundingRect(c)
    x1 = max(1, x - 15)
    y1 = max(1, y - 15)
    x2 = min(x + w + 15, rgb_resized.shape[1])
    y2 = min(y + h + 15, rgb_resized.shape[0])

    area_c = w * h
    if 2000 < area_c < 30000:
        rectangle_points.append({"x": x1, "y": y1, "h": y2 - y1, "w": x2 - x1})
        cv2.rectangle(img=rgb_double_erode, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 225), thickness=5)

for c in contours_double_erode:
    (x, y, w, h) = cv2.boundingRect(c)

    x1 = max(1, x - 35)
    y1 = max(1, y - 35)
    x2 = min(x + w + 35, rgb_resized.shape[1])
    y2 = min(y + h + 35, rgb_resized.shape[0])
    area_c = w * h
    if area_c < 2000:
        continue
    else:
        rectangle_points.append({"x": x1, "y": y1, "h": y2 - y1, "w": x2 - x1})
        cv2.rectangle(img=rgb_double_erode, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 225), thickness=5)

print(len(rectangle_points))
# cv2.imwrite("rgb_resized.jpg", rgb_resized)
# cv2.imwrite("rgb_single_erode.jpg", rgb_single_erode)
# cv2.imwrite("rgb_double_erode.jpg", rgb_double_erode)
print(len(rectangle_points))