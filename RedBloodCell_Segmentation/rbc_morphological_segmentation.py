# This is a file where we fist test our code then implement it into other file
from custom_classes import path, cv_iml
import cv2
import numpy as np

image_path = "/Users/qaziammar/Downloads/IMG_4452.JPG"
# image_segments = 500
rgb = cv2.imread(image_path)
clone = rgb.copy()
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Sharpen image
sharp_image = cv2.filter2D(gray, -1, kernel)

blurred = cv2.GaussianBlur(gray, (7, 7), 0)
# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(blurred, 0, 18)

kernel = np.ones((5, 5), np.uint8)

closing = cv2.morphologyEx(wide, cv2.MORPH_CLOSE, kernel)
cv_iml.image_show(closing, cmap='gray')

contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
cv2.drawContours(rgb, contours, -1, (0, 255, 0), 1)
print("length of contours is ", len(contours))
cv_iml.image_show(rgb, cmap='gray')

# create all-black mask image
mask = np.zeros(shape=rgb.shape, dtype="uint8")
# Draw rectangle on detected contours.
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(img=mask, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=-1)

red_blood_cells = []
result_path = path.result_folder_path + "rbc/"
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    crop_image = clone[y:y + h, x: x + w]
    if (w > 70 and h > 70) and (w < 180 and h < 180):
        red_blood_cells.append(crop_image)

for i in range(len(red_blood_cells)):
    cv2.imwrite(result_path + str(i) + ".jpg", red_blood_cells[i])
