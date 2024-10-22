# This is a file where we fist test our code then implement it into other file
# This code run on python 3.7
"""
Description:
This is very first code that we used to extract individual RBC from our dataset. This method
segments sufficient number of red blood cells form blood slides. Till 05-05-2020 We perform all experiments
on red blood cell by using this segmentation code.
We then compare this code with watershed algorithm which performs much better than this code and
new segmentation code almost extract each cell form complete slide image that is captured at 1000X or
450X.
"""

from custom_classes import path, cv_iml
import cv2
import numpy as np

dataset_path = path.result_folder_path + "microscope_test/sample_images/"
images_name = path.read_all_files_name_from(dataset_path, '.JPG')
result_path = path.result_folder_path + "microscope_test/morphological/"


for image in images_name:

    # image_segments = 500
    rgb = cv2.imread(dataset_path + image)
    clone = rgb.copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # Sharpen image
    sharp_image = cv2.filter2D(gray, -1, kernel)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    wide = cv2.Canny(blurred, 0, 14)

    kernel = np.ones((7, 7), np.uint8)

    closing = cv2.morphologyEx(wide, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 1)

    # create all-black mask image
    mask = np.zeros(shape=rgb.shape, dtype="uint8")

    # Draw rectangle on detected contours.
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 0, 0), 2)

    red_blood_cells = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        crop_image = clone[y:y + h, x: x + w]
        # if (w > 180 and h > 180) and (w < 620 and h < 620):
        #     cv2.imwrite(result_path + image[:-4] + "_" + str(x) + "_" + str(y)
        #                 + "_" + str(h) + "_" + str(w) + ".JPG", crop_image)
        if (w > 30 and h > 30) and (w < 70 and h < 70):
            red_blood_cells.append(crop_image)

    for i in range(len(red_blood_cells)):
        cv2.imwrite(result_path + image[:-4] + "_" + str(i) + ".jpg", red_blood_cells[i])
