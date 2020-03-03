# This is a file where we fist test our code then implement it into other file
from custom_classes import path, cv_iml
import cv2
import numpy as np
import numpy


def new_morphological_operation(img=None):
    rgb = img.copy()
    clone = img.copy()
    # keep resizing your image so it appropriately identifies the RBC's
    img = cv2.resize(img, (0, 0), fx=5, fy=5)
    # it's always easier if the image is copied for long pieces of code.
    # we're copying it twice for reasons you'll see soon enough.
    wol = img.copy()
    gg = img.copy()
    # convert to grayscale
    img_gray = cv2.cvtColor(gg, cv2.COLOR_BGR2GRAY)
    # enhance contrast (helps makes boundaries clearer)
    clache = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray = clache.apply(img_gray)
    # threshold the image and apply morphological transforms
    _, img_bin = cv2.threshold(img_gray, 50, 255,
                               cv2.THRESH_OTSU)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
                               numpy.ones((3, 3), dtype=int))
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_DILATE,
                               numpy.ones((3, 3), dtype=int), iterations=1)
    # call the 'segment' function (discussed soon)
    # dt, result = segment(img, img_bin)
    return img_bin


def old_morphological_operation(img=None):
    rgb = img
    clone = rgb.copy()
    if len(img.shape) < 3:
        gray = img
    else:
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
        cv2.rectangle(img=mask, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=-1)

    red_blood_cells = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        crop_image = clone[y:y + h, x: x + w]
        if (w > 70 and h > 70) and (w < 180 and h < 180):
            red_blood_cells.append(crop_image)

    return closing


dataset_path = path.dataset_path + "IML_new_microscope/p.f/100X_crop/"
img = cv2.imread(dataset_path + "IMG_3340.JPG")
old_black_mask = old_morphological_operation(img)
new_black_mask = new_morphological_operation(img)
combine_mask = old_morphological_operation(new_black_mask)
cv_iml.image_show(old_black_mask, suptitle="old mask", cmap='gray')
cv_iml.image_show(new_black_mask, suptitle="new mask", cmap='gray')
cv_iml.image_show(combine_mask, suptitle="combile mask", cmap='gray')

