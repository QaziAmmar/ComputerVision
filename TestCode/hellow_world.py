# This is a file where we fist test our code then implement it into other file
from custom_classes import path, cv_iml
import cv2
import psycopg2
import numpy
import numpy as np
from scipy.ndimage import label


def segment(im1, img):
    # morphological transformations
    border = cv2.dilate(img, None, iterations=10)
    border = border - cv2.erode(border, None, iterations=1)
    # invert the image so black becomes white, and vice versa
    img = -img
    # applies distance transform and shows visualization
    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    # reapply contrast to strengthen boundaries
    clache = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dt = clache.apply(dt)
    # rethreshold the image
    _, dt = cv2.threshold(dt, 40, 255, cv2.THRESH_BINARY)

    lbl, ncc = label(dt)
    lbl = lbl * (255 / ncc)
    # Complete the markers
    lbl[border == 255] = 255

    lbl = lbl.astype(numpy.int32)
    # apply watershed
    cv2.watershed(im1, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(numpy.uint8)
    # return the image as one list, and the labels as another.
    return dt, lbl


# load your image, I called mine 'rbc'
img = cv2.imread('/Users/qaziammar/Downloads/original_image.jpg')
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
dt, result = segment(img, img_bin)

print("end of fucntion")