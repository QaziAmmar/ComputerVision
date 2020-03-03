# This is a file where we fist test our code then implement it into other file
from custom_classes import path, cv_iml
import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import label

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table, find_contours
from skimage.transform import rotate


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

dataset_path = path.dataset_path + "/cell_images/train/malaria/"
save_path = path.dataset_path + "/cell_images/train/malaria/"
images_name = path.read_all_files_name_from(dataset_path, ".JPG")
counter = 0
# file_path = "/home/itu/Desktop/Qazi/Model_Result_Dataset/Dataset/cell_images/rgb_test.txt"
# image_tag = "malaria"
for image_name in images_name:
    img = cv2.imread(dataset_path + image_name)
    flipVertical = cv2.flip(img, 0)
    flipHorizontal = cv2.flip(img, 1)
    flipBoth = cv2.flip(img, -1)
    cv2.imwrite(save_path + image_name[:-4] + "fv.JPG", flipVertical)
    cv2.imwrite(save_path + image_name[:-4] + "fh.JPG", flipHorizontal)
    cv2.imwrite(save_path + image_name[:-4] + "fvh.JPG", flipBoth)

# for image_name in images_name:
#     img = cv2.imread(dataset_path + image_name)
#     if (image_name.find('_90.JPG') == -1) and (image_name.find('_270.JPG') == -1) \
#             and (image_name.find('_180.JPG') == -1):
#         cv2.imwrite(save_path + image_name, img)


# i = 0
# for image in images_name:
#     img = cv2.imread(dataset_path + image)
#     rgb = img.copy()
#     clone = img.copy()
#     # keep resizing your image so it appropriately identifies the RBC's
#     img = cv2.resize(img, (0, 0), fx=5, fy=5)
#     # it's always easier if the image is copied for long pieces of code.
#     # we're copying it twice for reasons you'll see soon enough.
#     wol = img.copy()
#     gg = img.copy()
#     # convert to grayscale
#     img_gray = cv2.cvtColor(gg, cv2.COLOR_BGR2GRAY)
#     # enhance contrast (helps makes boundaries clearer)
#     clache = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     img_gray = clache.apply(img_gray)
#     # threshold the image and apply morphological transforms
#     _, img_bin = cv2.threshold(img_gray, 50, 255,
#                                cv2.THRESH_OTSU)
#     img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
#                                numpy.ones((3, 3), dtype=int))
#     img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_DILATE,
#                                numpy.ones((3, 3), dtype=int), iterations=1)
#     # call the 'segment' function (discussed soon)
#     # dt, result = segment(img, img_bin)
#
#     cv2.imwrite(save_path + image, img_bin)
#     if i % 20 == 0:
#         print(i)
#     i = i + 1
#   save image into other folder.
