import matplotlib.pyplot as plt
import path
import cv2
import imutils
import time
import numpy as np
import os

path.init()


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def remove_black_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = -1
    best_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

    x, y, w, h = cv2.boundingRect(best_cnt)
    crop = img[y:y + h, x:x + w]
    return crop


folder_name = path.dataset_path + "Malaria_Dataset_self/SHIF_images/"

img1_path = folder_name + "IMG_4158.JPG"
img2_path = folder_name + "IMG_4159.JPG"

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

img1 = remove_black_region(img1)
img2 = remove_black_region(img2)

img1 = image_resize(img1, height=350)
img2 = image_resize(img2, height=350)

# img1 = img1[1000: 3000, 1000: 6500, :]
# img2 = img2[1000: 3000, 1000: 7000, :]

key_frames = [img1, img2]
time1 = time.time()
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()

(status, stitched) = stitcher.stitch(key_frames)
time2 = time.time()
print('Stitching: ', time2 - time1, ' sec')

cv2.imwrite(folder_name + "fernStitch_panaroma.jpg", stitched)
# cv2.imshow("img", stitched)
# cv2.waitKey(0)
plt.imshow(stitched)
plt.show()

