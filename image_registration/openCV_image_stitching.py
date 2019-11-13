# !/usr/bin/env python

"""
Run this file to perform stitching operation on videos.


"""

import matplotlib.pyplot as plt
import path
import cv2
import imutils
import time
import numpy as np
import os

path.init()

__author__ = "Qazi Ammar Arshad"
__email__ = "qaziammar.g@gmail.com"
__status__ = "This is working code of python that stitched python images automatically. Link of code: " \
             "https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/ "
__description__ = "This Code stitch both video frames and multiple image form same folder. If selection == 1 then " \
                  "this code stitch the images form images folder. other selection will stitch the video frames " \
                  "together."


def img_show(img):
    plt.imshow(img)
    plt.show()


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


def read_all_images_name(folder_path):
    """

    :param folder_path: this is the path of folder in which we pick all images.
    :return: this function return sorted name of images with complete path
    """
    # Reading all images form file.
    all_images_name = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(folder_path):
        for file in f:
            if '.JPG' in file:
                all_images_name.append(r + file)
    return sorted(all_images_name)


def define_row_col_crop_limit(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    row, col = gray.shape
    row_limit = 0
    col_limit = 0
    # select row range.
    for i in range(0, row, 10):
        temp_img_row = gray[i, :]
        # Zero counter does not work because the zero has value form 0 - 10.
        zero_count = np.sum(temp_img_row < 10)
        non_zero_count = col - zero_count
        percentage = (zero_count / col) * 100
        if percentage < 6:
            row_limit = i
            break

        # select col range.
        for j in range(0, col, 10):
            temp_img_col = gray[:, j]
            # Zero counter does not work because the zero has value form 0 - 10.
            zero_count = np.sum(temp_img_col < 10)
            non_zero_count = col - zero_count
            percentage = (zero_count / row) * 100
            if percentage < 6:
                col_limit = j
                break
    return row_limit, col_limit


def remove_non_zero_row_col_of(image):
    row_limit, col_limit = define_row_col_crop_limit(image)
    row, col, ch = image.shape
    cropped_image = image[row_limit: row - row_limit, col_limit: col - col_limit, :]
    return cropped_image


def read_all_images_form(images_names):
    # Reading all images form file.
    all_images_array = []
    for image_name in images_names:
        img = cv2.imread(image_name)
        img = remove_black_region(img)
        img = remove_non_zero_row_col_of(img)
        all_images_array.append(img)
    return all_images_array


def extract_key_frames_from_movie(movie_path):
    cap = cv2.VideoCapture(movie_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    key_frames = []
    for i in range(0, int(frame_count / 4), 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        # print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        _, frame = cap.read()
        frame = cv2.transpose(frame)
        frame = image_resize(frame, height=300)
        frame = remove_black_region(frame)
        key_frames.append(frame)

    return key_frames


selection = 1

out_image_name = "stitched_image.jpg"
time1 = time.time()
if selection == 1:
    results_folder = path.dataset_path + "Malaria_Dataset_self/SHIF_images/miscrscope_panaroma.2/"
    print("[INFO] key frames extraction of images...")
    # This function perform stitching on images.
    images_name = read_all_images_name(results_folder)
    key_frames = read_all_images_form(images_name)
    time2 = time.time()
else:
    # This function perform stitching combining video frames.
    movie_path = path.dataset_path + "Malaria_Dataset_self/SHIF_images/fern_video/IMG_4194.MOV"
    results_folder = movie_path
    key_frames = extract_key_frames_from_movie(movie_path)
    key_frames.reverse()
    time2 = time.time()

# initialize OpenCV's image sticher object and then perform the image
# stitching
print("[INFO] stitching images...")
# PANORAMA is 0, SCANS is 1 as defined
# enum  	Mode {
#   PANORAMA = 0,
#   SCANS = 1
# }
stitcher = cv2.Stitcher.create(mode=1)

(status, stitched) = stitcher.stitch(key_frames)

if status == 1:
    print(" ERR_NEED_MORE_IMGS = 1,")
elif status == 2:
    print("ERR_HOMOGRAPHY_EST_FAIL = 2,")
elif status == 3:
    print("ERR_CAMERA_PARAMS_ADJUST_FAIL = 3")

time3 = time.time()
cv2.imwrite(results_folder + out_image_name, stitched)

# convert BGR to RGB so that color shown right.
stitched = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
plt.imshow(stitched)
plt.show()

print('key frames extraction:', time2 - time1, ' sec')
print('Stitching: ', time3 - time2, ' sec')
