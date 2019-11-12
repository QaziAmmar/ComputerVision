# !/usr/bin/env python

"""
Run this file to perform stitching operation on videos.


"""

import matplotlib.pyplot as plt
import path
import cv2
import imutils
import time
import os

path.init()

__author__ = "Qazi Ammar Arshad"
__email__ = "qaziammar.g@gmail.com"
__status__ = "This is working code of python that stitched python images automatically. Link of code: " \
             "https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/ "
__description__ = "This Code stitch both video frames and multiple image form same folder. If selection == 1 then " \
                  "this code stitch the images form images folder. other selection will stitch the video frames " \
                  "together."


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


def read_all_images_form(images_names):
    # Reading all images form file.
    all_images_array = []
    for image_name in images_names:
        img = cv2.imread(image_name)
        # img = image_resize(img, height=900)
        img = remove_black_region(img)
        all_images_array.append(img)
    return all_images_array


def extract_key_frames_from_movie(movie_path):
    cap = cv2.VideoCapture(movie_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    key_frames = []
    for i in range(0, int(frame_count / 4), 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        # print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        _, frame = cap.read()
        frame = cv2.transpose(frame)
        key_frames.append(frame)

    return key_frames


selection = 2

out_image_name = "stitched_image_2f.jpg"
time1 = time.time()
if selection == 1:
    results_folder = path.dataset_path + "Malaria_Dataset_self/SHIF_images/miscrscope_panaroma.2/"

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

plt.imshow(stitched)
plt.show()

print('key frames extraction:', time2 - time1, ' sec')
print('Stitching: ', time3 - time2, ' sec')

# if the status is '0', then OpenCV successfully performed image
# stitching
# if status == 0:
#     # check to see if we supposed to crop out the largest rectangular
#     # region from the stitched image
#     if 10 > 0:
#         # create a 10 pixel border surrounding the stitched image
#         print("[INFO] cropping...")
#         stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
#                                       cv2.BORDER_CONSTANT, (0, 0, 0))
#
#         # convert the stitched image to grayscale and threshold it
#         # such that all pixels greater than zero are set to 255
#         # (foreground) while all others remain 0 (background)
#         gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
#         thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
#         # find all external contours in the threshold image then find
#         # the *largest* contour which will be the contour/outline of
#         # the stitched image
#         cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#                                 cv2.CHAIN_APPROX_SIMPLE)
#         cnts = imutils.grab_contours(cnts)
#         c = max(cnts, key=cv2.contourArea)
#
#         # allocate memory for the mask which will contain the
#         # rectangular bounding box of the stitched image region
#         mask = np.zeros(thresh.shape, dtype="uint8")
#         (x, y, w, h) = cv2.boundingRect(c)
#         cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
#         # create two copies of the mask: one to serve as our actual
#         # minimum rectangular region and another to serve as a counter
#         # for how many pixels need to be removed to form the minimum
#         # rectangular region
#         minRect = mask.copy()
#         sub = mask.copy()
#
#         # keep looping until there are no non-zero pixels left in the
#         # subtracted image
#         while cv2.countNonZero(sub) > 0:
#             # erode the minimum rectangular mask and then subtract
#             # the thresholded image from the minimum rectangular mask
#             # so we can count if there are any non-zero pixels left
#             minRect = cv2.erode(minRect, None)
#             sub = cv2.subtract(minRect, thresh)
#             # find contours in the minimum rectangular mask and then
#             # extract the bounding box (x, y)-coordinates
#             cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
#                                     cv2.CHAIN_APPROX_SIMPLE)
#             cnts = imutils.grab_contours(cnts)
#             c = max(cnts, key=cv2.contourArea)
#             (x, y, w, h) = cv2.boundingRect(c)
#
#             # use the bounding box coordinates to extract the our final
#             # stitched image
#             stitched = stitched[y:y + h, x:x + w]
#             # write the output stitched image to disk
#             # cv2.imwrite(args["output"], stitched)
#
#             # display the output stitched image to our screen
#             # cv2.imshow("Stitched", stitched)
#             # cv2.waitKey(0)
#
#         # otherwise the stitching failed, likely due to not enough keypoints)
#         # being detected
#         else:
#             print("[INFO] image stitching failed ({})".format(status))
