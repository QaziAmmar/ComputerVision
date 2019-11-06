# !/usr/bin/env python

"""
Run this file to perform stitching operation on videos.


"""
import time
import argparse
import cv2
# import the necessary packages
import imutils
import numpy as np
import os
import matplotlib.pyplot as plt

import path

path.init()

__author__ = "Qazi Ammar Arshad"
__email__ = "qaziammar.g@gmail.com"
__status__ = "This is working code of python that stitched python images automatically. Link of code: " \
             "https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/ "


def extract_key_frames_from_movie():
    folder_name = path.dataset_path + "Malaria_Dataset_self/SHIF_images/"
    video_name = "IMG_4123.MOV"
    key_frames = []
    cap = cv2.VideoCapture(folder_name + video_name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(0, frame_count, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        _, frame = cap.read()
        frame = cv2.transpose(frame)
        key_frames.append(frame)

    return key_frames


results_folder = path.dataset_path + "Malaria_Dataset_self/SHIF_images/"
out_image_name = "final_stitched_image.jpg"


time1 = time.time()
key_frames = extract_key_frames_from_movie()
key_frames.reverse()
time2 = time.time()

# initialize OpenCV's image sticher object and then perform the image
# stitching
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(key_frames)
time3 = time.time()
filename = 'savedImage.jpg'
cv2.imwrite(results_folder + out_image_name, stitched)
plt.imshow(stitched)
plt.show()

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

print('key frames extraction:', time2 - time1, ' sec')
print('Stitching: ', time3 - time2, ' sec')
