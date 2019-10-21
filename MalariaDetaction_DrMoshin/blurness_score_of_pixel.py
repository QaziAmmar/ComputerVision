# //  Created by Qazi Ammar Arshad on 21/10/2019.
# //  Copyright © 2019 Qazi Ammar Arshad. All rights reserved.

from imutils import paths
import argparse
import numpy as np
import cv2
import path
import matplotlib.pyplot as plt

path.init()

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


folder_name = path.dataset_path + "Malaria_Dataset_self/SHIF_images/"
complete_image = folder_name + "IMG_4030.JPG"
crop_image = folder_name + "croped_image.JPG"
unBlur_image = folder_name + "unBlur_image.JPG"

#
images = [complete_image, crop_image, unBlur_image]

threshold = 100
# loop over the input images

for imagePath in images:
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    print(fm)
    text = "Not Blurry"
    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < threshold:
        text = "Blurry"

# show the image
# cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
# cv2.imshow("Image", image)
# key = cv2.waitKey(10)

# %%
