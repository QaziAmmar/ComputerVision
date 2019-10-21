import cv2
import numpy as np
import path
import matplotlib.pyplot as plt

path.init()

# folder_name = path.dataset_path +  "Malaria_Dataset_self/SHIF_images/"
# image1_name = "IMG_4030.JPG"
# image2_name = "IMG_4031.JPG"
#
# img1 = cv2.imread(folder_name + image1_name)
# img2 = cv2.imread(folder_name + image2_name)
#
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# We extract the key points and sift descriptors for both the images as follows:
# sift = cv2.xfeatures2d.SIFT_create()
# # find the key points and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)


# plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
# plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
# plt.show()

from scipy.io import loadmat

annots = loadmat('/Users/qaziammar/Documents/MATLAB/Waseem_MSCS18051_04/Code/SIFT_Features.mat')

I1 = annots["I1"]
I2 = annots["I2"]
d1 = annots["d1"]
d2 = annots["d2"]
f1 = annots["f1"]
f2 = annots["f2"]
image1Gray = annots["image1Gray"]
image1RGB = annots["image1RGB"]
image1_path = annots["image1_path"]
image2Gray = annots["image2Gray"]
image2RGB = annots["image2RGB"]
image2_path = annots["image2_path"]
matches = annots["matches"]
scores = annots["scores"]

cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(image2RGB, f1, None))


plt.show()
