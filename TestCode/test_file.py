# This is a file where we fist test our code then implement it into other file
# Annotation draw testing code.
from custom_classes import path, cv_iml
import re
import os
import cv2

folder_path = "Malaria_Dataset_self/crop_images/Microscope/p_falciprim_plus_450X/"
dataset_path = path.dataset_path + folder_path
images_name = path.read_all_files_name_from(dataset_path, ".jpg")
for image in images_name:
    img = cv2.imread(dataset_path + image)
    h, w, c = img.shape
    crop = img[300: w - 300, 300: h - 300, :]
    cv2.imwrite(dataset_path + image, crop)


