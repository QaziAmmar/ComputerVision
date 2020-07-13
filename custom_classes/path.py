# //  Created by Qazi Ammar Arshad on 21/10/2019.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

import sys
import os
import cv2
from glob import glob

# This file contains the paths to all required dataset, saved model and other results.

base_path = ""
dataset_path = ""
save_models_path = ""
download_path = ""
result_folder_path = ""

if sys.platform == ("linux" or "linux1" or "linux2"):
    base_path = "/home/iml/Desktop/qazi/Model_Result_Dataset/"
elif sys.platform == "darwin":
    base_path = "/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/"

dataset_path = base_path + "Dataset/"
save_models_path = base_path + "SavedModel/"
download_path = "/Users/qaziammar/Downloads/"
result_folder_path = base_path + "Results/"


def make_folder_with(folder_name):
    # this function make a folder with folder name if does not exist.
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def read_all_files_name_from(folder_path, file_extension):
    """

    :param file_extension: is the type of file that you want to extract form the folder.
    :param folder_path: this is the path of folder from which we pick all images.
    :return: this function return sorted name of images with complete path
    """

    # images = glob('/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/Dataset/Malaria_Dataset_self/crop_images
    # /*/*/*.jpg') Reading all images form file.

    all_images_name = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(folder_path):
        for file in f:
            if file_extension in file:
                all_images_name.append(file)
    # return sorted(all_images_name, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return sorted(all_images_name)


def move_images(required_images_names=[], from_folder_path="", to_folder_path=""
                , file_extension=""):
    """
    This function move selected images form one folder to other. your have to mention images name in
    required_images_names = []. this folder find that imags and move them into your required folder.
    Version = 1.0
    -> This is not stable function. change required init. working fine.
    :param required_images_names: these are the name of image that you want to find from the
    required folder
    :param from_folder_path: folder where you want to find the images.
    :param to_folder_path: folder where you want to save your matched images.
    :param file_extension: extension of file to be find.
    :return: None
    """
    all_images_name = read_all_files_name_from(from_folder_path, file_extension)

    for single_image_name in required_images_names:
        res = [i for i in all_images_name if single_image_name in i]
        index = all_images_name.index(res[0])
        img = cv2.imread(from_folder_path + res[0])
        cv2.imwrite(to_folder_path + res[0], img)
