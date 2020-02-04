# This file contain the paths to all required dataset, saved model and other reults
import sys
import os
from glob import glob

base_path = ""
dataset_path = ""
save_models_path = ""
download_path = ""
result_folder_path = ""

if sys.platform == ("linux1" or "linux2"):
    base_path = "/home/ali/Desktop/ammar/Thesis/Model_Result_Dataset/"
elif sys.platform == "darwin":
    base_path = "/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/"

dataset_path = base_path + "Dataset/"
save_models_path = base_path + "SavedModel/"
download_path = "/Users/qaziammar/Downloads/"
result_folder_path = "/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/Results/"


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
