# //  Created by Qazi Ammar Arshad on 18/05/2021.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This function will generates parts of image and its respective mask.
"""

import cv2
import numpy as np
from custom_classes import path, cv_iml

divided_imgs_path = "/home/iml/Desktop/qazi/Model_Result_Dataset/Results/unet_segmentaion_crop_result/"


def combine_images(all_images_name, num_of_parts):
    """
    This function combine images that have its spacial information with it. Name of image contains
    the information of row, col and step size that it must be followd.
    :param image_path:
    :param num_of_parts:
    :return:
    """

    otsu_binary_mask = np.zeros((960, 1280, 3), np.uint8)

    for image_name in all_images_name:
        name_without_extension = image_name.split(".")[0]
        combined_row_col = name_without_extension.split("_")
        row_counter = int(combined_row_col[1])
        col_counter = int(combined_row_col[2])
        r_step = 96
        c_step = 128

        rgb = cv2.imread(divided_imgs_path + image_name)

        otsu_binary_mask[row_counter: row_counter + r_step, col_counter: col_counter + c_step] = rgb

        #  divide image into 10 small parts and then compute background and foreground threshold for these
        #  image and then combine these images.

    return otsu_binary_mask


def combine_images_without_spacial_information(image_path, num_of_parts):
    """
    This function combine images that have its spacial information with it. Name of image contains
    the information of row, col and step size that it must be followd.
    :param image_path:
    :param num_of_parts:
    :return:
    """

    all_images_name = path.read_all_files_name_from(image_path, '.JPG')

    otsu_binary_mask = np.zeros((960, 1280, 3), np.uint8)

    for image_name in all_images_name:
        name_without_extension = image_name.split(".")[0]
        combined_row_col = name_without_extension.split("_")

        row_counter = int(combined_row_col[1])
        col_counter = int(combined_row_col[2])

        r_step = 96
        c_step = 128

        rgb = cv2.imread(image_path + image_name)

        otsu_binary_mask[row_counter: row_counter + r_step, col_counter: col_counter + c_step] = rgb

        #  divide image into 10 small parts and then compute background and foreground threshold for these
        #  image and then combine these images.

    return otsu_binary_mask


save_combined_imgs_path = "/home/iml/Desktop/qazi/Model_Result_Dataset/Results/unet_combine_crop_images/"
# combined_img = combine_images(parts_imgs_path, 10)
test_imgs_name = path.read_all_files_name_from(path.dataset_path + "shalamar_segmentation_data/images/test", ".JPG")
all_images_name = path.read_all_files_name_from(divided_imgs_path, '.JPG')

# %%

for test_img in test_imgs_name:
    split_img_name = test_img.split(".")[0]
    res = [i for i in all_images_name if split_img_name in i]
    combined_img = combine_images(res, 10)
    cv2.imwrite(save_combined_imgs_path + test_img, combined_img)

# cv_iml.image_show(combined_img)
