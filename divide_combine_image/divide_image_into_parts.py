# //  Created by Qazi Ammar Arshad on 18/05/2021.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This function will generates parts of image and its respective mask.
"""

import cv2
import numpy as np
from custom_classes import path, cv_iml

# base path of folder where images and annotaion are saved.
image_path = path.dataset_path + "shalamar_croped_segmentation_data/images/val/"
mask_path = path.dataset_path + "shalamar_croped_segmentation_data/mask/val/"

save_crop_imgs_path = path.dataset_path + "shalamar_croped_segmentation_data/crop_images/val/"
save_crop_mask_path = path.dataset_path + "shalamar_croped_segmentation_data/crop_mask/val/"


# path of folder where all images are save.

def divide_image(image_name, num_of_parts):
    # This function get gray image divide it into sub parts, apply otsu thresh on these images part
    # and return binary mask.
    # Read Image
    rgb = cv2.imread(image_path + image_name)
    image = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # Load revelant mask of image
    mask = cv2.imread(mask_path + image_name)

    # divide image into 10 equal parts.
    img_clone = rgb
    image_parts = num_of_parts

    r, c = image.shape
    r_step = int(r / image_parts)
    c_step = int(c / image_parts)

    #  divide image into 10 small parts and then compute background and foreground threshold for these
    #  image and then combine these images.
    if r % image_parts == 0:
        row_range = (r - r_step) + 1
    else:
        row_range = (r - r_step)

    if c % image_parts == 0:
        col_range = (c - c_step) + 1
    else:
        col_range = (c - c_step)

    for temp_row in range(0, row_range, r_step):
        for temp_col in range(0, col_range, c_step):
            # separating image part for complete image.
            crop_imge = img_clone[temp_row: temp_row + r_step, temp_col: temp_col + c_step]
            crop_mask = mask[temp_row: temp_row + r_step, temp_col: temp_col + c_step]

            save_img_name = save_crop_imgs_path + image_name.split(".")[0] \
                            + "_" + str(temp_row) + "_" \
                            + str(temp_col) + ".JPG"
            save_mask_name = save_crop_mask_path + image_name.split(".")[0] \
                             + "_" + str(temp_row) + "_" \
                             + str(temp_col) + ".JPG"

            cv2.imwrite(save_img_name, crop_imge)
            cv2.imwrite(save_mask_name, crop_mask)


all_images_name = path.read_all_files_name_from(image_path, ".JPG")

for image_name in all_images_name:
    print(image_name)
    divide_image(image_name, 10)
