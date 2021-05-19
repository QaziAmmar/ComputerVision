# //  Created by Qazi Ammar Arshad on 18/06/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.


# Generate cell localization on Shalamar Dataset.

from custom_classes import path, cv_iml
from RedBloodCell_Loclization.seg_dr_waqas_watershed_microscope_single_image import watershed_labels,\
    preprocess_image
import cv2

# base path of folder where images and annotation are saved.

folder_base_path = path.dataset_path + "shalamar_segmentation_data/image/FirstSlide/"
# path of folder where all images are save.
original_images_path = folder_base_path

# Save generated Mask
save_images_path = path.dataset_path + "shalamar_segmentation_data/mask/FirstSlideMask/"


all_images_name = path.read_all_files_name_from(original_images_path, '.JPG')

# %%
# Read image
json_dictionary = []

for image_name in all_images_name:
    # reading images form the folder
    # get annotation box on images.
    print(original_images_path + image_name)
    img = cv2.imread(original_images_path + image_name)
    forground_background_mask = preprocess_image(img)
    # classify each boxes as healthy or malaria
    contours, hierarchy = cv2.findContours(forground_background_mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # %%
    # filling the "holes" of the cells
    for cnt in contours:
        cv2.drawContours(forground_background_mask, [cnt], 0, 255, -1)

    cv2.imwrite(save_images_path + image_name, forground_background_mask)
