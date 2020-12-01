# //  Created by Qazi Ammar Arshad on 16/07/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This code first detect the cells in image and then check the accuracy against the ground truth.
"""

import cv2
import json
# import torch
from custom_classes import path, cv_iml
from RedBloodCell_Loclization.seg_dr_waqas_watershed_microscope_single_image import get_detected_segmentaion


# base path of folder where images and annotaion are saved.
original_images_path = path.dataset_path + "100OLYMP/"
# path of folder where all images are save.
all_images_name = path.read_all_files_name_from(original_images_path, '.JPG')

ground_truth_labels_path = path.dataset_path + "shalamar_dataset.json"

save_image_folder =  path.dataset_path + "100OLYMP_a/"

# %%
with open(ground_truth_labels_path) as annotation_path:
    ground_truth = json.load(annotation_path)

# %%
# iterate through all images and find TF and FP.
annotation_image = []
for single_image_ground_truth in ground_truth:
    annotation_image.append(single_image_ground_truth['image_name'])

#%%
import collections
print([item for item, count in collections.Counter(all_images_name).items() if count > 1])


#%%
a = set(all_images_name)
b = set(annotation_image)
