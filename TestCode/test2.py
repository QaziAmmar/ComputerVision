# //  Created by Qazi Ammar Arshad on 16/07/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
Draw Json Annotation on cell images
"""

import cv2
import json
from custom_classes import path
from segmentation_accuracy.check_loclization_accuracy import convert_points_into_boxes
# base path of folder where images and annotaion are saved.
folder_base_path = path.dataset_path + "100OLYMP/"
# path of folder where all images are save.
original_images_path = folder_base_path

all_images_name = path.read_all_files_name_from(original_images_path, '.JPG')
ground_truth_labels_path = path.dataset_path + "shalamar_dataset.json"
save_image_path = path.dataset_path + "Shalamar_Captured_Malaria/json_loclization/"

# %%
with open(ground_truth_labels_path) as annotation_path:
    ground_truth = json.load(annotation_path)

# %%
# iterate through all images and find TF and FP.
for single_image_ground_truth in ground_truth:
    if not(single_image_ground_truth['image_name'] in all_images_name) :
        continue;
    print(single_image_ground_truth["image_name"])
    # single_image_ground_truth = ground_truth[0]
    original_img = cv2.imread(original_images_path + single_image_ground_truth["image_name"])
    ground_truth_boxes = convert_points_into_boxes(single_image_ground_truth["objects"])
    for cell_location in single_image_ground_truth["objects"]:
        cell_location = cell_location['bbox']
        x = int(cell_location['x'])
        y = int(cell_location['y'])
        h = int(cell_location['h'])
        w = int(cell_location['w'])
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 5)

    cv2.imwrite(save_image_path + single_image_ground_truth["image_name"], original_img)

