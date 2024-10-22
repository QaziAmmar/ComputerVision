# //  Created by Qazi Ammar Arshad on 10/05/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
# Link of dataset: https://www.kaggle.com/kmader/malaria-bounding-boxes

######################### Description: #########################
This file contains code that draw boxes on the cell annotated by LabelBox editor.
"""



import json

import cv2

from RedBloodCell_Loclization.seg_dr_waqas_watershed_microscope_single_image import get_detected_segmentaion
from custom_classes import path
from localization_annotations_generator.model_classes.iml_labelbox_model import welcome_from_dict

# base path of folder where images and annotaion are saved.
folder_base_path = path.dataset_path + "IML_PV_65/p.v/"
# path of folder where all images are save.
images_path = folder_base_path + "100X_crop/"
# save annotaion path.
annotation_path = folder_base_path + "iml_pv_65.json"
# save result path.
save_result_path = folder_base_path + "annotation/"
path.make_folder_with(save_result_path)

with open(annotation_path) as annotation_path:
    json_annotation = json.load(annotation_path)

# convert Json object into python objectf
result = welcome_from_dict(json_annotation)
# %%

# read each individual images form the folder and draw the annotation on it.
counter = 1
for image_cells in result:
    image_name = image_cells.external_id
    print(image_name, image_cells.created_by, counter)
    counter += 1
    # reading images form the folder
    img = cv2.imread(images_path + image_name)
    _, _, json_object = get_detected_segmentaion(images_path + image_name)
    detected_points = []
    groundtruth_points = []
    # draw detection points on images
    for point in json_object:
        x = point['x']
        y = point['y']
        h = point['h']
        w = point['w']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # detected_points.append([x, y, x+w, y+h])
    # draw annotation points on the image
    for point in image_cells.label.red_blood_cell:
        # getting points form the folder and draw it on the image
        x1 = point.geometry[0].x
        y1 = point.geometry[0].y
        x2 = point.geometry[2].x
        y2 = point.geometry[2].y
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # groundtruth_points.append([x1, y1, x2, y2])

    # print(detected_points)
    # print(groundtruth_points)
    cv2.imwrite(save_result_path + image_name, img)
