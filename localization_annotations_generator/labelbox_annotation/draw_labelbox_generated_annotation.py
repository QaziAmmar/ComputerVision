# //  Created by Qazi Ammar Arshad on 10/05/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
# Link of dataset: https://www.kaggle.com/kmader/malaria-bounding-boxes

######################### Description: #########################
This file contains code that draw boxes on the cell annotated by LabelBox editor.
"""

import json
import cv2

from custom_classes import path
from localization_annotations_generator.model_classes.iml_labelbox_model import welcome_from_dict

# base path of folder where images and annotaion are saved.

folder_base_path = path.dataset_path + "chughati_slides_shalamar_annotated_complete/"
# path of folder where all images are save.
images_path = folder_base_path + "p.f_plus_p.v/"
# save annotaion path.
annotation_path = folder_base_path + "pv_pf.json"
# save result path.
save_result_path = folder_base_path + "annotation/"
path.make_folder_with(save_result_path)

with open(annotation_path) as dr_asma_annotation_file:
    json_annotation = json.load(dr_asma_annotation_file)

# convert Json object into python objectf
result = welcome_from_dict(json_annotation)
# %%

# read each individual images form the folder and draw the annotation on it.
# iml_json = []

counter = 1
for image in result:
    # img_dict = {}
    image_name = image.image_id
    # print(image_name, image.created_by, counter)
    counter += 1
    # reading images form the folder
    img = cv2.imread(images_path + image_name)

    detected_points = []
    groundtruth_points = []
    # draw detection points on images
    # draw annotation points on the image
    objects = []
    if image.objects is not None:
        for object in image.objects:
            # getting points form the folder and draw it on the image
            x = object.bbox.x
            y = object.bbox.y
            w = object.bbox.w
            h = object.bbox.h
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    #         objects.append({
    #             "category": object.value.value,
    #             "bbox": {"x": x, "y": y, "h": h, "w": w}
    #         })
    #
    # img_dict = {
    #     "image_id": image.external_id,
    #     "created_by": image.created_by.value,
    #     "dataset_name": "Chughati P.F",
    #     "objects": objects
    # }
    cv2.imwrite(save_result_path + image_name, img)
    # iml_json.append(img_dict)

# save annotation file

# save_json_path = folder_base_path + "pv_pf.json"
# with open(save_json_path, "w") as outfile:
#     json.dump(iml_json, outfile)
