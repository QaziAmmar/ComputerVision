# //  Created by Qazi Ammar Arshad on 09/07/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This code read the annotation file and extract individual RBC form the whole image and save them into different
folder
"""
from custom_classes import path, predefine_models
import json
import numpy as np
import cv2
import os
import threading
from concurrent import futures
import glob
from custom_classes import path, cv_iml, predefine_models
from sklearn.preprocessing import LabelEncoder


# %%
# name of folder where you want to find the images
# folder base path.
folder_base_path = path.dataset_path + "IML_multiclass_classification/p.v/"
binary_classification_annotaion = folder_base_path + "rbc_binary_classification_json" \
                                                     "/pv_binary_classification_annotation.json"
# read json file
with open(binary_classification_annotaion) as annotation_path:
    final_binary_classification_annotation = json.load(annotation_path)

malaria_json_dictionary = []

# %%

for image_annotation in final_binary_classification_annotation:
    # we are testing on subset of images so first we check if images
    img_name = image_annotation["image_name"]
    img_path = folder_base_path + "100X_crop/" + img_name
    if not os.path.isfile(img_path):
        print("No file Found")
        continue
    print(img_name)
    # load image for testing either annotation are combining in correct way
    img = cv2.imread(folder_base_path + "100X_crop/" + image_annotation["image_name"])
    image = img.copy()
    malaria_cells_json = []
    # counter = 1
    for point in image_annotation["objects"]:
        x = int(point['x'])
        y = int(point['y'])
        h = int(point['h'])
        w = int(point['w'])
        category = point['category']
        cell_name = point['cell_name']
        if category == "healthy":
            continue
        # crop the cell form the image
        roi = image[y:y + h, x:x + w]
        # malaria_cells_json.append(point)
        cv2.imwrite(folder_base_path + "malaria_cell_by_cnn/ring/" + cell_name, roi)
        # counter += 1

    # append the malaria infected cell into separate json file
#     malaria_json_dictionary.append({
#         "image_name": img_name,
#         "objects": malaria_cells_json
#     })
#
# print("saving annotation files in json")
# save cell json file.
# add the name of json file at the end in which you want to save the classification annotations.
# save_json_image_path = folder_base_path + "malaria_cell/" + "pf_malaria_infected_cells_annotation.json"
# with open(save_json_image_path, "w") as outfile:
#     json.dump(malaria_json_dictionary, outfile)
