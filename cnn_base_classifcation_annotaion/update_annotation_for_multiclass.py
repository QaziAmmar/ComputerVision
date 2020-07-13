# //  Created by Qazi Ammar Arshad on 09/07/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
We have separate the red blood cell according to their life cycle stage manually. This code read each cell form it
 folder and update the annotation files accordingly.
"""
import os
import json
from custom_classes import path

#  Read all sub folders form a folder and then read all files form the folder.
folder_base_path = path.dataset_path + "IML_multiclass_classification/p.v/"
binary_classification_annotaion_path = folder_base_path + "rbc_binary_classification_json" \
                                                          "/pv_binary_classification_annotation.json"
malaria_cell_folders_path = folder_base_path + "malaria_cell_by_cnn/"
# read json file
with open(binary_classification_annotaion_path) as annotation_path:
    binary_classification_annotation_dictonary = json.load(annotation_path)

# %%

multiclass_files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(malaria_cell_folders_path):
    for file in f:
        if '.JPG' in file:
            multiclass_files.append({
                "cell_name": file,
                "category": r.split('/')[-1]
            })

for f in multiclass_files:
    print(f)

# %%

for image_annotation in binary_classification_annotation_dictonary:
    # we are testing on subset of images so first we check if images
    img_name = image_annotation["image_name"]
    for point in image_annotation["objects"]:
        # check this cell name into multiclass file and update the annotation acceding to it folder name
        cell_name = point['cell_name']
        print(cell_name)
        classified_cell = next((item for item in multiclass_files if item['cell_name'] == cell_name), None)
        if classified_cell is None:
            continue
        point['category'] = classified_cell["category"]

# save the multi class classification into separate folder.
# %%
print("saving annotation files in json")
# save cell json file
# add the name of json file at the end in which you want to save the classification annotations.
save_json_image_path = folder_base_path + "rbc_multiclass_classification_json/" + \
                       "pv_multiclass_classification_annotation.json"
with open(save_json_image_path, "w") as outfile:
    json.dump(binary_classification_annotation_dictonary, outfile)
