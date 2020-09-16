# //  Created by Qazi Ammar Arshad on 16/10/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
we found some misclassified sample in our dataset so we need to update our annotation file. For this purpose first we
separate misclassified cells with the help of any model and the update the annotation file accordingly. Now this
code is only updating the cells which are classified as healthy while actually they are malaria.
"""



import json
from custom_classes import path

# name of folder where you want to find the images
# folder base path

final_annotation_path = path.dataset_path + "IML_binary_classification_final/p.f/rbc_binary_classification_json/pf_binary_classification_annotation.json"
incorrected_classified_cells_path = path.dataset_path + "IML_binary_classification_final/p.f/malaria_healthy"
incorrect_files_name = path.read_all_files_name_from(incorrected_classified_cells_path, '.JPG')
# read json file
with open(final_annotation_path) as annotation_path:
    final_loclization_annotaion = json.load(annotation_path)

json_dictionary = []

# %%
counter = 1
for image_annotation in final_loclization_annotaion:
    # we are testing on subset of images so first we check if images
    img_name = image_annotation["image_name"]

    # print(img_name)
    # load image for testing either annotation are combining in correct way
    json_object = []

    for point in image_annotation["objects"]:
        x = int(point['x'])
        y = int(point['y'])
        h = int(point['h'])
        w = int(point['w'])
        category = point['category']
        cell_name = point['cell_name']

        # here we change the cell category
        if cell_name in incorrect_files_name:
            if category == 'healthy':
                category = 'malaria'
                print(counter)
                counter += 1

        json_object.append({
            "cell_name": cell_name,
            "x": str(x),
            "y": str(y),
            "h": str(h),
            "w": str(w),
            'category': category
        })

    #   save cell location in json file.
    json_dictionary.append({
        "image_name": img_name,
        "objects": json_object
    })



#%%
print("saving annotation files in json")
# save cell json file.
# add the name of json file at the end in which you want to save the classification annotations.
save_json_image_path = path.dataset_path + "IML_binary_classification_final/p.f/rbc_binary_classification_json/new_pf_binary_classification_annotation.json"
with open(save_json_image_path, "w") as outfile:
    json.dump(json_dictionary, outfile)

print("end of code")