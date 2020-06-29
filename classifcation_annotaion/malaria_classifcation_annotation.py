# //  Created by Qazi Ammar Arshad on 28/06/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This file try to capture malaria cells.
"""

from custom_classes import path, cv_iml
import json
from classifcation_annotaion.pvivax_classification_model import welcome_from_dict

# image_name = "IMG_4536.JPG"
folder_base_path = path.dataset_path + "LabelBox_classification/"

label_box_annotation_path = folder_base_path + "classifaction_malaria.json"
code_annotation_path = folder_base_path + "CodePlusLabelBox_annotation.json"

json_dictionary = []

with open(label_box_annotation_path) as annotation_path:
    label_box_json_annotation = json.load(annotation_path)

with open(code_annotation_path) as annotation_path:
    code_annotaion_python_object_array = json.load(annotation_path)

label_box_result_python_object_array = welcome_from_dict(label_box_json_annotation)

# %%
expectedResult = [sub for sub in label_box_json_annotation if sub['External ID'] == "IMG_4430.JPG"]

