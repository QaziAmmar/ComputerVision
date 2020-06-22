from custom_classes import path, cv_iml
from dataset_annotations_draw.labelbox_annotation.labelbox_annotation_model import welcome_from_dict
import json
import cv2


folder_base_path = path.dataset_path + "LabelBox_annotation_test_label_box/"
image_name = "IMG_4536.JPG"

# test code for drawing annotation
code_annotation_path = folder_base_path + "final_annotation.json"

with open(code_annotation_path) as annotation_path:
    code_json_annotation = json.load(annotation_path)

# %%
img = cv2.imread(folder_base_path + image_name)
code_annotaion_python_object = code_json_annotation

for point in code_annotaion_python_object:
    x = point['x']
    y = point['y']
    h = point['h']
    w = point['w']
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


    # code_detected_points.append([x, y, x + w, y + h])

cv2.imwrite(folder_base_path + "final_annotaion.jpg", img)