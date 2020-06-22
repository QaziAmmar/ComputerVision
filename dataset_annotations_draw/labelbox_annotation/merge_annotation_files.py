# //  Created by Qazi Ammar Arshad on 18/06/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This code merge 2 JSON annotation files into single file.
"""
from custom_classes import path, cv_iml
from dataset_annotations_draw.labelbox_annotation.labelbox_annotation_model import welcome_from_dict
import json
import cv2

image_name = "IMG_4536.JPG"
folder_base_path = path.dataset_path + "LabelBox_annotation_test_label_box/"
img = cv2.imread(folder_base_path + image_name)

label_box_annotation_path = folder_base_path + "labelBox_IMG_4536.json"
code_annotation_path = folder_base_path + "IMG_4536.json"

with open(label_box_annotation_path) as annotation_path:
    label_box_json_annotation = json.load(annotation_path)

with open(code_annotation_path) as annotation_path:
    code_json_annotation = json.load(annotation_path)

# %%
code_detected_points = []
labelbox_points = []
# convert Json object into python object
label_box_result_python_object = welcome_from_dict(label_box_json_annotation)
code_annotaion_python_object = code_json_annotation

for point in code_annotaion_python_object:
    x = point['x']
    y = point['y']
    h = point['h']
    w = point['w']
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    code_detected_points.append([x, y, w, h])
# draw annotation points on the image
for point in label_box_result_python_object[0].label.objects:
    # getting points form the folder and draw it on the image
    x = point.bbox.left
    y = point.bbox.top
    h = point.bbox.height
    w = point.bbox.width
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    labelbox_points.append([x, y, w, h])


# %%

def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def getBoxpoints(points1, points2):
    x1 = points1[0]
    y1 = points1[1]
    h1 = points1[2]
    w1 = points1[3]
    box1 = [x1, y1, x1 + w1, y1 + h1]
    x2 = points2[0]
    y2 = points2[1]
    h2 = points2[2]
    w2 = points2[3]
    box2 = [x2, y2, x2 + w2, y2 + h2]
    return box1, box2


# %%
matched_pointes_in_lablebox_bounding_boxes = []
matched_pointes_in_code_detected_bounding_boxes = []
# compute the iOU of all the points and remove the points that very much overlaping.
for temp_bounding_boxes_lablebox in labelbox_points:
    for code_temp_point in code_detected_points:
        # finding the matched boxes form both points and then remove these points form both arrays.
        box1, box2 = getBoxpoints(temp_bounding_boxes_lablebox, code_temp_point)
        iou = intersection_over_union(box1, box2)
        if iou > 0.93:
            # append both bounding boxes into annotaion files
            print(iou)
            matched_pointes_in_lablebox_bounding_boxes.append(temp_bounding_boxes_lablebox)
            matched_pointes_in_code_detected_bounding_boxes.append(code_temp_point)

# %%
unique_bounding_boxes_array = []
#  Remove matched points from both annotations
for temp_bounding_boxes_lablebox in labelbox_points:
    for temp_matched in matched_pointes_in_lablebox_bounding_boxes:
        if temp_bounding_boxes_lablebox != temp_matched:
            unique_bounding_boxes_array.append(temp_bounding_boxes_lablebox)

for temp_bounding_boxes_code in code_detected_points:
    for temp_matched in matched_pointes_in_code_detected_bounding_boxes:
        if temp_bounding_boxes_code != temp_matched:
            unique_bounding_boxes_array.append(temp_bounding_boxes_code)

# %%
json_object = []

for temp_final_points in unique_bounding_boxes_array:
    x = temp_final_points[0]
    y = temp_final_points[1]
    w = temp_final_points[2]
    h = temp_final_points[3]

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    json_object.append({"x": x, "y": y, "h": h, "w": w})

save_json_image_path = folder_base_path + "final_annotation.json"

cv2.imwrite(folder_base_path + "final_annotaion.jpg", img)

with open(save_json_image_path, "a") as outfile:
    json.dump(json_object, outfile)
# %%
# json_dictionary = [{
#     "image_name": image_name,
#     "objects": json_object
# }]
