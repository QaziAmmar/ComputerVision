# //  Created by Qazi Ammar Arshad on 18/06/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This code combine the label box annotations with code detection annotation that only include the malaria samples
"""
from custom_classes import path
from localization_annotations_generator.labelbox_annotation.draw_labelbox_generated_annotation import welcome_from_dict
import json
import cv2
import os


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
    x2 = int(points2[0])
    y2 = int(points2[1])
    h2 = int(points2[2])
    w2 = int(points2[3])
    box2 = [x2, y2, x2 + w2, y2 + h2]
    return box1, box2


def check_point_in_array(point, array):
    for temp_point in array:
        if point == temp_point:
            return True
    return False


# image_name = "IMG_4536.JPG"
folder_base_path = path.dataset_path + "chughati_slides_shalamar_annotated_complete/"

label_box_annotation_path = folder_base_path + "pv_pf.json"
code_annotation_path = folder_base_path + "pf_plus_binary_classification_annotation.json"

json_dictionary = []

with open(label_box_annotation_path) as annotation_path:
    label_box_json_annotation = json.load(annotation_path)

# you need to change this "welcome_from_dict" class each time when you write new object.
label_box_result_python_array = welcome_from_dict(label_box_json_annotation)

with open(code_annotation_path) as annotation_path:
    code_annotaion_python_dict = json.load(annotation_path)

# %%

for label_box_result_python_object in label_box_result_python_array:

    print(label_box_result_python_object.image_id)
    # load image for testing either annotation are combining in correct way
    code_annotaion_python_object = next(item for item in code_annotaion_python_dict if item["image_name"] ==
                                        label_box_result_python_object.image_id)

    code_detected_points = []
    labelbox_points = []
    # convert Json object into python object

    for point in code_annotaion_python_object['objects']:
        x = point['x']
        y = point['y']
        h = point['h']
        w = point['w']
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        code_detected_points.append([x, y, w, h])
    # draw annotation points on the image
    if label_box_result_python_object.objects is None:
        print("continue")
        continue
    for point in label_box_result_python_object.objects:
        # getting points form the folder and draw it on the image
        x = point.bbox.x
        y = point.bbox.y
        h = point.bbox.h
        w = point.bbox.w
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        labelbox_points.append([x, y, w, h])

    # %%
    matched_pointes_in_lablebox_bounding_boxes = []
    matched_pointes_in_code_detected_bounding_boxes = []
    # compute the iOU of all the points and rempvivax_labelBox_annotationove the points that very much overlaping.
    for temp_labelBox_point in labelbox_points:
        for code_temp_point in code_detected_points:
            # finding the matched boxes form both points and then remove these points form both arrays.
            box1, box2 = getBoxpoints(temp_labelBox_point, code_temp_point)
            iou = intersection_over_union(box1, box2)
            if iou > 0.75:
                # append both bounding boxes into matched array so that these points can be removed form
                # annotation file
                print(iou)
                matched_pointes_in_lablebox_bounding_boxes.append(temp_labelBox_point)
                matched_pointes_in_code_detected_bounding_boxes.append(code_temp_point)

# %%

    unique_bounding_boxes_array = []
    #  Remove matched points from both annotations array
    for temp_labelBox_point in labelbox_points:
        if check_point_in_array(temp_labelBox_point, matched_pointes_in_lablebox_bounding_boxes):
            None
        else:
            unique_bounding_boxes_array.append(temp_labelBox_point)

    for temp_code_annotation_point in code_detected_points:
        if check_point_in_array(temp_code_annotation_point, matched_pointes_in_code_detected_bounding_boxes):
            None
        else:
            unique_bounding_boxes_array.append(temp_code_annotation_point)

    # %%
    json_object = []

    for temp_final_points in unique_bounding_boxes_array:
        x = temp_final_points[0]
        y = temp_final_points[1]
        w = temp_final_points[2]
        h = temp_final_points[3]

        type = "healthy"
        if temp_final_points in labelbox_points:
            type = "malaria"

        bbox = {
            "type": type,
            "bbox": {
                "x": str(x),
                "y": str(y),
                "h": str(h),
                "w": str(w)}

        }
        json_object.append(bbox)

    json_dictionary.append({
        "image_name": label_box_result_python_object.image_id,
        "objects": json_object
    })

# %%
save_json_image_path = folder_base_path + "pf_pv_loclization.json"
with open(save_json_image_path, "w") as outfile:
    json.dump(json_dictionary, outfile)

print("Code end")
