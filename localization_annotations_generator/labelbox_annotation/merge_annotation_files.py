# //  Created by Qazi Ammar Arshad on 18/06/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This code merge 2 JSON annotation files into single file.
"""
from custom_classes import path
from localization_annotations_generator.model_classes.merge_file_model import welcome_from_dict
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
    x2 = points2[0]
    y2 = points2[1]
    h2 = points2[2]
    w2 = points2[3]
    box2 = [x2, y2, x2 + w2, y2 + h2]
    return box1, box2


def check_point_in_array(point, array):
    for temp_point in array:
        if point == temp_point:
            return True
    return False


# image_name = "IMG_4536.JPG"
folder_base_path = path.dataset_path + "Shalamar_Captured_Malaria/"

label_box_annotation_path = folder_base_path + "labelbox_generated_annotation/" \
                                               "labelbox_annotation_on_code_generated_annotation.json"
code_annotation_path = folder_base_path + "code_generated_annotation/code_generated_localization.json"

json_dictionary = []

# load label box generated annotations files
with open(label_box_annotation_path) as annotation_path:
    label_box_json_annotation = json.load(annotation_path)

# you need to change this "welcome_from_dict" class each time when you write new object.
label_box_result_python_object_array = welcome_from_dict(label_box_json_annotation)

# load code generated annotations .json files
with open(code_annotation_path) as annotation_path:
    code_generated_annotations = json.load(annotation_path)


# %%
counter = 0
for label_box_result_python_object in label_box_result_python_object_array:
    if label_box_result_python_object.external_id == "PA171993.JPG":
        continue
    img_name = label_box_result_python_object.external_id
    print(img_name)
    # load image for testing either annotation are combining in correct way
    img = cv2.imread(folder_base_path + "images/" + img_name)
    # save all points that are detected by code and labelbox.
    code_detected_points = []
    labelbox_points = []

    code_generated_annotation_object = next((item for item in code_generated_annotations if item["image_name"] == img_name), None)
    if code_generated_annotation_object is not None:
        # if we found any None object the we move forward to label box loop.
        for object in code_generated_annotation_object['objects']:
            category = object["type"]
            bbox = object["bbox"]
            x = int(bbox['x'])
            y = int(bbox['y'])
            h = int(bbox['h'])
            w = int(bbox['w'])
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            code_detected_points.append([x, y, w, h])
    # draw annotation points on the image
    if label_box_result_python_object.label.objects is not None:
        for point in label_box_result_python_object.label.objects:
            # getting points form the folder and draw it on the image
            x = point.bbox.left
            y = point.bbox.top
            h = point.bbox.height
            w = point.bbox.width
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

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        json_object.append({
            "x": str(x),
            "y": str(y),
            "h": str(h),
            "w": str(w),
        })

    json_dictionary.append({
        "image_name": label_box_result_python_object.external_id,
        "objects": json_object
    })
    cv2.imwrite(folder_base_path + "loclization_results/" + label_box_result_python_object.external_id, img)
    counter += 1

# %%
save_json_image_path = folder_base_path + "code_plus_labelBox_annotation.json"
with open(save_json_image_path, "w") as outfile:
    json.dump(json_dictionary, outfile)

print("Code end with count", counter)
