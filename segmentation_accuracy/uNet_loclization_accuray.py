# //  Created by Qazi Ammar Arshad on 16/07/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This code first detect the cells in image and then check the accuracy against the ground truth.
"""

import cv2
import json
# import torch
from custom_classes import path, cv_iml
from RedBloodCell_Loclization.seg_dr_waqas_watershed_microscope_single_image import getBoundingBoxes_from_UNet_detected_segmentaion


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


def convert_points_into_boxes(points):
    # this code convert the points into boxes
    boxes_array = []
    for point in points:
        if 'x' in point:
            x1 = int(point['x'])
            y1 = int(point['y'])
            h1 = int(point['h'])
            w1 = int(point['w'])
        else:
            bbox = point['bbox']
            x1 = int(bbox['x'])
            y1 = int(bbox['y'])
            h1 = int(bbox['h'])
            w1 = int(bbox['w'])
        boxes_array.append([x1, y1, x1 + w1, y1 + h1])
    return boxes_array


# base path of folder where images and annotaion are saved.

original_images_path = path.dataset_path + "test_imgs/test/"
masked_images_path = path.result_folder_path + "unet_combine_crop_segmentation_results/"
# path of folder where all images are save.

all_images_name = path.read_all_files_name_from(original_images_path, '.JPG')

ground_truth_labels_path = path.dataset_path + "shalamar_segmentation_data/shalamar_dataset.json"

# %%
with open(ground_truth_labels_path) as annotation_path:
    ground_truth = json.load(annotation_path)
# %%
true_positive_count = 0
false_positive = 0
false_negative = 0

# %%
# iterate through all images and find TF and FP.
for single_image_ground_truth in ground_truth:
    if not (single_image_ground_truth['image_name'] in all_images_name):
        continue;
    print(single_image_ground_truth["image_name"])
    # single_image_ground_truth = ground_truth[0]
    original_img = cv2.imread(original_images_path + single_image_ground_truth["image_name"])
    image_mask = cv2.imread(masked_images_path + single_image_ground_truth["image_name"])
    image_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)

    detected_annotated_img, _, json_object = getBoundingBoxes_from_UNet_detected_segmentaion(original_img, image_mask)
    cv2.imwrite(path.result_folder_path + "unet_combine_crop_loclization_shamalar/" + single_image_ground_truth["image_name"], detected_annotated_img)

    detected_boxes = convert_points_into_boxes(json_object)
    ground_truth_boxes = convert_points_into_boxes(single_image_ground_truth["objects"])
    # ground_truth_boxes = detected_boxes

    # %%

    iou_score_2d_array = []
    # make a 2d matrix of iou of all points with other points
    for i, temp_ground_truth_box in zip(range(len(ground_truth_boxes)), ground_truth_boxes):
        temp_iou_score = []
        for j, temp_detected_box in zip(range(len(detected_boxes)), detected_boxes):
            iou = intersection_over_union(temp_ground_truth_box, temp_detected_box)
            temp_iou_score.append(iou)
        if len(temp_iou_score) > 0:
            iou_score_2d_array.append(temp_iou_score)

    # %%
    # find maximum matching index from ground truth.
    ground_truth_matched_index_in_detect_boxes_array = []
    for temp_iou_score in iou_score_2d_array:
        max_index = temp_iou_score.index(max(temp_iou_score))
        max_iou = max(temp_iou_score)
        # all detected points in
        ground_truth_matched_index_in_detect_boxes_array.append([max_index, max_iou])

    # %%
    # Now separated the true positive, false positive and false negative
    true_positive = []
    for temp_matched_box in ground_truth_matched_index_in_detect_boxes_array:
        if temp_matched_box[1] >= 0.5:
            true_positive.append(temp_matched_box)
    # boxes that have value grater then 0.49 is called as true positive and other are called as false_positive

    # %%
    # calculate MioU of labels
    true_positive_sum = 0
    for temp_instance in true_positive:
        true_positive_sum += temp_instance[1]
    # false_instance = false positive + false negative
    true_positive_count += len(true_positive)
    false_positive += (len(detected_boxes) - len(true_positive))
    false_negative += (len(ground_truth_boxes) - len(true_positive))
    false_instances = false_positive + false_negative

print("Total True Positives:", true_positive_count)
print("Total False Negative", false_negative)
print("Total False Positives:", false_positive)

precision = true_positive_count / (true_positive_count + false_positive)
print("Precision: ", precision)
recall = true_positive_count / (true_positive_count + false_negative)
print("Recall:", recall)
# find F1 score for
F1 = (2 * precision * recall) / (precision + recall)
print("F1 Score", F1)
