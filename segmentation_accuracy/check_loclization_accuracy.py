# //  Created by Qazi Ammar Arshad on 16/07/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This code first detect the cells in image and then check the accuracy against the ground truth.
"""

import cv2
import json
# import torch
from custom_classes import path, cv_iml
from RedBloodCell_Segmentation.seg_dr_waqas_watershed_microscope_single_image import get_detected_segmentaion


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
        x1 = int(point['x'])
        y1 = int(point['y'])
        h1 = int(point['h'])
        w1 = int(point['w'])
        boxes_array.append([x1, y1, x1 + w1, y1 + h1])
    return boxes_array


#
# def intersect(box_a, box_b):
#     """ We resize both tensors to [A,B,2] without new malloc:
#     [A,2] -> [A,1,2] -> [A,B,2]
#     [B,2] -> [1,B,2] -> [A,B,2]
#     Then we compute the area of intersect between box_a and box_b.
#     Args:
#       box_a: (tensor) bounding boxes, Shape: [A,4].
#       box_b: (tensor) bounding boxes, Shape: [B,4].
#     Return:
#       (tensor) intersection area, Shape: [A,B].
#     """
#     A = box_a.size(0)
#     B = box_b.size(0)
#     max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
#                        box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
#     min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
#                        box_b[:, :2].unsqueeze(0).expand(A, B, 2))
#     inter = torch.clamp((max_xy - min_xy), min=0)
#     return inter[:, :, 0] * inter[:, :, 1]


#
# def jaccard(box_a, box_b):
#     """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
#     is simply the intersection over union of two boxes.  Here we operate on
#     ground truth boxes and default boxes.
#     E.g.:
#         A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
#     Args:
#         box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
#         box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
#     Return:
#         jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
#     """
#     inter = intersect(box_a, box_b)
#     area_a = ((box_a[:, 2] - box_a[:, 0]) *
#               (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
#     area_b = ((box_b[:, 2] - box_b[:, 0]) *
#               (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
#     union = area_a + area_b - inter
#     return inter / union  # [A,B]
#
#
# def get_region_props(image):
#     im = image.copy()
#
#     if len(im.shape) == 3 and im.shape[2] == 3:
#         im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#
#     label_image = skimage.measure.label(im)
#     region_props = skimage.measure.regionprops(label_image)
#     return region_props
#
#
# def get_iou_score(mask_1: np.ndarray, mask_2: np.ndarray):
#     props_mask_1 = get_region_props(mask_1)
#     props_mask_2 = get_region_props(mask_2)
#
#     IOU_bbx_mul = np.zeros((props_mask_1.__len__(), props_mask_2.__len__()))
#
#     # returning -1 only if there is no gt bbox found
#     # handle it in your code
#     if len(props_mask_1) == 0:
#         return -1
#
#     for g_b in range(0, props_mask_1.__len__()):
#         for p_b in range(0, props_mask_2.__len__()):
#             IOU_bbx_mul[g_b, p_b] = bb_intersection_over_union(props_mask_1[g_b].bbox, props_mask_2[p_b].bbox)
#
#     row_ind, col_ind = linear_sum_assignment(1 - IOU_bbx_mul)
#
# # if you calculate true positive, false positives etc using the IoU score. You won't need this calculated IoU. I
# just made it for calculating average of all calculated_IoU = [] for ir in range(0, len(row_ind)): IOU_bbx_s =
# IOU_bbx_mul[row_ind[ir], col_ind[ir]]
#
#         calculated_IoU.append(IOU_bbx_s)
#
#         # if IOU_bbx_s >= 0.5:
#         #     TP = TP + 1
#         # else:
#         #     FP = FP + 1
#         #     # FN = FN + 1
#         #     FP_loc = 1
#     # if (props_im.__len__() - props_gt.__len__()) > 0:
#     #     FP = FP + (props_im.__len__() - props_gt.__len__())
#     #     FP_loc = 1
#
#     if len(calculated_IoU) > 0:
#         calculated_IoU_mean = np.mean(calculated_IoU)
#     else:
#         calculated_IoU_mean = 0.0


# base path of folder where images and annotaion are saved.
folder_base_path = path.dataset_path + "IML_loclization_final/p.v/"
# path of folder where all images are save.
original_images_path = folder_base_path + "100X_crop/"

all_images_name = path.read_all_files_name_from(original_images_path, '.JPG')

ground_truth_labels_path = folder_base_path + "pv_loclization_annotation_code_plus_labelbox.json"

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
    if not(single_image_ground_truth['image_name'] in all_images_name) :
        continue;
    print(single_image_ground_truth["image_name"])
    # single_image_ground_truth = ground_truth[0]
    original_img = cv2.imread(original_images_path + single_image_ground_truth["image_name"])

    detected_annotated_img, _, json_object = get_detected_segmentaion(
        original_images_path + single_image_ground_truth["image_name"])

    detected_boxes = convert_points_into_boxes(json_object)
    ground_truth_boxes = convert_points_into_boxes(single_image_ground_truth["objects"])
    # ground_truth_boxes = detected_boxes

    # %%
    if 'IMG_4533.JPG' == single_image_ground_truth['image_name']:
        print("debug")

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
# precision = len(true_positive_count) / (len(true_positive_count) + (len(detected_boxes) - len(true_positive_count)))
# recall = len(true_positive_count) / (len(true_positive_count) + false_negative)
# # find F1 score for
# F1 = (2 * precision * recall) / (precision + recall)


# print("Precision: ", precision)
# print("Recall:", recall)
# print("F1 Score", F1)

# MioU = true_positive_sum / (len(true_positive) + false_instances)
