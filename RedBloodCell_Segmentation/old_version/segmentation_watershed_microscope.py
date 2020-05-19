# //  Created by Qazi Ammar Arshad on 15/04/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
# This code required python 3.7.

"""
Description:
This is the 3rd version of cell segmentation code. This code use watershed algorithm to extract each single
cell from complete blood slide. It saves each cell separate cell and complete annotation of blood slide in
separate folder. It also saves the coordinate of each cell in separate .txt file.
This a stable code.
"""

import cv2
import json
import os.path
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from custom_classes import path, cv_iml


def image_thresh_with_divide(image, num_of_parts):
    # This function get gray image divide it into sub parts, apply otsu thresh on these images part
    # and return binary mask.
    # divide image into 10 equal parts.
    img_clone = image
    image_parts = num_of_parts

    r, c = darker.shape
    r_step = int(r / image_parts)
    c_step = int(c / image_parts)

    otsu_binary_mask = np.zeros((r, c), np.uint8)

    #  divide image into 10 small parts and then compute background and foreground threshold for these
    #  image and then combine these images.
    if r % image_parts == 0:
        row_range = (r - r_step) + 1
    else:
        row_range = (r - r_step)

    if c % image_parts == 0:
        col_range = (c - c_step) + 1
    else:
        col_range = (c - c_step)

    for temp_row in range(0, row_range, r_step):
        for temp_col in range(0, col_range, c_step):
            # separating image part for complete image.
            temp_imge = img_clone[temp_row: temp_row + r_step, temp_col: temp_col + c_step]
            # Otsu's threshold
            ret, thresh = cv2.threshold(temp_imge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # invert
            invert_thresh = cv2.bitwise_not(thresh)
            # combining image.
            otsu_binary_mask[temp_row: temp_row + r_step, temp_col: temp_col + c_step] = invert_thresh

    return otsu_binary_mask


def watershed_labels(binary_mask):
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(binary_mask)
    localMax = peak_local_max(D, indices=False, min_distance=20,
                              labels=binary_mask)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=binary_mask)

    return labels


def save_cells_annotation(annotated_img, mask, labels, image_name):
    clotted_cell_image = mask.copy()
    json_object = []
    # loop over the unique labels returned by the Watershed algorithm
    cell_count = 0
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        # saving every single cell as a rectangular image.
        x, y, w, h = cv2.boundingRect(c)
        if (w < 20 or h < 20) or (w > 70 or h > 70):
            continue
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 0), 2)
        roi = image[y:y + h, x:x + w]
        clotted_cell_image[y:y + h, x:x + w] = 0
        cell_save_name = image_name[:-4] + "_" + str(cell_count) + ".JPG"

        #  Make JSON object to save annotation file.
        json_object.append({
            "cell_name": cell_save_name,
            "x": str(x),
            "y": str(y),
            "h": str(h),
            "w": str(w),
            "category": "red blood cell"
        })
        cv2.imwrite(save_individual_cell_path + cell_save_name, roi)
        cell_count += 1

    clotted_cell_image = clotted_cell_image * 255

    return annotated_img, clotted_cell_image, json_object


def is_new_cell_segments_found(new_count, pre_count):
    # this function terminates the loop when new generated cell are less than 10
    new_cells = new_count - pre_count
    if new_cells > 10:
        return True
    else:
        return False


def make_folder_with(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


#########################################################################
# You need to only specify these 2 parameters. folder_base_path, directory where your original images are
# saved.

# base path of folder where you save all related annotation.
folder_base_path = path.result_folder_path + "Malaria_2010_dataset/"
# where you want to read images. Microscopic captured images.
directory = folder_base_path + "original_images/"


# path where you want to save images on which rectangles are drawn.
save_annotated_image_path = folder_base_path + "annotated_img/"
make_folder_with(save_annotated_image_path)

# path of folder where each individual cell is save.
save_individual_cell_path = folder_base_path + "rbc/"
make_folder_with(save_individual_cell_path)
# path of annotation file that save the coordinate of each individual cell.
annotation_file_path = folder_base_path + "cells.json"
# append all json object for writing in json file.
json_dictionary = []

# read all images form a foler.
all_images_name = path.read_all_files_name_from(directory, '.jpg')

if all_images_name is None:
    print("No images are found!")


for image_name in all_images_name:
    print(image_name)
    image = cv2.imread(directory + image_name)

    annotated_img = image.copy()
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    darker = clahe.apply(gray)

    forground_background_mask = image_thresh_with_divide(darker, 8)
    kernel = np.ones((20, 20), np.uint8)
    forground_background_mask = cv2.morphologyEx(forground_background_mask, cv2.MORPH_OPEN, kernel)

    # %%
    # find contours of newimg
    contours, hierarchy = cv2.findContours(forground_background_mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    # %%
    # filling the "holes" of the cells
    for cnt in contours:
        cv2.drawContours(forground_background_mask, [cnt], 0, 255, -1)

    # %%
    # get labels with watershed algorithms.
    labels = watershed_labels(forground_background_mask)

    # %%
    # plot annotation on image
    annotated_img, clotted_cell_image, json_object = save_cells_annotation(annotated_img,
                                                                           forground_background_mask,
                                                                           labels, image_name)
    json_dictionary.append({
        "image_name": image_name,
        "objects": json_object
    })
    # save annotation of each image into json file.

    # save annotated_img in separate folder.
    cv2.imwrite(save_annotated_image_path + image_name, annotated_img)

with open(annotation_file_path, "a") as outfile:
    json.dump(json_dictionary, outfile)

