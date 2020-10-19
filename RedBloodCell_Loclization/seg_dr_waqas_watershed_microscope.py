# //  Created by Qazi Ammar Arshad on 06/05/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
# This code required python 3.7.

"""
Description:
This is the 4rd version of cell segmentation code. This code is a combination of dr waqas code plus
watershed algorithm to extract each single cell from complete blood slide.
It saves each cell separate cell and complete annotation
of blood slide in separate folder. It also saves the coordinate of each cell in separate .txt file.
Currently, we are using this code for classification of blood cells form chughati labs.
"""

import cv2
import json
import os.path
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from custom_classes import path, cv_iml


def preprocess_image(image, mean_gray):
    mean_gray_resized = cv2.resize(mean_gray, (image.shape[1], image.shape[0]))

    # Convert RGB to gray scale and improve contrast of the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imge_clahe = clahe.apply(gray)

    # Subtract the Background (mean) image
    mean_subtracted = imge_clahe - mean_gray_resized[:, :, 0]
    # clone = mean_subtracted.copy()

    # Remove the pixels which are very close to the mean. 60 is selected after watching a few images
    # mean_subtracted[mean_subtracted < 60] = 0
    mean_subtracted[mean_subtracted < mean_subtracted.mean()] = 0
    # to remove noise data form the image.
    kernel = np.ones((12, 12), np.uint8)
    mean_subtracted_open = cv2.morphologyEx(mean_subtracted, cv2.MORPH_OPEN, kernel)

    # To separate connected cells, do the Erosion. The kernal parameters are randomly selected.
    kernel = np.ones((6, 6), np.uint8)
    forground_background_mask = cv2.erode(mean_subtracted_open, kernel)

    return forground_background_mask


def image_thresh_with_divide(image, num_of_parts):
    # This function get gray image divide it into sub parts, apply otsu thresh on these images part
    # and return binary mask.
    # divide image into 10 equal parts.
    img_clone = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    darker = cv2.equalizeHist(gray)

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
    gray = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2GRAY)
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
        # Change these coordinate according to microscope resolution.

        x = x - 10
        y = y - 10
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        w = w + 15
        h = h + 15
        # need to fix these parametes automatically
        if (w < 70 or h < 70) or (w > 200 or h > 200):
            continue
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
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
        # cv2.imwrite(save_individual_cell_path + cell_save_name, roi)
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


def get_mean_gray_image(directory, images_name):
    # This function check if mean gray image is found in file then get this image otherwise compute
    # Mean image
    print("Computing mean Image ...")
    mean_rgb_path = directory + "mean_image.png"
    # this loop is use to get diminsion of mean_gray_image
    for image in images_name:
        rgb_first = cv2.imread(directory + image)
        break

    if os.path.exists(mean_rgb_path):
        mean_gray = cv2.imread(mean_rgb_path)
        return mean_gray
    else:

        # Compute Mean Image
        mean_gray = np.zeros((rgb_first.shape[0], rgb_first.shape[1]))
        count = 0
        for image in images_name:
            count = count + 1

            rgb = cv2.imread(directory + image)
            # Resize all images to be of the same size
            rgb1 = cv2.resize(rgb, (rgb_first.shape[1], rgb_first.shape[0]))

            # Convert RGB to gray scale and improve contrast of the image
            gray = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            imge_clahe = clahe.apply(gray)

            mean_gray = mean_gray + imge_clahe

        mean_gray = mean_gray / count
        cv2.imwrite(mean_rgb_path, mean_gray)
        # This code gives error if use this image directaly so we have to first save that image and then
        # load that image again so error does not occure in our code.
        mean_gray = cv2.imread(mean_rgb_path)
        return mean_gray


#########################################################################
# You need to only specify these 2 parameters.
# 1.folder_base_path,
# 2.directory
# where your original images are saved.

# base path of folder where you save all related annotation.
folder_base_path = path.result_folder_path + "microscope_test/"
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
all_images_name = path.read_all_files_name_from(directory, '.JPG')

if all_images_name is None:
    print("No images are found!")

mean_gray = get_mean_gray_image(directory, all_images_name)

for image_name in all_images_name:
    print(image_name)
    image = cv2.imread(directory + image_name)

    annotated_img = image.copy()

    forground_background_mask = preprocess_image(image.copy(), mean_gray)

    # %%
    # find contours of newimg
    contours, hierarchy = cv2.findContours(forground_background_mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    # %%
    # filling the "holes" of the cells
    for cnt in contours:
        cv2.drawContours(forground_background_mask, [cnt], 0, 255, -1)

    cv2.imwrite(save_annotated_image_path + "mask" + image_name, forground_background_mask)
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

exit(0)
# The code belwo this part is under testing phase.
# %%
# Iterate through the image until no cell left behind
kernel = np.ones((3, 3), np.uint8)
counter = 0

while counter < 10:
    # Apply erosion on the image so that large cell can also be seprated
    clotted_cell_image = cv2.resize(clotted_cell_image, image.shape[1::-1])
    remaining_image = cv2.bitwise_and(image, image, mask=clotted_cell_image)
    remaining_image_sharp = cv_iml.apply_sharpening_on(remaining_image)

    gray = cv2.cvtColor(remaining_image_sharp, cv2.COLOR_BGR2GRAY)
    # improve the contrast of our images
    darker = cv2.equalizeHist(gray)

    ret, thresh = cv2.threshold(darker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    forground_background_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=kernel)

    labels = watershed(forground_background_mask)
    prev_count = total_cell_count

    annotated_img, clotted_cell_image, total_cell_count = save_cells_annotation(annotated_img,
                                                                                labels, total_cell_count, image_name)
    counter += 1
    # this condition terminates the loop if no new cells are found.
    print(total_cell_count)

    if is_new_cell_segments_found(total_cell_count, prev_count):
        continue
    else:
        print("code is terminated because no new cells found")
        break

print(counter)
