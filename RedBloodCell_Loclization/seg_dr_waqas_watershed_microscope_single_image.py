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
from custom_classes import cv_iml
# from skimage.morphology import watershed
from skimage.segmentation import watershed


def preprocess_image(image, mean_gray):
    mean_gray_resized = cv2.resize(mean_gray, (image.shape[1], image.shape[0]))

    # Convert RGB to gray scale and improve contrast of the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imge_clahe = clahe.apply(gray)

    # Subtract the Background (mean) image
    mean_subtracted = imge_clahe - mean_gray_resized
    # clone = mean_subtracted.copy()

    # Remove the pixels which are very close to the mean. 60 is selected after watching a few images
    # mean_subtracted[mean_subtracted < 60] = 0
    ret, thresh = cv2.threshold(imge_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    invert_thresh = cv2.bitwise_not(thresh)
    mean_subtracted[mean_subtracted < mean_subtracted.mean()] = 0
    # for IML microscope kernal size is 12 for opening operation
    # kernel = np.ones((12, 12), np.uint8)
    # to remove noise data form the image.
    kernel = np.ones((5, 5), np.uint8)
    mean_subtracted_open = cv2.morphologyEx(mean_subtracted, cv2.MORPH_OPEN, kernel)


    # To separate connected cells, do the Erosion. The kernal parameters are randomly selected.
    # For IML images kernal size is 6
    # kernel = np.ones((6, 6), np.uint8)
    kernel = np.ones((4, 4), np.uint8)
    forground_background_mask = cv2.erode(invert_thresh, kernel)

    return invert_thresh


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


def save_cells_annotation(annotated_img, labels):
    original_image = annotated_img.copy()

    individual_cell_images = []
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
        # these parameters are need to be set automatically

        x = x - 10
        y = y - 10
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        w = w + 15
        h = h + 15
        if (w < 65 or h < 65) or (w > 110 or h > 110):
            continue
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi = original_image[y:y + h, x:x + w, :]

        #  Make JSON object to save annotation file.
        json_object.append({"x": x, "y": y, "h": h, "w": w})
        individual_cell_images.append(roi)
        # json_object.append({
        #     # "cell_name": cell_save_name,
        #     "x": str(x),
        #     "y": str(y),
        #     "h": str(h),
        #     "w": str(w),
        #     "status": "healthy"
        # })

        # cv2.imwrite(save_individual_cell_path + cell_save_name, roi)
        cell_count += 1

    return annotated_img, individual_cell_images, json_object


# def get_mean_gray_image(directory, images_name):
#     # This function check if mean gray image is found in file then get this image otherwise compute
#     # Mean image
#     print("Computing mean Image ...")
#     mean_rgb_path = directory + "mean_image.png"
#     # this loop is use to get diminsion of mean_gray_image
#     for image in images_name:
#         rgb_first = cv2.imread(directory + image)
#         break
#
#     if os.path.exists(mean_rgb_path):
#         mean_gray = cv2.imread(mean_rgb_path)
#         return mean_gray
#     else:
#
#         # Compute Mean Image
#         mean_gray = np.zeros((rgb_first.shape[0], rgb_first.shape[1]))
#         count = 0
#         for image in images_name:
#             count = count + 1
#
#             rgb = cv2.imread(directory + image)
#             # Resize all images to be of the same size
#             rgb1 = cv2.resize(rgb, (rgb_first.shape[1], rgb_first.shape[0]))
#
#             # Convert RGB to gray scale and improve contrast of the image
#             gray = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#             imge_clahe = clahe.apply(gray)
#
#             mean_gray = mean_gray + imge_clahe
#
#         mean_gray = mean_gray / count
#         cv2.imwrite(mean_rgb_path, mean_gray)
#         # This code gives error if use this image directaly so we have to first save that image and then
#         # load that image again so error does not occure in our code.
#         mean_gray = cv2.imread(mean_rgb_path)
#         return mean_gray

def get_mean_gray_image(img):
    # This function convert image into gray and the by applying opening morphological operation computer the mean image.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # we need to computer the kernerl size by the ration of image size.
    #  for IML microscope images kernal size should be 250
    # kernel = np.ones((250, 250), np.uint8)
    kernel = np.ones((75, 75), np.uint8)
    mean_gray = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    return mean_gray


#########################################################################

def get_detected_segmentaion(image_name):
    # base path of folder where you save all related annotation.
    # folder_base_path = path.result_folder_path + "microscope_test/"
    #
    # # path of folder where each individual cell is save.
    # save_individual_cell_path = folder_base_path + "rbc/"
    # make_folder_with(save_individual_cell_path)

    # reading image from folder.
    image = cv2.imread(image_name)
    mean_gray = get_mean_gray_image(image)
    print(image_name)

    annotated_img = image.copy()
    forground_background_mask = preprocess_image(image.copy(), mean_gray)
    base_path = "/home/iml/Desktop/qazi/Model_Result_Dataset/Dataset/Shalamar_Captured_Malaria/mask/"
    cv2.imwrite(base_path+ image_name.split ('/')[-1], forground_background_mask)
    # find contours of newimg
    contours, hierarchy = cv2.findContours(forground_background_mask, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

    # %%
    # filling the "holes" of the cells
    for cnt in contours:
        cv2.drawContours(forground_background_mask, [cnt], 0, 255, -1)

    # %%
    # get labels with watershed algorithms.
    labels = watershed_labels(forground_background_mask)

    # %%
    # plot annotation on image
    annotated_img, individual_cell_images, json_object = save_cells_annotation(annotated_img,
                                                                               labels)

    return annotated_img, individual_cell_images, json_object