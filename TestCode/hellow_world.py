# //  Created by Qazi Ammar Arshad on 06/05/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
# This code required python 3.7.

"""
Description:
This a testing code for segmentaion of single image. this code will be added into API.
"""
# //  Created by Qazi Ammar Arshad on 06/05/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
# This code required python 3.7.

"""
Description:
This a testing code for segmentaion of single image. this code will be added into API.
"""

import cv2
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
    mean_subtracted = imge_clahe - mean_gray_resized
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
        w = w + 15
        h = h + 15
        if (w < 70 or h < 70) or (w > 200 or h > 200):
            continue
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi = original_image[y:y + h, x:x + w]

        #  Make JSON object to save annotation file.
        json_object.append({
            # "cell_name": cell_save_name,
            "x": str(x),
            "y": str(y),
            "h": str(h),
            "w": str(w),
            "status": "healthy"
        })
        individual_cell_images.append(roi)
        # cv2.imwrite(save_individual_cell_path + cell_save_name, roi)
        cell_count += 1

    return annotated_img, individual_cell_images, json_object


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


def get_mean_gray_image(img):
    # This function convert image into gray and the by applying opening morphological operation computer the mean image.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # we need to computer the kernerl size by the ration of image size.
    kernel = np.ones((250, 250), np.uint8)
    mean_gray = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    return mean_gray


#########################################################################
# You need to only specify these 2 parameters.
# 1.folder_base_path,
# 2.directory
# where your original images are saved.
# This is the main function of this class that calls all other functions.
def cell_segmentation(image_name):
    # base path of folder where you save all related annotation.
    folder_base_path = path.result_folder_path + "microscope_test/"

    # path of folder where each individual cell is save.
    save_individual_cell_path = folder_base_path + "rbc/"
    make_folder_with(save_individual_cell_path)

    # reading image from folder.
    image = cv2.imread(image_name)
    mean_gray = get_mean_gray_image(image)
    print(image_name)

    annotated_img = image.copy()
    forground_background_mask = preprocess_image(image.copy(), mean_gray)

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

    cv_iml.image_show(annotated_img)
    print(len(individual_cell_images))
    return annotated_img, individual_cell_images, json_object


image_path = "/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/Dataset/IML_dataset/new_microcsope" \
             "/p.v/100X_crop/IMG_4657.JPG"

cell_segmentation(image_path)
