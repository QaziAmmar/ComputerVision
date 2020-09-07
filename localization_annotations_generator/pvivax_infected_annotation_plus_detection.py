# //  Created by Qazi Ammar Arshad on 20/03/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

from custom_classes import path
import json
import cv2
import os
import os.path
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from custom_classes import path, cv_iml

from localization_annotations_generator.pvivax_model import Annotation_Model, Objects, Bounding_box


# Link of dataset: https://www.kaggle.com/kmader/malaria-bounding-boxes

######################### Description: #########################
# This code is used to check the accuracy of segmentaion code on pvivax dataset taken for kaggel.
# this code draws both detection results and ground truth on an image.


def draw_annotation(ground_truth_annotation, annotated_images_save_path):
    # This function draw rectangles on images.
    # this counter break the loop according to requirements.
    count = 0

    for image in ground_truth_annotation:
        if count == 50:
            break
        # we need to split image path because it has also folder namd apped with it.
        image_name = image.image.path_name.split('/')[2]
        # '', 'images', '8d02117d-6c71-4e47-b50a-6cc8d5eb1d55.png']
        # img = cv2.imread(images_path + image_name)
        detected_cell_img = get_detected_segmentaion(image_name, images_path)
        img = detected_cell_img
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # counter variable to append with image name to uniquely save the image name.

        for object in image.objects:
            # separating x and y coordinate for crop image.

            x1 = object.bounding_box.x1
            y1 = object.bounding_box.y1
            x2 = object.bounding_box.x2
            y2 = object.bounding_box.y2
            # Crop image form give point.
            # draw green box on images according to annotation.
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # for object in detected_segmentations.objects:
        #     # separating x and y coordinate for crop image.
        #
        #     x1 = object.bounding_box.x1
        #     y1 = object.bounding_box.y1
        #     x2 = object.bounding_box.x2
        #     y2 = object.bounding_box.y2
        #     # Crop image form give point.
        #     # draw green box on images according to annotation.
        #     img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

        cv2.imwrite(annotated_images_save_path + image_name, img)
        count += 1
        print(count)


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
    image = annotated_img.copy()
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
        w = w + 15
        h = h + 15
        if (w < 70 or h < 70) or (w > 200 or h > 200):
            continue
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        clotted_cell_image[y:y + h, x:x + w] = 0

        #  Make JSON object to save annotation file.
        json_object.append({
            "cell_name": "",
            "x": str(x),
            "y": str(y),
            "h": str(h),
            "w": str(w),
            "category": "red blood cell"
        })
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


def get_detected_segmentaion(image_name, directory):
    # This function take a single image and detect is segmented regions.
    print(image_name)

    image = cv2.imread(directory + image_name)

    annotated_img = image.copy()

    forground_background_mask = preprocess_image(image.copy(), mean_gray)

    # find contours of newimg
    contours, hierarchy = cv2.findContours(forground_background_mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    # filling the "holes" of the cells
    for cnt in contours:
        cv2.drawContours(forground_background_mask, [cnt], 0, 255, -1)

    # get labels with watershed algorithms.
    labels = watershed_labels(forground_background_mask)

    # plot annotation on image
    annotated_img, clotted_cell_image, json_object = save_cells_annotation(annotated_img,
                                                                           forground_background_mask,
                                                                           labels, image_name)

    # detecte_segments = Annotation_Model()
    # detecte_segments.image.path_name = image_name
    # for temp_json in json_object:
    #     temp_annotation = Objects()
    #     temp_annotation.category = temp_json["category"]
    #     temp_box = Bounding_box()
    #     # (x, y), (x + w, y + h)
    #     temp_box.x1 = temp_json['x']
    #     temp_box.y1 = temp_json['y']
    #     temp_box.x2 = temp_json['x'] + temp_json['w']
    #     temp_box.y2 = temp_json['y'] + temp_json['h']
    #     temp_annotation.bounding_box = temp_box
    #     detecte_segments.objects.append(temp_annotation)

    return annotated_img

    # return annotation Object as ground truth object.


# %%

# folder name can be healthy or malaria.
folder_name = "healthy/"
# defining path for all images.
dataset_path = path.dataset_path + "p_vivax_malaria_bounding_boxes/"
images_path = dataset_path + folder_name + "images/"
train_image_annotation_path = dataset_path + folder_name + "training.json"
test_image_annotation_path = dataset_path + folder_name + "test.json"
save_crop_images_path = path.result_folder_path + "p_vivax_malaria_bounding_boxes/ground_truth/"

mean_gray = cv2.imread(images_path + "mean_image.png")
# Reading training json annotation path
with open(train_image_annotation_path) as train_image_annotation_path:
    train_annotation = json.load(train_image_annotation_path)
# Reading test json annotation path
# with open(test_image_annotation_path) as test_image_annotation_path:
#     test_annotation = json.load(test_image_annotation_path)

# %%
# Parsing JSON data into python object for easy use. combine both test and train
# images together, and the crop red blood cell from a whole image.
groundtruth_annotations = []

for annotation in train_annotation:
    groundtruth_annotations.append(Annotation_Model(annotation))

# for annotation in test_annotation:
#     groundtruth_annotations.append(Annotation_Model(annotation))

#  Get detected cell annotaion array.


draw_annotation(groundtruth_annotations, save_crop_images_path)
