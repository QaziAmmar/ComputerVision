# This is a file where we fist test our code then implement it into other file
# This Code run on python 3.6
from custom_classes import path, cv_iml
import cv2
import os
import numpy as np
import copy
import matplotlib.pyplot as plt

save_folder_path = path.result_folder_path + "color_consistency/stretch/"

# folder_path = "Malaria_dataset/malaria/"
# dataset_path = path.dataset_path + folder_path
dataset_path = save_folder_path
images_name = path.read_all_files_name_from(dataset_path, ".jpg")
# result_path = path.result_folder_path + "morphological_drwaqas_malaria_online/"
# result_path = save_folder_path
mean_rgb_path = save_folder_path + "mean_image.png"

for image in images_name:
    # image_segments = 500
    rgb_first = cv2.imread(dataset_path + image)
    break

if os.path.exists(mean_rgb_path):
    mean_gray = cv2.imread(mean_rgb_path)
else:

    # Compute Mean Image
    mean_gray = np.zeros((rgb_first.shape[0], rgb_first.shape[1]))
    tot_images = len(images_name)
    count = 0
    for image in images_name:
        print(image)
        print(count)
        count = count + 1
        # if count < 10:
        #     continue
        # Just to avoid last few dense color images. Not sure whether it is useful or not!

        rgb = cv2.imread(dataset_path + image)
        # Resize all images to be of the same size
        rgb1 = cv2.resize(rgb, (rgb_first.shape[1], rgb_first.shape[0]))

        # Convert RGB to gray scale and improve contrast of the image
        gray = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imge_clahe = clahe.apply(gray)

        mean_gray = mean_gray + imge_clahe
        # if count > tot_images - 10:
        #     break
        # if count>370:

        #  break
    # mean Image
    mean_gray = mean_gray / count
    # save_path = path.dataset_path + "Malaria_dataset/"
    cv2.imwrite(mean_rgb_path, mean_gray)

count = 0
for image in images_name:
    count = count + 1

    if count < 1:
        continue

    rgb = cv2.imread(dataset_path + image)

    # Resize all images to be of the same size
    rgb_resized = cv2.resize(rgb, (mean_gray.shape[1], mean_gray.shape[0]))

    # Convert RGB to gray scale and improve contrast of the image
    gray = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imge_clahe = clahe.apply(gray)

    # Subtract the Background (mean) image
    mean_subtracted = imge_clahe - mean_gray[:, :, 0]
    clone = mean_subtracted.copy()

    # Remove the pixels which are very close to the mean. 60 is selected after watching a few images
    mean_subtracted[mean_subtracted < 60] = 0

    # To separate connected cells, do the Erosion. The kernal parameters are randomly selected.
    kernel = np.ones((20, 20), np.uint8)
    mean_subtracted_erode = cv2.erode(mean_subtracted, kernel)
    kernel = np.ones((7, 7), np.uint8)
    closing = cv2.morphologyEx(mean_subtracted_erode, cv2.MORPH_CLOSE, kernel)
    _, contours_single_erode, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the regions which are still large and strongle collected and apply errosion again only to those regions
    mean_subtracted_erode_forLarge = copy.deepcopy(mean_subtracted_erode)
    area_c = []
    for c in contours_single_erode:
        (x, y, w, h) = cv2.boundingRect(c)
        area_c = w * h
        if area_c < 30000:
            mean_subtracted_erode_forLarge[y:y + h, x:x + w] = 0

    kernel = np.ones((20, 20), np.uint8)
    mean_subtracted_doubleerode_forLarge = cv2.erode(mean_subtracted_erode_forLarge, kernel)

    closing = cv2.morphologyEx(mean_subtracted_doubleerode_forLarge, cv2.MORPH_CLOSE, kernel)
    _, contours_double_erode, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Plot results with single errosion
    rgb_single_erode = copy.deepcopy(rgb_resized)
    for c in contours_single_erode:
        (x, y, w, h) = cv2.boundingRect(c)
        x1 = max(1, x - 15)
        y1 = max(1, y - 15)
        x2 = min(x + w + 25, rgb_resized.shape[1])
        y2 = min(y + h + 25, rgb_resized.shape[0])

        area_c = w * h
        # if area_c < 2000:
        #    continue
        # else:
        # cv2.rectangle(img=rgb_single_erode, pt1=(max(1,x-15), max(1,y-15) ), pt2=(x + w+25 , y + h+25), color=(255, 0, 225), thickness=5)
        cv2.rectangle(img=rgb_single_erode, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 225), thickness=2)

    # Plot results with double errosion
    rgb_double_erode = copy.deepcopy(rgb_resized)

    for c in contours_single_erode:
        (x, y, w, h) = cv2.boundingRect(c)
        x1 = max(1, x - 15)
        y1 = max(1, y - 15)
        x2 = min(x + w + 25, rgb_resized.shape[1])
        y2 = min(y + h + 25, rgb_resized.shape[0])

        area_c = w * h
        if 2000 < area_c < 30000:
            # cv2.rectangle(img=rgb_double_erode,  pt1=(max(x-15,1), max(y-15,1) ), pt2=(x + w+25 , y + h+25), color=(255, 0, 225), thickness=5)
            cv2.rectangle(img=rgb_double_erode, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 225), thickness=2)

    for c in contours_double_erode:
        (x, y, w, h) = cv2.boundingRect(c)

        x1 = max(1, x - 35)
        y1 = max(1, y - 35)
        x2 = min(x + w + 35, rgb_resized.shape[1])
        y2 = min(y + h + 35, rgb_resized.shape[0])
        area_c = w * h
        if area_c < 2000:
            continue
        else:
            cv2.rectangle(img=rgb_double_erode, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 225), thickness=2)

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    # ax1.imshow(rgb_resized)
    # ax2.imshow(rgb_single_erode)
    # ax3.imshow(rgb_double_erode)
    # plt.show()
    ####################################
    cv2.imwrite(save_folder_path + image, rgb_single_erode)
    # continue
