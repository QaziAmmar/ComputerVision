# //  Created by Qazi Ammar Arshad on 20/03/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

from custom_classes import path
import json
import cv2
import os
from dataset_annotations.pvivax_model import Annotation_Model

# Link of dataset: https://www.kaggle.com/kmader/malaria-bounding-boxes

######################### Description: #########################
# This code separate the red blood cell from the whole sample slide
# according to the annotation and save them into a relevant folder.


# folder name can be healthy or malaria.
folder_name = "malaria/"
# defining path for all images.
dataset_path = path.dataset_path + "malaria_bounding_boxes/"
images_path = dataset_path + folder_name + "images/"
train_image_annotation_path = dataset_path + folder_name + "training.json"
test_image_annotation_path = dataset_path + folder_name + "test.json"
save_crop_images_path = path.result_folder_path + "pvivax_malaria_cells/"

# Reading training json annotation path
with open(train_image_annotation_path) as train_image_annotation_path:
    train_annotation = json.load(train_image_annotation_path)
# Reading test json annotation path
with open(test_image_annotation_path) as test_image_annotation_path:
    test_annotation = json.load(test_image_annotation_path)

# %%
# Parsing JSON data into python object for easy use. combine both test and train
# images together, and the crop red blood cell from a whole image.
python_annotations = []

for annotation in train_annotation:
    python_annotations.append(Annotation_Model(annotation))

for annotation in test_annotation:
    python_annotations.append(Annotation_Model(annotation))


# %%
# Crop red blood cells from image form image according to annotation and save them in
# required folder.


def separate_rbs(images_annotations, save_images_path):
    # these are two folders in which we save our image on the base of categories.
    # healthy = "healthy/"
    # malaria = "malaria/"

    # This function save the cell according to their categories.
    difficult = "difficult"
    gametocyte = "gametocyte"
    leukocyte = "leukocyte"
    red_blood_cell = "red_blood_cell"
    ring = "ring"
    schizont = "schizont"
    trophozoite = "trophozoite"

    for image in images_annotations:
        # we need to split image path because it has also folder namd apped with it.
        image_name = image.image.path_name.split('/')[2]
        # '', 'images', '8d02117d-6c71-4e47-b50a-6cc8d5eb1d55.png']
        img = cv2.imread(images_path + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # counter variable to append with image name to uniquely save the image name.
        count = 0
        for object in image.objects:
            # separating x and y coordinate for crop image.

            x1 = object.bounding_box.x1
            y1 = object.bounding_box.y1
            x2 = object.bounding_box.x2
            y2 = object.bounding_box.y2
            # Crop image form give point.
            crop_image = img[y1:y2, x1:x2, :]
            # if object category is red blood cell then does save it into a healthy folder.
            # if object category is not red blood cell then save it into malaria folder.
            image_tag = "/" + str(count)
            if object.category == difficult:
                save_name = save_images_path + difficult + image_tag + "_" + image_name
            elif object.category == ring:
                save_name = save_images_path + ring + image_tag + "_" + image_name
            elif object.category == schizont:
                save_name = save_images_path + schizont + image_tag + "_" + image_name
            elif object.category == trophozoite:
                save_name = save_images_path + trophozoite + image_tag + "_" + image_name
            elif object.category == gametocyte:
                save_name = save_images_path + gametocyte + image_tag + "_" + image_name
            elif object.category == leukocyte:
                save_name = save_images_path + leukocyte + image_tag + "_" + image_name
            else:
                continue
            cv2.imwrite(save_name, crop_image)
            count += 1


# In this dataset some images have .jpg extension, so we change their extension into .png
# also, change their extension form annotation.json file. Other method can be you can change
# a segmented cell extension on the annotation time in upper mention function.
def change_extension_of_image(images_annotations):
    counter = 0
    for image in images_annotations:
        # we need to split image path because it has also folder namd apped with it.
        image_name = image.image.path_name.split('/')[2]
        # '', 'images', '8d02117d-6c71-4e47-b50a-6cc8d5eb1d55.png']
        img = cv2.imread(images_path + image_name)
        if img is None:
            image_name = image_name.split('.')[0] + ".jpg"
            img = cv2.imread(images_path + image_name)
        # remove first image and then save other because both .png and .jpg file remains in
        # the folder
        os.remove(images_path + image_name)
        # append .png extension on all dataset.
        image_name = image_name.split('.')[0] + ".png"
        cv2.imwrite(images_path + image_name, img)
        if counter % 20 == 0:
            print(counter)
        counter += 1


# change_extension_of_image(python_annotations)
separate_rbs(python_annotations, save_crop_images_path)
