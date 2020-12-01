# //  Created by Qazi Ammar Arshad on 19/11/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This file contains code that take complete slide image, extract single cells from each image and
then classify each cell as healthy or malaria
"""
from custom_classes import path, cv_iml, predefine_models
from RedBloodCell_Loclization.seg_dr_waqas_watershed_microscope_single_image import get_detected_segmentaion
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

INPUT_SHAPE = (125, 125, 3)
classes = 2

model = predefine_models.get_resnet50v2(INPUT_SHAPE, classes=classes)
model.load_weights("/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/SavedModel/binary_resnet50v2.h5")

def image_normalization(img):
    IMG_DIMS = (125, 125)
    img = cv2.resize(img, dsize=IMG_DIMS, interpolation=cv2.INTER_CUBIC)
    img = np.array(img, dtype=np.float32)
    return img


def check_image_malaria(images):
    # Model 1: CNN from Scratch

    img_list = []
    print("length of images" + str(len(images)))
    for temp_img in images:
        img = image_normalization(temp_img)
        img_list.append(img)

    test_data = np.array(list(img_list))

    print("img", len(test_data))

    test_img_scaled = test_data / 255.
    basic_cnn_preds = model.predict(test_img_scaled)

    train_labels = ["healthy", "malaria"]
    le = LabelEncoder()
    le.fit(train_labels)

    basic_cnn_preds_argmax = basic_cnn_preds.argmax(1)
    basic_cnn_preds_labels = le.inverse_transform(basic_cnn_preds_argmax)
    prediction_scores = [max(max_pred) for max_pred in basic_cnn_preds]

    prediction = []

    for temp_prediction, temp_confidance in zip(basic_cnn_preds_labels, prediction_scores):
        prediction.append({"prediction": temp_prediction, "confidence": str(temp_confidance)})

    return prediction


# base path of folder where images and annotation are saved.
folder_base_path = path.dataset_path
# path of folder where all images are save.
original_images_path = folder_base_path + "100OLYMP/"
save_images_path = folder_base_path + "loclization_results/"

all_images_name = path.read_all_files_name_from(original_images_path, '.JPG')

# %%
# Read image
json_dictionary = []

for image_name in all_images_name:
    # reading images form the folder
    cell_prediction_json = []
    img = cv2.imread(original_images_path + image_name)
    # get annotation box on images.
    _, individual_cell_images, json_object = get_detected_segmentaion(original_images_path + image_name)
    # classify each boxes as healthy or malaria
    prediction = check_image_malaria(individual_cell_images)
    # Merging cell segmentation and classification retuls
    for temp_json, temp_pred in zip(json_object, prediction):
        temp_json.update(temp_pred)
        cell_prediction_json.append(temp_json)

    # here we get our predicted response.
    annotated_img = img.copy()
    # after the we have to draw our response images according to prediction.
    for temp_prediction in cell_prediction_json:
        x = temp_prediction['x']
        y = temp_prediction['y']
        h = temp_prediction['h']
        w = temp_prediction['w']
        confidence = temp_prediction['confidence']
        prediction = temp_prediction['prediction']
        if prediction == 'healthy':
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
        else:
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 5)

    cv2.imwrite(save_images_path + image_name, annotated_img)
