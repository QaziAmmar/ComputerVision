# //  Created by Qazi Ammar Arshad on 30/06/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This code pass each cell from the cnn and generate json file that separate the malaria and healthy
cells. Model used in this code is basic cnn that is traind on cell_imaegs dataset and then fint tune on IML dataset.
"""
from custom_classes import path, predefine_models
import json
import numpy as np
import cv2
import os
import threading
from concurrent import futures
import glob
from custom_classes import path, cv_iml, predefine_models
from sklearn.preprocessing import LabelEncoder

INPUT_SHAPE = (125, 125, 3)
IMG_DIMS = (125, 125)


def get_model():
    model = predefine_models.get_basic_CNN_for_malaria(INPUT_SHAPE)
    save_weights_path = path.save_models_path + "MalariaDetaction_DrMoshin/best_resutls/basic_cnn_IML_fineTune.h5"
    model.load_weights(save_weights_path)
    return model


def get_prediction_from_model(img, model):
    img = cv2.resize(img, dsize=IMG_DIMS, interpolation=cv2.INTER_CUBIC)
    img = np.array(img, dtype=np.float32)

    img_list = [img]
    test_data = np.array(list(img_list))

    test_imgs_scaled = test_data / 255.

    train_labels = ["healthy", "malaria"]
    le = LabelEncoder()
    le.fit(train_labels)

    train_labels_enc = le.transform(train_labels)
    # Model Performance Evaluation

    basic_cnn_preds = model.predict(test_imgs_scaled)
    basic_cnn_preds_labels = le.inverse_transform([1 if pred > 0.6 else 0
                                                   for pred in basic_cnn_preds.ravel()])
    return basic_cnn_preds_labels


# %%
# name of folder where you want to find the images
folder_name = "p.f"
# folder base path.
folder_base_path = path.dataset_path + "IML_binary_classification_final/" + folder_name + "/"
final_annotation_path = folder_base_path + "pf_loclization_annotation_code_plus_labelbox.json"
# read json file
with open(final_annotation_path) as annotation_path:
    final_loclization_annotaion = json.load(annotation_path)

json_dictionary = []
model = get_model()

# %%

for image_annotation in final_loclization_annotaion:
    # we are testing on subset of images so first we check if images
    img_name = image_annotation["image_name"]
    img_path = folder_base_path + "100X_crop/" + img_name
    if not os.path.isfile(img_path):
        print("No file Found")
        continue
    print(img_name)
    # load image for testing either annotation are combining in correct way
    img = cv2.imread(folder_base_path + "100X_crop/" + image_annotation["image_name"])
    image = img.copy()
    json_object = []
    counter = 1
    for point in image_annotation["objects"]:
        x = int(point['x'])
        y = int(point['y'])
        h = int(point['h'])
        w = int(point['w'])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi = image[y:y + h, x:x + w]
        # get image prediction from cnn and save this prediction in json file.
        prediction = get_prediction_from_model(roi, model)

        cell_name = img_name[:-4] + "_" + str(counter) + ".JPG"
        json_object.append({
            "cell_name": cell_name,
            "x": str(x),
            "y": str(y),
            "h": str(h),
            "w": str(w),
            'category': prediction[0]
        })

        # cv2.imwrite(folder_base_path + "rbc/" + cell_name, roi)
        counter += 1

    #     save image annotated image
    # cv2.imwrite(folder_base_path + "final_image_code/" + img_name, img)
    #   save cell location in json file.
    json_dictionary.append({
        "image_name": img_name,
        "objects": json_object
    })

print("saving annotation files in json")
# save cell json file.
# add the name of json file at the end in which you want to save the classification annotations.
save_json_image_path = folder_base_path + "rbc_classification_json/" + "pf_binary_classification_annotation.json"
with open(save_json_image_path, "w") as outfile:
    json.dump(json_dictionary, outfile)

print("end of code")