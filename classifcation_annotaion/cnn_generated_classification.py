# //  Created by Qazi Ammar Arshad on 30/06/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This code pass each cell from the cnn and generate json file that seprate the malaria and healthy
cells.
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
    save_weights_path = path.save_models_path + "MalariaDetaction_DrMoshin/basic_cnn_IML_fineTune.h5"
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


# folder base path.
folder_base_path = path.dataset_path + "IML_dataset/new_microcsope/p.v/"
final_annotation_path = folder_base_path + "CodePlusLabelBox_annotation.json"
# read json file
with open(final_annotation_path) as annotation_path:
    final_annotaion = json.load(annotation_path)

json_dictionary = []
model = get_model()

# %%

for image_annotation in final_annotaion:
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

# save cell json file.
save_json_image_path = folder_base_path + "rbc_classification_json/" + "cell_classification_annotation.json"
with open(save_json_image_path, "w") as outfile:
    json.dump(json_dictionary, outfile)
