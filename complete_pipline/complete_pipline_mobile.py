# //  Created by Qazi Ammar Arshad on 05/08/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This file contains the code of complete pipeline of our purposed method.
1. we separate each cell from microscopic image.
2. Separate healthy and malaria cells.
3. classify the life-cycle stage of each malaria cell.
"""
# this code need to match the plot colors with legend color in image plot.

import cv2
import json
import tensorflow as tf
import threading
import numpy as np
from concurrent import futures
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from custom_classes import path, cv_iml
from RedBloodCell_Loclization.seg_dr_waqas_watershed_microscope_single_image import get_detected_segmentaion


def get_basic_CNN_for_binary(INPUT_SHAPE):
    inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    flat = tf.keras.layers.Flatten()(pool3)

    hidden1 = tf.keras.layers.Dense(512, activation='relu')(flat)
    drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
    hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

    out = tf.keras.layers.Dense(2, activation='sigmoid')(drop2)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    # model.load_weights(save_weight_path)
    return model


def get_basic_CNN_for_multiclass(INPUT_SHAPE, classes=1):
    """
    Model 1: CNN from Scratch.
    This model server for both binary and multiple-class classification of malaria. if you want to do
    multi-class classification then you need to false the binary_classification and need to mention
    number of classes.

    :param INPUT_SHAPE: Shape of input vector
    :param binary_classification: this is for binary classification in our case it is malaria and
    healthy cells.
    :param classes: Number of classes for different stage of malaria.
    :return: model tensorflow model
    """

    inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    flat = tf.keras.layers.Flatten()(pool3)

    hidden1 = tf.keras.layers.Dense(128, activation='relu')(flat)
    drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
    hidden2 = tf.keras.layers.Dense(64, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

    out = tf.keras.layers.Dense(classes, activation='softmax')(drop2)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    # model.load_weights(save_weight_path)
    return model


INPUT_SHAPE = (125, 125, 3)


def normalized_data(input_images):
    # INPUT_SHAPE = (125, 125, 3)
    IMG_DIMS = (125, 125)

    def get_img_data_parallel(idx, img, total_imgs):
        if idx % 5000 == 0 or idx == (total_imgs - 1):
            print('{}: working on img num {}'.format(threading.current_thread().name, idx))
        img = cv2.resize(img, dsize=IMG_DIMS, interpolation=cv2.INTER_CUBIC)
        img = np.array(img, dtype=np.float32)

        return img

    ex = futures.ThreadPoolExecutor(max_workers=None)
    test_data_inp = [(idx, img, len(input_images)) for idx, img in enumerate(input_images)]

    print('\nLoading Test Images:')
    test_data_map = ex.map(get_img_data_parallel,
                           [record[0] for record in test_data_inp],
                           [record[1] for record in test_data_inp],
                           [record[2] for record in test_data_inp])

    test_data = np.array(list(test_data_map))

    print(test_data.shape)

    # normalized data
    test_imgs_scaled = test_data / 255.

    # encoding labels into 0 and 1 form
    test_labels = ["Parasitized", "Uninfected"]
    le = LabelEncoder()
    le.fit(test_labels)

    test_labels_enc = le.transform(test_labels)
    test_labels_enc = to_categorical(test_labels_enc, num_classes=len(test_labels))
    return test_imgs_scaled, test_labels_enc, le


def get_multiclass_labels():
    train_labels = ['gametocyte', 'healthy', 'ring', 'schizont', 'trophozoite']

    number_of_classes = len(train_labels)
    le = LabelEncoder()
    le.fit(train_labels)

    train_labels_enc = le.transform(train_labels)
    train_labels_enc = to_categorical(train_labels_enc, num_classes=number_of_classes)

    print(train_labels[:6], train_labels_enc[:6])
    return le


# read image name.

image_path = path.dataset_path + "IML_binary_classification_final/p.v/100X_crop/IMG_4467.JPG"
# "IML_dataset/new_microcsope/p.f_plus/100X_crop/IMG_2571.JPG"


# localize cell location.
img = cv2.imread(image_path)
annotated_img, individual_cell_images, json_object = get_detected_segmentaion(image_path)
# %%
# normalized input data.
test_imgs_scaled, test_labels_enc, le = normalized_data(individual_cell_images)

# %%
# load binary mode.
model = get_basic_CNN_for_binary(INPUT_SHAPE)

load_weights_path = path.save_models_path + "IML_binary_CNN_experimtents/basicCNN_binary/pv_basic_binary_cnn.h5"
model.load_weights(load_weights_path)

# %%
# binary prediction
basic_cnn_preds = model.predict(test_imgs_scaled)
basic_cnn_preds = basic_cnn_preds.argmax(1)
basic_cnn_preds_labels = le.inverse_transform(basic_cnn_preds)

# basic_cnn_preds_labels = le.inverse_transform([1 if pred > 0.5 else 0
#                                                for pred in basic_cnn_preds.ravel()])

# %%
# update annotation according to model's prediction.
counter = 0

for prediction_label in basic_cnn_preds_labels:
    json_object[counter]["prediction"] = prediction_label
    (x, y, w, h) = (json_object[counter]['x'], json_object[counter]['y'], json_object[counter]['w'],
                    json_object[counter]['h'])
    if prediction_label == "Parasitized":
        cv2.rectangle(img=annotated_img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=3)
    else:
        cv2.rectangle(img=annotated_img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=3)
    counter = counter + 1

# %%

cv_iml.image_show(annotated_img)
# now predict the life-cycle stage of each malaria cell.
multi_le = get_multiclass_labels()
multi_class_model = get_basic_CNN_for_multiclass(INPUT_SHAPE, classes=5)

# %%
# change the path of your save model.
load_weights_path = path.save_models_path + "IML_multiclass_CNN_experimtents/BasicCNN/pv_basicCNN_multiclass.h5"
multi_class_model.load_weights(load_weights_path)

# %%
# separate malaria images. and them into a seperate array
malaria_cells = []
for d, temp_test_img in zip(json_object, test_imgs_scaled):
    if d['prediction'] == "Parasitized":
        malaria_cells.append(temp_test_img)
malaria_cells = np.array(malaria_cells)
# %%
# Model Performance Evaluation
basic_cnn_preds = multi_class_model.predict(malaria_cells)
basic_cnn_preds = basic_cnn_preds.argmax(1)
prediction_labels = multi_le.inverse_transform(basic_cnn_preds)

# %%
# update labels according to new multi-class prediction
prediction_counter = 0
for i in range(len(json_object)):
    if json_object[i]['prediction'] == "Parasitized":
        json_object[i]['prediction'] = prediction_labels[prediction_counter]
        prediction_counter += 1

# %%

# draw different color for each category of life cycle stage of malaria.
for cell in json_object:
    x = cell['x']
    y = cell['y']
    w = cell['w']
    h = cell['h']
    if cell['prediction'] == 'gametocyte':
        cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 255), thickness=3)
        print(cell['prediction'])
    elif cell['prediction'] == 'trophozoite':
        cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 0), thickness=3)
        print(cell['prediction'])
    elif cell['prediction'] == 'ring':
        cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=3)
        print(cell['prediction'])
    elif cell['prediction'] == "schizont":
        cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 255), thickness=3)
        print(cell['prediction'])
    else:
        cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 129, 0), thickness=3)

cv2.imwrite("/Users/qaziammar/Desktop/test.jpg", img)

# %%

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

red_patch = mpatches.Patch(color='green', label='healthy')
blue_patch = mpatches.Patch(color='blue', label='ring')
cyan_patch = mpatches.Patch(color='cyan', label='schizont')
magenta_patch = mpatches.Patch(color='yellow', label='trophozoite')
silver_patch = mpatches.Patch(color='magenta', label='gametocyte')
plt.imshow(img)
plt.legend(handles=[red_patch, blue_patch, cyan_patch, magenta_patch, silver_patch], loc=4, fontsize=7)
plt.show()


