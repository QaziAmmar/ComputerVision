# //  Created by Qazi Ammar Arshad on 30/05/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

import tensorflow as tf
from custom_classes import path


# This file contained all testing models for malaria detection.

# def get_basic_CNN_for_malaria(INPUT_SHAPE, save_weight_path=None, binary_classification=True, classes=2):
def get_2_CNN(INPUT_SHAPE):
    # Model 1: 1-layer CNN from Scratch

    # if save_weight_path is None:
    #     save_weight_path = path.save_models_path + "malaria_binaryclass_DrMoshin/basic_cnn.h5"

    inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    # conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    # pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    flat = tf.keras.layers.Flatten()(pool2)

    hidden1 = tf.keras.layers.Dense(512, activation='relu')(flat)
    drop1 = tf.keras.layers.Dropout(rate=0.5)(hidden1)
    hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.4)(hidden2)
    hidden2 = tf.keras.layers.Dense(128, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    # model.load_weights(save_weight_path)
    return model
