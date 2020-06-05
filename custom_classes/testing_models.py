# //  Created by Qazi Ammar Arshad on 30/05/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

import tensorflow as tf
from custom_classes import path


# This file contained all testing models for malaria detection.


def get_1_CNN(INPUT_SHAPE):
    """
    Dataset: Malaria Cell Dataset
    Results :
    Test loss: 2.7911472206371372e+32
    Test accuracy: 0.9611756
    Accuracy: 0.961176
    Precision: 0.963096
    Recall: 0.960343
    F1 score: 0.961717
    [[2610  103]
     [ 111 2688]]

    :param INPUT_SHAPE:
    :return:
    """
    # Model 1: 1-layer CNN from Scratch

    inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

    conv1 = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(inp)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    flat = tf.keras.layers.Flatten()(pool4)

    hidden1 = tf.keras.layers.Dense(512, activation='relu')(flat)
    drop1 = tf.keras.layers.Dropout(rate=0.5)(hidden1)
    hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.4)(hidden2)
    hidden3 = tf.keras.layers.Dense(128, activation='relu')(drop2)
    drop3 = tf.keras.layers.Dropout(rate=0.3)(hidden3)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(drop3)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    # model.load_weights(save_weight_path)
    return model


def get_2_CNN(INPUT_SHAPE):
    """
    Dataset: Malaria Cell Dataset
    Results :


    :param INPUT_SHAPE:
    :return:
    """
    # Model 1: 1-layer CNN from Scratch

    inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

    conv1 = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(inp)
    conv2 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv4 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3))(conv5)

    flat = tf.keras.layers.Flatten()(pool3)

    hidden1 = tf.keras.layers.Dense(120, activation='relu')(flat)
    drop1 = tf.keras.layers.Dropout(rate=0.5)(hidden1)
    hidden2 = tf.keras.layers.Dense(60, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.4)(hidden2)
    hidden3 = tf.keras.layers.Dense(10, activation='relu')(drop2)
    drop3 = tf.keras.layers.Dropout(rate=0.3)(hidden3)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(drop3)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    # model.load_weights(save_weight_path)
    return model
