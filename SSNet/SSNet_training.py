# %%

from keras.models import Sequential
from keras.utils import plot_model
import keras as k
import keras.backend as kb
# import pydot
# import graphviz
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, Input, Reshape, Permute, \
    GlobalAveragePooling2D
from keras import optimizers
# from chainer import serializers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.misc import imresize
import csv
import scipy.io as sio
from PIL import Image
import scipy
import pickle
import time
from custom_classes import path

# %%

def softmax_cross_entropy(y_true, y_pred):
    #     y_hat = kb.mean(y_pred, axis = (1, 2))
    #     y_hat = y_pred
    #     return y_true * -kb.log(y_hat) + (1.0 - y_true) * -kb.log(1.0 - y_hat)
    #     if y_true == 1:
    #         return -kb.log(y_hat)
    #     else:
    #         return -kb.log(1 - y_hat)

    #     y_hat = np.average(y_pred[0, :, :, 1])
    #     loss = 0
    #     if y_true == 1:
    #         loss = -log(y_hat)
    #     elif y_true == 0:
    #         loss = -log(1 - y_hat)
    #     return np.float32(loss)
    return kb.categorical_crossentropy(y_true, y_pred)


# %%

def create_model():
    model = Sequential()

    # conv1_1
    model.add(Conv2D(64, kernel_size=3, strides=1,
                     padding='SAME', use_bias=False,
                     activation='relu', name='conv1_1', batch_input_shape=(None, None, None, 3)))
    # conv1_2
    model.add(Conv2D(64, kernel_size=3, strides=1,
                     padding='SAME', use_bias=False,
                     activation='relu', name='conv1_2'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # conv2_1
    model.add(Conv2D(128, kernel_size=3, strides=1,
                     padding='SAME', use_bias=False,
                     activation='relu', name="conv2_1"))

    # conv2_2
    model.add(Conv2D(128, kernel_size=3, strides=1,
                     padding='SAME', use_bias=False,
                     activation='relu', name='conv2_2'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # conv3_1
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='SAME', use_bias=False,
                     activation='relu', name='conv3_1'))

    # conv3_2
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='SAME', use_bias=False,
                     activation='relu', name='conv3_2'))

    # conv3_3
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='SAME', use_bias=False,
                     activation='relu', name='conv3_3'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # conv6
    model.add(Conv2D(2048, kernel_size=8, strides=1, use_bias=False,
                     activation='relu', name='conv6'))

    # conv7
    model.add(Conv2D(256, kernel_size=1, strides=1,
                     use_bias=False,
                     activation='relu', name='conv7'))

    # conv8
    model.add(Conv2D(2, kernel_size=1, strides=1,
                     use_bias=False,
                     name='conv8'))

    model.add(Activation('softmax'))

    model.add(GlobalAveragePooling2D())

    sgd = optimizers.SGD(lr=0.0001)

    model.compile(loss=softmax_cross_entropy, optimizer=sgd, metrics=['accuracy'])

    return model


# %%

model = create_model()
model.load_weights(path.save_models_path + "SSNet/base_weights.hdf5")

# %%

kb.set_image_data_format('channels_last')
M = [108, 104, 94]


def load_dataset(patches_path, counts_path):
    counts = open(counts_path, 'r').read().split('\n')
    counts = [row.split(' ') for row in counts]
    patches = []
    labels = []
    for row in counts:
        image = Image.open(patches_path + row[0])
        image = np.asarray(image, dtype=np.float32)
        image[:, :, 0] = image[:, :, 0] - M[0]
        image[:, :, 1] = image[:, :, 1] - M[1]
        image[:, :, 2] = image[:, :, 2] - M[2]
        patches.append(image)
        label = np.asarray([0, 1])
        #         label = np.ones((25, 25))
        if row[1] == '0':
            label = np.asarray([1, 0])
        #             np.zeros((25, 25))
        label = np.float32(label)
        labels.append(label)
    return patches, labels


# %%

patches, labels = load_dataset(path.dataset_path + "foldscope_dataset/patches/",
                               path.dataset_path + "foldscope_dataset/patches_annotation.txt")
train_data = np.zeros((len(patches), 256, 256, 3))
train_labels = np.zeros((len(labels), 2))

for i in range(len(patches)):
    train_data[i, :, :, :] = patches[i]
    train_labels[i] = labels[i]
print(train_data.shape, train_labels.shape)

# %%

time_0 = time.time()
history = model.fit(x=train_data, y=train_labels, epochs=20, verbose=1, batch_size=32, shuffle=True,
                    validation_split=0.1)
time_1 = time.time()
print(time_1 - time_0)


# %%

def smooth(signal, n=2):
    smooth_signal = []
    for i in range(len(signal) - n + 1):
        smooth_signal.append(sum(signal[i: i + n]) / n)
    return smooth_signal


plt.plot(smooth(history.history['acc'][1:]))
plt.plot(smooth(history.history['val_acc'][1:]))
plt.legend(['acc', 'val_acc'], loc='upper left')
plt.show()

plt.plot(smooth(history.history['loss'][1:]))
plt.plot(smooth(history.history['val_loss'][1:]))
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.show()

# %%

model.save_weights(path.save_models_path + "SSNet/Busara_Vicinity_40_2.hdf5")
