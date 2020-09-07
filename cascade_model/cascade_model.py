import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import pathlib
import matplotlib.pyplot as plt
from collections import Counter
from custom_classes import path, cv_iml, predefine_models
from custom_classes.images_loader import *
from class_imbalance_loss import class_balanced_loss
from keras.utils import to_categorical
import os
import keras.backend as K

# Achieving peak performance requires an efficient input pipeline that delivers data
# for the next step before the current step has finished
AUTOTUNE = tf.data.experimental.AUTOTUNE

# %%
# Directory data

data_dir = path.dataset_path + "BBBC041/BBBC041_loss_test/"
# data_dir = pathlib.Path(data_dir)
save_weights_path = path.save_models_path + "BBBC041/basic_cnn_loss_test.h5"

train_files, train_labels, test_files, test_labels, val_files, val_labels = \
    load_train_test_val_images_from(data_dir, file_extension=".png")

# %%
# show the number of train, test and val files in dataset folder
print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest', Counter(test_labels))

# %%
import cv2
from concurrent import futures
import threading

# Load image data and resize on 125, 125 pixel.
IMG_DIMS = (125, 125)


def get_img_data_parallel(idx, img, total_imgs):
    if idx % 5000 == 0 or idx == (total_imgs - 1):
        print('{}: working on img num {}'.format(threading.current_thread().name, idx))
    img = cv2.imread(img)
    img = cv2.resize(img, dsize=IMG_DIMS, interpolation=cv2.INTER_CUBIC)
    img = np.array(img, dtype=np.float32)

    return img


ex = futures.ThreadPoolExecutor(max_workers=None)
train_data_inp = [(idx, img, len(train_files)) for idx, img in enumerate(train_files)]
val_data_inp = [(idx, img, len(val_files)) for idx, img in enumerate(val_files)]
test_data_inp = [(idx, img, len(test_files)) for idx, img in enumerate(test_files)]
print('Loading Train Images:')
train_data_map = ex.map(get_img_data_parallel,
                        [record[0] for record in train_data_inp],
                        [record[1] for record in train_data_inp],
                        [record[2] for record in train_data_inp])
train_data = np.array(list(train_data_map))

print('\nLoading Validation Images:')
val_data_map = ex.map(get_img_data_parallel,
                      [record[0] for record in val_data_inp],
                      [record[1] for record in val_data_inp],
                      [record[2] for record in val_data_inp])
val_data = np.array(list(val_data_map))

print('\nLoading Test Images:')
test_data_map = ex.map(get_img_data_parallel,
                       [record[0] for record in test_data_inp],
                       [record[1] for record in test_data_inp],
                       [record[2] for record in test_data_inp])
test_data = np.array(list(test_data_map))

train_data.shape, val_data.shape, test_data.shape

# %%

import matplotlib.pyplot as plt

plt.figure(1, figsize=(8, 8))
n = 0

for i in range(16):
    n += 1
    r = np.random.randint(0, train_data.shape[0], 1)
    plt.subplot(4, 4, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.imshow(train_data[r[0]] / 255.)
    plt.title('{}'.format(train_labels[r[0]]))
    plt.xticks([]), plt.yticks([])
plt.show()

# %%
number_of_classes = len(list(np.unique(train_labels)))

BATCH_SIZE = 16
NUM_CLASSES = 2
EPOCHS = 25
INPUT_SHAPE = (125, 125, 3)

train_imgs_scaled = train_data / 255.
val_imgs_scaled = val_data / 255.
test_imgs_scaled = test_data / 255.

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)

train_labels_enc = le.transform(train_labels)
val_labels_enc = le.transform(val_labels)
test_labels_enc = le.transform(test_labels)

train_labels_enc_multiclass = to_categorical(train_labels_enc, num_classes=number_of_classes)
val_labels_enc_multiclass = to_categorical(val_labels_enc, num_classes=number_of_classes)
test_labels_enc_multiclass = to_categorical(test_labels_enc, num_classes=number_of_classes)

print(train_labels[:6], train_labels_enc[:6])

# get the index of healthy label so that loss is not backpropagation when label is healthy.
healthy_label_index = list(le.transform(['healthy']))[0]
# %%
#  skip images that does not fit on batch size, this cause run time error in code.
train_reminder = len(train_labels) % BATCH_SIZE

train_imgs_scaled = train_imgs_scaled[train_reminder:]
train_labels_enc_multiclass = train_labels_enc_multiclass[train_reminder:]


# %%

def binary_loss_function(true, pred):
    loss = tf.losses.binary_crossentropy(true, pred)
    return loss


def multiclass_loss_function(true, pred):
    loss = tf.losses.categorical_crossentropy(true, pred)
    return loss


def custom_loss(y_true, y_pred):
    # calculate loss, using y_pred
    # proto_tensor = tf.make_tensor_proto(y_true)  # convert `tensor a` to a proto tensor
    # nd_array = tf.make_ndarray(proto_tensor)
    # indexs = np.argmax(nd_array, axis=1)
    # binary_index = (np.equal(indexs, 2)).astype(int)
    # binary_index = np.logical_not(binary_index).astype(int)
    yp = y_pred[1]
    yp_1 = y_pred[0]
    print(tf.shape(yp), tf.shape(yp_1))
    # y_pred[0]
    my_loss = []
    for i in range(BATCH_SIZE - 1):  # this will run in the range of the batch
        if tf.argmax(y_true[i]) == 1:  # 1 for healthy label
            # append the special loss to my_loss
            # for healthy label only
            # binary loss is returned
            binary_loss = binary_loss_function(y_true[i], y_pred[i])
            tower_loss = binary_loss
            # print(my_loss)
            # print(tower_loss.eval())
        else:
            # for the case of malaria
            # return towerloss = binary + multiclass loss
            binary_loss = binary_loss_function(y_true[i], y_pred[i])
            multiclass_loss = multiclass_loss_function(y_true[i], y_pred[i])

            tower_loss = binary_loss + multiclass_loss

        my_loss.append(tower_loss)
    # convert the my_loss to tensor
    my_loss = tf.math.reduce_sum(my_loss)
    # we need to normalized our loss function with batch size
    return my_loss


inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

#
flat = tf.keras.layers.Flatten()(pool3)

hidden1 = tf.keras.layers.Dense(512, activation='relu')(flat)
drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

out = tf.keras.layers.Dense(number_of_classes, activation='softmax')(drop2)
out1 = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)

model = tf.keras.Model(inputs=inp, outputs=[out, out1])
model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
model.summary()

# %%

history = model.fit(x=train_imgs_scaled, y=train_labels_enc_multiclass,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(val_imgs_scaled, val_labels_enc_multiclass)
                    )
