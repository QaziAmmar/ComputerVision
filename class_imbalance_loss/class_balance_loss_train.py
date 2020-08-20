# //  Created by Qazi Ammar Arshad on 01/08/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.


import os
import tensorflow as tf
import keras.backend as K
import numpy as np
from collections import Counter
from custom_classes import path, predefine_models, cv_iml, testing_models
from malaria_binaryclass_DrMoshin.image_loader import load_train_test_val_images_from
from class_imbalance_loss import class_balanced_loss
from keras.utils import to_categorical

samples_per_cls = []
number_of_classes = 2


def get_model(INPUT_SHAPE, binary_classification=True, classes=1):
    def custom_loss(y_true, y_pred):
        # calculate loss, using y_pred
        # logits = y_pred
        # labels = y_true
        # beta = 0.9999
        # gamma = 2.0
        # loss_type = "softmax"
        # loss = class_balanced_loss.tensor_loss_func(labels, logits, samples_per_cls, number_of_classes, loss_type, beta,
        #                                             gamma)

        loss = K.square(y_pred - y_true)  # (batch_size, 2)

        # multiplying the values with weights along batch dimension
        loss = loss * [0.3, 0.7]  # (batch_size, 2)

        # summing both loss values along batch dimension
        loss = K.sum(loss, axis=1)

        return loss

    inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    flat = tf.keras.layers.Flatten()(pool3)

    hidden1 = tf.keras.layers.Dense(512, activation='relu')(flat)
    drop1 = tf.keras.layers.Dropout(rate=0.4)(hidden1)
    hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.4)(hidden2)

    out = tf.keras.layers.Dense(classes, activation='softmax')(drop2)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])

    model.summary()

    return model


# hard_negative_mining_experiments parameter specify the type of experiment. In hard negative mining images are
# just separated into train, test and validation so their read style is just different.

save_weights_path = path.save_models_path + "binary_classification_test_CNN/pf_binary_basic.h5"
load_weights_path = path.save_models_path + "binary_classification_test_CNN/cell_images_basic_cnn.h5"

data_set_base_path = path.dataset_path + "IML_training_data/binary_classifcation_train_test_seperate/p.v"

train_files, train_labels, test_files, test_labels, val_files, val_labels = \
    load_train_test_val_images_from(data_set_base_path)
#
# train_files = train_files[: int(len(train_files) * 0.2)]
# train_labels = train_labels[: int(len(train_labels) * 0.2)]
# test_files = test_files[: int(len(test_files) * 0.2)]
# test_labels = test_labels[: int(len(test_labels) * 0.2)]
# val_files = val_files[: int(len(val_files) * 0.2)]
# val_labels = val_labels[: int(len(val_labels) * 0.2)]
# # %%
# for num_of_imgs in Counter(train_labels).values():
#     samples_per_cls.append(num_of_imgs)


# %%

print(train_files.shape, val_files.shape, test_files.shape)
print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest', Counter(test_labels))

# %%
import cv2
from concurrent import futures
import threading


def get_img_shape_parallel(idx, img, total_imgs):
    if idx % 5000 == 0 or idx == (total_imgs - 1):
        print('{}:working on img num {}'.format(threading.current_thread().name, idx))
    return cv2.imread(img).shape


ex = futures.ThreadPoolExecutor(max_workers=None)
data_inp = [(idx, img, len(train_files)) for idx, img in enumerate(train_files)]
print('Starting Img shape computation:')
train_img_dims_map = ex.map(get_img_shape_parallel,
                            [record[0] for record in data_inp],
                            [record[1] for record in data_inp],
                            [record[2] for record in data_inp])
# this part of code is getting dimensions of all image and save in train_img_dims.
train_img_dims = list(train_img_dims_map)
print('Min Dimensions:', np.min(train_img_dims, axis=0))
print('Average Dimensions: ', np.mean(train_img_dims, axis=0))
print('Median Dimensions:', np.median(train_img_dims, axis=0))
print('Max Dimensions:', np.max(train_img_dims, axis=0))

# %%
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

print("train, test and validation shape", train_data.shape, val_data.shape, test_data.shape)


# %%
BATCH_SIZE = 64
NUM_CLASSES = 2
EPOCHS = 50
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

train_labels_enc = to_categorical(train_labels_enc, num_classes=number_of_classes)
val_labels_enc = to_categorical(val_labels_enc, num_classes=number_of_classes)
test_labels_enc = to_categorical(test_labels_enc, num_classes=number_of_classes)

print(train_labels[:6], train_labels_enc[:6])

# %%
# load model according to your choice.
model = get_model(INPUT_SHAPE, binary_classification=False, classes=number_of_classes)

# %%

import datetime

logdir = os.path.join(path.save_models_path,
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=2, min_lr=0.000001)
callbacks = [reduce_lr, tensorboard_callback]

# if os.path.isfile(load_weights_path):
#     model.load_weights(load_weights_path)


history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(val_imgs_scaled, val_labels_enc),
                    callbacks=callbacks,
                    verbose=1)

# %%
# This cell shows the accuracy and loss graph and save the model for next time usage.
model.save(save_weights_path)
# model.load_weights(save_weights_path)
score = model.evaluate(test_imgs_scaled, test_labels_enc)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# model.save('basic_cnn.h5')


# %%
import matplotlib.pyplot as plt
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(history.history['accuracy']) + 1
epoch_list = list(range(1, max_epoch))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
plt.show()

# %%
# Model Performance Evaluation
basic_cnn_preds = model.predict(test_imgs_scaled, batch_size=512)
basic_cnn_preds_labels = le.inverse_transform([1 if pred > 0.5 else 0
                                               for pred in basic_cnn_preds.ravel()])
cv_iml.get_f1_score(test_labels, basic_cnn_preds_labels, pos_label="malaria")
