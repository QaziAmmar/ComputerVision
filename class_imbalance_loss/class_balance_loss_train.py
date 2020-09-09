# //  Created by Qazi Ammar Arshad on 01/08/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.


import tensorflow as tf
import cv2
from concurrent import futures
import threading
import numpy as np
from collections import Counter
from custom_classes import cv_iml, path
from custom_classes.dataset_loader import *
from class_imbalance_loss import class_balanced_loss
from keras.utils import to_categorical
from custom_classes.dataset_loader import *


def get_cnn_pretrained_weights_model(INPUT_SHAPE, classes=1):
    inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    model = tf.keras.Model(inputs=inp, outputs=pool3)

    model.load_weights(path.save_models_path + "IML_binary_CNN_experimtents/cell_images_basic_no_top.h5")
    pool3 = model.output

    flat = tf.keras.layers.Flatten()(pool3)

    hidden1 = tf.keras.layers.Dense(512, activation='relu')(flat)
    drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
    hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

    out = tf.keras.layers.Dense(classes, activation='softmax')(drop2)
    model = tf.keras.Model(inputs=model.input, outputs=out)
    model.compile(optimizer="adam", loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])

    model.summary()

    return model


def custom_loss(y_true, y_pred):
    # calculate loss, using y_pred
    loss = class_balanced_loss.get_CB_loss(number_of_classes, samples_per_cls, y_true, y_pred)
    return loss


def get_model(INPUT_SHAPE, classes=1):
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

    out = tf.keras.layers.Dense(classes, activation='softmax')(drop2)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])

    model.summary()
    # model.load_weights(path.save_models_path + "IML_binary_CNN_experimtents/cell_images_basic_cnn.h5")
    return model


def get_vgg_model(INPUT_SHAPE, classes=2):
    save_weight_path = path.save_models_path + "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False,
                                            weights=save_weight_path,
                                            input_shape=INPUT_SHAPE)
    # Freeze the layers
    vgg.trainable = True

    set_trainable = False
    for layer in vgg.layers:
        if layer.name in ['block5_conv1', 'block4_conv1']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    base_vgg = vgg
    base_out = base_vgg.output
    pool_out = tf.keras.layers.Flatten()(base_out)
    hidden1 = tf.keras.layers.Dense(64, activation='relu')(pool_out)
    drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
    hidden2 = tf.keras.layers.Dense(32, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)
    model = tf.keras.Model(inputs=base_vgg.input, outputs=out)

    # opt = SGD(lr=0.00001)
    # # model.compile(loss="categorical_crossentropy", optimizer=opt)
    model.compile(optimizer="adam",
                  loss=tf.losses.binary_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    return model


# hard_negative_mining_experiments parameter specify the type of experiment. In hard negative mining images are
# just separated into train, test and validation so their read style is just different.
# load_weights_path =  path.save_models_path + "IML_binary_CNN_experimtents/basicCNN_binary/pv_binary_basic_cnn.h5"
save_weights_path = path.save_models_path + "IML_binary_CNN_experimtents/pv_basic_binary_cnn.h5"
data_set_base_path = path.dataset_path + "IML_training_data/binary_classifcation_train_test_seperate/p.v"

train_files, train_labels, test_files, test_labels, val_files, val_labels = \
    load_train_test_val_images_from(data_set_base_path, show_train_data=Fasle)

# %%

print(train_files.shape, val_files.shape, test_files.shape)
print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest', Counter(test_labels))

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
# section for class balance loss
number_of_classes = len(list(np.unique(train_labels)))

BATCH_SIZE = 64
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

train_labels_enc = to_categorical(train_labels_enc, num_classes=number_of_classes)
val_labels_enc = to_categorical(val_labels_enc, num_classes=number_of_classes)
test_labels_enc = to_categorical(test_labels_enc, num_classes=number_of_classes)

print(train_labels[:6], train_labels_enc[:6])

# %%
# section for class balance loss count number of sample for each class in one hot label order
samples_per_cls = list(np.zeros(number_of_classes))

for train_item in Counter(train_labels).items():
    class_name = train_item[0]
    class_count = train_item[1]
    label_enc = le.transform([class_name])
    labe_index = int(label_enc)
    samples_per_cls[labe_index] = class_count

# %%
# load model according to your choice.
# model = get_vgg_model(INPUT_SHAPE, classes=1)
model = get_cnn_pretrained_weights_model(INPUT_SHAPE, classes=number_of_classes)
# model = get_model(INPUT_SHAPE, classes=number_of_classes)

# %%

import datetime

logdir = os.path.join(path.save_models_path,
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=2, min_lr=0.000001)
callbacks = [reduce_lr, tensorboard_callback]

history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(val_imgs_scaled, val_labels_enc),
                    callbacks=callbacks,
                    verbose=1)

# %%
# This cell shows the accuracy and loss graph and save the model for next time usage.
model.save(save_weights_path)
# model.load_weights(load_weights_path)
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
# # Making prediction lables for multiclass
basic_cnn_preds = basic_cnn_preds.argmax(1)
prediction_labels = le.inverse_transform(basic_cnn_preds)
cv_iml.get_f1_score(test_labels, prediction_labels, binary_classifcation=False, pos_label='malaria',
                    plot_confusion_matrix=True)

# Making predication labels for binary classification
# basic_cnn_preds_labels = le.inverse_transform([1 if pred > 0.5 else 0
#                                                for pred in basic_cnn_preds.ravel()])
# cv_iml.get_f1_score(test_labels, basic_cnn_preds_labels, binary_classifcation=True, pos_label="malaria")
