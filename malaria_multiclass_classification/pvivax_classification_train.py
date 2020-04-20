# //  Created by Qazi Ammar Arshad on 01/10/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

# import the necessary packages
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import pathlib
from custom_classes import path, cv_iml, predefine_models
from keras.utils import to_categorical
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE



# %%
# Directory data
data_dir = path.result_folder_path + "pvivax_malaria_cells/"
data_dir = pathlib.Path(data_dir)
save_weights_path = path.save_models_path + "pvivax_malaria_multi_class/" + "densnet121_MC_TL.h5"
# image_count = len(list(data_dir.glob('*/*.png')))
#
# print(image_count)

# CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])
# print(CLASS_NAMES)

# %%
files_df = None
np.random.seed(42)
number_of_classes = 0
for folder_name in data_dir.glob('*'):
    # '.DS_Store' file is automatically created by mac which we need to exclude form your code.
    if '.DS_Store' == str(folder_name).split('/')[-1]:
        continue
    files_in_folder = glob.glob(str(folder_name) + '/*.png')
    df2 = pd.DataFrame({
        'filename': files_in_folder,
        'label': [folder_name.name] * len(files_in_folder)
    })
    number_of_classes += 1
    if files_df is None:
        files_df = df2
    else:
        files_df = files_df.append(df2, ignore_index=True)

files_df.sample(frac=1, random_state=42).reset_index(drop=True)

files_df.head()

# %%
from sklearn.model_selection import train_test_split
from collections import Counter

# Generating tanning and testing data.
train_files, test_files, train_labels, test_labels = train_test_split(files_df['filename'].values,
                                                                      files_df['label'].values,
                                                                      test_size=0.2,
                                                                      random_state=42)
# Generating validation data form tanning data.
train_files, val_files, train_labels, val_labels = train_test_split(train_files,
                                                                    train_labels,
                                                                    test_size=0.2,
                                                                    random_state=42)

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

train_labels_enc = to_categorical(train_labels_enc, num_classes=6)
val_labels_enc = to_categorical(val_labels_enc, num_classes=6)
test_labels_enc = to_categorical(test_labels_enc, num_classes=6)

print(train_labels[:6], train_labels_enc[:6])

# %%


# model = predefine_models.get_basic_CNN_for_malaria(INPUT_SHAPE, binary_classification=False,
# classes=number_of_classes)
model = predefine_models.get_dennet121_transfer_learning(INPUT_SHAPE, binary_classification=False,
                                                         classes=number_of_classes)
# %%

# Model training
import datetime

logdir = os.path.join(path.save_models_path,
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=2, min_lr=0.000001)
callbacks = [reduce_lr, tensorboard_callback]

# if os.path.isfile(save_weights_path):
#     model.load_weights(save_weights_path)

history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(val_imgs_scaled, val_labels_enc),
                    callbacks=callbacks,
                    verbose=1)

# %%

# model.save(save_weights_path)
model.load_weights(save_weights_path)
score = model.evaluate(test_imgs_scaled, test_labels_enc)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# model.save('basic_cnn.h5')

# %%

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
from sklearn.metrics import confusion_matrix
# This portion need to be updated accoruding to multiclass
# Model Performance Evaluation
basic_cnn_preds = model.predict(test_imgs_scaled, batch_size=512)
# Making prediction lables for multiclass
basic_cnn_preds = basic_cnn_preds.argmax(1)
prediction_labels = le.inverse_transform(basic_cnn_preds)
cv_iml.get_f1_score(test_labels, prediction_labels, plot_confusion_matrix=True)
