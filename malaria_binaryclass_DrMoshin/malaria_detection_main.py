# %%
# //  Created by Qazi Ammar Arshad on 15/07/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

import os
import tensorflow as tf
import numpy as np
from collections import Counter
from custom_classes import path, predefine_models, cv_iml
from custom_classes.dataset_loader import *

# hard_negative_mining_experiments parameter specify the type of experiment. In hard negative mining images are
# just separated into train, test and validation so their read style is just different.

save_weights_path = path.save_models_path + "shamalar_data/binaryclass_vgg.h5"

data_set_base_path = path.dataset_path + "shalamar_training_data/train_test_seprate_binary/"

train_files, train_labels, test_files, test_labels, val_files, val_labels = \
    load_train_test_val_images_from(data_set_base_path, show_train_data=True)

# %%

print(train_files.shape, val_files.shape, test_files.shape)
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

print("train, test and validation shape", train_data.shape, val_data.shape, test_data.shape)

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

print(train_labels[:6], train_labels_enc[:6])

# %%
# load model according to your choice.
# model = testing_models.get_1_CNN(INPUT_SHAPE)
# model = predefine_models.get_basic_CNN_for_malaria(INPUT_SHAPE)
model = predefine_models.get_vgg_19_fine_tune(INPUT_SHAPE)
# model = predefine_models.get_vgg_19_transfer_learning(INPUT_SHAPE)
# model = predefine_models.get_resnet50_transferLearning(INPUT_SHAPE)
# model = predefine_models.get_dennet121_transfer_learning(INPUT_SHAPE)
# %%
# Model training
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
cv_iml.get_f1_score(test_labels, basic_cnn_preds_labels, binary_classifcation=True, pos_label="malaria",
                    plot_confusion_matrix=True)

