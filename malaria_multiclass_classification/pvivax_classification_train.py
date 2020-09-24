# //  Created by Qazi Ammar Arshad on 01/10/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

# import the necessary packages
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import pathlib
import matplotlib.pyplot as plt
from collections import Counter
from custom_classes import path, cv_iml, predefine_models
from custom_classes.dataset_loader import *
from class_imbalance_loss import class_balanced_loss
from keras.utils import to_categorical
import os
# Achieving peak performance requires an efficient input pipeline that delivers data
# for the next step before the current step has finished
AUTOTUNE = tf.data.experimental.AUTOTUNE


# %%
# Directory data

data_dir = path.dataset_path + "BBBC041/train_test_val/"

save_weights_path = path.save_models_path + "BBBC041/densenet_multiclass.h5"
# load_weights_path = path.save_models_path + "IML_binary_CNN_experimtents/cell_images_basic_cnn.h5"

train_imgs_scaled, train_labels, test_imgs_scaled, test_labels, val_imgs_scaled, val_labels = \
    load_train_test_val_images_from(data_dir, file_extension=".png", show_train_data=True)

# %%
# show the number of train, test and val files in dataset folder
print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest', Counter(test_labels))


# %%
number_of_classes = len(list(np.unique(train_labels)))

BATCH_SIZE = 64
NUM_CLASSES = 2
EPOCHS = 25
INPUT_SHAPE = (125, 125, 3)


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
# Custom loss function for class imbalance loss, we need to make our predefine_model function as so that they can
# take loss function as parameters.
def custom_loss(y_true, y_pred):
    # calculate loss, using y_pred
    loss = class_balanced_loss.get_CB_loss(number_of_classes, samples_per_cls, y_true, y_pred)
    return loss


# %%
model = predefine_models.get_basic_CNN_for_malaria(INPUT_SHAPE, binary_classification=False,
                                                   classes=number_of_classes)
# model = predefine_models.get_dennet121_transfer_learning(INPUT_SHAPE, classes=number_of_classes)
# model = predefine_models.get_resnet50(INPUT_SHAPE=INPUT_SHAPE, classes=number_of_classes)

model = predefine_models.get_vgg_19_fine_tune(INPUT_SHAPE=INPUT_SHAPE, binary_classification=False,
                                              classes=number_of_classes)
model.load_weights("/home/iml/Desktop/qazi/Model_Result_Dataset/SavedModel/IML_binary_CNN_experimtents/cell_images_basic_cnn.h5")
print("Total Layers:", len(model.layers))
print("Total trainable layers:", sum([1 for l in model.layers if l.trainable]))
# %%

# Model training
import datetime

logdir = os.path.join(path.save_models_path + "/resnet50_BBBC041_checkpoints/",
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
from sklearn.metrics import confusion_matrix

# This portion need to be updated accoruding to multiclass
# Model Performance Evaluation
basic_cnn_preds = model.predict(test_imgs_scaled, batch_size=512)
# Making prediction lables for multiclass
basic_cnn_preds = basic_cnn_preds.argmax(1)
prediction_labels = le.inverse_transform(basic_cnn_preds)
cv_iml.get_f1_score(test_labels, prediction_labels, binary_classifcation=False, plot_confusion_matrix=True)



