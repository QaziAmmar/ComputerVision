# //  Created by Qazi Ammar Arshad on 01/10/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

import os
from collections import Counter

import matplotlib.pyplot as plt
# import the necessary packages
import tensorflow as tf
from keras.utils import to_categorical

from class_imbalance_loss import class_balanced_loss
from custom_classes import path, predefine_models
from custom_classes.dataset_loader import *

# Achieving peak performance requires an efficient input pipeline that delivers data
# for the next step before the current step has finished
AUTOTUNE = tf.data.experimental.AUTOTUNE

fpr = dict()
tpr = dict()
roc_auc = dict()
model_name = []

# %%
# Directory data
# save_weights_path = path.save_models_path + "shamalar_data/multiclass/multiclass_dense201.h5"
data_dir = path.dataset_path + "shalamar_training_data_balanced/train_test_seprate/"

# load_weights_path = path.save_models_path + "IML_binary_CNN_experimtents/cell_images_basic_cnn.h5"

train_imgs_scaled, train_labels, test_imgs_scaled, test_labels, val_imgs_scaled, val_labels = \
    load_train_test_val_images_from(data_dir, file_extension=".JPG", show_train_data=True)

# %%
# show the number of train, test and val files in dataset folder
print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest', Counter(test_labels))

# %%
number_of_classes = len(list(np.unique(train_labels)))

BATCH_SIZE = 64
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
print(samples_per_cls)


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
# model = predefine_models.get_vgg16(INPUT_SHAPE, classes=number_of_classes)
# model = predefine_models.get_vgg_19_fine_tune(INPUT_SHAPE=INPUT_SHAPE, binary_classification=False,
#                                                       classes=number_of_classes)
# model = getvgg19(INPUT_SHAPE, classes=number_of_classes)
# model = predefine_models.get_resnet50v2(INPUT_SHAPE=INPUT_SHAPE, classes=number_of_classes)
# model = predefine_models.get_densenet121(INPUT_SHAPE, classes=number_of_classes)
# model = predefine_models.get_densenet169(INPUT_SHAPE, classes=number_of_classes)
# model = predefine_models.get_densenet201(INPUT_SHAPE, classes=number_of_classes)

print("Total Layers:", len(model.layers))
print("Total trainable layers:", sum([1 for l in model.layers if l.trainable]))
# %%
# Model training

import datetime

#
logdir = os.path.join(path.save_models_path + "resnet50_BBBC041_checkpoints/",
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=2, min_lr=0.000001)
# This callback will stop the training when there is no improvement in
# the validation loss for three consecutive epochs.
earlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

callbacks = [reduce_lr, tensorboard_callback, earlyStop]

history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(val_imgs_scaled, val_labels_enc),
                    callbacks=callbacks,
                    verbose=1)

# %%
save_weights_path = path.save_models_path + "shamalar_data/balance_multiclass_basicCNN.h5"

model.load_weights(save_weights_path)
# score = model.evaluate(test_imgs_scaled, test_labels_enc)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# model.save('basic_cnn.h5')

# %%
#
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
# t = f.suptitle('Basic CNN Performance', fontsize=12)
# f.subplots_adjust(top=0.85, wspace=0.3)
#
# max_epoch = len(history.history['accuracy']) + 1
# epoch_list = list(range(1, max_epoch))
# ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
# ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
# ax1.set_xticks(np.arange(1, max_epoch, 5))
# ax1.set_ylabel('Accuracy Value')
# ax1.set_xlabel('Epoch')
# ax1.set_title('Accuracy')
# l1 = ax1.legend(loc="best")
#
# ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
# ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
# ax2.set_xticks(np.arange(1, max_epoch, 5))
# ax2.set_ylabel('Loss Value')
# ax2.set_xlabel('Epoch')
# ax2.set_title('Loss')
# l2 = ax2.legend(loc="best")
# plt.show()

# %%

# This portion need to be updated accoruding to multiclass
# Model Performance Evaluation
basic_cnn_preds = model.predict(test_imgs_scaled, batch_size=512)
# Making prediction lables for multiclass
basic_cnn_preds = basic_cnn_preds.argmax(1)
prediction_labels = le.inverse_transform(basic_cnn_preds)
cv_iml.get_f1_score(test_labels, prediction_labels, pos_label='malaria', binary_classifcation=False,
                    confusion_matrix_title="Single Stage confusion matrix")

# %%

# plot ROC Curve for the code.

#
# macro_roc_auc_ovr = roc_auc_score(test_labels_enc, basic_cnn_preds, multi_class="ovr",
#                                   average="macro")

# print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),".format(macro_roc_auc_ovr))

# %%

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

# basic_cnn_preds = model.predict(test_imgs_scaled, batch_size=512)

lw = 2

# label = "basic CNN"
# class_name.append(label)
# fpr[label], tpr[label], _ = roc_curve(test_labels_enc, basic_cnn_preds, pos_label='malaria')
# roc_auc[label] = auc(fpr[label], tpr[label])
label = "densenet201"
model_name.append(label)
fpr[label], tpr[label], _ = roc_curve(test_labels_enc.ravel(), basic_cnn_preds.ravel())
roc_auc[label] = auc(fpr[label], tpr[label])

# for i in range(len(np.unique(test_labels))):
#     label = le.inverse_transform([i])[0]
#     model_name.append(label)
#     fpr[label], tpr[label], _ = roc_curve(test_labels_enc[:, i], basic_cnn_preds[:, i])
#     roc_auc[label] = auc(fpr[label], tpr[label])

# %%
colors = ['aqua', 'darkorange', 'cornflowerblue', 'navy', 'deeppink', 'aqua', 'darkorange', 'cornflowerblue']
color_counter = 0
for key in model_name:
    plt.plot(list(fpr[key]), list(tpr[key]), color=colors[color_counter], lw=lw,
             label='ROC curve of Class {0} (area = {1:0.4f})'
                   ''.format(key, roc_auc[key]))
    color_counter += 1

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of Single Stage Classification')
plt.legend(loc="lower right")
plt.show()
