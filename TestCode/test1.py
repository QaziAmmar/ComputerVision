# //  Created by Qazi Ammar Arshad on 01/08/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

#  Traing code

import tensorflow as tf
from collections import Counter
from custom_classes import path, predefine_models
from keras.utils import to_categorical
from custom_classes.dataset_loader import *

# hard_negative_mining_experiments parameter specify the type of experiment. In hard negative mining images are
# just separated into train, test and validation so their read style is just different.

# %%

data_set_base_path = path.dataset_path + "cell_images/"

INPUT_SHAPE = (125, 125, 3)

train_imgs_scaled, train_labels, test_imgs_scaled, test_labels, val_imgs_scaled, val_labels = \
    load_train_test_val_images_from(data_set_base_path, file_extension=".png", show_train_data=True)

# %%

print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest', Counter(test_labels))


# %%
# First complete the binary cycle

def _replaceitem(x):
    if x == 'healthy':
        return 'healthy'
    else:
        return "malaria"


# binary_train_labels = list(map(_replaceitem, train_labels))
# binary_test_labels = list(map(_replaceitem, test_labels))
# binary_val_labels = list(map(_replaceitem, val_labels))
binary_train_labels = train_labels
binary_test_labels = test_labels
binary_val_labels = val_labels

number_of_binary_classes = len(np.unique(binary_train_labels))

BATCH_SIZE = 64

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(binary_test_labels)

binary_train_labels_enc = le.transform(binary_train_labels)
binary_train_labels_enc = to_categorical(binary_train_labels_enc, num_classes=number_of_binary_classes)

binary_test_labels_enc = le.transform(binary_test_labels)
binary_test_labels_enc = to_categorical(binary_test_labels_enc, num_classes=number_of_binary_classes)

binary_val_labels_enc = le.transform(binary_val_labels)
binary_val_labels_enc = to_categorical(binary_val_labels_enc, num_classes=number_of_binary_classes)

print(binary_test_labels[:6], binary_test_labels_enc[:6])

# %%
# load model according to your choice.
# model = predefine_models.get_vgg_19_fine_tune(INPUT_SHAPE=INPUT_SHAPE, binary_classification=False,
#                                                classes=number_of_binary_classes)
model = predefine_models.get_resnet50v2(INPUT_SHAPE=INPUT_SHAPE,
                                        classes=number_of_binary_classes)
# model = predefine_models.get_dennet121_transfer_learning(INPUT_SHAPE=INPUT_SHAPE, classes=number_of_binary_classes)
# model.load_weights("/home/iml/Desktop/qazi/Model_Result_Dataset/SavedModel/shamalar_data/binary/binary_resnet50v2.h5")

# %%
import datetime

#
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=2, min_lr=0.000001)
# This callback will stop the training when there is no improvement in
# the validation loss for three consecutive epochs.
earlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

callbacks = [reduce_lr, earlyStop]

history = model.fit(x=train_imgs_scaled, y=binary_train_labels_enc,
                    batch_size=BATCH_SIZE,
                    epochs=25,
                    validation_data=(val_imgs_scaled, binary_val_labels_enc),
                    callbacks=callbacks,
                    verbose=1)

#%%
import matplotlib.pyplot as plt
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(history.history['accuracy'])+1
epoch_list = list(range(1,max_epoch))
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
# This cell shows the accuracy and loss graph and save the model for next time usage.

binary_load_weights_path = path.save_models_path + "BBBC041/balance_binary_resnet50v2.h5"

model.load_weights(binary_load_weights_path)

# %%
# Model Performance Evaluation
cnn_preds_binary = model.predict(test_imgs_scaled, batch_size=512)
# # Making prediction lables for multiclass
cnn_preds_binary = cnn_preds_binary.argmax(1)
prediction_labels_binary = le.inverse_transform(cnn_preds_binary)
cv_iml.get_f1_score(binary_test_labels, prediction_labels_binary, binary_classifcation=False, pos_label='malaria',
                    confusion_matrix_title="Binary classification Result")

# %%
# only chose those test images which are classified as malaria by binary cnn
cnn_indicated_test_imgs_scaled = test_imgs_scaled[prediction_labels_binary == "malaria"]
cnn_indicated_test_labels = test_labels[prediction_labels_binary == "malaria"]
print(Counter(cnn_indicated_test_labels))
# %%
# section for multiclass classification.
number_of_classes = len(list(np.unique(train_labels)))

le = LabelEncoder()
le.fit(train_labels)

cnn_indicated_test_labels_enc = le.transform(cnn_indicated_test_labels)
cnn_indicated_test_labels_enc = to_categorical(cnn_indicated_test_labels_enc, num_classes=number_of_classes)

print(cnn_indicated_test_labels[:6], cnn_indicated_test_labels_enc[:6])

# %%
# load model according to your choice.
model = predefine_models.get_basic_CNN_for_malaria(INPUT_SHAPE, binary_classification=False,
                                                   classes=number_of_classes)
# model = predefine_models.get_vgg16(INPUT_SHAPE, classes=number_of_classes)
# model = predefine_models.get_vgg_19_fine_tune(INPUT_SHAPE=INPUT_SHAPE, binary_classification=False,
#                                                classes=number_of_classes)
# model = predefine_models.get_resnet50v2(INPUT_SHAPE=INPUT_SHAPE, classes=number_of_classes)
# model = predefine_models.get_densenet121(INPUT_SHAPE, classes=number_of_classes)
# model = predefine_models.get_densenet169(INPUT_SHAPE, classes=number_of_classes)
# model = predefine_models.get_densenet201(INPUT_SHAPE, classes=number_of_classes)

# %%
# This cell shows the accuracy and loss graph and save the model for next time usage.

load_weights_path = path.save_models_path + "shamalar_data/balance_multiclass/balance_multiclass_resnet50v2.h5"

model.load_weights(load_weights_path)
score = model.evaluate(cnn_indicated_test_imgs_scaled, cnn_indicated_test_labels_enc)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# get all the images which are healthy but our model predicts them as healthy

# %%

# Model Performance Evaluation
cnn_preds = model.predict(cnn_indicated_test_imgs_scaled, batch_size=512)
# # Making prediction lables for multiclass
cnn_preds = cnn_preds.argmax(1)
prediction_labels = le.inverse_transform(cnn_preds)
# cv_iml.get_f1_score(cnn_indicated_test_labels, prediction_labels, binary_classifcation=False, pos_label='malaria',
#                     confusion_matrix_title="Two Stage classification")

# combined F1 score for stage 1 and stage 2
cnn_indicated_healthy_labels = test_labels[prediction_labels_binary == "healthy"]
prediction_labels_binary = prediction_labels_binary[prediction_labels_binary == "healthy"]
print(Counter(cnn_indicated_test_labels))

cv_iml.get_f1_score(np.concatenate((cnn_indicated_test_labels, cnn_indicated_healthy_labels)),
                    np.concatenate((prediction_labels, prediction_labels_binary)), binary_classifcation=False,
                    pos_label='malaria', confusion_matrix_title="Two Stage classification")
