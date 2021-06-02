# //  Created by Qazi Ammar Arshad on 01/08/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
 This code perform 2 stage classification of malaria cells. This file is only for testing purpose, we train
 both model in "Malaria_binaryClass and Malaria_multiclass" folder and then load these models
 into this file to predict the cells.
"""
# this code is a testing code that separate the cells after prediction.


import tensorflow as tf
from collections import Counter
from custom_classes import path, predefine_models
from keras.utils import to_categorical
from custom_classes.dataset_loader import *


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
    hidden1 = tf.keras.layers.Dense(512, activation='relu')(pool_out)
    drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
    hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

    out = tf.keras.layers.Dense(classes, activation='softmax')(drop2)
    model = tf.keras.Model(inputs=base_vgg.input, outputs=out)

    # opt = SGD(lr=0.00001)
    # # model.compile(loss="categorical_crossentropy", optimizer=opt)
    model.compile(optimizer="adam",
                  loss=tf.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    return model


# hard_negative_mining_experiments parameter specify the type of experiment. In hard negative mining images are
# just separated into train, test and validation so their read style is just different.

# %%
data_set_base_path = path.dataset_path + "shalamar_training_data_balanced/train_test_seprate/"

INPUT_SHAPE = (125, 125, 3)

train_imgs_scaled, train_labels, test_imgs_scaled, test_labels, val_imgs_scaled, val_labels = \
    load_train_test_val_images_from(data_set_base_path, file_extension=".JPG", show_train_data=True)

# %%

print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest', Counter(test_labels))


# %%
# First complete the binary cycle

def _replaceitem(x):
    if x == 'healthy':
        return 'healthy'
    else:
        return "malaria"


binary_test_labels = list(map(_replaceitem, test_labels))

number_of_binary_classes = 2

BATCH_SIZE = 64

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(binary_test_labels)

binary_test_labels_enc = le.transform(binary_test_labels)
binary_test_labels_enc = to_categorical(binary_test_labels_enc, num_classes=number_of_binary_classes)

print(binary_test_labels[:6], binary_test_labels_enc[:6])

# %%
# load model according to your choice.
# model = get_vgg_model(INPUT_SHAPE, classes=number_of_binary_classes)
model = predefine_models.get_resnet50v2(INPUT_SHAPE=INPUT_SHAPE,
                                        classes=number_of_binary_classes)
# model = predefine_models.get_dennet121_transfer_learning(INPUT_SHAPE=INPUT_SHAPE, classes=number_of_binary_classes)

# %%
# This cell shows the accuracy and loss graph and save the model for next time usage.
binary_load_weights_path = path.save_models_path + "shamalar_data/binary/binary_resnet50v2.h5"

model.load_weights(binary_load_weights_path)
# score = model.evaluate(test_imgs_scaled, binary_test_labels_enc)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# model.save('basic_cnn.h5')


# %%
# Model Performance Evaluation
cnn_preds_binary = model.predict(test_imgs_scaled, batch_size=512)
# # Making prediction lables for multiclass
cnn_preds_binary = cnn_preds_binary.argmax(1)
prediction_labels_binary = le.inverse_transform(cnn_preds_binary)
cv_iml.get_f1_score(binary_test_labels, prediction_labels_binary, binary_classifcation=True, pos_label='malaria',
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

