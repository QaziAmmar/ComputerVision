# //  Created by Qazi Ammar Arshad on 01/08/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

# this code is a testing code that separate the cells after prediction.


import tensorflow as tf
from collections import Counter
from custom_classes import path
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
# load_weights_path =  path.save_models_path + "IML_binary_CNN_experimtents/basicCNN_binary/pv_binary_basic_cnn.h5"
#%%

load_weights_path = path.save_models_path + "IML_binary_CNN_experimtents/vgg_2hidden_units/pf_vgg_binary_2hiddenUnit.h5"
data_set_base_path = path.dataset_path + "IML_training_data/binary_classifcation_train_test_seperate/p.f/"

INPUT_SHAPE = (125, 125, 3)

train_files, train_labels, test_files, test_labels, val_files, val_labels = \
    load_train_test_val_images_from(data_set_base_path, file_extension=".JPG", show_train_data=True)

# %%

print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest', Counter(test_labels))

# %%
train_data, val_data, test_data = load_img_data_parallel(train_files=train_files, test_files=test_files,
                                                         val_files=val_files)

# Normalized input data.
train_imgs_scaled = train_data / 255.
val_imgs_scaled = val_data / 255.
test_imgs_scaled = test_data / 255.


#%%

# section for class balance loss
number_of_classes = len(list(np.unique(train_labels)))

BATCH_SIZE = 64
NUM_CLASSES = number_of_classes
EPOCHS = 25


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
model = get_vgg_model(INPUT_SHAPE, classes=number_of_classes)

# %%
# This cell shows the accuracy and loss graph and save the model for next time usage.

model.load_weights(load_weights_path)
score = model.evaluate(test_imgs_scaled, test_labels_enc)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# model.save('basic_cnn.h5')


# %%
# Model Performance Evaluation
basic_cnn_preds = model.predict(test_imgs_scaled, batch_size=512)
# # Making prediction lables for multiclass
basic_cnn_preds = basic_cnn_preds.argmax(1)
prediction_labels = le.inverse_transform(basic_cnn_preds)
cv_iml.get_f1_score(test_labels, prediction_labels, binary_classifcation=True, pos_label='malaria',
                    plot_confusion_matrix=True)

#%%
# get all the images which are healthy but our model predicts them as healthy
counter = 0
mis_predicted_cell = []
for temp_train_lbl, temp_pred_lbl in zip(test_labels, prediction_labels):
    if temp_train_lbl == "healthy" and temp_pred_lbl == 'malaria':
        # print(test_files[counter])
        mis_predicted_cell.append(test_files[counter])
    counter += 1

print(len(mis_predicted_cell))

#%%
import cv2
save_new_cell_path = "/home/iml/Desktop/qazi/Model_Result_Dataset/Dataset/IML_training_data/cnn_generated_classifcation_data/p.f/test/malaria/"

for temp_cell in mis_predicted_cell:
    img = cv2.imread(temp_cell)
    save_name = temp_cell.split('/')[-1]
    cv2.imwrite(save_new_cell_path +  save_name, img)
    print(save_name)