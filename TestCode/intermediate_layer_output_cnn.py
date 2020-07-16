

# Dont remove this code.

from custom_classes import path
from keras.models import Model
import tensorflow as tf
import cv2
import numpy as np


def get_model(INPUT_SHAPE):
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

    out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


INPUT_SHAPE = (125, 125, 3)
model = get_model(INPUT_SHAPE)

save_weights_path = path.save_models_path + "MalariaDetaction_DrMoshin/basic_cnn_IML_fineTune.h5"
model.load_weights(save_weights_path)
layer_name = 'flatten_2'

# %%

folder_base_path = path.dataset_path + "LabelBox_classification/"
chughati_cell_img = cv2.imread(folder_base_path + "pf.jpg")
img = cv2.resize(chughati_cell_img, (125, 125))
img_list = [img]
test_data = np.array(list(img_list))

print("img", len(test_data))
test_img_scaled = test_data / 255.
basic_cnn_preds = model.predict(test_img_scaled)



# %%
# getting intermidiate layres data/

# this how we get intermedicate layers output form net work
layer_name = 'dense_4'

intermediate_from_a = model.get_layer(layer_name).output

intermediate_model = tf.keras.models.Model(inputs=model.input,outputs=intermediate_from_a)
intermediate_model.predict([test_img_scaled])


