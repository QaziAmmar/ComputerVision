

# Dont remove this code.

from custom_classes import path, predefine_models
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


INPUT_SHAPE = (125, 125, 3)
# model = get_vgg_model(INPUT_SHAPE)
model = predefine_models.get_dennet121_transfer_learning(INPUT_SHAPE)

load_weight_path = path.save_models_path + "IML_binary_CNN_experimtents/vgg_19_binary_temp/pfPlusPv/pfpv_densnet_binaryClassification.h5"
save_weights_path = path.save_models_path + "IML_binary_CNN_experimtents/vgg_19_binary_temp/pfPlusPv/pv_densnet_binary_no_top.h5"
model.load_weights(load_weight_path)
layer_name = 'block5_pool'
#%%
model.summary()

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
layer_name = 'dense_1'

intermediate_from_a = model.get_layer(layer_name).output

intermediate_model = tf.keras.models.Model(inputs=model.input,outputs=intermediate_from_a)

#%%
intermediate_model.predict([test_img_scaled])


