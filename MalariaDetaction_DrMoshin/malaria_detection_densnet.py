import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import pickle
import csv

from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from PIL import Image
from custom_classes import path, cv_iml

import os

data_set_base_path = path.dataset_path + "IML_cell_images/"
# Hard Negative mining. (HNM)
save_weights_path = path.save_models_path + "MalariaDetaction_DrMoshin/basic_cnn_IML_fineTune.h5"

base_dir = os.path.join(data_set_base_path)
infected_dir = os.path.join(base_dir, "malaria")
healthy_dir = os.path.join(base_dir, "healthy")

print(os.listdir(data_set_base_path))

# %%
# Now we set the base path for the car dataset and show one image of the dataset.

path_base = data_set_base_path

class_names = ['malaria', 'healthy']

# %%

# Building the model
# Now we will define the model architecture. First of all, load the Keras libraries.

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import densenet
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers
from keras import backend as K

K.set_learning_phase(1)

# %%
img_width, img_height = 224, 224
nb_train_samples = 9752
nb_validation_samples = 2199
epochs = 10
batch_size = 32
n_classes = 2

# %%
train_data_dir = path_base + 'train/'
validation_data_dir = path_base + 'validation/'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # shear_range=0.2,
    zoom_range=0.2,
    # fill_mode = 'constant',
    # cval = 1,
    rotation_range=5,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# %%
def build_model():
    base_model = densenet.DenseNet121(input_shape=(img_width, img_height, 3),
                                      weights=path.save_models_path + 'densenet121_weights_tf.h5',
                                      include_top=False,
                                      pooling='avg')
    for layer in base_model.layers:
        print(layer)
        layer.trainable = True

    x = base_model.output
    x = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    predictions = Dense(n_classes, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

#%%
model = build_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
