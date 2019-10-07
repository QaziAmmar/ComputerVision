import coremltools
import pickle
import tensorflow as tf
import os
import glob
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
from concurrent import futures
import threading
import sklearn
# encode text category labels
from sklearn.preprocessing import LabelEncoder

data_set_base_path = "/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/Dataset/cell_images/"
save_weights_path = "/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/SavedModel/MalariaDetaction_DrMoshin/basic_cnn.h5"

base_dir = os.path.join(data_set_base_path)
infected_dir = os.path.join(base_dir, "Parasitized")
healthy_dir = os.path.join(base_dir, "Uninfected")

infected_files = glob.glob(infected_dir + '/*.png')
healthy_files = glob.glob(healthy_dir + '/*.png')

# %%
# cell 3
# Let’s build a dataframe from this which will be of use to us shortly as
# we start building our dataset

import numpy as np
import pandas as pd

np.random.seed(42)

files_df = pd.DataFrame({
    'filename': infected_files + healthy_files,
    'label': ['malaria'] * len(infected_files) + ['healthy'] * len(healthy_files)
}).sample(frac=1, random_state=42).reset_index(drop=True)

files_df.head()

# %%
# Generating tanning and testing data.

train_files, test_files, train_labels, test_labels = train_test_split(files_df['filename'].values,
                                                                      files_df['label'].values,
                                                                      test_size=0.3,
                                                                      random_state=42)
# Generating validation data form tanning data.
train_files, val_files, train_labels, val_labels = train_test_split(train_files,
                                                                    train_labels,
                                                                    test_size=0.1,
                                                                    random_state=42)

print(train_files.shape, val_files.shape, test_files.shape)
print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest', Counter(test_labels))

# %%
# Load image data and resize on 125, 125 pixel.
IMG_DIMS = (125, 125)
NUM_CLASSES = 2
EPOCHS = 25
INPUT_SHAPE = (125, 125, 3)

le = LabelEncoder()
le.fit(train_labels)

train_labels_enc = le.transform(train_labels)
val_labels_enc = le.transform(val_labels)
test_labels_enc = le.transform(test_labels)

print(train_labels[:6], train_labels_enc[:6])

# Model 1: CNN from Scratch
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
model.summary()

model.load_weights(save_weights_path)
# score = model.evaluate(test_imgs_scaled, test_labels_enc)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


class_label = pd.unique(val_labels_enc.tolist())
coreml_model = coremltools.converters.keras.convert(model=model,
                                                    input_names="image",
                                                    image_input_names="image",
                                                    image_scale=1 / 255.0,
                                                    class_labels=class_label,
                                                    is_bgr=True)
