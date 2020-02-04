#%%

import numpy as np
import os
import cv2
from keras.optimizers import SGD
import keras.backend as K
import scipy.io as sio
from keras.models import Model
from DenseNet.densenet121 import DenseNet
import glob
import pickle
from custom_classes import path

#%%

weights_path = '/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/SavedModel/densenet121_weights_tf.h5'
model = DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)

#%%

part_model = Model(inputs=model.input, outputs=model.get_layer('relu5_blk').output)
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
part_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#%%

root_path = path.dataset_path
features_path = root_path + 'features/'
images_path = root_path + 'RBCs_counting_LR/'
locations = open(root_path + 'patches_locations.txt', 'r').read().split('\n')
locations = [row.split(' ') for row in locations]
image_extension = '.jpg'

#%%

image_height = 519
image_width = 775

for row in locations:
    image = cv2.imread(images_path + row[0])
    test_image = np.zeros((1, image_height, image_width, 3))
    test_image[0, :, :, :] = image
    feature_map = part_model.predict(test_image)
    filename = "{}/{}.npy".format(features_path, row[0])
    np.save(open(filename, 'wb'), feature_map)
