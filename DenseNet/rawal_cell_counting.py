# %%

import numpy as np
import os
import cv2
import pandas as pd
import scipy.io as sio

import os
from keras.optimizers import SGD, Adagrad
import keras.backend as K
import tensorflow as tf
import scipy.io as sio
from keras.models import Model
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Concatenate, LeakyReLU, Dropout
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.callbacks import Callback, ModelCheckpoint

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# %%

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# %%

# K.tensorflow_backend._get_available_gpus()

# %%

training_labels_path = 'E:/ITU/Dr_Mohsen/houseCounting/sh_new_mix_train.txt'

test_labels_path = 'E:/ITU/Dr_Mohsen/houseCounting/sh_new_mix_test.txt'

# fmaps_path = 'E:/ITU/Dr_Mohsen/houseCounting/all_data_new_mix_jittered_maps/pool5'
# final_results_path = 'E:/ITU/Dr_Mohsen/houseCounting/hamza_experiments/isprs/drc_original/drc.txt'

# creating folder to save weights
weights_savepath = 'E:/ITU/Dr_Mohsen/houseCounting/hamza_experiments/isprs/drc_original/weights/'
if not os.path.exists(weights_savepath):
    os.makedirs(weights_savepath)
weights_savepath = weights_savepath + 'drc_'

# %%

# #concatinating training files pool5 layer maps
# lines = [line.rstrip('\n') for line in open(training_labels_path)]
# fmaps_all = np.empty((1024,))
# for idx,line in enumerate(lines):
#     image_path = os.path.join(fmaps_path,line.split()[0] + '.mat')
#     fmap1 = sio.loadmat(image_path)
#     fmap1 = fmap1['fmap']
#     fmaps_all = np.vstack((fmaps_all, fmap1[0]))
#     print(idx)
# fmaps_all = fmaps_all[1:]
# sio.savemat('E:/ITU/Dr_Mohsen/houseCounting/hamza_experiments/isprs/fmaps_train_rawalOnly_nonAug.mat',{'fmaps_all':fmaps_all})

# training feature maps
fmaps_all = sio.loadmat('E:/ITU/Dr_Mohsen/houseCounting/hamza_experiments/isprs/fmaps_train.mat')
fmaps_all = fmaps_all['fmaps_all']

# loading test feature maps
fmaps_test = sio.loadmat('E:/ITU/Dr_Mohsen/houseCounting/hamza_experiments/isprs/fmaps_test.mat')
fmaps_test = fmaps_test['fmaps_all']

# %%

# fasi large data
# loading test feature maps
test_labels_path = 'F:/fasi_patches/fasi_largeData.txt'
fmaps_test = sio.loadmat('F:/fasi_patches/fasi_largeData.mat')
fmaps_test = fmaps_test['fmap']

# %%

# print(fmaps_all.shape)
print(fmaps_test.shape)

# %%

# training data labels
training_labels = np.array([np.int(line.rstrip('\n').split()[1]) for line in open(training_labels_path)])
test_labels = np.array([np.int(line.rstrip('\n').split()[1]) for line in open(test_labels_path)])


# %%

def generator(train_indices, batch_size, itr, lines, image_size=336):
    batch_images = []
    batch_labels = []
    startInd = itr * batch_size
    endInd = startInd + batch_size

    batch_images = fmaps_all[startInd:endInd]
    batch_labels = training_labels[startInd:endInd]

    return batch_images, batch_labels


# %%

def create_model():
    model = Sequential()
    model.add(
        Dense(512, input_dim=1024, init='glorot_normal', W_regularizer=l2(0.001), activation='relu', name='dense1'))
    model.add(Dropout(0.6, name='dropout1'))
    model.add(Dense(32, init='glorot_normal', W_regularizer=l2(0.001), activation='relu', name='dense2'))
    model.add(Dropout(0.6, name='dropout2'))
    model.add(Dense(1, init='glorot_normal', W_regularizer=l2(0.001), activation='linear', name='dense3'))

    adagrad = Adagrad(lr=0.01, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=adagrad, metrics=['accuracy'])

    return model


# %%

model = create_model()
model.summary()

# %%

import time

batch_size = 60
nb_epoch = 100

model = create_model()

lines = [line.rstrip('\n') for line in open(training_labels_path)]

indices = np.arange(len(lines))
train_indices = indices[:]  # split training
print(train_indices.shape)
iterations_t = np.int(len(train_indices) / batch_size)
t_loss_ep = []

for epoch in range(nb_epoch):
    ## training
    start = time.time()
    t_loss = []
    for i in range(iterations_t):
        X_train, y_train = generator(train_indices, batch_size, i, lines, image_size=336)
        history = model.fit(X_train, y_train, verbose=0)
        t_loss.append(history.history['loss'])

    stop = time.time()
    duration = stop - start
    t_loss_ep.append(np.mean(np.array(t_loss, dtype=np.float32)))
    print("Epoch: {}  Training Loss: {}  Time: {} sec".format(epoch, t_loss_ep[epoch], duration))
    model_name = weights_savepath + str(epoch) + "_Epochs.hdf5"
    model.save_weights(model_name)

model.save_weights(weights_savepath + '_final.hdf5')

# %%

import time

lines = [line.rstrip('\n') for line in open(test_labels_path)]

# ----------------------------
# Load Model weights
# ---------------------------
model.load_weights(weights_savepath + '_final.hdf5')

# test data filenames
filenames = np.array([line.rstrip('\n').split()[0] for line in open(test_labels_path)])
filenames_split = np.array([line.rstrip('\n').split()[0].split('_')[0] for line in open(test_labels_path)])

data = ""
data_low = ""
data_low_gt = ""
data_med = ""
data_med_gt = ""
data_high = ""
data_high_gt = ""

mean_abs_error = 0
mean_sqrd_error = 0

mean_abs_error_low = 0
mean_sqrd_error_low = 0

mean_abs_error_med = 0
mean_sqrd_error_med = 0

mean_abs_error_high = 0
mean_sqrd_error_high = 0

total_data = len(lines)
count = 0
count_low = 0
count_med = 0
count_high = 0

print(len(lines))

start = time.time()

# ----------------------------
# Loop through all test data
# ---------------------------
for i in range(len(lines)):

    line = lines[i].split(' ')
    gt_count = np.int(line[1])

    fmap = fmaps_test[i]
    test_image = np.zeros((1, 1024))
    test_image[0, :] = fmap

    # ------------------------
    # Calculate Predictions
    # ------------------------
    output = model.predict(test_image)
    if output < 0:
        output = 0
    print([filenames[i], gt_count, output])

    # ------------------------
    # Calculate Errors
    # ------------------------
    abs_error = np.abs(gt_count - output)
    mean_abs_error = mean_abs_error + abs_error

    sqrd_error = (gt_count - output) * (gt_count - output)
    mean_sqrd_error = mean_sqrd_error + sqrd_error

    if gt_count <= 30:
        count_low = count_low + 1
        mean_abs_error_low = mean_abs_error_low + abs_error
        mean_sqrd_error_low = mean_sqrd_error_low + sqrd_error
        data_low = data_low + line[0] + " " + str(float(output)) + "\n"
        data_low_gt = data_low_gt + line[0] + " " + str(float(gt_count)) + "\n"

    if 30 < gt_count <= 60:
        count_med = count_med + 1
        mean_abs_error_med = mean_abs_error_med + abs_error
        mean_sqrd_error_med = mean_sqrd_error_med + sqrd_error
        data_med = data_med + line[0] + " " + str(float(output)) + "\n"
        data_med_gt = data_med_gt + line[0] + " " + str(float(gt_count)) + "\n"

    if gt_count > 60:
        count_high = count_high + 1
        mean_abs_error_high = mean_abs_error_high + abs_error
        mean_sqrd_error_high = mean_sqrd_error_high + sqrd_error
        data_high = data_high + line[0] + " " + str(float(output)) + "\n"
        data_high_gt = data_high_gt + line[0] + " " + str(float(gt_count)) + "\n"

    data = data + line[0] + " " + str(float(output)) + "\n"

stop = time.time()
duration = stop - start

print('Total time for all images: ', duration)

f = open(final_results_path, 'w')
f.write(data)

f_mean_abs_error = mean_abs_error / total_data
f_mean_sqrd_error = mean_sqrd_error / total_data

f_mean_abs_error_low = mean_abs_error_low / count_low
f_mean_sqrd_error_low = mean_sqrd_error_low / count_low

f_mean_abs_error_med = mean_abs_error_med / count_med
f_mean_sqrd_error_med = mean_sqrd_error_med / count_med

f_mean_abs_error_high = mean_abs_error_high / count_high
f_mean_sqrd_error_high = mean_sqrd_error_high / count_high

print('##################################### ERROR VALUES ###########################################')
print("Total Absolute Error: {} ".format(mean_abs_error))
print("Mean Squared Error: {} ".format(f_mean_sqrd_error))
print("Mean Absolute Error: {} ".format(f_mean_abs_error))
print("Number of Instances: {} ".format(total_data))
print('##############################################################################################')

print('##################################### ERROR VALUES-LOW ###########################################')
print("Total Absolute Error: {} ".format(mean_abs_error_low))
print("Mean Squared Error: {} ".format(f_mean_sqrd_error_low))
print("Mean Absolute Error: {} ".format(f_mean_abs_error_low))
print("Number of Instances: {} ".format(count_low))
print('##############################################################################################')

print('##################################### ERROR VALUES-MEDIUM ###########################################')
print("Total Absolute Error: {} ".format(mean_abs_error_med))
print("Mean Squared Error: {} ".format(f_mean_sqrd_error_med))
print("Mean Absolute Error: {} ".format(f_mean_abs_error_med))
print("Number of Instances: {} ".format(count_med))
print('##############################################################################################')

print('##################################### ERROR VALUES-HIGH ###########################################')
print("Total Absolute Error: {} ".format(mean_abs_error_high))
print("Mean Squared Error: {} ".format(f_mean_sqrd_error_high))
print("Mean Absolute Error: {} ".format(f_mean_abs_error_high))
print("Number of Instances: {} ".format(count_high))
print('##############################################################################################')

# %%

# test script
import time

lines = [line.rstrip('\n') for line in open(test_labels_path)]

# ----------------------------
# Load Model weights
# ---------------------------
model.load_weights(weights_savepath + '_final.hdf5')

# # test data filenames
filenames = np.array([line.rstrip('\n').split()[0] for line in open(test_labels_path)])
filenames_split = np.array([line.rstrip('\n').split()[0].split('_')[0] for line in open(test_labels_path)])

print(len(lines))

start = time.time()

data = ''

# ----------------------------
# Loop through all test data
# ---------------------------
for i in range(len(lines)):

    line = lines[i].split(' ')

    fmap = fmaps_test[i]
    test_image = np.zeros((1, 1024))
    test_image[0, :] = fmap

    # ------------------------
    # Calculate Predictions
    # ------------------------
    output = model.predict(test_image)
    if output < 0:
        output = 0
    print([filenames[i], output])

    data = data + line[0] + " " + str(float(output)) + "\n"

stop = time.time()
duration = stop - start

print('Total time for all images: ', duration)

f = open('fasi_largeDataResults.txt', 'w')
f.write(data)

# %%

f = open('fasi_largeDataResults.txt', 'w')
f.write(data)

# %%


