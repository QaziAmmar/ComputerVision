# %%

from keras.models import Sequential
from keras.utils import plot_model
import keras as k
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, Input, Reshape, Permute, \
    GlobalAveragePooling2D
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.misc import imresize, imsave
import csv
import scipy.io as sio
from PIL import Image
import scipy
import pickle
from custom_classes import cv_iml

# %%

def create_model():
    model = Sequential()

    # conv1_1
    model.add(Conv2D(64, kernel_size=3, strides=1,
                     padding='SAME', use_bias=False,
                     activation='relu', name='conv1_1', batch_input_shape=(None, None, None, 3)))
    # conv1_2
    model.add(Conv2D(64, kernel_size=3, strides=1,
                     padding='SAME', use_bias=False,
                     activation='relu', name='conv1_2'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # conv2_1
    model.add(Conv2D(128, kernel_size=3, strides=1,
                     padding='SAME', use_bias=False,
                     activation='relu', name="conv2_1"))

    # conv2_2
    model.add(Conv2D(128, kernel_size=3, strides=1,
                     padding='SAME', use_bias=False,
                     activation='relu', name='conv2_2'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # conv3_1
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='SAME', use_bias=False,
                     activation='relu', name='conv3_1'))

    # conv3_2
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='SAME', use_bias=False,
                     activation='relu', name='conv3_2'))

    # conv3_3
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='SAME', use_bias=False,
                     activation='relu', name='conv3_3'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # conv6
    model.add(Conv2D(2048, kernel_size=8, strides=1, use_bias=False,
                     activation='relu', name='conv6'))

    # conv7
    model.add(Conv2D(256, kernel_size=1, strides=1,
                     use_bias=False,
                     activation='relu', name='conv7'))

    # conv8
    model.add(Conv2D(2, kernel_size=1, strides=1,
                     use_bias=False,
                     name='conv8'))

    model.add(Activation('softmax'))
    #     model.add(GlobalAveragePooling2D())

    return model


# %%


root_path = 'E:/ITU/Dr_Mohsen/houseCounting/'
# features_path = root_path + 'all_new_mix_feature_maps/dmaps'
images_path = "/home/itu/Desktop/Qazi/foldscope_dataset/test_patch/"
locations = open("/home/itu/Desktop/Qazi/foldscope_dataset/patches_annotatin_test.txt", 'r').read().split('\n')
locations = [row.split(' ') for row in locations]

save_overlays = True
# overlays_path = "{}/all_new_mix_feature_maps/overlays/".format(root_path)

# image_extension = '.jpg'

# %%

model = create_model()
# model.load_weights("/home/itu/Desktop/Qazi/SSNet_weigth/Busara_Vicinity_40_2.hdf5")
model.load_weights("/home/itu/Desktop/Qazi/SSNet_weigth/base_weights.hdf5")


img_h = 256
img_w = 256

img_h = np.int(img_h)
img_w = np.int(img_w)

M = [108, 104, 94]  # mean values
result_path = "/home/itu/Desktop/Qazi/results/"

for idx, row in enumerate(locations):
    print(idx)
    f = Image.open(images_path + row[0])
    image = np.asarray(f, dtype=np.float32)
    image = cv2.resize(image, (256, 256))

    if len(image.shape) == 3:
        image = image.transpose(2, 0, 1)

    test_img = np.zeros((1, img_h, img_w, 3))
    overlay = np.zeros((img_h, img_w, 3))

    test_img[0, :, :, 0] = image[0, :, :] - M[0]
    test_img[0, :, :, 1] = image[1, :, :] - M[1]
    test_img[0, :, :, 2] = image[2, :, :] - M[2]
    test_img = np.float32(test_img)

    output = model.predict(test_img)
    # save the sample 0 with image name and
    sample0 = imresize(output[0, :, :, 0], [img_h, img_w], interp='bilinear', mode='F')
    sample1 = imresize(output[0, :, :, 1], [img_h, img_w], interp='bilinear', mode='F')

    # overlay[:, :, 0] = image[0, :, :]
    # overlay[:, :, 1] = image[1, :, :]
    # temp = image[2, :, :]
    #
    # vis = (np.double(temp) * np.double(sample0)) + np.double(255 * (sample1))
    # overlay[:, :, 2] = vis
    #

    filename = "{}/{}".format(result_path, row[0])
    cv2.imwrite(filename, sample0 * 255)
    # sio.savemat(filename, {'prob': sample0})
    #
    # #     filename = "{}/{}.npy".format(features_path, row[0])
    # #     np.save(open(filename, 'wb'), sample1)
    #
    # if save_overlays:
    #     overlay_path = "{}/{}.png".format(overlays_path, row[0])
    #     imsave(overlay_path, overlay)

# %%


