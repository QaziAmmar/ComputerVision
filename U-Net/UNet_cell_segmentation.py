import pandas as pd
import numpy as np
import os
from custom_classes import path

import matplotlib.pyplot as plt
# %matplotlib inline

from skimage.io import imread, imshow
from skimage.transform import resize

# Don't Show Warning Messages
import warnings

warnings.filterwarnings('ignore')

# %%
IMG_HEIGHT = 96
IMG_WIDTH = 128
IMG_CHANNELS = 3

# %%
folder_base_path = path.dataset_path + 'shalamar_croped_segmentation_data/'

train_img_list = os.listdir(folder_base_path + 'images/train/')
train_mask_list = os.listdir(folder_base_path + 'mask/train/')

test_img_list = os.listdir(folder_base_path + '/images/test/')
test_mask_list = os.listdir(folder_base_path + '/mask/test/')

val_img_list = os.listdir(folder_base_path + '/images/val/')
val_mask_list = os.listdir(folder_base_path + '/mask/val/')

# %%
# create a dataframe
train_df_images = pd.DataFrame(train_img_list, columns=['image_id'])
test_df_images = pd.DataFrame(test_img_list, columns=['image_id'])
val_df_images = pd.DataFrame(val_img_list, columns=['image_id'])

# %%
# ======================================================
# Add a column showing how many cells are on each image
# ======================================================

train_df_masks = train_df_images
test_df_masks = test_df_images
val_df_masks = val_df_images

# create a new column called mask_id that is just a copy of image_id
train_df_masks['mask_id'] = train_df_images['image_id']
test_df_masks['mask_id'] = test_df_images['image_id']
val_df_masks['mask_id'] = val_df_images['image_id']

train_df_masks.shape

# %%
# create a test set
df_test = test_df_masks

# Reset the index.
# This is so that we can use loc to access mask id's later.

# create a list of test images
test_images_list = list(df_test['image_id'])

# Select only rows that are not part of the test set.
# Note the use of ~ to execute 'not in'.
df_masks = train_df_masks

print(df_masks.shape)
print(df_test.shape)

# %%
# Get lists of images and their masks.
image_id_list = list(df_masks['image_id'])
mask_id_list = list(df_masks['mask_id'])
test_id_list = list(df_test['image_id'])

# Create empty arrays

X_train = np.zeros((len(image_id_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

Y_train = np.zeros((len(image_id_list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

X_test = np.zeros((len(test_id_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

Y_test = np.zeros((len(test_id_list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

# %%

for i, image_id in enumerate(image_id_list):
    path_image = folder_base_path + 'images/train/' + image_id

    # read the image using skimage
    image = imread(path_image)

    # resize the image
    # image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
    # image = np.expand_dims(image, axis=-1)

    # insert the image into X_train
    X_train[i] = image

X_train.shape


# Y_train

for i, mask_id in enumerate(mask_id_list):
    path_mask = folder_base_path + 'mask/train/' + mask_id

    # read the image using skimage
    mask = imread(path_mask, True)

    # resize the image
    # mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
    mask = np.expand_dims(mask, axis=-1)

    # insert the image into Y_Train
    Y_train[i] = mask

Y_train.shape

# X_test

for i, image_id in enumerate(test_id_list):
    path_image = folder_base_path + 'images/test/' + image_id

    # read the image using skimage
    image = imread(path_image)

    # resize the image
    # image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
    # image = np.expand_dims(image, axis=-1)

    # insert the image into X_test
    X_test[i] = image

X_test.shape

# Y_test

for i, mask_id in enumerate(test_id_list):
    path_mask = folder_base_path + 'mask/test/' + mask_id

    # read the image using skimage
    mask = imread(path_mask, True)

    # resize the image
    # mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
    mask = np.expand_dims(mask, axis=-1)

    # insert the image into Y_Train
    Y_test[i] = mask

Y_test.shape

# %%
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

# %% Preparing model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

s = Lambda(lambda x: x / 255)(inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
c5 = Dropout(0.3)(c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()

# %%
filepath = path.save_models_path + "unet_shalamar.h5"

# earlystopper = EarlyStopping(patience=5, verbose=1)

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
                             save_best_only=True, mode='min')

callbacks_list = [checkpoint]

history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,
                    callbacks=callbacks_list)

# %%

# Make a prediction
# use the best epoch
model.load_weights(path.save_models_path + "unet_shalamar.h5")

test_preds = model.predict(X_test)

preds_test_thresh = (test_preds >= 0.5).astype(np.uint8)

test_img = preds_test_thresh[5, :, :, 0]

plt.imshow(test_img, cmap='gray')
plt.show()

# %%
# loop trhour all images and save them into a separate folder
import cv2

save_segmantaion_path = path.result_folder_path + "unet_segmentaion_crop_result/"
count = 0
for img, img_name in zip(preds_test_thresh, test_img_list):
    test_img = preds_test_thresh[count, :, :, 0] * 255
    # resized = cv2.resize(test_img, (1280, 960), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(save_segmantaion_path + img_name, test_img)
    count += 1

# %% # set up the canvas for the subplots
plt.figure(figsize=(10, 10))
plt.axis('Off')

# Our subplot will contain 3 rows and 3 columns
# plt.subplot(nrows, ncols, plot_number)


# == row 1 ==

# image
plt.subplot(3, 3, 1)
test_image = X_test[1, :, :, 0]
plt.imshow(test_image)
plt.title('Test Image', fontsize=14)
plt.axis('off')

# true mask
plt.subplot(3, 3, 2)
mask_id = df_test.loc[1, 'mask_id']
path_mask = path.dataset_path + 'mask/mask/' + mask_id
mask = imread(path_mask)
mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
plt.imshow(mask, cmap='gray')
plt.title('True Mask', fontsize=14)
plt.axis('off')

# predicted mask
plt.subplot(3, 3, 3)
test_mask = preds_test_thresh[1, :, :, 0]
plt.imshow(test_mask, cmap='gray')
plt.title('Pred Mask', fontsize=14)
plt.axis('off')

# == row 2 ==

# image
plt.subplot(3, 3, 4)
test_image = X_test[2, :, :, 0]
plt.imshow(test_image)
plt.title('Test Image', fontsize=14)
plt.axis('off')

# true mask
plt.subplot(3, 3, 5)
mask_id = df_test.loc[2, 'mask_id']
path_mask = path.dataset_path + 'mask/images/' + mask_id
mask = imread(path_mask)
mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
plt.imshow(mask, cmap='gray')
plt.title('True Mask', fontsize=14)
plt.axis('off')

# predicted mask
plt.subplot(3, 3, 6)
test_mask = preds_test_thresh[2, :, :, 0]
plt.imshow(test_mask, cmap='gray')
plt.title('Pred Mask', fontsize=14)
plt.axis('off')

# == row 3 ==

# image
plt.subplot(3, 3, 7)
test_image = X_test[3, :, :, 0]
plt.imshow(test_image)
plt.title('Test Image', fontsize=14)
plt.axis('off')

# true mask
plt.subplot(3, 3, 8)
mask_id = df_test.loc[3, 'mask_id']
path_mask = path.dataset_path + 'mask/mask/' + mask_id
mask = imread(path_mask)
mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
plt.imshow(mask, cmap='gray')
plt.title('True Mask', fontsize=14)
plt.axis('off')

# predicted mask
plt.subplot(3, 3, 9)
test_mask = preds_test_thresh[3, :, :, 0]
plt.imshow(test_mask, cmap='gray')
plt.title('Pred Mask', fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.show()
