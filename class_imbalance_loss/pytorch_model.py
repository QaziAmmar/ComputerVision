# //  Created by Qazi Ammar Arshad on 01/10/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

# import the necessary packages
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import pathlib
import tensorflow.keras.backend as kb
from custom_classes import path, cv_iml, predefine_models
from keras.utils import to_categorical
from class_imbalance_loss.class_balanced_loss import *
import os

# importing the libraries
import pandas as pd
import numpy  as  np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
# % matplotlib  inline

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout
from torch.optim import Adam, SGD

AUTOTUNE = tf.data.experimental.AUTOTUNE

samples_per_cls = []
number_of_classes = 0



# %%
# Directory data
data_dir = path.dataset_path + "IML_training_data/IML_multiclass_classification/p.f/"
data_dir = pathlib.Path(data_dir)
load_weight_path = path.save_models_path + "pvivax_malaria_multi_class/" + "basic_cnn_MC_TL.h5"

# %%
files_df = None
np.random.seed(42)
for folder_name in data_dir.glob('*'):
    # '.DS_Store' file is automatically created by mac which we need to exclude form your code.
    if '.DS_Store' == str(folder_name).split('/')[-1]:
        continue
    files_in_folder = glob.glob(str(folder_name) + '/*.JPG')
    df2 = pd.DataFrame({
        'filename': files_in_folder,
        'label': [folder_name.name] * len(files_in_folder)
    })
    number_of_classes += 1
    samples_per_cls.append(len(files_in_folder))
    if files_df is None:
        files_df = df2
    else:
        files_df = files_df.append(df2, ignore_index=True)

files_df.sample(frac=1, random_state=42).reset_index(drop=True)

files_df.head()

# %%
from sklearn.model_selection import train_test_split
from collections import Counter

# Generating tanning and testing data.
train_files, test_files, train_labels, test_labels = train_test_split(files_df['filename'].values,
                                                                      files_df['label'].values,
                                                                      test_size=0.2,
                                                                      random_state=42)
# Generating validation data form tanning data.
train_files, val_files, train_labels, val_labels = train_test_split(train_files,
                                                                    train_labels,
                                                                    test_size=0.2,
                                                                    random_state=42)

print(train_files.shape, val_files.shape, test_files.shape)
print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest', Counter(test_labels))

# %%
import cv2
from concurrent import futures
import threading


def get_img_shape_parallel(idx, img, total_imgs):
    if idx % 5000 == 0 or idx == (total_imgs - 1):
        print('{}:working on img num {}'.format(threading.current_thread().name, idx))
    return cv2.imread(img).shape


ex = futures.ThreadPoolExecutor(max_workers=None)
data_inp = [(idx, img, len(train_files)) for idx, img in enumerate(train_files)]
print('Starting Img shape computation:')
train_img_dims_map = ex.map(get_img_shape_parallel,
                            [record[0] for record in data_inp],
                            [record[1] for record in data_inp],
                            [record[2] for record in data_inp])
# this part of code is getting dimensions of all image and save in train_img_dims.
train_img_dims = list(train_img_dims_map)
print('Min Dimensions:', np.min(train_img_dims, axis=0))
print('Average Dimensions: ', np.mean(train_img_dims, axis=0))
print('Median Dimensions:', np.median(train_img_dims, axis=0))
print('Max Dimensions:', np.max(train_img_dims, axis=0))

# %% # Load image data and resize on 125, 125 pixel.
IMG_DIMS = (125, 125)


def get_img_data_parallel(idx, img, total_imgs):
    if idx % 5000 == 0 or idx == (total_imgs - 1):
        print('{}: working on img num {}'.format(threading.current_thread().name, idx))
    img = cv2.imread(img)
    img = cv2.resize(img, dsize=IMG_DIMS, interpolation=cv2.INTER_CUBIC)
    img = np.array(img, dtype=np.float32)

    return img


ex = futures.ThreadPoolExecutor(max_workers=None)
train_data_inp = [(idx, img, len(train_files)) for idx, img in enumerate(train_files)]
val_data_inp = [(idx, img, len(val_files)) for idx, img in enumerate(val_files)]
test_data_inp = [(idx, img, len(test_files)) for idx, img in enumerate(test_files)]
print('Loading Train Images:')
train_data_map = ex.map(get_img_data_parallel,
                        [record[0] for record in train_data_inp],
                        [record[1] for record in train_data_inp],
                        [record[2] for record in train_data_inp])
train_data = np.array(list(train_data_map))

print('\nLoading Validation Images:')
val_data_map = ex.map(get_img_data_parallel,
                      [record[0] for record in val_data_inp],
                      [record[1] for record in val_data_inp],
                      [record[2] for record in val_data_inp])
val_data = np.array(list(val_data_map))

print('\nLoading Test Images:')
test_data_map = ex.map(get_img_data_parallel,
                       [record[0] for record in test_data_inp],
                       [record[1] for record in test_data_inp],
                       [record[2] for record in test_data_inp])
test_data = np.array(list(test_data_map))

train_data.shape, val_data.shape, test_data.shape

# %%
BATCH_SIZE = 64
NUM_CLASSES = 2
EPOCHS = 25
INPUT_SHAPE = (3, 125, 125)

train_imgs_scaled = train_data / 255.
val_imgs_scaled = val_data / 255.
test_imgs_scaled = test_data / 255.

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)

train_labels_enc = le.transform(train_labels)
val_labels_enc = le.transform(val_labels)
test_labels_enc = le.transform(test_labels)

print(train_labels[:6], train_labels_enc[:6])

# %% converting training images into torch format
train_x = train_imgs_scaled.reshape(len(train_data), 3, 125, 125)
train_x = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_labels_enc.astype(int);
train_y = torch.from_numpy(train_y)

# shape of training data
print(train_x.shape, train_y.shape)


# converting training images into torch format
val_x = val_imgs_scaled.reshape(len(val_imgs_scaled), 3, 125, 125)
val_x = torch.from_numpy(val_x)

# converting the target into torch format
val_y = val_labels_enc.astype(int);
val_y = torch.from_numpy(val_y)

# shape of training data
print(val_x.shape, val_y.shape)


# %% Implementing CNNs using PyTorch

class Net(Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.pool3 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(21632, 512)
        self.dropout1 = nn.Dropout2d(0.3)
        self.fc2 = nn.Linear(512, 5)
        self.dropout2 = nn.Dropout2d(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        output = F.log_softmax(x, dim=1)
        return output


# defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.07)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
# if torch.cuda.is_available():
#     model = model.cuda()
#     criterion = criterion.cuda()

print(model)


#%%

def train(epoch):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    x_val, y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format
    # if torch.cuda.is_available():
    #     x_train = x_train.cuda()
    #     y_train = y_train.cuda()
    #     x_val = x_val.cuda()
    #     y_val = y_val.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()

    # prediction for training and validation set
    output_train = model(x_train)
    output_val = model(x_val)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch % 2 == 0:
        # printing the validation loss
        print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val)


# defining the number of epochs
n_epochs = 25
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# training the model
for epoch in range(n_epochs):
    train(epoch)