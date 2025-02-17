"""
//  Created by Qazi Ammar Arshad on 8/09/2020.
//  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

This is a dataset load class that load dataset from any folder path.
"""

import os
import pandas as pd
import glob
import pathlib
import numpy as np
import cv2
from concurrent import futures
import threading
from custom_classes import cv_iml


def convert_multiclass_lbl_to_binary_lbl(multiclass_labels):
    """
    This function convert the multiclass labels into binary labels as healthy and malaria.
    :param multiclass_labels: ['ring', 'schizont', ...]
    :return: ['healthy', 'malaria']
    """
    binary_label = []
    for tempLabel in multiclass_labels:
        if tempLabel == "healthy":
            binary_label.append("healthy")
        else:
            binary_label.append("malaria")
    return binary_label


def process_path(data_dir, file_extension=".JPG"):
    """
    This function go through the entire folder path and return all files in that folder.
    :param data_dir: directory of data where you want to load all image
    :param file_extension:
    :return: return files path and labels of each files in pandas frame.
    """
    data_dir = pathlib.Path(data_dir)
    files_df = None
    for folder_name in data_dir.glob('*'):
        # '.DS_Store' file is automatically created by mac which we need to exclude form your code.
        if '.DS_Store' == str(folder_name).split('/')[-1]:
            continue
        files_in_folder = glob.glob(str(folder_name) + '/*' + file_extension)
        df2 = pd.DataFrame({
            'filename': files_in_folder,
            'label': [folder_name.name] * len(files_in_folder)
        })

        if files_df is None:
            files_df = df2
        else:
            files_df = files_df.append(df2, ignore_index=True)
    # to shuffle the data
    files_df = files_df.sample(frac=1).reset_index(drop=True)

    files = files_df['filename'].values
    labels = files_df['label'].values

    return files, labels


def load_images_data_parallel(images_file_name):
    # Load image data and resize on 125, 125 pixel.
    IMG_DIMS = (125, 125)

    def get_img_data_parallel(idx, img, total_imgs):
        if idx % 5000 == 0 or idx == (total_imgs - 1):
            print('{}: working on img num {}'.format(threading.current_thread().name, idx))
        img = cv2.imread(img)
        img = cv2.resize(img, dsize=IMG_DIMS, interpolation=cv2.INTER_CUBIC)
        img = np.array(img)

        return img

    ex = futures.ThreadPoolExecutor(max_workers=None)
    data_inp = [(idx, img, len(images_file_name)) for idx, img in enumerate(images_file_name)]

    data_map = ex.map(get_img_data_parallel,
                      [record[0] for record in data_inp],
                      [record[1] for record in data_inp],
                      [record[2] for record in data_inp])
    data = np.array(list(data_map))

    return data


def load_train_test_val_images_from(folder_path, file_extension=".JPG", show_train_data=False):
    """

    IMG_DIMS = (125, 125) all images will be resized on this size
    :param show_train_data:
    :param folder_path:
    :param file_extension:
    :return:
    """

    train_dir = os.path.join(folder_path, 'train')
    test_dir = os.path.join(folder_path, 'test')
    validation_dir = os.path.join(folder_path, 'val')

    # load train img file path with label
    train_files, train_labels = process_path(train_dir, file_extension)
    test_files, test_labels = process_path(test_dir, file_extension)
    val_files, val_labels = process_path(validation_dir, file_extension)
    # get img form image path and stack data into a array.

    print('Loading Train Images:')
    train_data = load_images_data_parallel(images_file_name=train_files)
    print('Loading Test Images:')
    test_data = load_images_data_parallel(images_file_name=test_files)
    print('Loading Val Images:')
    val_data = load_images_data_parallel(images_file_name=val_files)

    # show train images for view
    if show_train_data:
        cv_iml.show_train_images(train_data, train_labels)

    # Normalized image data between 0 - 1
    train_imgs_scaled = train_data / 255.
    val_imgs_scaled = val_data / 255.
    test_imgs_scaled = test_data / 255.

    # convert multiclass labels to binary labels.
    # train_labels = convert_multiclass_lbl_to_binary_lbl(train_labels)
    # test_labels = convert_multiclass_lbl_to_binary_lbl(test_labels)
    # val_labels = convert_multiclass_lbl_to_binary_lbl(val_labels)

    return train_imgs_scaled, train_labels, test_imgs_scaled, test_labels, val_imgs_scaled, val_labels
    # return train_files, train_labels, test_files, test_labels, val_files, val_labels
