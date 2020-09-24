# //  Created by Qazi Ammar Arshad on 24/08/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
# This code separate the dataset into train, test and validation folder.

# import the necessary packages
import numpy as np
import pandas as pd
import glob
from custom_classes import path
import pathlib
import cv2

# Directory data
images_extension = ".png"
data_dir = "/home/iml/Desktop/qazi/Model_Result_Dataset/Dataset/BBBC041/train_test_separate/train/"
data_dir = pathlib.Path(data_dir)

# %%
files_df = None
for folder_name in data_dir.glob('*'):
    # '.DS_Store' this hidden file is automatically created by mac/windows which we need to exclude form your data.
    if '.DS_Store' == str(folder_name).split('/')[-1]:
        continue
    files_in_folder = glob.glob(str(folder_name) + '/*' + images_extension)
    df2 = pd.DataFrame({
        'filename': files_in_folder,
        'label': [folder_name.name] * len(files_in_folder)
    })
    if files_df is None:
        files_df = df2
    else:
        files_df = files_df.append(df2, ignore_index=True)

files_df = files_df.sample(frac=1).reset_index(drop=True)

files_df.head()

# %%
from sklearn.model_selection import train_test_split
from collections import Counter

# Generating tanning and testing data.
# train_files, test_files, train_labels, test_labels = train_test_split(files_df['filename'].values,
#                                                                       files_df['label'].values,
#                                                                       test_size=0.2,
#                                                                       random_state=42)
# Generating validation data form tanning data.
train_files, val_files, train_labels, val_labels = train_test_split(files_df['filename'].values,
                                                                    files_df['label'].values,
                                                                    test_size=0.1,
                                                                    random_state=42)

# print(train_files.shape, val_files.shape, test_files.shape)
# print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest', Counter(test_labels))
print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest')
# %%


base_train_test_path = "/home/iml/Desktop/qazi/Model_Result_Dataset/Dataset/BBBC041/new_val_train/"
# save train data
train_folder = base_train_test_path + "train/"
# save test data
# test_folder = base_train_test_path + "test/"
# save val data
val_folder = base_train_test_path + "val/"

for temp_trainfile in train_files:
    img_name = temp_trainfile.split('/')[-1]
    cell_category = temp_trainfile.split('/')[-2]
    img = cv2.imread(temp_trainfile)
    cv2.imwrite(train_folder + cell_category + "/" + img_name, img)

# for temp_test_file in test_files:
#     img_name = temp_test_file.split('/')[-1]
#     cell_category = temp_test_file.split('/')[-2]
#     img = cv2.imread(temp_test_file)
#     cv2.imwrite(test_folder + cell_category + "/" + img_name, img)

for temp_val_file in val_files:
    img_name = temp_val_file.split('/')[-1]
    cell_category = temp_val_file.split('/')[-2]
    img = cv2.imread(temp_val_file)
    cv2.imwrite(val_folder + cell_category + "/" + img_name, img)
