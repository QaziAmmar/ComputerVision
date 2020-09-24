# //  Created by Qazi Ammar Arshad on 30/06/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

# this is a test file that is use to plot the distribution of data.


from custom_classes import path, cv_iml
from collections import Counter
import random
import cv2
from custom_classes.dataset_loader import load_train_test_val_images_from

# name of folder where you want to find the images
# folder base path

data_set_base_path = path.dataset_path + "IML_training_data/IML_multiclass_healthy_after_first_stage/p.v"

INPUT_SHAPE = (125, 125, 3)

train_imgs_scaled, train_labels, test_imgs_scaled, test_labels, val_imgs_scaled, val_labels = \
    load_train_test_val_images_from(data_set_base_path, file_extension=".JPG", show_train_data=True)

# %%
# print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest', Counter(test_labels))
print('Train:', Counter(train_labels) + Counter(val_labels) + Counter(test_labels))

# %%

cv_iml.hist_of_label_count(list(train_labels))

# %%

# randomly select train 227 files form the healthy
# test selected items = 89
# val selected items = 59

healthy_folder_path = "/home/iml/Desktop/qazi/Model_Result_Dataset/Dataset/IML_training_data/IML_multiclass_healthy_after_first_stage/p.v/train/healthy/"
healthy_cells = path.read_all_files_name_from(healthy_folder_path, ".JPG")

sampled_list = random.sample(healthy_cells, 2086)

save_folder_path = "/home/iml/Desktop/qazi/Model_Result_Dataset/Dataset/IML_training_data/IML_multiclass_healthy_after_first_stage/p.v/train/new_healthy/"

for temp_img in sampled_list:
    complete_path = healthy_folder_path + temp_img
    img = cv2.imread(complete_path)
    cv2.imwrite(save_folder_path + temp_img, img)
