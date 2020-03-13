# This is a file where we fist test our code then implement it into other file
from custom_classes import path, cv_iml
import os

test_dataset_path = path.dataset_path + "/IML_unaugmented_images/test/"
# train_healthy_images_path = path.dataset_path + "/IML_unaugmented_images/healthy/"
train_malaria_images_path = path.dataset_path + "/IML_unaugmented_images/malaria/"
test_images_name = path.read_all_files_name_from(test_dataset_path, ".JPG")
train_images_name = path.read_all_files_name_from(train_malaria_images_path, ".JPG")
# train_images_name.append(path.read_all_files_name_from(train_malaria_images_path, ".JPG"))

# matched_images = [a for a, b in zip(train_images_name, test_images_name) if a==b]
matched_images = []
for train_image in train_images_name:
    for test_image in test_images_name:
        if train_image == test_image:
            matched_images.append(test_image)
print(len(matched_images))#%%
# Now remove matched images form test folder.
try:
    for image in matched_images:
        os.remove(test_dataset_path + image)
except: pass





