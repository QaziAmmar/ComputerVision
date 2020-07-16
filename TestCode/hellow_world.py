# transfer imges form result folder to train folder for hard negative mining.

import os
import cv2
from custom_classes import path

prediction_folder_path = path.result_folder_path + "hard_negative_mining/p.v/actual_malaria/"
train_folder_path = path.dataset_path + "/IML_training_data/binary_classifcation_HardNegative_mining/p.v/train/"
#  move images form predction folder to train folder and avoid duplication of data.
# 1. delete images form train healthy and then move these images to malaria folder
pred_all_images_name = path.read_all_files_name_from(prediction_folder_path, ".JPG")
#     a. check these images in healthy folder if found then delete these images from there
healthy_images_name = path.read_all_files_name_from(train_folder_path + "Uninfected" , ".JPG")
# Compare these 2 arrays
matched_malaria_images_in_healthy_folder = set(pred_all_images_name) & set(healthy_images_name)
#     b. move these malaria images to malaria folder.
for malaria_imgs in matched_malaria_images_in_healthy_folder:
    img = cv2.imread(train_folder_path + "Uninfected/" + malaria_imgs)
    print(os.remove(train_folder_path + "Uninfected/" + malaria_imgs))
#     save this img into malaria folder
    cv2.imwrite(train_folder_path + "Parasitized/" + malaria_imgs, img)


# 2. delete images form train malaria and then move these images to healthy folder path
# same code is applied for moving actual healthy images to train healthy folder and remove them from train malaria
# folder
