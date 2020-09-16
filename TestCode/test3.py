# //  Created by Qazi Ammar Arshad on 10/05/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

# this file is used to copy files form one folder to other.

from custom_classes import path
import os

healthy_path = "/home/iml/Desktop/qazi/Model_Result_Dataset/Dataset/IML_training_data/binary_classifcation/p.f/healthy/"
malaria_healthy = "/home/iml/Desktop/qazi/Model_Result_Dataset/Dataset/IML_training_data/binary_classifcation/p.f/malaria_healthy/"


healthy_files_name = path.read_all_files_name_from(healthy_path, ".JPG")
malaria_healthy_files_name = path.read_all_files_name_from(malaria_healthy, ".JPG")

#%% check unique files name in these folders

unique_file_name = set(healthy_files_name) & set(malaria_healthy_files_name)

#%%
for img in unique_file_name:
    img_path = healthy_path + img
    print(img_path)
    os.remove(img_path)



