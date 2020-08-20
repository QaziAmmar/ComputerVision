# //  Created by Qazi Ammar Arshad on 10/05/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

import cv2
from custom_classes import path, cv_iml
# plot boxes on images

folder_base_path = path.dataset_path + "1000X-100x/all_images/"
all_images_name = path.read_all_files_name_from(folder_base_path, ".JPG")

#%%
# find substring from the images name

for image_name in all_images_name:
    if image_name.find("100x") != -1:
        img = cv2.imread(folder_base_path + image_name)
        cv2.rectangle(img, (735, 1340), (1000, 1560), (255, 0, 0), 2)
        cv2.imwrite(folder_base_path+ image_name, img)
        continue
    if image_name.find("400x") != -1:
        img = cv2.imread(folder_base_path + image_name)
        cv2.rectangle(img, (451, 1204), (1260, 2060), (0, 0, 255), 2)
        cv2.imwrite(folder_base_path + image_name, img)




