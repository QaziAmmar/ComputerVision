# //  Created by Qazi Ammar Arshad on 10/05/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
from custom_classes import path, cv_iml
from dataset_annotations_draw.model_classes.iml_pvivax_65_model import welcome_from_dict
from RedBloodCell_Segmentation.seg_dr_waqas_watershed_microscope_single_image import get_detected_segmentaion
import json
import cv2

# base path of folder where images and annotaion are saved.if (w < 70 or h < 70) or (w > 200 or h > 200):
folder_base_path = path.dataset_path + "Malaria_2010_dataset/"
# path of folder where all images are save.
images_path = folder_base_path + "Part 4/"
save_annotated_imgs_path = folder_base_path + "annotated_img/"

images_name = path.read_all_files_name_from(images_path, ".jpg")

# save annotaion path.
# %%
for img_name in images_name:
    annotated_img, _, json_object = get_detected_segmentaion(images_path + img_name)
    cv2.imwrite(save_annotated_imgs_path + img_name, annotated_img)