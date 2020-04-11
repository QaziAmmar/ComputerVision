from custom_classes import path, cv_iml
import json
import cv2
from dataset_annotations.pvivax_model import Annotation_Model

# Link of dataset: https://www.kaggle.com/kmader/malaria-bounding-boxes

######################### Description: #########################
# This code separate the red blood cell form the whole sample slide
# according to the annotation and save them into a relevant folder.


# folder name can be healthy or malaria.
folder_name = "healthy/"
# defining path for all images.
dataset_path = path.dataset_path + "malaria/"
images_path = dataset_path + folder_name + "images/"
train_image_annotation_path = dataset_path + folder_name + "training.json"
test_image_annotation_path = dataset_path + folder_name + "test.json"
save_crop_images_path = path.result_folder_path + "pvivax_malaria_rbc/" + folder_name

# Reading training json annotation path
with open(train_image_annotation_path) as train_image_annotation_path:
    train_annotation = json.load(train_image_annotation_path)
# Reading test json annotation path
with open(test_image_annotation_path) as test_image_annotation_path:
    test_annotation = json.load(test_image_annotation_path)

# %%
# Parsing JSON data into python object for easy use. combine both test and train
# images together, and the crop red blood cell from whole image.
python_annotations = []

for annotation in train_annotation:
    python_annotations.append(Annotation_Model(annotation))

for annotation in test_annotation:
    python_annotations.append(Annotation_Model(annotation))

# %%
# Crop red blood cells from image form image according to annotation and save them in
# required folder.

for image in python_annotations:
    # we need to split image path because it has also folder namd apped with it.
    image_name = image.image.path_name.split('/')[2]
    # '', 'images', '8d02117d-6c71-4e47-b50a-6cc8d5eb1d55.png']
    img = cv2.imread(images_path + image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # counter variable to append with image name to uniquely save the image name.
    count = 0
    for object in image.objects:
        x1 = object.bounding_box.x1
        y1 = object.bounding_box.y1
        x2 = object.bounding_box.x2
        y2 = object.bounding_box.y2

        crop_image = img[y1:y2, x1:x2, :]
        save_name = save_crop_images_path + str(count) + "_" + image_name
        cv2.imwrite(save_name, crop_image)
        count += 1
