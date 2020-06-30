# //  Created by Qazi Ammar Arshad on 28/06/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This file try to capture malaria cells.
"""
# 1. compute dense feature of malaria images annotated by chughati lab.

from custom_classes import path, cv_iml, predefine_models
import cv2
import tensorflow as tf
import numpy as np
from scipy.spatial import distance
from numpy.linalg import norm


def getDensFeatuer(img):
    INPUT_SHAPE = (125, 125, 3)
    img = cv2.resize(img, (125, 125))
    # Model
    save_weight_path = path.save_models_path + "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"
    dn121 = tf.keras.applications.DenseNet121(weights=save_weight_path, include_top=False, input_shape=INPUT_SHAPE)

    img_data = np.expand_dims(img, axis=0)

    features = np.array(dn121.predict(img_data))
    features = features.flatten()
    return features


def getVgg19Feature(img):
    img = cv2.resize(img, (125, 125))
    INPUT_SHAPE = (125, 125, 3)
    save_weight_path = path.save_models_path + "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False,
                                            weights=save_weight_path,
                                            input_shape=INPUT_SHAPE)

    img_data = np.expand_dims(img, axis=0)

    features = np.array(vgg.predict(img_data))
    features = features.flatten()
    return features


def getResnetFeature(img):
    img = cv2.resize(img, (125, 125))
    INPUT_SHAPE = (125, 125, 3)
    save_weight_path = path.save_models_path + "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=save_weight_path,
                                                       input_shape=INPUT_SHAPE)
    img_data = np.expand_dims(img, axis=0)

    features = np.array(resnet50.predict(img_data))
    features = features.flatten()
    return features


def getBasicCnnFeature(img):
    print("test")


def l1_norm(feature):
    # this function normalized the feature vector with l1 norm
    l1 = norm(feature)
    normalized_feature = feature / l1
    return normalized_feature


# image_name = "IMG_4536.JPG"
folder_base_path = path.dataset_path + "LabelBox_classification/"
chughati_cell_img = cv2.imread(folder_base_path + "pf.jpg")

cell_images_name = path.read_all_files_name_from(folder_base_path + "dens_net/", ".npy")

chughati_cell_feature = getDensFeatuer(chughati_cell_img)

# %%
counter = 1
features = []
for img_name in cell_images_name:
    print(img_name)
    counter += 1
    # tempimg = cv2.imread(folder_base_path + "rbc/" + img_name)
    # tempfeature = getDensFeatuer(tempimg)
    # save feature vector
    # np.save(folder_base_path + "dens_net/" + img_name, tempfeature)
    tempfeature = np.load(folder_base_path + "dens_net/" + img_name)
    features.append(tempfeature)

# %%
# compute L2 norm of each feature vector with chughati image feature
distance_array = []
chughati_cell_feature = l1_norm(chughati_cell_feature)
for f in features:
    # normalized the feature vector.
    f = l1_norm(f)
    d = distance.euclidean(f, chughati_cell_feature)
    distance_array.append(d)
# %%
# on the base of distance matrix sort the images array.
sorted_images_on_distance = [x for _, x in sorted(zip(distance_array, cell_images_name))]

# %%
# save sorted images
counter = 1
for img_name in sorted_images_on_distance:
    img = cv2.imread(folder_base_path + "rbc/" + img_name[:-4])
    cv2.imwrite(folder_base_path + "sorted/" + "0000" + str(counter) + ".JPG", img)
    counter += 1

print("eoc")

# label_box_annotation_path = folder_base_path + "classifaction_malaria.json"
# code_annotation_path = folder_base_path + "CodePlusLabelBox_annotation.json"
#
# json_dictionary = []
#
# with open(label_box_annotation_path) as annotation_path:
#     label_box_json_annotation = json.load(annotation_path)
#
# with open(code_annotation_path) as annotation_path:
#     code_annotaion_python_object_array = json.load(annotation_path)
#
# label_box_result_python_object_array = welcome_from_dict(label_box_json_annotation)
#
# expectedResult = [sub for sub in label_box_json_annotation if sub['External ID'] == "IMG_4430.JPG"]
