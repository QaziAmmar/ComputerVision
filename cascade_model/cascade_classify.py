# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import cv2
from custom_classes import path
from custom_classes.dataset_loader import *

save_model_path = path.save_models_path + "BBBC041/cascade.h5"
path_to_binary_label = path.save_models_path + "label_encoder_path/binaryclass_lb.pickle"
path_to_mutliclass_label = path.save_models_path + "label_encoder_path/multiclass_lb.pickle"

data_dir = path.dataset_path + "BBBC041/BBBC041_train_test_separate/"

train_imgs_scaled, train_labels, test_imgs_scaled, test_labels, val_imgs_scaled, val_labels = \
    load_train_test_val_images_from(data_dir, file_extension=".png", show_train_data=True)

# %%

# load the trained convolutional neural network from disk, followed
# by the category and color label binarizers, respectively
print("[INFO] loading network...")
model = load_model(save_model_path, custom_objects={"tf": tf})
categoryLB = pickle.loads(open(path_to_mutliclass_label, "rb").read())
colorLB = pickle.loads(open(path_to_binary_label, "rb").read())

# %%


# classify the input image using Keras' multi-output functionality
print("[INFO] classifying image...")
(multiclass_Proba, binary_Proba) = model.predict(test_imgs_scaled)
# find indexes of both the category and color outputs with the
# largest probabilities, then determine the corresponding class
# labels

multiclass_preds = multiclass_Proba.argmax(1)
multiclass_prediction_labels = categoryLB.inverse_transform(multiclass_preds)

binary_preds = binary_Proba.argmax(1)
binary_prediction_labels = colorLB.inverse_transform(binary_preds)

cv_iml.get_f1_score(test_labels, multiclass_prediction_labels, binary_classifcation=False, pos_label='malaria',
                    plot_confusion_matrix=True)

