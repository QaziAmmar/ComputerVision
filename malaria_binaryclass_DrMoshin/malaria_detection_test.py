import tensorflow as tf
import cv2
import threading
import numpy as np
from concurrent import futures
import glob
import os
from custom_classes import path, cv_iml, predefine_models
from sklearn.preprocessing import LabelEncoder

INPUT_SHAPE = (125, 125, 3)
IMG_DIMS = (125, 125)

#  load test data.
# test_data_set_base_path = path.dataset_path + "IML_training_data/binary_classifcation_HardNegative_mining/p.v/train/"
# adding data form both healthy and infected folder
test_data_set_base_path = "/home/iml/Downloads/"

infected_dir = os.path.join(test_data_set_base_path, "Parasitized")
healthy_dir = os.path.join(test_data_set_base_path, "Uninfected")

infected_files_names = path.read_all_files_name_from(infected_dir, ".png")
healthy_files_names = path.read_all_files_name_from(healthy_dir, ".png")

test_files = []

for name in infected_files_names:
    test_files.append(infected_dir + "/" + name)

for name in healthy_files_names:
    test_files.append(healthy_dir + "/" + name)


def get_img_shape_parallel(idx, img, total_imgs):
    if idx % 5000 == 0 or idx == (total_imgs - 1):
        print('{}:working on img num {}'.format(threading.current_thread().name, idx))
    return cv2.imread(img).shape


def get_img_data_parallel(idx, img, total_imgs):
    if idx % 5000 == 0 or idx == (total_imgs - 1):
        print('{}: working on img num {}'.format(threading.current_thread().name, idx))
    img = cv2.imread(img)
    img = cv2.resize(img, dsize=IMG_DIMS, interpolation=cv2.INTER_CUBIC)
    img = np.array(img, dtype=np.float32)

    return img


ex = futures.ThreadPoolExecutor(max_workers=None)
test_data_inp = [(idx, img, len(test_files)) for idx, img in enumerate(test_files)]

print('\nLoading Test Images:')
test_data_map = ex.map(get_img_data_parallel,
                       [record[0] for record in test_data_inp],
                       [record[1] for record in test_data_inp],
                       [record[2] for record in test_data_inp])
test_data = np.array(list(test_data_map))

print(test_data.shape)

test_imgs_scaled = test_data / 255.

train_labels = ["healthy", "malaria"]
le = LabelEncoder()
le.fit(train_labels)

train_labels_enc = le.transform(train_labels)

# %%
# Model 1: CNN from Scratch

model = predefine_models.get_basic_CNN_for_malaria(INPUT_SHAPE)

# %%
load_weights_path = path.save_models_path + "MalariaDetaction_DrMoshin/best_resutls/basic_cnn_IML_fineTune.h5"
model.load_weights(load_weights_path)

# %%
# Model Performance Evaluation
print("start prediction")
basic_cnn_preds = model.predict(test_imgs_scaled)
print("end prediction")
basic_cnn_preds_labels = le.inverse_transform([1 if pred > 0.5 else 0
                                               for pred in basic_cnn_preds.ravel()])

# %%
# Save image into separate folder base of the prediction.

for i in range(len(basic_cnn_preds_labels)):
    label = basic_cnn_preds_labels[i]
    img = cv2.imread(test_files[i])
    if label == 'healthy':
        image_save_path = path.result_folder_path + "test/healthy/" + test_files[i].split('/')[-1]
        cv2.imwrite(image_save_path, img)
    else:
        image_save_path = path.result_folder_path + "test/malaria/" + test_files[i].split('/')[-1]
        cv2.imwrite(image_save_path, img)

# end of testing code.
# exit(0)
# %%
# # Start calculating F1 score of our dataset.
# import os
#
# f1_score_base_path = path.dataset_path + "cell_images/"
#
# base_dir = os.path.join(f1_score_base_path)
# infected_dir = os.path.join(base_dir, "Parasitized")
# healthy_dir = os.path.join(base_dir, "Uninfected")
#
# infected_files = glob.glob(infected_dir + '/*.png')
# healthy_files = glob.glob(healthy_dir + '/*.png')
#
# print(len(infected_files), len(healthy_files))
# # %%
#
# import numpy as np
# import pandas as pd
#
# np.random.seed(42)
#
# files_df = pd.DataFrame({
#     'filename': infected_files + healthy_files,
#     'label': ['malaria'] * len(infected_files) + ['healthy'] * len(healthy_files)
# }).sample(frac=1, random_state=42).reset_index(drop=True)
#
# files_df.head()
#
# test_files = files_df['filename'].values
# test_labels = files_df['label'].values
#
# print(test_files.shape)
#
# # %%
# ex = futures.ThreadPoolExecutor(max_workers=None)
# data_inp = [(idx, img, len(test_files)) for idx, img in enumerate(test_files)]
# test_data_map = ex.map(get_img_data_parallel,
#                        [record[0] for record in data_inp],
#                        [record[1] for record in data_inp],
#                        [record[2] for record in data_inp])
# test_data = np.array(list(test_data_map))
#
# # %%
# # predict probabilities for test set
# # train_labels = ["healthy", "malaria"]
# le = LabelEncoder()
# le.fit(test_labels)
#
# test_labels_enc = le.transform(test_labels)
# test_imgs_scaled = test_data / 255.
# basic_cnn_preds = model.predict(test_imgs_scaled)
#
# basic_cnn_preds_labels = le.inverse_transform([1 if pred > 0.6 else 0
#                                                for pred in basic_cnn_preds.ravel()])
#
# # %%
#
# cv_iml.get_f1_score(test_labels, basic_cnn_preds_labels, pos_label='malaria')
