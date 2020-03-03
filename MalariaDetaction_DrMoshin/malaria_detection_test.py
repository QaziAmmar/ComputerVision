import tensorflow as tf
import cv2
import threading
import numpy as np
from concurrent import futures
import glob
from custom_classes import path, cv_iml
from sklearn.preprocessing import LabelEncoder

INPUT_SHAPE = (125, 125, 3)
IMG_DIMS = (125, 125)

#  load test data.

test_data_set_base_path = path.dataset_path + "IML_cell_images/test/"
files_names = path.read_all_files_name_from(test_data_set_base_path, ".JPG")
test_files = []
for name in files_names:
    test_files.append(test_data_set_base_path + name)


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

inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

flat = tf.keras.layers.Flatten()(pool3)

hidden1 = tf.keras.layers.Dense(512, activation='relu')(flat)
drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)

model = tf.keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# %%

save_weights_path = path.save_models_path + "MalariaDetaction_DrMoshin/basic_cnn_IML_fineTune.h5"
model.load_weights(save_weights_path)

# %%
# Model Performance Evaluation

basic_cnn_preds = model.predict(test_imgs_scaled)

basic_cnn_preds_labels = le.inverse_transform([1 if pred > 0.6 else 0
                                               for pred in basic_cnn_preds.ravel()])

# %%
# Save image into separate folder base of the prediction.

for i in range(len(basic_cnn_preds_labels)):
    label = basic_cnn_preds_labels[i]
    img = cv2.imread(test_files[i])
    if label == 'healthy':
        image_save_path = path.result_folder_path + "IML_finetune_rbc_result/healthy/" + files_names[i]
        cv2.imwrite(image_save_path, img)
    else:
        image_save_path = path.result_folder_path + "IML_finetune_rbc_result/malaria/" + files_names[i]
        cv2.imwrite(image_save_path, img)

# end of testing code.

# %%
# Start calculating F1 score of our dataset.
# import os
#
# f1_score_base_path = path.dataset_path + "IML_cell_images/test_f1/"
#
# base_dir = os.path.join(f1_score_base_path)
# infected_dir = os.path.join(base_dir, "malaria")
# healthy_dir = os.path.join(base_dir, "healthy")
#
# infected_files = glob.glob(infected_dir + '/*.JPG')
# healthy_files = glob.glob(healthy_dir + '/*.JPG')
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
# # demonstration of calculating metrics for a neural network model using sklearn
#
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
#
#
# accuracy = accuracy_score(test_labels, basic_cnn_preds_labels)
# print('Accuracy: %f' % accuracy)
# # # precision tp / (tp + fp)
# precision = precision_score(test_labels, basic_cnn_preds_labels, pos_label='malaria')
# print('Precision: %f' % precision)
# # recall: tp / (tp + fn)
# recall = recall_score(test_labels, basic_cnn_preds_labels, pos_label='malaria')
# print('Recall: %f' % recall)
# # f1: 2 tp / (2 tp + fp + fn)
# f1 = f1_score(test_labels, basic_cnn_preds_labels, pos_label='malaria')
# print('F1 score: %f' % f1)
