import cv2
import threading
import numpy as np
from concurrent import futures
import glob
from keras.utils import to_categorical
from custom_classes import path, cv_iml, predefine_models
from sklearn.preprocessing import LabelEncoder

INPUT_SHAPE = (125, 125, 3)
IMG_DIMS = (125, 125)

#  load test data.

test_dataset_path = path.dataset_path + "IML_cell_images/test/malaria/"
files_names = path.read_all_files_name_from(test_dataset_path, ".JPG")
test_files = []
for name in files_names:
    test_files.append(test_dataset_path + name)


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

train_labels = ['gametocyte', 'leukocyte', 'ring', 'schizont', 'trophozoite']

number_of_classes = len(train_labels)
le = LabelEncoder()
le.fit(train_labels)

train_labels_enc = le.transform(train_labels)
train_labels_enc = to_categorical(train_labels_enc, num_classes=number_of_classes)

print(train_labels[:6], train_labels_enc[:6])
# %%
# Model 1: CNN from Scratch

model = predefine_models.get_basic_CNN_for_malaria(INPUT_SHAPE, binary_classification=False,
                                                   classes=number_of_classes)

# %%
# change the path of your save model.
save_weights_path = path.save_models_path + "malaria_binaryclass_DrMoshin/basic_cnn_finetune_IMLdata.h5"
model.load_weights(save_weights_path)

# %%
# Model Performance Evaluation

basic_cnn_preds = model.predict(test_imgs_scaled)
basic_cnn_preds = basic_cnn_preds.argmax(1)
prediction_labels = le.inverse_transform(basic_cnn_preds)

# %%
# Save image into separate folder base of the prediction.

for i in range(len(prediction_labels)):
    label = prediction_labels[i]
    img = cv2.imread(test_files[i])
    if label == 'gametocyte':
        image_save_path = path.result_folder_path + "multi_class_cnn_iml/gametocyte/" + files_names[i]
        cv2.imwrite(image_save_path, img)
    elif label == 'leukocyte':
        image_save_path = path.result_folder_path + "multi_class_cnn_iml/leukocyte/" + files_names[i]
        cv2.imwrite(image_save_path, img)
    elif label == 'ring':
        image_save_path = path.result_folder_path + "multi_class_cnn_iml/ring/" + files_names[i]
        cv2.imwrite(image_save_path, img)
    elif label == "schizont":
        image_save_path = path.result_folder_path + "multi_class_cnn_iml/schizont/" + files_names[i]
        cv2.imwrite(image_save_path, img)
    else:
        # trophozoite
        image_save_path = path.result_folder_path + "multi_class_cnn_iml/trophozoite/" + files_names[i]
        cv2.imwrite(image_save_path, img)

# end of testing code.