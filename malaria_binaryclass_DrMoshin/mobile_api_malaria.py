# //  Created by Qazi Ammar Arshad on 01/02/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
# This is a file where we fist test our code then implement it into other file
from custom_classes import path, cv_iml
import cv2
import tensorflow as tf
import threading
import numpy as np
from concurrent import futures
from custom_classes import path, cv_iml
from sklearn.preprocessing import LabelEncoder

image_path = path.dataset_path + "IML_new_microscope/p.f_plus/100X_crop/IMG_2458.JPG"
# IMG_4456 = 4 Malaria
# IMG_4462 =

# image_segments = 500
rgb = cv2.imread(image_path)
clone = rgb.copy()
annotated_image = rgb.copy()
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Sharpen image
sharp_image = cv2.filter2D(gray, -1, kernel)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(blurred, 0, 14)

kernel = np.ones((7, 7), np.uint8)

closing = cv2.morphologyEx(wide, cv2.MORPH_CLOSE, kernel)

contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(rgb, contours, -1, (0, 255, 0), 1)

# create all-black mask image
mask = np.zeros(shape=rgb.shape, dtype="uint8")

red_blood_images = []
red_blood_cells_locations = []
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    if (w > 70 and h > 70) and (w < 180 and h < 180):
        cv2.rectangle(img=annotated_image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=3)

cv_iml.image_show(annotated_image)

for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    if (w > 70 and h > 70) and (w < 180 and h < 180):
        location = (x, y, w, h)
        crop_image = clone[y:y + h, x: x + w]
        # save data for CNN.
        red_blood_images.append(crop_image)
        # Saving data for rectangle draw.
        red_blood_cells_locations.append({
            'red_blood_cell': crop_image,
            'location': location,
            'isMalaria': False
        })

print(len(red_blood_cells_locations))

INPUT_SHAPE = (125, 125, 3)
IMG_DIMS = (125, 125)


def get_img_data_parallel(idx, img, total_imgs):
    if idx % 5000 == 0 or idx == (total_imgs - 1):
        print('{}: working on img num {}'.format(threading.current_thread().name, idx))
    img = cv2.resize(img, dsize=IMG_DIMS, interpolation=cv2.INTER_CUBIC)
    img = np.array(img, dtype=np.float32)

    return img


ex = futures.ThreadPoolExecutor(max_workers=None)
test_data_inp = [(idx, img, len(red_blood_images)) for idx, img in enumerate(red_blood_images)]

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

save_weights_path = path.save_models_path + "malaria_binaryclass_DrMoshin/best_resutls/multi_finetune_basic_cnn.h5"
model.load_weights(save_weights_path)

# %%
# Model Performance Evaluation

basic_cnn_preds = model.predict(test_imgs_scaled)
basic_cnn_preds_labels = le.inverse_transform([1 if pred > 0.5 else 0
                                               for pred in basic_cnn_preds.ravel()])

# %%

# Draw prediction on Image.
counter = 0

for prediction_label in basic_cnn_preds_labels:
    if prediction_label == "malaria":
        red_blood_cells_locations[counter]['isMalaria'] = True
        (x, y, w, h) = red_blood_cells_locations[counter]['location']
        cv2.rectangle(img=clone, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=3)
    counter = counter + 1
cv_iml.image_show(clone)
cv2.imwrite(path.result_folder_path + "IMG_2458_actual_image.png", annotated_image)
cv2.imwrite(path.result_folder_path + "IMG_2458_malaria_image.png", clone)

