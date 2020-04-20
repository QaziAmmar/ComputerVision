# //  Created by Qazi Ammar Arshad on 01/04/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.

from custom_classes import path
import tensorflow as tf
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import pathlib
import glob

# Cluster cell on the base of k-means clustring.

data_dir = path.result_folder_path + "pvivax_malaria_cells/"
data_dir = pathlib.Path(data_dir)
files_in_folder = []
for folder_name in data_dir.glob('*'):
    files_in_folder = files_in_folder + (glob.glob(str(folder_name) + '/*.png'))

# dataset_path = path.result_folder_path + "/rbc_countors/"
targetdir = path.result_folder_path + "kmeans_clustrin_pvivax_malaria/"
# image_list = path.read_all_files_name_from(dataset_path, file_extension='jpg')

save_weight_path = path.save_models_path + "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
vgg = tf.keras.applications.vgg19.VGG19(include_top=False,
                                        weights=save_weight_path)

featurelist = []
for image_name in files_in_folder:
    img = cv2.imread(image_name)
    img = cv2.resize(img, (125, 125))
    img_data = np.expand_dims(img, axis=0)

    features = np.array(vgg.predict(img_data))
    features = features.flatten()
    featurelist.append(features.flatten())
    print(image_name)

# %%
# Clustering
number_clusters = 6
kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(featurelist))
# %%

for i, m in enumerate(kmeans.labels_):
    try:
        os.makedirs(targetdir + str(m))
    except OSError:
        pass
    img = cv2.imread(files_in_folder[i])
    image_name = files_in_folder[i].split('/')[9]
    cv2.imwrite(targetdir + str(m) + "/" + image_name, img)

# %%
path1 = path.result_folder_path + "pvivax_malaria_cells/difficult"
path2 = path.result_folder_path + "kmeans_clustrin_pvivax_malaria/5"

path1_files = path.read_all_files_name_from(path1, '.png')
path2_files = path.read_all_files_name_from(path2, '.png')
print("Different = " + str(len(set(path1_files).union(set(path2_files)))))
print("Matching = " + str(len((set(path1_files).intersection(set(path2_files))))))

#%%
