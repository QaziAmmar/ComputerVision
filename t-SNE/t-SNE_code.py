# you can use interactive python interpreter, jupyter notebook, spyder or python code
# I am using interactive python interpreter (Python 3.7)
# import pandas dataframe formatted digits dataset for t-SNE analysis 
# (dataset available at scikit-learn)
import os
import tensorflow as tf
import numpy as np
from collections import Counter
from custom_classes import path, predefine_models, cv_iml
from custom_classes.dataset_loader import *

# hard_negative_mining_experiments parameter specify the type of experiment. In hard negative mining images are
# just separated into train, test and validation so their read style is just different.


data_set_base_path = path.dataset_path + "shalamar_training_data_balanced/train_test_seprate/train"

test_files, test_labels = process_path(data_set_base_path, ".JPG")


# %%
def get_img_data_parallel(idx, img, total_imgs):
    if idx % 5000 == 0 or idx == (total_imgs - 1):
        print('{}: working on img num {}'.format(threading.current_thread().name, idx))
    img = cv2.imread(img)
    img = cv2.resize(img, dsize=IMG_DIMS, interpolation=cv2.INTER_CUBIC)
    img = np.array(img)

    return img


IMG_DIMS = (125, 125)

ex = futures.ThreadPoolExecutor(max_workers=None)
test_data_inp = [(idx, img, len(test_files)) for idx, img in enumerate(test_files)]
print('\nLoading Test Images:')
test_data_map = ex.map(get_img_data_parallel,
                       [record[0] for record in test_data_inp],
                       [record[1] for record in test_data_inp],
                       [record[2] for record in test_data_inp])
test_data = np.array(list(test_data_map))

# %%
print(test_data.shape)
print('\nTest', Counter(test_labels))
# %%
BATCH_SIZE = 64
NUM_CLASSES = 2
EPOCHS = 25
INPUT_SHAPE = (125, 125, 3)

test_imgs_scaled = test_data / 255.

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(test_labels)

test_labels_enc = le.transform(test_labels)

print(test_labels[:6], test_labels_enc[:6])
# %%
# model = testing_models.get_1_CNN(INPUT_SHAPE)
# model = predefine_models.get_basic_CNN_for_malaria(INPUT_SHAPE)
# model = predefine_models.get_vgg_19_fine_tune(INPUT_SHAPE)
# model = predefine_models.get_vgg_19_transfer_learning(INPUT_SHAPE)
# model = predefine_models.get_resnet50_transferLearning(INPUT_SHAPE)
# model = predefine_models.get_dennet121_transfer_learning(INPUT_SHAPE)
# %%
#save_weights_path = path.save_models_path + "shamalar_data/balance_multiclass/balance_multiclass_dense201.h5"
save_weights_path = path.save_models_path + "shamalar_data/multiclass/multiclass_dense201.h5"
model = predefine_models.get_densenet201(INPUT_SHAPE,
                                                   classes=len(np.unique(test_labels)))
model.load_weights(save_weights_path)

# %%
# this how we get intermedicate layers output form net work
layer_name = 'dense_13'

intermediate_from_a = model.get_layer(layer_name).output

intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=intermediate_from_a)

# %%
test_data_feature_map = intermediate_model.predict([test_imgs_scaled])

# %%
RS = 123
# run t-SNE
from sklearn.manifold import TSNE

# perplexity parameter can be changed based on the input datatset
# dataset with larger number of variables requires larger perplexity
# set this value between 5 and 50 (sklearn documentation)
# verbose=1 displays run time messages
# set n_ite sufficiently high to resolve the well stabilized cluster
# get embeddings
tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(test_data_feature_map)
# %%
# plot t-SNE clusters
from bioinfokit.visuz import cluster

# cluster.tsneplot(score=tsne_em)
# plot will be saved in same directory (tsne_2d.png)

color_class = test_labels
cluster.tsneplot(score=tsne_em, colorlist=color_class, legendpos='upper right', legendanchor=(1.35, 1))
