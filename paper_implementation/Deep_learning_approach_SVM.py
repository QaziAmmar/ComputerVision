from custom_classes import path, predefine_models, cv_iml
from custom_classes.dataset_loader import *
import tensorflow as tf
from collections import Counter
from sklearn import svm
from custom_classes import path, predefine_models
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

INPUT_SHAPE = (224, 224, 3)


def get_model():
    save_weight_path = path.save_models_path + "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False,
                                            weights=save_weight_path,
                                            input_shape=(INPUT_SHAPE))
    # Freeze the layers
    vgg.trainable = False

    base_vgg = vgg
    return vgg


# %%
save_weights_path = path.save_models_path + "shamalar_data/papers/deep_learning_based_classification.h5"

data_set_base_path = path.dataset_path + "shalamar_training_data/train_test_separate_balanced/"

EPOCHS = 25

train_imgs_scaled, train_labels, test_imgs_scaled, test_labels, val_imgs_scaled, val_labels = \
    load_train_test_val_images_from(data_set_base_path, file_extension=".JPG", show_train_data=True)

# %%

print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest', Counter(test_labels))


# %%
# First complete the binary cycle

def _replaceitem(x):
    if x == 'healthy':
        return 'healthy'
    else:
        return "malaria"


# binary_train_labels = list(map(_replaceitem, train_labels))
# binary_test_labels = list(map(_replaceitem, test_labels))
# binary_val_labels = list(map(_replaceitem, val_labels))

binary_train_labels = train_labels
binary_test_labels = test_labels
binary_val_labels = val_labels

number_of_binary_classes = len(np.unique(train_labels))

BATCH_SIZE = 64

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(binary_test_labels)

binary_train_labels_enc = le.transform(binary_train_labels)
# binary_train_labels_enc = to_categorical(binary_train_labels_enc, num_classes=number_of_binary_classes)

binary_test_labels_enc = le.transform(binary_test_labels)
# binary_test_labels_enc = to_categorical(binary_test_labels_enc, num_classes=number_of_binary_classes)

binary_val_labels_enc = le.transform(binary_val_labels)
# binary_val_labels_enc = to_categorical(binary_val_labels_enc, num_classes=number_of_binary_classes)

print(binary_test_labels[:6], binary_test_labels_enc[:6])

# %%
# load model according to your choice.
model = get_model()

# %%
# get vgg19 features
train_features = model.predict([train_imgs_scaled])
test_features = model.predict([test_imgs_scaled])

# %%
# Train an svm on these features

train_flatten = []
for i in range(0, len(train_features)):
    train_flatten.append(train_features[i].flatten())
train_flatten = np.array(train_flatten)

test_flatten = []
for i in range(0, len(test_features)):
    test_flatten.append(test_features[i].flatten())
test_flatten = np.array(test_flatten)


# %%
# Create a svm Classifier
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovr', probability=True, degree=2,
              gamma='auto', kernel='sigmoid', verbose=False)

# Train the model using the training sets
clf.fit(train_flatten, binary_train_labels_enc)

# %%
import pickle
from  custom_classes import path
filename = path.save_models_path + 'SVM_on_binary_shalamar.sav'
# save model to disk
pickle.dump(clf, open(filename, 'wb'))
# load the model from disk
clf = pickle.load(open(filename, 'rb'))

#%%

# Predict the response for test dataset
y_pred = clf.predict(test_flatten)

# %%
cnn_preds_binary = y_pred
# # Making prediction labels for multiclass
prediction_labels_binary = le.inverse_transform(cnn_preds_binary)
cv_iml.get_f1_score(binary_test_labels, prediction_labels_binary, binary_classifcation=False, pos_label='malaria',
                    confusion_matrix_title="Leveraging Deep Learning Techniques for Malaria")

