# import the necessary packages
from tensorflow.keras.models import load_model
import pickle
from custom_classes import path
from cascade_model import dr_mohsen_cascade, fashionNet
from custom_classes.dataset_loader import *


def convert_multiclass_lbl_to_binary_lbl(multiclass_labels):
    binary_label = []
    for tempLabel in multiclass_labels:
        if tempLabel == "healthy":
            binary_label.append("healthy")
        else:
            binary_label.append("malaria")
    return binary_label


save_model_path = path.save_models_path + "BBBC041/loss_test.h5"
path_to_binary_label = path.save_models_path + "label_encoder_path/binaryclass_lb.pickle"
path_to_mutliclass_label = path.save_models_path + "label_encoder_path/multiclass_lb.pickle"

data_dir = path.dataset_path + "BBBC041/BBBC041_train_test_separate/"

train_imgs_scaled, train_labels, test_imgs_scaled, test_labels, val_imgs_scaled, val_labels = \
    load_train_test_val_images_from(data_dir, file_extension=".png", show_train_data=True)

# %%

#  converting labels to binary class labels.
train_binary_labels = convert_multiclass_lbl_to_binary_lbl(train_labels)
test_binary_labels = convert_multiclass_lbl_to_binary_lbl(test_labels)
val_binary_labels = convert_multiclass_lbl_to_binary_lbl(val_labels)

# convert binary_labels to numpy arrays.
train_binary_labels = np.array(train_binary_labels)
test_binary_labels = np.array(test_binary_labels)
val_binary_labels = np.array(val_binary_labels)

num_of_multiclass_labels = len(list(np.unique(train_labels)))
num_of_binaryClass_labels = len(list(np.unique(train_binary_labels)))

# %%

# load the trained convolutional neural network from disk, followed
# by the category and color label binarizers, respectively
print("[INFO] loading network...")
model = dr_mohsen_cascade.build_dr_mohsen_cascade(125, 125,
                                                  numOfMulticlassLbls=num_of_multiclass_labels,
                                                  numOfBinaryClassLbls=num_of_binaryClass_labels,
                                                  finalAct="softmax")
# model = fashionNet.build(125, 125,
#                          numOfMulticlassLbls=num_of_multiclass_labels,
#                          numOfBinaryClassLbls=num_of_binaryClass_labels,
#                          finalAct="softmax")
model.load_weights(save_model_path)
multiclassLE = pickle.loads(open(path_to_mutliclass_label, "rb").read())
binaryLB = pickle.loads(open(path_to_binary_label, "rb").read())

# %%


# classify the input image using Keras' multi-output functionality
print("[INFO] classifying image...")
(multiclass_Proba, binary_Proba) = model.predict(test_imgs_scaled)
# find indexes of both the category and color outputs with the
# largest probabilities, then determine the corresponding class
# labels

multiclass_preds = multiclass_Proba.argmax(1)
multiclass_prediction_labels = multiclassLE.inverse_transform(multiclass_preds)

cv_iml.get_f1_score(test_labels, multiclass_prediction_labels, binary_classifcation=False, pos_label='malaria',
                    plot_confusion_matrix=True)


# binary_preds = binary_Proba.argmax(1)
# binary_prediction_labels = binaryLB.inverse_transform(binary_preds)
#
# cv_iml.get_f1_score(test_binary_labels, binary_prediction_labels, binary_classifcation=True, pos_label='malaria',
#                     plot_confusion_matrix=True)
