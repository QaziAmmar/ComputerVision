import pickle
import tensorflow as tf
from collections import Counter
from custom_classes import path
from custom_classes.dataset_loader import *
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from cascade_model.dr_mohsen_cascade import build_dr_mohsen_cascade
from cascade_model import fashionNet
import matplotlib.pyplot as plt


def convert_multiclass_lbl_to_binary_lbl(multiclass_labels):
    binary_label = []
    for tempLabel in multiclass_labels:
        if tempLabel == "healthy":
            binary_label.append("healthy")
        else:
            binary_label.append("malaria")
    return binary_label


# Achieving peak performance requires an efficient input pipeline that delivers data
# for the next step before the current step has finished
AUTOTUNE = tf.data.experimental.AUTOTUNE

EPOCHS = 50
INIT_LR = 1e-3
BS = 16

# %% # grab the image paths and randomly shuffle them
save_model_path = path.save_models_path + "BBBC041/loss_test.h5"

data_dir = path.dataset_path + "BBBC041/loss_test2_folder/"
# save_weights_path = path.save_models_path + "BBBC041/basic_cnn_loss_test.h5"

train_imgs_scaled, train_labels, test_imgs_scaled, test_labels, val_imgs_scaled, val_labels = \
    load_train_test_val_images_from(data_dir, file_extension=".png", show_train_data=True)

# %% make train images that fit on batch size
# extra_train_images = len(train_labels) % BS
# train_imgs_scaled = train_imgs_scaled[extra_train_images:]
# train_labels = train_labels[extra_train_images:]
# extra_val_images = len(val_labels) % BS
# val_imgs_scaled = val_imgs_scaled[extra_val_images:]
# val_labels = val_labels[extra_val_images:]

# %%# show the number of train, test and val files in dataset folder
print('Train:', Counter(train_labels), '\nVal', Counter(val_labels), '\nTest', Counter(test_labels))
# %%
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions

# IMAGE_DIMS = (96, 96, 3)

# Assigning value to multi-class labels.
train_multiclass_labels = train_labels
test_multiclass_labels = test_labels
val_multiclass_labels = val_labels

#  converting labels to binary class labels.
train_binary_labels = convert_multiclass_lbl_to_binary_lbl(train_labels)
test_binary_labels = convert_multiclass_lbl_to_binary_lbl(test_labels)
val_binary_labels = convert_multiclass_lbl_to_binary_lbl(val_labels)

# convert binary_labels to numpy arrays.
train_binary_labels = np.array(train_binary_labels)
test_binary_labels = np.array(test_binary_labels)
val_binary_labels = np.array(val_binary_labels)

num_of_multiclass_labels = len(list(np.unique(train_multiclass_labels)))
num_of_binaryClass_labels = len(list(np.unique(train_binary_labels)))

# %%
# convert the labels to one hot labels.

binaryclass_LE = LabelEncoder()
multiclass_LE = LabelEncoder()

binaryclass_LE.fit(train_binary_labels)
multiclass_LE.fit(train_multiclass_labels)

# converting multiclass labels to one hot labels.
train_labels_enc_binary = binaryclass_LE.transform(train_binary_labels)
val_labels_enc_binary = binaryclass_LE.transform(val_binary_labels)
test_labels_enc_binary = binaryclass_LE.transform(test_binary_labels)

train_labels_enc_binary = to_categorical(train_labels_enc_binary, num_classes=num_of_binaryClass_labels)
val_labels_enc_binary = to_categorical(val_labels_enc_binary, num_classes=num_of_binaryClass_labels)
test_labels_enc_binary = to_categorical(test_labels_enc_binary, num_classes=num_of_binaryClass_labels)

print(train_binary_labels[:6], train_labels_enc_binary[:6])

# converting binary labels to one hot labels.
train_labels_enc_multiclass = multiclass_LE.transform(train_multiclass_labels)
val_labels_enc_multiclass = multiclass_LE.transform(val_multiclass_labels)
test_labels_enc_multiclass = multiclass_LE.transform(test_multiclass_labels)

train_labels_enc_multiclass = to_categorical(train_labels_enc_multiclass, num_classes=num_of_multiclass_labels)
val_labels_enc_multiclass = to_categorical(val_labels_enc_multiclass, num_classes=num_of_multiclass_labels)
test_labels_enc_multiclass = to_categorical(test_labels_enc_multiclass, num_classes=num_of_multiclass_labels)

print(train_multiclass_labels[:6], train_labels_enc_multiclass[:6])

# %%

# initialize our FashionNet multi-output network
model = build_dr_mohsen_cascade(125, 125,
                                numOfMulticlassLbls=num_of_multiclass_labels,
                                numOfBinaryClassLbls=num_of_binaryClass_labels,
                                finalAct="softmax")

# model = fashionNet.build(125, 125,
#                          numOfMulticlassLbls=num_of_multiclass_labels,
#                          numOfBinaryClassLbls=num_of_binaryClass_labels,
#                          finalAct="softmax")


# def combined_loss(y_true, y_pred):
#     # tf.shape(y_pred)
#     multi_pred, binary_pred = y_pred
#     multi_true, binary_true = y_true
#     multiloss = tf.losses.categorical_crossentropy(y_true, y_pred)
#     binary_loss = tf.losses.binary_crossentropy(binary_true, binary_pred)
#
#     loss = multiloss + binary_loss
#     return multiloss


def binary_loss(y_true, y_pred):
    loss = tf.losses.categorical_crossentropy(y_true, y_pred)
    return loss


def multi_loss(y_true, y_pred):
    my_loss = []
    for i in range(BS - 1):  # this will run in the range of the batch
        if tf.argmax(y_pred[i]) == 1:  # healthy
            # append to the my_loss
            tower_loss = tf.constant(0.0)
        else:
            # malaria case
            tower_loss = tf.losses.categorical_crossentropy(y_true[i], y_pred[i])
            # print(tower_loss.eval())
        my_loss.append(tower_loss)
    # convert the my_loss to tensor
    my_loss = tf.math.reduce_sum(my_loss)
    my_loss = tf.math.divide(my_loss, BS)
    return my_loss


losses = {
    "binary_output": binary_loss,
    # https://github.com/keras-team/keras/issues/2488
    # combining 2 loss together.
    "multiclass_output": [multi_loss, {"binary_output": binary_loss}],  # how can I sum multi_loss + binary_loss
}

lossWeights = {"multiclass_output": 1.0, "binary_output": 1.0}
# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
              metrics=["accuracy"])

# %%

# train the network to perform multi-output classification
H = model.fit(x=train_imgs_scaled,
              y={"multiclass_output": train_labels_enc_multiclass, "binary_output": train_labels_enc_binary},
              validation_data=(val_imgs_scaled,
                               {"multiclass_output": val_labels_enc_multiclass,
                                "binary_output": val_labels_enc_binary}),
              epochs=EPOCHS,
              verbose=1)
# save the model to disk
print("[INFO] serializing network...")
model.save(save_model_path)

# %%
# save the category binarizer to disk
path_to_mutliclass_label = "/home/iml/Desktop/qazi/Model_Result_Dataset/SavedModel/label_encoder_path/multiclass_lb.pickle"
path_to_binary_label = "/home/iml/Desktop/qazi/Model_Result_Dataset/SavedModel/label_encoder_path/binaryclass_lb.pickle"

print("[INFO] serializing category label binarizer...")
f = open(path_to_mutliclass_label, "wb")
f.write(pickle.dumps(multiclass_LE))
f.close()
# save the color binarizer to disk
print("[INFO] serializing color label binarizer...")
f = open(path_to_binary_label, "wb")
f.write(pickle.dumps(binaryclass_LE))
f.close()

# %%

# plot the total loss, category loss, and color loss
lossNames = ["loss", "multiclass_output_loss", "binary_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
# loop over the loss names
for (i, l) in enumerate(lossNames):
    # plot the loss for both the training and validation data
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
               label="val_" + l)
    ax[i].legend()
# save the losses figure
plt.tight_layout()
plt.show()
# plt.savefig("{}_losses.png".format(args["plot"]))
plt.close()

# %%

# create a new figure for the accuracies
accuracyNames = ["multiclass_output_accuracy", "binary_output_accuracy"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))
# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
    # plot the loss for both the training and validation data
    ax[i].set_title("Accuracy for {}".format(l))
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Accuracy")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
               label="val_" + l)
    ax[i].legend()
# save the accuracies figure
plt.tight_layout()
plt.show()
# plt.savefig("{}_accs.png".format(args["plot"]))
plt.close()

# %%
# evulate test
# classify the input image using Keras' multi-output functionality
print("[INFO] classifying image...")
(multiclass_Proba, binary_Proba) = model.predict(test_imgs_scaled)
# find indexes of both the category and color outputs with the
# largest probabilities, then determine the corresponding class
# labels

multiclass_preds = multiclass_Proba.argmax(1)
multiclass_prediction_labels = multiclass_LE.inverse_transform(multiclass_preds)

binary_preds = binary_Proba.argmax(1)
binary_prediction_labels = binaryclass_LE.inverse_transform(binary_preds)

cv_iml.get_f1_score(test_labels, multiclass_prediction_labels, binary_classifcation=False, pos_label='malaria',
                    plot_confusion_matrix=True)
