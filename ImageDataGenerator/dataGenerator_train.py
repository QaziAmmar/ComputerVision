import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from custom_classes import cv_iml, path, predefine_models
from custom_classes.dataset_loader import *


save_weights_path = path.save_models_path + "BBBC041/basic_multiclass_densenest.h5"
data_set_base_path = path.dataset_path + "BBBC041/train_test_val/"
INPUT_SHAPE = (125, 125, 3)


# %%
# featurewise_center = calculate mean of input data.
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, featurewise_center=True, featurewise_std_normalization=True,
                                   horizontal_flip=True, vertical_flip=True)
datagen = ImageDataGenerator(rescale=1.0 / 255.0, featurewise_center=True, featurewise_std_normalization=True)
# prepare an iterators to scale images
train = train_datagen.flow_from_directory(data_set_base_path + "train/", class_mode="categorical", shuffle=True
                                          , batch_size=32, target_size=(125, 125))

val = datagen.flow_from_directory(data_set_base_path + "val/",class_mode="categorical", shuffle=True
                                          , batch_size=32, target_size=(125, 125))

test = datagen.flow_from_directory(data_set_base_path + "test/", class_mode="categorical", shuffle=True
                                          , batch_size=32, target_size=(125, 125))


# %%
# load model according to your choice.
# model = get_cnn_pretrained_weights_model(INPUT_SHAPE=INPUT_SHAPE, classes=number_of_classes)
# model = get_vgg_model(INPUT_SHAPE, classes=number_of_classes)
# model = predefine_models.get_resnet50(INPUT_SHAPE=INPUT_SHAPE, classes=5)
model = predefine_models.get_dennet121_transfer_learning(INPUT_SHAPE, 5)
# model.load_weights(path.save_models_path + "IML_binary_CNN_experimtents/vgg_2hidden_units"
#                                            "/pf_plus_vgg_binary_2hiddenUnit.h5")

# %%

import datetime

logdir = os.path.join(path.save_models_path,
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=2, min_lr=0.000001)
callbacks = [reduce_lr, tensorboard_callback]
# fit model with generator

history = model.fit(train, validation_data=val,
                              epochs=25,
                              verbose=1)

# %%
# This cell shows the accuracy and loss graph and save the model for next time usage.
model.load_weights(save_weights_path)

score = model.evaluate(test, verbose=1)

# model.save(save_weights_path)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# model.save('basic_cnn.h5')


# %%
import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(history.history['accuracy']) + 1
epoch_list = list(range(1, max_epoch))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
plt.show()

# %%
# Model Performance Evaluation
from sklearn.metrics import classification_report, confusion_matrix

basic_cnn_preds = model.predict(test, batch_size=512)
# # Making prediction lables for multiclass
basic_cnn_preds = basic_cnn_preds.argmax(1)
confusion_matrix(test.classes, basic_cnn_preds)
