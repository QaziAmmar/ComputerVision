# %%
# //  Created by Qazi Ammar Arshad on 22/05/2021.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.


from collections import Counter
from custom_classes import path, predefine_models, cv_iml
from custom_classes.dataset_loader import *
import tensorflow as tf
from collections import Counter
from custom_classes import path, predefine_models
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt


def getModel(INPUT_SHAPE, classes=1):
    """
    Model that is used in this paper.
    Leveraging Deep Learning Techniques for Malaria Parasite Detection Using Mobile Application.
    https://www.hindawi.com/journals/wcmc/2020/8895429/

    :param INPUT_SHAPE: Shape of input vector
    :param binary_classification: this is for binary classification in our case it is malaria and
    healthy cells.
    :param classes: Number of classes for different stage of malaria.
    :return: model tensorflow model
    """

    inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same')(inp)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    batch1 = BatchNormalization()(pool1)
    drop1 = tf.keras.layers.Dropout(rate=0.15)(batch1)

    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(drop1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    batch2 = BatchNormalization()(pool2)
    drop2 = tf.keras.layers.Dropout(rate=0.15)(batch2)

    conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(drop2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    batch3 = BatchNormalization()(pool3)
    drop3 = tf.keras.layers.Dropout(rate=0.15)(batch3)

    GAP = tf.keras.layers.GlobalAveragePooling2D()(drop3)
    drop4 = tf.keras.layers.Dropout(rate=0.15)(GAP)

    hidden1 = tf.keras.layers.Dense(1000, activation='relu')(drop4)
    drop5 = tf.keras.layers.Dropout(rate=0.015)(hidden1)

    out = tf.keras.layers.Dense(number_of_binary_classes, activation='softmax')(drop5)
    model = tf.keras.Model(inputs=inp, outputs=out)
    opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    model.compile(optimizer=opt, loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])

    model.summary()
    return model


# %%
save_weights_path = path.save_models_path + "shamalar_data/papers/leveraging_deep_learning_techniques.h5"

data_set_base_path = path.dataset_path + "shalamar_training_data/train_test_separate_balanced/"

INPUT_SHAPE = (125, 125, 3)
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


binary_train_labels = list(map(_replaceitem, train_labels))
binary_test_labels = list(map(_replaceitem, test_labels))
binary_val_labels = list(map(_replaceitem, val_labels))

# binary_train_labels = train_labels
# binary_test_labels = test_labels
# binary_val_labels = val_labels

number_of_binary_classes = len(np.unique(binary_train_labels))

BATCH_SIZE = 64

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(binary_test_labels)

binary_train_labels_enc = le.transform(binary_train_labels)
binary_train_labels_enc = to_categorical(binary_train_labels_enc, num_classes=number_of_binary_classes)

binary_test_labels_enc = le.transform(binary_test_labels)
binary_test_labels_enc = to_categorical(binary_test_labels_enc, num_classes=number_of_binary_classes)

binary_val_labels_enc = le.transform(binary_val_labels)
binary_val_labels_enc = to_categorical(binary_val_labels_enc, num_classes=number_of_binary_classes)

print(binary_test_labels[:6], binary_test_labels_enc[:6])

# %%
# load model according to your choice.
model = getModel(INPUT_SHAPE, number_of_binary_classes)

# %%
# Model training
import datetime

logdir = os.path.join(path.save_models_path,
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=2, min_lr=0.000001)
callbacks = [reduce_lr]

# if os.path.isfile(load_weights_path):
#     model.load_weights(load_weights_path)


history = model.fit(x=train_imgs_scaled, y=binary_train_labels_enc,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(val_imgs_scaled, binary_val_labels_enc),
                    callbacks=callbacks,
                    verbose=1)

# %%
# This cell shows the accuracy and loss graph and save the model for next time usage.
# model.save(save_weights_path)
model.load_weights(save_weights_path)
score = model.evaluate(test_imgs_scaled, binary_test_labels_enc)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# model.save('basic_cnn.h5')


# %%
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
cnn_preds_binary = model.predict(test_imgs_scaled, batch_size=512)
# # Making prediction lables for multiclass
cnn_preds_binary = cnn_preds_binary.argmax(1)
prediction_labels_binary = le.inverse_transform(cnn_preds_binary)
cv_iml.get_f1_score(binary_test_labels, prediction_labels_binary, binary_classifcation=True, pos_label='malaria',
                    confusion_matrix_title="Leveraging Deep Learning Techniques for Malaria")
