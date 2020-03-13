from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os
from custom_classes import path, cv_iml
import glob

MODEL_NAME = 'dn121_transfer_cell_images_finetune_PlasmodiumVivax'
TRAIN_DIR = path.dataset_path + "IML_cell_images/train_p.v/"
VAL_DIR = path.dataset_path + "IML_cell_images/val/"
TEST_DIR = path.dataset_path + "IML_cell_images/test"
IMG_HEIGHT = 96
IMG_WIDTH = 96
IMG_CHANNELS = 3
BATCH_SIZE = 128
TEST_BATCH_SIZE = 100
NUM_CLASSES = 2
CHECKPOINT_DIR = path.save_models_path + 'checkpoints/'
CLASS_NAMES_LOC = path.dataset_path + 'IML_cell_images/class_names.txt'
LAYER_UNITS = (64, 16)
LR = 1e-4
EPOCHS = 10
NUM_SAMPLES = 5000
NUM_VAL_SAMPLES = 256
NUM_TRAINING_SAMPLES = NUM_SAMPLES - NUM_VAL_SAMPLES

save_weight_path = path.save_models_path + "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"
pretraind_weights_on_cell_images = path.save_models_path + "checkpoints/dn121_transfer_cell_images.h5"

# %%

def build_model(base, layer_units, num_classes):
    for layer in base.layers:
        layer.trainable = False

    x = base.output
    x = Flatten()(x)
    for num_units in layer_units:
        x = Dense(num_units, activation='relu')(x)

    predictions = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=predictions)
    return model


# Data Feed
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)
train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)
val_gen = datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)
test_gen = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=TEST_BATCH_SIZE
)
#%%
# NUM_SAMPLES = 5000
# NUM_VAL_SAMPLES = 256
# NUM_TRAINING_SAMPLES = NUM_SAMPLES - NUM_VAL_SAMPLES

#%%

# Class Names
class_names = dict()
with open(CLASS_NAMES_LOC, 'r') as f:
    for i in range(NUM_CLASSES):
        class_names[i] = f.readline().strip()

label_map = {value: class_names[value]
             for key, value in train_gen.class_indices.items()}

# Model
adam = Adam(lr=LR)
dn121 = DenseNet121(weights=save_weight_path, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
model = build_model(dn121, LAYER_UNITS, NUM_CLASSES)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.load_weights(pretraind_weights_on_cell_images)
#%%

# Checkpoints

if not os.path.isdir(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
checkpoint_path = CHECKPOINT_DIR + MODEL_NAME + '.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc',
                             verbose=1, save_weights_only=True, mode='max')
callbacks = [checkpoint]

# Check for pretrained weights
if os.path.isfile(checkpoint_path):
    model.load_weights(checkpoint_path)
else:
    # Train
    _ = model.fit_generator(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch= NUM_TRAINING_SAMPLES // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps= NUM_VAL_SAMPLES // BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks)

#%%

# Predict
test_batch_x, test_batch_y = test_gen.next()
pred_batch = model.predict(test_batch_x)

test_labels = np.argmax(test_batch_y, axis=1)
test_pred = np.argmax(pred_batch, axis=1)

test_acc = sum(test_labels == test_pred) / len(test_labels)
print('Accuracy: %.3f' % test_acc)

fig, axes = plt.subplots(3, 3, figsize=(10, 12))
for i in range(3):
    for j in range(3):
        ind = np.random.randint(0, len(test_labels))
        axes[i, j].imshow((test_batch_x[ind] + 2.5) / 5)
        axes[i, j].set_title('pred = %s, label = %s' % (label_map[test_pred[ind]], label_map[test_labels[ind]]))

plt.show()

#%%
cv_iml.get_f1_score(test_labels, test_pred, pos_label=1)

