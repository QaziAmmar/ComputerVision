# This file contained all models that are working perfectly.

import tensorflow as tf
from custom_classes import path
from keras.optimizers import Adam


# def get_basic_CNN_for_malaria(INPUT_SHAPE, save_weight_path=None, binary_classification=True, classes=2):
def get_basic_CNN_for_malaria(INPUT_SHAPE, binary_classification=True, classes=2):
    # Model 1: CNN from Scratch
    # This model server for both binary and multiple-class classification of malaria. if you want to do
    # multi-class classification then you need to false the binary_classification and need to mention
    # number of classes.
    """

    :param INPUT_SHAPE: Shape of input vector
    :param binary_classification: this is for binary classification in our case it is malaria and
    healthy cells.
    :param classes: Number of classes for different stage of malaria.
    :return: model of your required type.
    """
    # if save_weight_path is None:
    #     save_weight_path = path.save_models_path + "malaria_binaryclass_DrMoshin/basic_cnn.h5"

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

    if binary_classification:
        out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)
        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        out = tf.keras.layers.Dense(classes, activation='softmax')(drop2)

        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    # model.load_weights(save_weight_path)
    return model


def get_vgg_19_fine_tune(INPUT_SHAPE, save_weight_path=None):
    # Model 3: Fine-tuned Pre-trained Model with Image Augmentation.
    # only last 2 layers are fine-tuned on our dataset.
    if save_weight_path is None:
        save_weight_path = path.save_models_path + "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False,
                                            weights=save_weight_path,
                                            input_shape=INPUT_SHAPE)
    # Freeze the layers
    vgg.trainable = True

    set_trainable = False
    for layer in vgg.layers:
        if layer.name in ['block5_conv1', 'block4_conv1']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    base_vgg = vgg
    base_out = base_vgg.output
    pool_out = tf.keras.layers.Flatten()(base_out)
    hidden1 = tf.keras.layers.Dense(512, activation='relu')(pool_out)
    drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
    hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)

    model = tf.keras.Model(inputs=base_vgg.input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

    # print("Total Layers:", len(model.layers))
    # print("Total trainable layers:", sum([1 for l in model.layers if l.trainable]))


def get_vgg_19_transfer_learning(INPUT_SHAPE, save_weight_path=None):
    # Using VGG-19 Network for transfer learning in malaria detection taks.
    # this network use only change the last 3 layer of model according to our requirements.
    if save_weight_path is None:
        save_weight_path = path.save_models_path + "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False,
                                            weights=save_weight_path,
                                            input_shape=INPUT_SHAPE)
    vgg.trainable = False
    # Freeze the layers

    for layer in vgg.layers:
        layer.trainable = False

    base_vgg = vgg
    base_out = base_vgg.output
    pool_out = tf.keras.layers.Flatten()(base_out)
    hidden1 = tf.keras.layers.Dense(512, activation='relu')(pool_out)
    drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
    hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)
    model = tf.keras.Model(inputs=base_vgg.input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model


def get_resnet50_transferLearning(INPUT_SHAPE, save_weight_path=None):
    if save_weight_path is None:
        save_weight_path = path.save_models_path + "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

    resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=save_weight_path,
                                                       input_shape=INPUT_SHAPE)
    resnet50.trainable = False

    # Freeze the layers

    for layer in resnet50.layers:
        layer.trainable = False

    base_resnet50 = resnet50
    base_out = base_resnet50.output
    pool_out = tf.keras.layers.Flatten()(base_out)
    hidden1 = tf.keras.layers.Dense(512, activation='relu')(pool_out)
    drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
    hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)
    model = tf.keras.Model(inputs=base_resnet50.input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model


def get_dennet121_transfer_learning(INPUT_SHAPE, save_weight_path=None):
    if save_weight_path is None:
        save_weight_path = path.save_models_path + "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"

    # Model
    dn121 = tf.keras.applications.DenseNet121(weights=save_weight_path, include_top=False, input_shape=INPUT_SHAPE)
    dn121.trainable = False

    # Freeze the layers
    for layer in dn121.layers:
        layer.trainable = False

    base_dn121 = dn121
    base_out = base_dn121.output
    pool_out = tf.keras.layers.Flatten()(base_out)
    hidden1 = tf.keras.layers.Dense(512, activation='relu')(pool_out)
    drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
    hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)
    model = tf.keras.Model(inputs=base_dn121.input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return model


def basic_CNN_for_multiclass(INPUT_SHAPE, save_weight_path=None):
    # Model 1: CNN from Scratch
    if save_weight_path is None:
        save_weight_path = path.save_models_path + "malaria_binaryclass_DrMoshin/basic_cnn.h5"

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

    out = tf.keras.layers.Dense(6, activation='softmax')(drop2)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.load_weights(save_weight_path)
    return model
