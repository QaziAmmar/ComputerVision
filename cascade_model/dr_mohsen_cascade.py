# import the necessary packages
from tensorflow.keras.models import Model
from custom_classes import path
import tensorflow as tf


def get_vgg_for_cascade(INPUT_SHAPE):
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
    hidden1 = tf.keras.layers.Dense(16, activation='relu')(pool_out)
    drop1 = tf.keras.layers.Dropout(rate=0.1)(hidden1)
    hidden2 = tf.keras.layers.Dense(8, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.1)(hidden2)

    return base_vgg.input, drop2


def get_basic_cnn_for_cascade(INPUT_SHAPE):

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

    return inp, drop2


def build_multiclass_branch(inputs, numOfMulticlassLbls, finalAct="softmax", chanDim=-1):
    # utilize a lambda layer to convert the 3 channel input to a
    out = tf.keras.layers.Dense(numOfMulticlassLbls, activation=finalAct, name="multiclass_output")(inputs)
    # return the category prediction sub-network
    return out


def build_binary_branch(inputs, numOfBinaryClassLbls, finalAct="softmax", chanDim=-1):
    out = tf.keras.layers.Dense(numOfBinaryClassLbls, activation=finalAct, name="binary_output")(inputs)
    return out


def build_dr_mohsen_cascade(width, height, numOfMulticlassLbls, numOfBinaryClassLbls,
                            finalAct="softmax"):
    # initialize the input shape and channel dimension (this code
    # assumes you are using TensorFlow which utilizes channels
    # last ordering)
    INPUT_SHAPE = (height, width, 3)
    chanDim = -1

    inp, drop2 = get_vgg_for_cascade(INPUT_SHAPE)

    # now separate multiclass networks
    multiclassBranch = build_multiclass_branch(drop2,
                                               numOfMulticlassLbls, finalAct=finalAct, chanDim=chanDim)
    # now separate binary networks
    binaryClassBranch = build_binary_branch(drop2,
                                            numOfBinaryClassLbls, finalAct=finalAct, chanDim=chanDim)
    # create the model using our input (the batch of images) and
    # two separate outputs -- one for the clothing category
    # branch and another for the color branch, respectively
    model = Model(
        inputs=inp,
        outputs=[multiclassBranch, binaryClassBranch],
        name="dr_mohsen_cascade_model")
    # return the constructed network architecture
    return model
