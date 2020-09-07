import tensorflow as tf
import numpy as np

a = tf.constant([[0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0, 0]])

proto_tensor = tf.make_tensor_proto(a)  # convert `tensor a` to a proto tensor
nd_array = tf.make_ndarray(proto_tensor)
indexs = np.argmax(nd_array, axis=1)
binary_index = (np.equal(indexs, 2)).astype(int)
binary_index = np.logical_not(binary_index).astype(int)
