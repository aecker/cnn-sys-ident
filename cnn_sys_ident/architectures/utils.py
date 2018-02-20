import numpy as np
import tensorflow as tf


def soft_threshold(x):
    return tf.log(tf.exp(x) + 1, name='soft_threshold')


def inv_soft_threshold(x):
    return np.log(np.exp(x) - 1)


def poisson(prediction, response):
    return tf.reduce_mean(tf.reduce_sum(
        prediction - response * tf.log(prediction + 1e-5), 1), name='poisson')


def rotate_weights(weights, num_rotations, first_layer=False):
    # shape = [filter_size, filter_size, num_rotations*num_inputs, num_outputs]
    shape = weights.get_shape().as_list()
    filter_size, _, num_inputs_total, num_outputs = shape
    num_inputs = num_inputs_total // num_rotations
    weights_flat = tf.reshape(weights, [filter_size, filter_size, num_inputs_total*num_outputs])
    weights_rotated = []
    for i in range(num_rotations):
        angle = i * 2 * np.pi / num_rotations
        w = tf.contrib.image.rotate(weights_flat, angle)
        w = tf.reshape(w, shape)
        if i and not first_layer:
            shift = num_inputs_total - i * num_inputs
            begin_a = [0, 0, shift, 0]
            size_a = [filter_size, filter_size, num_inputs_total-shift, num_outputs]
            begin_b = [0, 0, 0, 0]
            size_b = [filter_size, filter_size, shift, num_outputs]
            w = tf.concat([tf.slice(w, begin_a, size_a), tf.slice(w, begin_b, size_b)], axis=2)
        weights_rotated.append(w)
    weights_all_rotations = tf.concat(weights_rotated, axis=3, name='weights_all_rotations')
    return weights_all_rotations
