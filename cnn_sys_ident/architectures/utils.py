import numpy as np
import tensorflow as tf
from scipy import signal
from ..utils.hermite import rotation_matrix


def soft_threshold(x):
    return tf.log(tf.exp(x) + 1, name='soft_threshold')


def inv_soft_threshold(x):
    return np.log(np.exp(x) - 1)


def crop_responses(prediction, response):
    if len(prediction.shape) > 2:
        if type(response) is np.ndarray:
            response = response[...,response.shape[-2]-prediction.shape[-2]:,:]
        else:
            response = tf.slice(response,tf.shape(response)-tf.shape(prediction),tf.shape(prediction))
    return(response)


def poisson(prediction, response):
    response = crop_responses(prediction,response)
    return tf.reduce_mean(tf.reduce_sum(
        prediction - response * tf.log(prediction + 1e-5), -1), name='poisson') # test if -1 instead of 1 breaks sth.


def mean_sq_err(prediction, response):
    response = crop_responses(prediction,response)
    return tf.reduce_mean(tf.reduce_sum(
        (prediction - response)**2, -1), name='mean_sq_error')


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


def rotate_weights_hermite(H, desc, mu, coeffs, num_rotations, first_layer=False):
    num_coeffs, num_inputs_total, num_outputs = coeffs.shape.as_list()
    filter_size = int(H.shape[1])
    num_inputs = num_inputs_total // num_rotations
    weights_rotated = []
    for i in range(num_rotations):
        angle = i * 2 * np.pi / num_rotations
        R = rotation_matrix(desc, mu, angle)
        R = tf.constant(R, dtype=tf.float32, name='R')
        coeffs_rotated = tf.tensordot(R, coeffs, axes=[[1], [0]])
        w = tf.tensordot(H, coeffs_rotated, axes=[[0], [0]],
                         name='weights_rotated_{}'.format(i))
        if i and not first_layer:
            shift = num_inputs_total - i * num_inputs
            w = tf.concat([w[:,:,shift:,:], w[:,:,:shift,:]], axis=2)
        weights_rotated.append(w)
    weights_all_rotations = tf.concat(weights_rotated, axis=3)
    return weights_all_rotations


def downsample_weights(weights, factor=2):
    w = 0
    for i in range(factor):
        for j in range(factor):
            w += weights[i::factor,j::factor]
    return w


def envelope(w, k=51):
    t = np.linspace(-2.5, 2.5, k, endpoint=True)
    u, v = np.meshgrid(t, t)
    win = np.exp(-(u ** 2 + v ** 2) / 2) / k**2
    sub = lambda x: x - np.mean(x)
    return np.array([signal.convolve2d(sub(wi) ** 2, win, 'same') for wi in w])


def sta_init(x, y, k=51, alpha=10, max_val=0.1, sd=0.01):
    x = x[:,:,:,0]
    x = (x - x.mean()) / x.std()
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    w = np.tensordot(y, x, axes=[[0], [0]])
    e = envelope(w, k)
    e = (e / np.max(e, axis=(1, 2), keepdims=True)) ** alpha
    e *= max_val
    e += np.random.normal(size=e.shape) * sd
    return e
