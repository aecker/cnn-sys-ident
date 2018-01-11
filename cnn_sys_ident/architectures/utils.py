import numpy as np
import tensorflow as tf


def soft_threshold(x):
    return tf.log(tf.exp(x) + 1, name='soft_threshold')


def inv_soft_threshold(x):
    return np.log(np.exp(x) - 1)


def poisson(prediction, response):
    return tf.reduce_mean(tf.reduce_sum(prediction - response * tf.log(prediction + 1e-5), 1), name='poisson')
