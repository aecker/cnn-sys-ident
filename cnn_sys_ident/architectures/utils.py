import tensorflow as tf


def soft_threshold(x):
    return tf.log(tf.exp(x) + 1)
