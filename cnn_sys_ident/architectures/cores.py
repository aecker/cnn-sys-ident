import tensorflow as tf
from tensorflow.contrib import layers

from .utils import soft_threshold


ACTIVATION_FN = {
    'none': None,
    'relu': tf.nn.relu,
    'elu': tf.nn.elu,
    'soft': soft_threshold,
}


def smoothness_regularizer_2d(W, weight=1.0):
    with tf.variable_scope('smoothness'):
        lap = tf.constant([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]])
        lap = tf.expand_dims(tf.expand_dims(lap, 2), 3)
        num_filters = W.get_shape().as_list()[2]
        W_lap = tf.nn.depthwise_conv2d(tf.transpose(W, perm=[3, 0, 1, 2]),
                                       tf.tile(lap, [1, 1, num_filters, 1]),
                                       strides=[1, 1, 1, 1], padding='SAME')
        penalty = tf.reduce_sum(tf.reduce_sum(tf.square(W_lap), [1, 2, 3]) / tf.reduce_sum(tf.square(W), [0, 1, 2]))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('smoothness_regularizer_2d', penalty)
        return penalty


def group_sparsity_regularizer_2d(W, weight=1.0):
    with tf.variable_scope('group_sparsity'):
        penalty = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(W), [0, 1])))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('group_sparsity_regularizer_2d', penalty)
        return penalty


class StackedConv2dCore:
    def __init__(self,
                 base,
                 inputs,
                 filter_size=[13, 3],
                 num_filters=[16, 32],
                 stride=[1, 1],
                 rate=[1, 1],
                 padding=['VALID', 'VALID'],
                 activation_fn=['elu', 'elu'],
                 rel_smooth_weight=[1, 0],
                 rel_sparse_weight=[0, 1],
                 conv_smooth_weight=0.001,
                 conv_sparse_weight=0.001,
                 scope='core',
                 reuse=False,
                 **kwargs):
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                self.conv = []
                self.weights = []
                x = inputs
                for i, (fs, nf, st, rt, pd, fn, sm, sp) in enumerate(
                    zip(filter_size, num_filters, stride, rate, padding,
                        activation_fn, rel_smooth_weight, rel_sparse_weight)):
                    bn_params = {'decay': 0.98, 'is_training': base.is_training}
                    scope = 'conv{}'.format(i)
                    reg = lambda w: smoothness_regularizer_2d(w, conv_smooth_weight * sm) + \
                                    group_sparsity_regularizer_2d(w, conv_sparse_weight * sp)
                    x = layers.convolution2d(inputs=x,
                                             num_outputs=int(nf),
                                             kernel_size=int(fs),
                                             stride=int(st),
                                             rate=int(rt),
                                             padding=pd,
                                             activation_fn=ACTIVATION_FN[fn],
                                             normalizer_fn=layers.batch_norm,
                                             normalizer_params=bn_params,
                                             weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                             weights_regularizer=reg,
                                             scope=scope)
                    with tf.variable_scope(scope, reuse=True):
                        weights = tf.get_variable('weights')
                    self.weights.append(weights)
                    self.conv.append(x)

                self.output = tf.identity(self.conv[-1], name='output')
