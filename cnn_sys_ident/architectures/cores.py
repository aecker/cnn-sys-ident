import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

from .utils import soft_threshold, rotate_weights, rotate_weights_hermite, \
    downsample_weights
from ..utils.hermite import hermite_2d


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
        penalty = tf.reduce_sum(tf.reduce_sum(tf.square(W_lap), [1, 2, 3]) / \
                                (1e-8 + tf.reduce_sum(tf.square(W), [0, 1, 2])))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('smoothness_regularizer_2d', penalty)
        return penalty

def group_sparsity_regularizer_2d(W, weight=1.0):
    with tf.variable_scope('group_sparsity'):
        penalty = tf.reduce_sum(
            tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(W), [0, 1])), 0) / \
            tf.sqrt(1e-8 + tf.reduce_sum(tf.square(W), [0, 1, 2])))
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
                                             weights_initializer=tf.truncated_normal_initializer(
                                                 mean=0.0, stddev=0.01),
                                             weights_regularizer=reg,
                                             scope=scope)
                    with tf.variable_scope(scope, reuse=True):
                        weights = tf.get_variable('weights')
                    self.weights.append(weights)
                    self.conv.append(x)

                self.output = tf.identity(self.conv[-1], name='output')


class StackedRotEquiConv2dCore:
    def __init__(self,
                 base,
                 inputs,
                 filter_size=[13, 5, 5],
                 num_filters=[8, 16, 32],
                 num_rotations=8,
                 shared_biases=False,
                 stride=[1, 1, 1],
                 rate=[1, 1, 1],
                 padding=['VALID', 'VALID', 'VALID'],
                 activation_fn=['soft', 'soft', 'soft'],
                 rel_smooth_weight=[1, 0, 0],
                 rel_sparse_weight=[0, 1, 1],
                 conv_smooth_weight=0.001,
                 conv_sparse_weight=0.001,
                 scope='core',
                 reuse=False,
                 fused_bn=True,
                 **kwargs):
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                conv = inputs
                self.conv = []
                self.weights = []
                self.weights_all = []
                nf_in = 1
                for i, (fs, nf_out, st, rt, pd, fn, sm, sp) in enumerate(
                    zip(filter_size, num_filters, stride, rate, padding,
                        activation_fn, rel_smooth_weight, rel_sparse_weight)):
                    with tf.variable_scope('conv{:d}'.format(i+1)):
                        weights = tf.get_variable(
                            'weights',
                            shape=[fs, fs, nf_in, nf_out],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
                        self.weights.append(weights)
                        weights_all_rotations = rotate_weights(weights, num_rotations, first_layer=(not i))
                        self.weights_all.append(weights_all_rotations)

                        # apply regularization to all rotated versions
                        self.smooth_reg = smoothness_regularizer_2d(
                            weights_all_rotations, conv_smooth_weight * sm)
                        self.sparse_reg = group_sparsity_regularizer_2d(
                            weights_all_rotations, conv_sparse_weight * sp)
                        tf.losses.add_loss(
                            self.smooth_reg,
                            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
                        tf.losses.add_loss(
                            self.sparse_reg,
                            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

                        # conv = tf.nn.conv2d(conv, weights_all_rotations, strides=[1, st, st, 1],
                        #                     dilations=[1, rt, rt, 1], padding=pd)
                        assert rt == 1, 'Dilation not supported. Need to upgrade to TF 1.5'
                        conv = tf.nn.conv2d(conv, weights_all_rotations,
                                            strides=[1, st, st, 1], padding=pd)
                        if shared_biases:
                            s = conv.shape.as_list()
                            conv = tf.reshape(conv, [-1] + s[1:3] + [num_rotations, nf_out])

                        enable_scale = i < len(filter_size) - 1
                        conv = layers.batch_norm(conv, center=True, scale=enable_scale, decay=0.95,
                                                 is_training=base.is_training, fused=fused_bn)
                        if shared_biases:
                            conv = tf.reshape(conv, [-1] + s[1:])
                        if not (fn == 'none'):
                            conv = ACTIVATION_FN[fn](conv)
                        self.conv.append(conv)
                        nf_in = nf_out * num_rotations

                self.output = tf.identity(self.conv[-1], name='output')


class StackedRotEquiHermiteConv2dCore:
    def __init__(self,
                 base,
                 inputs,
                 num_rotations=8,
                 upsampling=2,
                 shared_biases=False,
                 filter_size=[13, 5, 5],
                 num_filters=[8, 16, 32],
                 stride=[1, 1, 1],
                 rate=[1, 1, 1],
                 padding=['VALID', 'VALID', 'VALID'],
                 activation_fn=['soft', 'soft', 'soft'],
                 rel_smooth_weight=[1, 0, 0],
                 rel_sparse_weight=[0, 1, 1],
                 conv_smooth_weight=0.001,
                 conv_sparse_weight=0.001,
                 scope='core',
                 reuse=False,
                 fused_bn=True,
                 **kwargs):
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                conv = inputs
                self.conv = []
                self.weights = []
                self.weights_all = []
                nf_in = 1
                for i, (fs, nf_out, st, rt, pd, fn, sm, sp) in enumerate(
                    zip(filter_size, num_filters, stride, rate, padding,
                        activation_fn, rel_smooth_weight, rel_sparse_weight)):
                    with tf.variable_scope('conv{:d}'.format(i+1)):
                        H, desc, mu = hermite_2d(fs, fs*upsampling, 2*np.sqrt(fs))
                        H = tf.constant(H, dtype=tf.float32, name='hermite_basis')
                        n_coeffs = fs * (fs + 1) // 2
                        coeffs = tf.get_variable(
                            'coeffs',
                            shape=[n_coeffs, nf_in, nf_out],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
                        weights = tf.tensordot(H, coeffs, axes=[[0], [0]])
                        weights = tf.identity(downsample_weights(weights, upsampling),
                                              name='weights')
                        self.weights.append(weights)
                        weights_all_rotations = rotate_weights_hermite(
                            H, desc, mu, coeffs, num_rotations, first_layer=(not i))
                        weights_all_rotations = tf.identity(
                            downsample_weights(weights_all_rotations, upsampling),
                            name='weights_all_rotations')
                        self.weights_all.append(weights_all_rotations)

                        # apply regularization to all rotated versions
                        self.smooth_reg = smoothness_regularizer_2d(
                            weights_all_rotations, conv_smooth_weight * sm)
                        self.sparse_reg = group_sparsity_regularizer_2d(
                            weights_all_rotations, conv_sparse_weight * sp)
                        tf.losses.add_loss(
                            self.smooth_reg,
                            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
                        tf.losses.add_loss(
                            self.sparse_reg,
                            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

                        # conv = tf.nn.conv2d(conv, weights_all_rotations, strides=[1, st, st, 1],
                        #                     dilations=[1, rt, rt, 1], padding=pd)
                        assert rt == 1, 'Dilation not supported. Need to upgrade to TF 1.5'
                        conv = tf.nn.conv2d(conv, weights_all_rotations,
                                            strides=[1, st, st, 1], padding=pd)
                        if shared_biases:
                            s = conv.shape.as_list()
                            conv = tf.reshape(conv, [-1] + s[1:3] + [num_rotations, nf_out])

                        enable_scale = i < len(filter_size) - 1
                        conv = layers.batch_norm(conv, center=True, scale=enable_scale, decay=0.9,
                                                 is_training=base.is_training, fused=fused_bn)
                        if shared_biases:
                            conv = tf.reshape(conv, [-1] + s[1:])
                        if not (fn == 'none'):
                            conv = ACTIVATION_FN[fn](conv)
                        self.conv.append(conv)
                        nf_in = nf_out * num_rotations

                self.output = tf.identity(self.conv[-1], name='output')
