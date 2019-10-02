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


def smoothness_regularizer_1d(W, weight=1.0):
    with tf.variable_scope('smoothness'):
        lap = tf.constant([-1, 2, -1], shape=(3, 1, 1, 1), dtype=tf.float32)
        W = tf.expand_dims(W, 1)
        num_filters = W.get_shape().as_list()[2]
        W_lap = tf.nn.depthwise_conv2d(tf.transpose(W, perm=[3, 0, 1, 2]),
                                       tf.tile(lap, [1, 1, num_filters, 1]),
                                       strides=[1, 1, 1, 1], padding='SAME')
        penalty = tf.reduce_sum(tf.reduce_sum(tf.square(W_lap), [1, 2, 3]) / \
                                (1e-8 + tf.reduce_sum(tf.square(W), [0, 1, 2])))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('smoothness_regularizer_1d', penalty)
        return penalty

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
    
def smoothness_regularizer_3d(W, spatial_weight=1.0, temporal_weight=1.0):
    shape = W.shape
    penalty = 0
    if shape[1] > 1 and shape[2] > 1: # spatial kernel
        for i in range(shape[0]):
            penalty += smoothness_regularizer_2d(
                W[i], weight=spatial_weight)
    if shape[0] > 1: # temporal kernel
        for i in range(shape[1]):
            for j in range(shape[2]):
                penalty += smoothness_regularizer_1d(
                    W[:,i,j,:,:], weight=temporal_weight)
    tf.add_to_collection('smoothness_regularizer_3d', penalty)
    return penalty

def group_sparsity_regularizer_1d(W, weight=1.0):
    with tf.variable_scope('group_sparsity'):
        W = tf.expand_dims(W,1)
        penalty = tf.reduce_sum(
            tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(W), [0, 1])), 0) / \
            tf.sqrt(1e-8 + tf.reduce_sum(tf.square(W), [0, 1, 2])))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('group_sparsity_regularizer_1d', penalty)
        return penalty

def group_sparsity_regularizer_2d(W, weight=1.0):
    with tf.variable_scope('group_sparsity'):
        penalty = tf.reduce_sum(
            tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(W), [0, 1])), 0) / \
            tf.sqrt(1e-8 + tf.reduce_sum(tf.square(W), [0, 1, 2])))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('group_sparsity_regularizer_2d', penalty)
        return penalty

def group_sparsity_regularizer_3d(W, spatial_weight=1.0, temporal_weight=1.0):
    shape = W.shape
    penalty = 0
    if shape[1] > 1 and shape[2] > 1: # spatial kernel
        for i in range(shape[0]):
            penalty += group_sparsity_regularizer_2d(
                W[i], weight=spatial_weight)
    if shape[0] > 1: # temporal kernel
        for i in range(shape[1]):
            for j in range(shape[2]):
                penalty += group_sparsity_regularizer_1d(
                    W[:,i,j,:,:], weight=temporal_weight)
    tf.add_to_collection('group_sparsity_regularizer_3d', penalty)
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


class MultiScanCore:
    def __init__(self,
                 base,
                 data,
                 inputs,
                 core_type,
                 scope='core',
                 **kwargs):
        # Make one core per scan (with weight sharing for kernels)
        self.cores = []
        for i in range(inputs.shape[0]):
#             self.cores.append(core_type(
#                 base, data, inputs[i], reuse=i>0, scope = '{}_{}'.format(scope, i), **kwargs))
            self.cores.append(core_type(
                base, data, inputs[i], reuse=i>0, scope = scope, **kwargs))
        self.output = [c.output for c in self.cores]


class StackedConv3dCore:
    def __init__(self,
                 base,
                 data,
                 inputs,
                 filter_size_spatial=[13, 13],
                 filter_size_temporal=[20, 20],
                 num_filters=[8, 16],
                 stride=[1, 1],
                 rate=[1, 1],
                 padding=['VALID', 'VALID'],
                 activation_fn=['elu', 'none'],
                 rel_smooth_weight=[1, 1],
                 rel_sparse_weight=[0, 1],
                 conv_smooth_weight_spatial=0.001,
                 conv_smooth_weight_temporal=0.001,
                 conv_sparse_weight=0.001,
                 scope='core',
                 reuse=False,
                 **kwargs):
        #calculate resulting steps in history that contribute to output
        self.steps_hist = 1+np.sum(filter_size_temporal)-len(filter_size_temporal)
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                self.conv = []
                self.weights = []
                x = inputs
                for i, (fs_s, fs_t, nf, st, rt, pd, fn, sm, sp) in enumerate(
                    zip(filter_size_spatial, filter_size_temporal, num_filters,
                        stride, rate, padding, activation_fn, rel_smooth_weight,
                        rel_sparse_weight)):
                    bn_params = {'decay': 0.98, 'is_training': base.is_training}
                    scope = 'conv{}'.format(i)
                    def reg(w):
                        penalty = smoothness_regularizer_3d(
                            w, conv_smooth_weight_spatial * sm,
                            conv_smooth_weight_temporal * sm)
                        penalty += group_sparsity_regularizer_3d(
                            w, conv_sparse_weight * sp)
                        return penalty

                    x = layers.conv3d(
                            inputs=x,
                            num_outputs=int(nf),
                            kernel_size=[fs_t,fs_s,fs_s],
                            stride=int(st),
                            padding=pd,
                            data_format='NDHWC',
                            activation_fn=ACTIVATION_FN[fn],
                            normalizer_fn=layers.batch_norm,
                            normalizer_params=bn_params,
                            weights_initializer=tf.truncated_normal_initializer(
                                mean=0.0, stddev=0.01),
                            weights_regularizer=reg,
                            scope=scope
                            )
                    with tf.variable_scope(scope, reuse=True):
                        weights = tf.get_variable('weights')
                    self.weights.append(weights)
                    self.conv.append(x)

                self.output = tf.identity(self.conv[-1], name='output')


class StackedFactorizedConv3dCore:
    def __init__(self,
                 base,
                 data,
                 inputs,
                 filter_size_spatial=[13, 13],
                 filter_size_temporal=[20, 20],
                 num_filters=[8, 16],
                 stride=[1, 1],
                 rate=[1, 1],
                 padding=['VALID', 'VALID'],
                 nonzero_padding=False,
                 padding_constant=0,
                 activation_fn=['elu', 'none'],
                 rel_smooth_weight=[1, 0],
                 rel_sparse_weight=[0, 1],
                 conv_smooth_weight_spatial=0.001,
                 conv_smooth_weight_temporal=0.001,
                 conv_sparse_weight=0.001,
                 scope='core',
                 reuse=False,
                 **kwargs):
        #calculate resulting steps in history that contribute to output
        self.steps_hist = 1+np.sum(filter_size_temporal)-len(filter_size_temporal)
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                self.conv = []
                self.weights_temporal = []
                self.weights_spatial = []
                self.weights_scan_bias = []
                self.weights_scan_scale = []
                # Put scans into batch dimension: SBDHWC -> (S*B)DHWC
                # input_list = [i[0] for i in tf.split(inputs, data.input_shape[0])]
                # x = tf.concat(input_list, 0, name = 'put_scan_in_batch')
                x = inputs
                for i, (fs_s, fs_t, nf, st, rt, pd, fn, sm, sp) in enumerate(
                        zip(filter_size_spatial, filter_size_temporal, num_filters,
                            stride, rate, padding, activation_fn, rel_smooth_weight,
                            rel_sparse_weight)):
                    with tf.variable_scope('conv{}'.format(i), reuse=tf.AUTO_REUSE):
                        # temporal
                        #filter: [filter_depth, filter_height, filter_width, in_channels, out_channels]
                        self.weights_temporal.append(tf.get_variable(
                                           name='weights_temporal_{}'.format(i),
                                           shape=[fs_t,1,1,x.shape[-1],int(nf)],
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)))

                        # spatial
                        self.weights_spatial.append(tf.get_variable(
                                          name='weights_spatial_{}'.format(i),
                                          shape=[1,fs_s,fs_s,x.shape[-1],int(nf)],
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)))

                    # combined
                    self.W_combined = tf.einsum('dabio,chwio->dhwio',
                                           self.weights_temporal[-1],
                                           self.weights_spatial[-1],
                                           name='weights_combined_{}'.format(i)
                    )
                    
#                     weights_mean=tf.math.reduce_mean(self.W_combined,axis=[0,1,2,3])
#                     self.W_combined = tf.subtract(self.W_combined,weights_mean)
#                     self.W_combined = tf.divide(self.W_combined,100*tf.math.reduce_std(self.W_combined,axis=[0,1,2,3]))
#                     self.W_combined = tf.add(self.W_combined,weights_mean)

                    if nonzero_padding:
                        x = tf.pad(x,[[0,0],[0,0],[fs_s//2,fs_s//2],[fs_s//2,fs_s//2],[0,0]],mode='CONSTANT',constant_values=padding_constant)

                    # Convolution
                    x = tf.nn.conv3d(
                        input=x,
                        filter=self.W_combined,
                        strides=[int(st)]*5,
                        padding=pd
                    )
                    x = tf.contrib.layers.batch_norm(
                            inputs=x,
                            decay=0.9,
                            is_training=base.is_training,
                        )

                    if not (fn == 'none'):
                        x = ACTIVATION_FN[fn](x)

                    self.conv.append(x)

                    # regularization
                    if not(reuse):
                        self.reg_loss = group_sparsity_regularizer_1d(
                                   self.weights_temporal[-1][:,0,0,:,:],
                                   conv_sparse_weight * sp
                        )
                        self.reg_loss += group_sparsity_regularizer_2d(
                                    self.weights_spatial[-1][0,:,:,:,:],
                                    conv_sparse_weight * sp
                        )
                        self.reg_loss += smoothness_regularizer_1d(
                                    self.weights_temporal[-1][:,0,0,:,:],
                                    conv_smooth_weight_temporal * sm
                        )
                        self.reg_loss += smoothness_regularizer_2d(
                                    self.weights_spatial[-1][0,:,:,:,:],
                                    conv_smooth_weight_spatial * sm
                        )
                        tf.losses.add_loss(self.reg_loss, tf.GraphKeys.REGULARIZATION_LOSSES)

                # split scans into scan wise outputs
                # self.output = tf.split(
                #     self.conv[-1],
                #     len(data.scans),
                #     axis=0,
                #     name='output'
                # )
                self.output = tf.identity(self.conv[-1], name='output')


class StackedFactorizedConv3dAdaptationCore:
    def __init__(self,
                 base,
                 data,
                 inputs,
                 filter_size_spatial=[13, 13],
                 filter_size_temporal=[20, 20],
                 num_filters=[8, 16],
                 stride=[1, 1],
                 rate=[1, 1],
                 padding=['VALID', 'VALID'],
                 nonzero_padding=False,
                 padding_constant=0,
                 activation_fn=['elu', 'none'],
                 rel_smooth_weight=[1, 0],
                 rel_sparse_weight=[0, 1],
                 conv_smooth_weight_spatial=0.001,
                 conv_smooth_weight_temporal=0.001,
                 conv_sparse_weight=0.001,
                 scope='core',
                 reuse=False,
                 **kwargs):
        #calculate resulting steps in history that contribute to output
        self.steps_hist = 1+np.sum(filter_size_temporal)-len(filter_size_temporal)
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                self.conv = []
                self.weights_temporal = []
                self.weights_spatial = []
                self.weights_scan_bias = []
                self.weights_scan_scale = []
                # Put scans into batch dimension: SBDHWC -> (S*B)DHWC
                # input_list = [i[0] for i in tf.split(inputs, data.input_shape[0])]
                # x = tf.concat(input_list, 0, name = 'put_scan_in_batch')
                x = inputs[...,:2]
                for i, (fs_s, fs_t, nf, st, rt, pd, fn, sm, sp) in enumerate(
                        zip(filter_size_spatial, filter_size_temporal, num_filters,
                            stride, rate, padding, activation_fn, rel_smooth_weight,
                            rel_sparse_weight)):
                    with tf.variable_scope('conv{}'.format(i), reuse=tf.AUTO_REUSE):
                        # temporal
                        #filter: [filter_depth, filter_height, filter_width, in_channels, out_channels]
                        self.weights_temporal.append(tf.get_variable(
                                           name='weights_temporal_{}'.format(i),
                                           shape=[fs_t,1,1,x.shape[-1],int(nf)],
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)))

                        # spatial
                        self.weights_spatial.append(tf.get_variable(
                                          name='weights_spatial_{}'.format(i),
                                          shape=[1,fs_s,fs_s,x.shape[-1],int(nf)],
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)))

                    # combined
                    self.W_combined = tf.einsum('dabio,chwio->dhwio',
                                           self.weights_temporal[-1],
                                           self.weights_spatial[-1],
                                           name='weights_combined_{}'.format(i)
                    )

                    if nonzero_padding:
                        x = tf.pad(x,[[0,0],[0,0],[fs_s//2,fs_s//2],[fs_s//2,fs_s//2],[0,0]],mode='CONSTANT',constant_values=padding_constant)

                    # Convolution
                    x = tf.nn.conv3d(
                        input=x,
                        filter=self.W_combined,
                        strides=[int(st)]*5,
                        padding=pd
                    )
                    x = tf.contrib.layers.batch_norm(
                            inputs=x,
                            decay=0.9,
                            is_training=base.is_training,
                        )

                    if not (fn == 'none'):
                        x = ACTIVATION_FN[fn](x)

                    self.conv.append(x)

                    # regularization
                    if not(reuse):
                        self.reg_loss = group_sparsity_regularizer_1d(
                                   self.weights_temporal[-1][:,0,0,:,:],
                                   conv_sparse_weight * sp
                        )
                        self.reg_loss += group_sparsity_regularizer_2d(
                                    self.weights_spatial[-1][0,:,:,:,:],
                                    conv_sparse_weight * sp
                        )
                        self.reg_loss += smoothness_regularizer_1d(
                                    self.weights_temporal[-1][:,0,0,:,:],
                                    conv_smooth_weight_temporal * sm
                        )
                        self.reg_loss += smoothness_regularizer_2d(
                                    self.weights_spatial[-1][0,:,:,:,:],
                                    conv_smooth_weight_spatial * sm
                        )
                        tf.losses.add_loss(self.reg_loss, tf.GraphKeys.REGULARIZATION_LOSSES)

                # no symmetric crop possible for even filter_size_spatial
                crop_start = int(np.floor((np.sum(filter_size_spatial)-len(filter_size_spatial))/2))
                crop_end = int(np.ceil((np.sum(filter_size_spatial)-len(filter_size_spatial))/2))
                self.cropped_adaptation_stim = inputs[:,np.sum(filter_size_temporal)-1:,crop_start:-crop_end,crop_start:-crop_end,:2]
                self.output = tf.concat([self.conv[-1],self.cropped_adaptation_stim], axis = -1, name='output')


class IdentityCore:
    def __init__(self,
                 base,
                 data,
                 inputs,
                 scope='core',
                 **kwargs):
        self.output = tf.identity(inputs, name='output')
