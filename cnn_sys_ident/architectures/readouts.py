import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, resampler

from .utils import soft_threshold, inv_soft_threshold, sta_init
from .cores import smoothness_regularizer_1d, smoothness_regularizer_2d, group_sparsity_regularizer_1d, group_sparsity_regularizer_2d


def grid_sample(input, grid):
    _, num_px_y, num_px_x, num_features = input.shape.as_list()
    i = tf.cast(num_px_y - 1, grid.dtype) * (grid[..., 0] + 1) / 2
    j = tf.cast(num_px_x - 1, grid.dtype) * (grid[..., 1] + 1) / 2
    grid_new = tf.stack([i, j], axis=-1)
    return resampler.resampler(input, grid)


class SpatialXFeatureJointL1Readout:
    def __init__(self,
                 base,
                 inputs,
                 positive_feature_weights=False,
                 readout_sparsity=0.02,
                 init_masks='sta',
                 scope='readout',
                 reuse=False,
                 **kwargs):
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                data = base.data
                _, num_px_y, num_px_x, num_features = inputs.shape.as_list()
                num_neurons = data.num_neurons

                # masks
                if init_masks == 'sta':
                    images_train, responses_train = data.train()
                    k = (images_train.shape[1] - num_px_y) // 2
                    mask_init = sta_init(images_train, responses_train,
                                         max_val=0.01, sd=0.001)[:,k:-k,k:-k]
                    mask_init = tf.constant_initializer(mask_init)
                else:
                    mask_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                self.masks = tf.get_variable(
                    'masks',
                    shape=[num_neurons, num_px_y, num_px_x],
                    initializer=mask_init)
                self.masks = tf.abs(self.masks, name='positive_masks')
                self.masked = tf.tensordot(inputs, self.masks, [[1, 2], [1, 2]], name='masked')

                # feature weights
                self.feature_weights = tf.get_variable(
                    'feature_weights',
                    shape=[num_neurons, num_features],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                if positive_feature_weights:
                    self.feature_weights = tf.abs(self.feature_weights, name='positive_feature_weights')
                self.h = tf.reduce_sum(self.masked * tf.transpose(self.feature_weights), 1)

                # L1 regularization for readout layer
                self.readout_reg = readout_sparsity * tf.reduce_sum(
                    tf.reduce_sum(tf.abs(self.masks), [1, 2]) * \
                    tf.reduce_sum(tf.abs(self.feature_weights), 1))
                tf.losses.add_loss(self.readout_reg, tf.GraphKeys.REGULARIZATION_LOSSES)

                # bias and output nonlinearity
                _, responses = data.train()
                bias_init = 0.5 * inv_soft_threshold(responses.mean(axis=0))
                self.biases = tf.get_variable(
                    'biases',
                    shape=[num_neurons],
                    initializer=tf.constant_initializer(bias_init))
                self.output = tf.identity(soft_threshold(self.h + self.biases), name='output')


class SpatialXFeatureSeparateL1Readout:
    def __init__(self,
                 base,
                 inputs,
                 positive_feature_weights=False,
                 mask_sparsity=0.02,
                 feature_sparsity=0.05,
                 init_masks='sta',
                 scope='readout',
                 reuse=False,
                 **kwargs):
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                data = base.data
                _, num_px_y, num_px_x, num_features = inputs.shape.as_list()
                num_neurons = data.num_neurons

                # masks
                if init_masks == 'sta':
                    images_train, responses_train = data.train()
                    k = (images_train.shape[1] - num_px_y) // 2
                    mask_init = sta_init(images_train, responses_train,
                                         max_val=0.01, sd=0.001)[:,k:-k,k:-k]
                    mask_init = tf.constant_initializer(mask_init)
                else:
                    mask_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                self.masks = tf.get_variable(
                    'masks',
                    shape=[num_neurons, num_px_y, num_px_x],
                    initializer=mask_init)
                self.masks = tf.abs(self.masks, name='positive_masks')
                self.masked = tf.tensordot(inputs, self.masks, [[1, 2], [1, 2]], name='masked')

                # feature weights
                self.feature_weights = tf.get_variable(
                    'feature_weights',
                    shape=[num_neurons, num_features],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                if positive_feature_weights:
                    self.feature_weights = tf.abs(self.feature_weights, name='positive_feature_weights')
                self.h = tf.reduce_sum(self.masked * tf.transpose(self.feature_weights), 1)

                # L1 regularization for masks and feature weights, with separate weights
                self.mask_sparsity_reg = mask_sparsity * tf.reduce_sum(tf.abs(self.masks))
                self.feature_sparsity_reg = feature_sparsity * tf.reduce_sum(tf.abs(self.feature_weights))
                self.readout_reg = self.mask_sparsity_reg + self.feature_sparsity_reg
                tf.losses.add_loss(self.readout_reg, tf.GraphKeys.REGULARIZATION_LOSSES)

                # bias and output nonlinearity
                _, responses = data.train()
                bias_init = 0.5 * inv_soft_threshold(responses.mean(axis=0))
                self.biases = tf.get_variable(
                    'biases',
                    shape=[num_neurons],
                    initializer=tf.constant_initializer(bias_init))
                self.output = tf.identity(soft_threshold(self.h + self.biases), name='output')


class SpatialSparseXFeatureDenseReadout:
    def __init__(self,
                 base,
                 inputs,
                 positive_feature_weights=False,
                 mask_sparsity=0.02,
                 init_masks='sta',
                 scope='readout',
                 reuse=False,
                 **kwargs):
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                data = base.data
                _, num_px_y, num_px_x, num_features = inputs.shape.as_list()
                num_neurons = data.num_neurons

                # masks
                if init_masks == 'sta':
                    images_train, responses_train = data.train()
                    k = (images_train.shape[1] - num_px_y) // 2
                    mask_init = sta_init(images_train, responses_train)[:,k:-k,k:-k]
                    mask_init = tf.constant_initializer(mask_init)
                else:
                    mask_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                self.masks = tf.get_variable(
                    'masks',
                    shape=[num_neurons, num_px_y, num_px_x],
                    initializer=mask_init)
                self.masks = tf.abs(self.masks, name='positive_masks')
                self.masked = tf.tensordot(inputs, self.masks, [[1, 2], [1, 2]], name='masked')

                # feature weights
                self.feature_weights = tf.get_variable(
                    'feature_weights',
                    shape=[num_neurons, num_features],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                if positive_feature_weights:
                    self.feature_weights = tf.abs(self.feature_weights, name='positive_feature_weights')
                self.h = tf.reduce_sum(self.masked * tf.transpose(self.feature_weights), 1)

                # L1 regularization for masks, L2 for feature weights
                self.readout_reg = mask_sparsity * tf.reduce_sum(
                    tf.reduce_sum(tf.abs(self.masks), [1, 2]) * \
                    tf.sqrt(tf.reduce_sum(tf.square(self.feature_weights), 1)))
                tf.losses.add_loss(self.readout_reg, tf.GraphKeys.REGULARIZATION_LOSSES)

                # bias and output nonlinearity
                _, responses = data.train()
                bias_init = 0.5 * inv_soft_threshold(responses.mean(axis=0))
                self.biases = tf.get_variable(
                    'biases',
                    shape=[num_neurons],
                    initializer=tf.constant_initializer(bias_init))
                self.output = tf.identity(soft_threshold(self.h + self.biases), name='output')


class SpatialSparseXFeatureDenseSeparateReadout:
    def __init__(self,
                 base,
                 inputs,
                 positive_feature_weights=False,
                 mask_sparsity=0.02,
                 feature_l2=0.05,
                 init_masks='sta',
                 scope='readout',
                 reuse=False,
                 **kwargs):
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                data = base.data
                _, num_px_y, num_px_x, num_features = inputs.shape.as_list()
                num_neurons = data.num_neurons

                # masks
                if init_masks == 'sta':
                    images_train, responses_train = data.train()
                    k = (images_train.shape[1] - num_px_y) // 2
                    mask_init = sta_init(images_train, responses_train)[:,k:-k,k:-k]
                    mask_init = tf.constant_initializer(mask_init)
                else:
                    mask_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                self.masks = tf.get_variable(
                    'masks',
                    shape=[num_neurons, num_px_y, num_px_x],
                    initializer=mask_init)
                self.masks = tf.abs(self.masks, name='positive_masks')
                self.masked = tf.tensordot(inputs, self.masks, [[1, 2], [1, 2]], name='masked')

                # feature weights
                self.feature_weights = tf.get_variable(
                    'feature_weights',
                    shape=[num_neurons, num_features],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                if positive_feature_weights:
                    self.feature_weights = tf.abs(self.feature_weights, name='positive_feature_weights')
                self.h = tf.reduce_sum(self.masked * tf.transpose(self.feature_weights), 1)

                # L1 regularization for masks, L2 for feature weights, with separate weights
                self.mask_sparsity_reg = mask_sparsity * tf.reduce_sum(tf.reduce_sum(tf.abs(self.masks)))
                self.feature_l2_reg = feature_l2 * tf.reduce_sum(tf.square(self.feature_weights))
                self.readout_reg = self.mask_sparsity_reg + self.feature_l2_reg
                tf.losses.add_loss(self.readout_reg, tf.GraphKeys.REGULARIZATION_LOSSES)

                # bias and output nonlinearity
                _, responses = data.train()
                bias_init = 0.5 * inv_soft_threshold(responses.mean(axis=0))
                self.biases = tf.get_variable(
                    'biases',
                    shape=[num_neurons],
                    initializer=tf.constant_initializer(bias_init))
                self.output = tf.identity(soft_threshold(self.h + self.biases), name='output')


class SpatialXFeatureJointL1TransferReadout:
    def __init__(self,
                 base,
                 inputs,
                 k_transfer,
                 positive_feature_weights=False,
                 readout_sparsity=0.017,
                 init_masks='sta',
                 scope='readout',
                 reuse=False,
                 **kwargs):
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                data = base.data
                _, num_px_y, num_px_x, num_features = inputs.shape.as_list()
                num_neurons = data.num_neurons

                # masks
                if init_masks == 'sta':
                    images_train, responses_train = data.train()
                    k = (images_train.shape[1] - num_px_y) // 2
                    mask_init = sta_init(images_train, responses_train,
                                         max_val=0.01, sd=0.001)[:,k:-k,k:-k]
                    mask_init = tf.constant_initializer(mask_init)
                else:
                    mask_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                self.masks = tf.get_variable(
                    'masks',
                    shape=[num_neurons, num_px_y, num_px_x],
                    initializer=mask_init)
                self.masks = tf.abs(self.masks, name='positive_masks')

                # split masks into neurons used for training the core and neurons
                # used for transfer learning (i.e. only readout weights are trained,
                # but they do not affect the core). We then insert a stop_gradient
                # for the units used only for transfer learning, apply the masks
                # and put things back together. This way, there is no gradient flow
                # from the units used for transfer learning back into the core.
                idx_train = tf.range(num_neurons, delta=k_transfer)
                idx_transfer, _ = tf.setdiff1d(tf.range(num_neurons), idx_train)
                idx_all = tf.invert_permutation(tf.concat([idx_train, idx_transfer], axis=0))
                masks_train = tf.gather(self.masks, idx_train)
                masks_transfer = tf.gather(self.masks, idx_transfer)
                masked_train = tf.tensordot(masks_train, inputs,
                                            [[1, 2], [1, 2]], name='masked_train')
                masked_transfer = tf.tensordot(masks_transfer, tf.stop_gradient(inputs),
                                               [[1, 2], [1, 2]], name='masked_transfer')
                self.masked = tf.transpose(tf.gather(tf.concat(
                            [masked_train, masked_transfer], axis=0), 
                        idx_all, name='masked'),
                    [1, 2, 0])

                # feature weights
                self.feature_weights = tf.get_variable(
                    'feature_weights',
                    shape=[num_neurons, num_features],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                if positive_feature_weights:
                    self.feature_weights = tf.abs(self.feature_weights, name='positive_feature_weights')
                self.h = tf.reduce_sum(self.masked * tf.transpose(self.feature_weights), 1)

                # L1 regularization for readout layer
                self.readout_reg = readout_sparsity * tf.reduce_sum(
                    tf.reduce_sum(tf.abs(self.masks), [1, 2]) * \
                    tf.reduce_sum(tf.abs(self.feature_weights), 1))
                tf.losses.add_loss(self.readout_reg, tf.GraphKeys.REGULARIZATION_LOSSES)

                # bias and output nonlinearity
                _, responses = data.train()
                bias_init = 0.5 * inv_soft_threshold(responses.mean(axis=0))
                self.biases = tf.get_variable(
                    'biases',
                    shape=[num_neurons],
                    initializer=tf.constant_initializer(bias_init))
                self.output = tf.identity(soft_threshold(self.h + self.biases), name='output')


class SpatialXFeature3dJointL1Readout:
    def __init__(self,
                 base,
                 data,
                 inputs,
                 positive_feature_weights=False,
#                  mask_sparsity=0.01,
#                  feature_sparsity=0.001,
                 readout_sparsity=0.01,
                 init_masks='sta',
                 scope='readout',
                 reuse=False,
                 nonlinearity=True,
                 ca_kernel=False,
                 output_sparsity=0,
                 **kwargs):
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                # data = base.data
                _, _, num_px_y, num_px_x, num_features = inputs.shape.as_list()
                num_neurons = data.num_neurons
                
                # masks
                if init_masks == 'sta':
                    try:
                        data.sta_space
                    except AttributeError:
                        if scope=='readout0':
                            print('Initialize masks randomly')
                        mask_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                    else: # to verify/test
                        tmp_indx = (data.px_x-num_px_x)//2
                        tmp_indy = (data.px_y-num_px_y)//2
                        tmp_sta = data.sta_space[:,tmp_indy:tmp_indy+num_px_y,tmp_indx:tmp_indx+num_px_x]
                        mask_init = np.random.normal(0,.01,tmp_sta.shape)
                        for n in range(num_neurons):
                            max_ind = np.unravel_index(np.argmax(abs(tmp_sta[n])),[num_px_y, num_px_x])
                            mask_init[n,max_ind[0],max_ind[1]] = .2
                        mask_init = tf.constant_initializer(mask_init)
                else:
                    mask_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                self.masks = tf.get_variable(
                    'masks',
                    shape=[num_neurons, num_px_y, num_px_x],
                    initializer=mask_init)
                self.masks = tf.abs(self.masks, name='positive_masks')
                self.masked = tf.tensordot(inputs, self.masks, [[2, 3], [1, 2]], name='masked')

                # feature weights
                self.feature_weights = tf.get_variable(
                    'feature_weights',
                    shape=[num_neurons, num_features],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                if positive_feature_weights:
                    self.feature_weights = tf.abs(self.feature_weights, name='positive_feature_weights')
                self.h = tf.reduce_sum(self.masked * tf.transpose(self.feature_weights), 2)  # 2?

                # L1 regularization for readout layer
                # summed, scales with neurons <<<< YOU WANT THIS!!!
                self.reg_loss = readout_sparsity * tf.reduce_sum(
                    tf.reduce_sum(tf.abs(self.masks), [1, 2]) * \
                    tf.reduce_sum(tf.abs(self.feature_weights), 1))

#                 self.mask_reg = mask_sparsity * tf.reduce_sum(tf.abs(self.masks))
#                 self.feature_reg = feature_sparsity * tf.reduce_sum(tf.abs(self.feature_weights))
#                 self.readout_reg = self.mask_reg + self.feature_reg
                tf.losses.add_loss(self.reg_loss, tf.GraphKeys.REGULARIZATION_LOSSES)
    
                self.sparsity_loss = output_sparsity * tf.reduce_sum(tf.abs(self.h))
                tf.losses.add_loss(self.sparsity_loss, tf.GraphKeys.REGULARIZATION_LOSSES)

                # bias and output nonlinearity
                _, responses = data.train()
                if nonlinearity:
#                     bias_init = 0.5 * inv_soft_threshold(responses.mean(axis=0))
                    bias_init = inv_soft_threshold(responses.mean(axis=0))
                else:
                    bias_init = responses.mean(axis=0)
                self.biases = tf.get_variable(
                    'biases',
                    shape=[num_neurons],
                    initializer=tf.constant_initializer(bias_init))
                if nonlinearity:
                    self.y = tf.identity(soft_threshold(self.h + self.biases), name='pre_output')
                else:
                    self.y = tf.identity(self.h + self.biases, name='pre_output')
                    
                                # optional calcium kernel
                if ca_kernel:
                    ca_kernel_size = 15
                    ca_timebase = tf.constant(np.linspace(0, 1, ca_kernel_size), dtype=tf.float32)
                    ca_kernel_tau = tf.get_variable('ca_kernel_tau',shape=2,initializer=tf.random_uniform_initializer(-20,-5))
                    ca_kernel_weight = tf.get_variable('ca_kernel_weights',shape=2,initializer=tf.constant_initializer(0.5))
                    # initialization with fixed kernel
#                     ca_kernel_tau = tf.constant([-3,-10],dtype='float32')
#                     ca_kernel_weight= tf.constant([0.5,0.5])
                    calcium_kernel = ca_kernel_weight[0]*tf.exp(tf.scalar_mul(ca_kernel_tau[0],ca_timebase)) \
                                    + ca_kernel_weight[1]*tf.exp(tf.scalar_mul(ca_kernel_tau[1],ca_timebase))
                    calcium_kernel = tf.reverse(calcium_kernel, axis=[0])
                    calcium_kernel = tf.expand_dims(calcium_kernel, axis = [-1])
                    calcium_kernel = tf.expand_dims(calcium_kernel, axis = [-1])
                    self.calcium_kernel = tf.expand_dims(calcium_kernel, axis = [-1])

                    # pad the input
                    paddings = tf.constant([[0, 0],
                                            [ca_kernel_size-1, 0],
                                            [0, 0]])
                    y = tf.pad(self.y, paddings, "CONSTANT")
                    self.ca_kernel_input = tf.expand_dims(y, axis = [-1])
                    ca_kernel_output = tf.nn.convolution(self.ca_kernel_input,
                                                         self.calcium_kernel,
                                                         name='output',
                                                         padding='VALID'
                                                         )
                    self.output = tf.squeeze(ca_kernel_output, -1, name='ca_output')
                else:
                    self.output = tf.identity(self.y, name='output')
                    
                    
class SpatialXFeature3dJointTemperatureReadout:
    def __init__(self,
                 base,
                 data,
                 inputs,
                 positive_feature_weights=False,
#                  mask_sparsity=0.01,
#                  feature_sparsity=0.001,
#                  readout_sparsity=0.01,
                 init_masks='sta',
                 scope='readout',
                 reuse=False,
                 nonlinearity=True,
                 ca_kernel=False,
                 output_sparsity=0,
                 **kwargs):
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                # data = base.data
                _, _, num_px_y, num_px_x, num_features = inputs.shape.as_list()
                num_neurons = data.num_neurons
                
                # masks
                if init_masks == 'sta':
                    try:
                        data.sta_space
                    except AttributeError:
                        if scope=='readout0':
                            print('Initialize masks randomly')
                        mask_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                    else: # to verify/test
                        tmp_indx = (data.px_x-num_px_x)//2
                        tmp_indy = (data.px_y-num_px_y)//2
                        tmp_sta = data.sta_space[:,tmp_indy:tmp_indy+num_px_y,tmp_indx:tmp_indx+num_px_x]
                        mask_init = np.random.normal(0,.01,tmp_sta.shape)
                        for n in range(num_neurons):
                            max_ind = np.unravel_index(np.argmax(abs(tmp_sta[n])),[num_px_y, num_px_x])
                            mask_init[n,max_ind[0],max_ind[1]] = .2
                        mask_init = tf.constant_initializer(mask_init)
                else:
                    mask_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                self.masks = tf.get_variable(
                    'masks',
                    shape=[num_neurons, num_px_y * num_px_x],
                    initializer=mask_init)
                self.beta = tf.placeholder_with_default(tf.constant(1, dtype=tf.float32), [])
                self.masks = tf.nn.softmax(self.masks*self.beta,axis=-1) # i temperature = runs, do the same for the weights?
#                 self.masks = tf.abs(self.masks, name='positive_masks')
                self.masked = tf.tensordot(inputs, tf.reshape(self.masks,[num_neurons, num_px_y, num_px_x]), [[2, 3], [1, 2]], name='masked')

                # feature weights
                self.feature_weights = tf.get_variable(
                    'feature_weights',
                    shape=[num_neurons, num_features],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                self.feature_weights = tf.nn.softmax(self.feature_weights*self.beta*5,axis=-1)
#                 if positive_feature_weights:
#                     self.feature_weights = tf.abs(self.feature_weights, name='positive_feature_weights')
                self.h = tf.reduce_sum(self.masked * tf.transpose(self.feature_weights), 2)  # 2?

                # L1 regularization for readout layer
                # summed, scales with neurons <<<< YOU WANT THIS!!!
#                 self.reg_loss = readout_sparsity * tf.reduce_sum(
#                     tf.reduce_sum(tf.abs(self.masks), [1, 2]) * \
#                     tf.reduce_sum(tf.abs(self.feature_weights), 1))

#                 self.mask_reg = mask_sparsity * tf.reduce_sum(tf.abs(self.masks))
#                 self.feature_reg = feature_sparsity * tf.reduce_sum(tf.abs(self.feature_weights))
#                 self.readout_reg = self.mask_reg + self.feature_reg
#                 tf.losses.add_loss(self.reg_loss, tf.GraphKeys.REGULARIZATION_LOSSES)
    
#                 self.sparsity_loss = output_sparsity * tf.reduce_sum(tf.abs(self.h))
#                 tf.losses.add_loss(self.sparsity_loss, tf.GraphKeys.REGULARIZATION_LOSSES)

                # bias and output nonlinearity
                _, responses = data.train()
                if nonlinearity:
#                     bias_init = 0.5 * inv_soft_threshold(responses.mean(axis=0))
                    bias_init = inv_soft_threshold(responses.mean(axis=0))
                else:
                    bias_init = responses.mean(axis=0)
                self.biases = tf.get_variable(
                    'biases',
                    shape=[num_neurons],
                    initializer=tf.constant_initializer(bias_init))
                self.scales = tf.get_variable(
                    'scales',
                    shape=[num_neurons],
                    initializer=tf.ones_initializer())
                if nonlinearity:
                    self.output = tf.identity(soft_threshold(self.h * self.scales + self.biases), name='output')
                else:
                    self.output = tf.identity(self.h * self.scales + self.biases, name='pre_output')


class ConstantReadout:
    def __init__(self,
                 base,
                 data,
                 inputs,
                 scope='readout',
                 reuse=False,
                 **kwargs):
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                shape = tf.shape(inputs)
                num_neurons = data.num_neurons
                _, responses = data.train()
                self.readout_reg = tf.constant(0)
                self.output = tf.constant(
                    responses.mean(axis=0),
                    dtype=tf.float32,
                    name='output',
                    shape=[1, 1, num_neurons]) + inputs[:,:,0:1,0,0] * 0
#                     shape=[1, 1, num_neurons]) + inputs[:,:-20,0:1,0,0] * 0
                print(self.output.shape)


class SpatialXFeature3dL1Readout:
    def __init__(self,
                 base,
                 data,
                 inputs,
                 positive_feature_weights=True,
                 mask_sparsity=0.01,
                 feature_sparsity=0.001,
                 init_masks='sta',
                 scope='readout',
                 final_nonlinearity=False,
                 final_bias=False,
                 final_scale=False,
                 reuse=False,
                 spread=0,
                 norm=[False,False],
                 prior=0,
                 **kwargs):
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                _, _, num_px_y, num_px_x, num_features = inputs.shape.as_list()
                num_neurons = data.num_neurons

                # masks
                if init_masks == 'sta':
                    try:
                        data.sta_space
                    except AttributeError:
                        if scope=='readout0':
                            print('Initialize masks randomly')
                        mask_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                    else:
                        tmp_indx = (data.px_x-num_px_x)//2
                        tmp_indy = (data.px_y-num_px_y)//2
                        tmp_sta = data.sta_space[:,tmp_indy:tmp_indy+num_px_y,tmp_indx:tmp_indx+num_px_x]
                        mask_init = np.random.normal(0,.01,tmp_sta.shape)
                        for n in range(num_neurons):
                            max_ind = np.unravel_index(np.argmax(abs(tmp_sta[n])),[num_px_y, num_px_x])
                            mask_init[n,max_ind[0],max_ind[1]] = .2
                        mask_init = tf.constant_initializer(mask_init)
                else:
                    if scope=='readout0':
                        print('Initialize masks randomly')
                    mask_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                self.masks = tf.get_variable(
                    'masks',
                    shape=[num_neurons, num_px_y, num_px_x],
                    initializer=mask_init)
                if norm[0]:
                    self.masks /= tf.norm(
                        self.masks,
                        ord='euclidean',
                        axis=[1,2],
                        keepdims=True,
                        name='normed_masks'
                    ) + 1e-8
                else:
                    self.masks = tf.abs(self.masks, name='positive_masks')
                self.masked = tf.tensordot(inputs, self.masks, [[2, 3], [1, 2]], name='masked')

                # feature weights
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                if np.sum(abs(prior))>0:
                    if scope=='readout0':
                        print('initialize feature to prior')
                    initializer = tf.constant_initializer(prior)
                self.feature_weights = tf.get_variable(
                    'feature_weights',
                    shape=[num_neurons, num_features],
                    initializer=initializer)
                if positive_feature_weights:
                    self.feature_weights = tf.abs(self.feature_weights, name='positive_feature_weights')
                if norm[1]:
                    self.feature_weights /= tf.norm(
                        self.feature_weights,
                        ord=1,#'euclidean',
                        axis=1,
                        keepdims=True,
                        name='normed_feature'
                    ) + 1e-8
                self.output = tf.reduce_sum(
                    self.masked * tf.transpose(self.feature_weights),
                    axis=2,
                    name='output')

                # L1 regularization for masks and feature weights
                # summed, scales with neurons <<<< YOU WANT THIS!!!
                self.mask_reg = mask_sparsity * tf.reduce_sum(tf.abs(self.masks))
                self.feature_reg = feature_sparsity * tf.reduce_sum(tf.abs(self.feature_weights))
                # mean, doesnt scale with neurons
                #self.mask_reg = mask_sparsity * tf.reduce_mean(tf.abs(self.masks))
                #self.feature_reg = feature_sparsity * tf.reduce_mean(tf.abs(self.feature_weights))
                self.readout_reg = self.mask_reg + self.feature_reg

                #spread (weights across all channels)
                if spread:
                    type_weights = tf.norm(self.feature_weights, axis=0)
                    self.spread = spread / (tf.reduce_sum(type_weights) / tf.norm(type_weights))
                    self.readout_reg += self.spread

                tf.losses.add_loss(self.readout_reg, tf.GraphKeys.REGULARIZATION_LOSSES)

                # final scale
                if final_scale:
                    self.scales = tf.get_variable(
                        'final_scale',
                        shape=[num_neurons],
                        initializer=tf.constant_initializer(np.ones(num_neurons))
                    )
                    tf.losses.add_loss(1e-5 * tf.norm(self.scales - 1 + 1e-8),
                                       tf.GraphKeys.REGULARIZATION_LOSSES)
                    self.output *= tf.abs(self.scales)

                # bias and output nonlinearity
                if final_nonlinearity:
                    self.biases = tf.get_variable(
                        'biases',
                        shape=[num_neurons],
                        initializer=tf.constant_initializer(np.zeros(num_neurons)))
                    tf.losses.add_loss(1e-5 * tf.norm(self.biases + 1e-8),
                                       tf.GraphKeys.REGULARIZATION_LOSSES)
                    self.output = tf.identity(soft_threshold(self.output + self.biases), name='output_NL')

                if final_bias:
                    self.biases2 = tf.get_variable(
                        'biases2',
                        shape=[num_neurons],
                        initializer=tf.constant_initializer(np.zeros(num_neurons)))
                    tf.losses.add_loss(1e-5 * tf.norm(self.biases2 + 1e-8),
                                       tf.GraphKeys.REGULARIZATION_LOSSES)
                    self.output += self.biases2


def smoothness_regularizer_1d_(W, weight=1.0):
    with tf.variable_scope('smoothness'):
        lap = tf.constant([-1, 2, -1], shape=(3, 1, 1, 1), dtype=tf.float32)
        W = tf.expand_dims(W, 1)
        num_filters = W.get_shape().as_list()[2]
        W_lap = tf.nn.depthwise_conv2d(tf.transpose(W, perm=[3, 0, 1, 2]),
                                       tf.tile(lap, [1, 1, num_filters, 1]),
                                       strides=[1, 1, 1, 1], padding='SAME')
        penalty = tf.reduce_sum(tf.square(W_lap))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('smoothness_regularizer_1d', penalty)
        return penalty

def smoothness_regularizer_2d_(W, weight=1.0):
    with tf.variable_scope('smoothness'):
        lap = tf.constant([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]])
        lap = tf.expand_dims(tf.expand_dims(lap, 2), 3)
        num_filters = W.get_shape().as_list()[2]
        W_lap = tf.nn.depthwise_conv2d(tf.transpose(W, perm=[3, 0, 1, 2]),
                                       tf.tile(lap, [1, 1, num_filters, 1]),
                                       strides=[1, 1, 1, 1], padding='SAME')
        penalty = tf.reduce_sum(tf.square(W_lap))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('smoothness_regularizer_2d', penalty)
        return penalty


class FactorizedConv3dReadout:
    def __init__(self,
                 base,
                 data,
                 inputs,
                 filter_size_temporal=20,
                 activation_fn=['none'],
                 conv_smooth_weight_spatial=0.001,
                 conv_smooth_weight_temporal=0.001,
                 conv_sparse_weight=0.001,
                 l2=0.0,
                 nonlinearity=True,
                 final_bias=False,
                 scope='core',
                 reuse=False,
                 **kwargs):
        _, _, num_px_y, num_px_x, num_features = inputs.shape.as_list()
        filter_size_spatial = num_px_x
        num_neurons = data.num_neurons
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                self.weights_temporal = []
                self.weights_spatial = []
                self.weights_scan_bias = []
                self.weights_scan_scale = []

                #filter: [filter_depth, filter_height, filter_width, in_channels, out_channels]
                # spatial
                # hack
#                 print('SPATIAL WEIGHTS INITIALIZATION FIXED TO FILTERS FROM BASELINE MODEL!')
#                 sp_weight_init = np.load('../../../RF_init_test/noise_RF_'+scope+'.npy')
#                 sp_weight_init = sp_weight_init.reshape(1,sp_weight_init.shape[0],sp_weight_init.shape[1],1,sp_weight_init.shape[2])
                self.weights_spatial=tf.get_variable(
                                    name='readout_weights_spatial',
                                    shape=[1, num_px_y, num_px_x,inputs.shape[-1],num_neurons],
#                                     initializer=tf.constant_initializer(sp_weight_init))
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

                #temporal
                self.weights_temporal=tf.get_variable(
                                    name='readout_weights_temporal',
                                    shape=[filter_size_temporal,1,1,inputs.shape[-1],num_neurons],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

                # combined
                self.W_combined = tf.einsum('dabio,chwio->dhwio',
                                        self.weights_temporal,
                                        self.weights_spatial,
                                        name='readout_weights_combined'
                )

                # Convolution
                self.conv = tf.nn.conv3d(input=inputs,
                                           filter=self.W_combined,
                                           strides=[1]*5,
                                           padding='VALID',
                                           name='conv'
                )
                self.h = tf.squeeze(self.conv, [2,3])

                # regularization
                if not(reuse):
                    self.reg_loss = group_sparsity_regularizer_1d(
                                self.weights_temporal[:,0,0,:,:],
                                conv_sparse_weight
                    )
                    self.reg_loss += group_sparsity_regularizer_2d(
                                self.weights_spatial[0,:,:,:,:],
                                conv_sparse_weight
                    )
                    print('TEMPORARILY CHANGED SMOOTHNESS REGULARIZER!!!')
#                     self.reg_loss += smoothness_regularizer_1d(
                    self.reg_loss += smoothness_regularizer_1d_(
                                self.weights_temporal[:,0,0,:,:],
                                conv_smooth_weight_temporal
                    )
#                     self.reg_loss += smoothness_regularizer_2d(
                    self.reg_loss += smoothness_regularizer_2d_(
                                self.weights_spatial[0,:,:,:,:],
                                conv_smooth_weight_spatial
                    )
                    self.reg_loss += l2 * tf.reduce_sum(tf.square(self.W_combined))
                    tf.losses.add_loss(self.reg_loss, tf.GraphKeys.REGULARIZATION_LOSSES)

                # bias and output nonlinearity
                _, responses = data.train()
                if nonlinearity:
                    bias_init = 0.5 * inv_soft_threshold(responses.mean(axis=0))
                else:
                    bias_init = responses.mean(axis=0)
                self.biases = tf.get_variable(
                    'biases',
                    shape=[num_neurons],
                    initializer=tf.constant_initializer(bias_init))
                if nonlinearity:
                    self.output = tf.identity(soft_threshold(self.h + self.biases), name='output')
                else:
                    self.output = tf.identity(self.h + self.biases, name='output')


class SpatialTransformerPooled3dReadout:
    def __init__(self,
                 base,
                 data,
                 inputs,
                 pool_steps=1,
                 positive_feature_weights=False,
                 feature_sparsity=0.001,
                 bias=True,
                 init_range=.05,
                 kernel_size=2,
                 stride=2,
                 grid=None,
                 stop_grad=False,
                 scope='readout',
                 reuse=False,
                 nonlinearity=False,
                 ca_kernel=False,
                 **kwargs):
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
				
                x = inputs
                self._pool_steps = pool_steps
                N, t, num_px_y, num_px_x, num_features = inputs.shape.as_list()
                num_neurons = data.num_neurons
				
                grid_init = tf.random_uniform_initializer(minval=-init_range, maxval=init_range)
                self.grid = tf.get_variable(
                    'grid',
                    shape=[1, num_neurons, 2],
                    initializer=grid_init)
					
                self.feature_weights = tf.get_variable(
                    'features',
                    shape=[num_neurons, num_features * (self._pool_steps + 1)],
#                     initializer=tf.constant_initializer(0.001))
                    # [AE] value seems way to high --> variance of output should be close to zero at init
                    initializer=tf.constant_initializer(1/num_features))
                self.mask = tf.get_variable(
                    'mask',
                    shape=self.feature_weights.shape,
                    initializer=tf.ones_initializer)

				# [AE]: what's the mask here?
                self.feature_weights *= self.mask
                if positive_feature_weights:
                    self.feature_weights = tf.clip_by_value(self.feature_weights,0,np.infty)
                    # [AE] I'd rather use absolute value than clipping
#                     self.feature_weights = tf.abs(self.feature_weights)
                self.grid = tf.clip_by_value(self.grid, -1, 1)
				
                self.z = tf.reshape(x,[-1,num_px_y,num_px_x,num_features])
                input_shape = tf.shape(inputs)
                grid = tf.tile(self.grid,[input_shape[0]*input_shape[1],1,1])

                pools = [grid_sample(self.z, grid)]
                for i in range(self._pool_steps):
                    self.z = tf.nn.avg_pool(self.z,ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding='VALID',  data_format='NHWC')
                    pools.append(grid_sample(self.z, grid))

                y = tf.concat(pools, axis=-1) # y.shape is B*D,N,C*(P+1)
                y = tf.reshape(y,[input_shape[0],input_shape[1],num_neurons,num_features * (self._pool_steps + 1)])
                y = tf.reduce_sum(y * self.feature_weights, -1)
		
				# add regularization loss for the readout layer
                self.reg_loss = feature_sparsity * tf.reduce_sum(tf.abs(self.feature_weights))
                tf.losses.add_loss(self.reg_loss, tf.GraphKeys.REGULARIZATION_LOSSES)

                # bias and output nonlinearity
                _, responses = data.train()
                if nonlinearity:
#                     bias_init = 0.5 * inv_soft_threshold(responses.mean(axis=0))
                    bias_init = inv_soft_threshold(responses.mean(axis=0))
                else:
                    bias_init = responses.mean(axis=0)
                self.biases = tf.get_variable(
                    'biases',
                    shape=[num_neurons],
                    initializer=tf.constant_initializer(bias_init))
                if nonlinearity:
#                     self.output = tf.identity(soft_threshold(y + self.biases), name='output')
                    y = soft_threshold(y + self.biases)
                else:
#                     self.output = tf.identity(y + self.biases, name='output')
                    y = y + self.biases
                    
                # optional calcium kernel
                if ca_kernel:
                    ca_kernel_size = 30
                    ca_timebase = tf.constant(np.linspace(0, 1, ca_kernel_size), dtype=tf.float32)
                    ca_kernel_tau = tf.get_variable('ca_kernel_tau',shape=2,initializer=tf.random_uniform_initializer(-50,-10))
                    ca_kernel_weight = tf.get_variable('ca_kernel_weights',shape=2,initializer=tf.constant_initializer(0.5))
                    # initialization with fixed kernel
#                     ca_kernel_tau = tf.constant([-3,-10],dtype='float32')
#                     ca_kernel_weight= tf.constant([0.5,0.5])
                    calcium_kernel = ca_kernel_weight[0]*tf.exp(tf.scalar_mul(ca_kernel_tau[0],ca_timebase)) \
                                    + ca_kernel_weight[1]*tf.exp(tf.scalar_mul(ca_kernel_tau[1],ca_timebase))
                    calcium_kernel = tf.reverse(calcium_kernel, axis=[0])
                    calcium_kernel = tf.expand_dims(calcium_kernel, axis = [-1])
                    calcium_kernel = tf.expand_dims(calcium_kernel, axis = [-1])
                    self.calcium_kernel = tf.expand_dims(calcium_kernel, axis = [-1])

                    # pad the input
                    paddings = tf.constant([[0, 0],
                                            [ca_kernel_size-1, 0],
                                            [0, 0]])
                    y = tf.pad(y, paddings, "CONSTANT")
                    self.ca_kernel_input = tf.expand_dims(y, axis = [-1])
                    ca_kernel_output = tf.nn.convolution(self.ca_kernel_input,
                                                         self.calcium_kernel,
                                                         name='output',
                                                         padding='VALID'
                                                         )
                    self.output = tf.squeeze(ca_kernel_output, -1, name='output')
                else:
                    self.output = tf.identity(y, name='output')


class SpatialTransformerPooled3dCalciumReadout:
    def __init__(self,
                base,
                data,
                inputs,
                pool_steps=1,
                positive_feature_weights=False,
                feature_sparsity=0.001,
                bias=True,
                init_range=.05,
                kernel_size=2,
                stride=2,
                grid=None,
                stop_grad=False,
                scope='readout',
                reuse=False,
                nonlinearity=False,
                **kwargs):
        with base.tf_session.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
				
                x = inputs
                self._pool_steps = pool_steps
                N, t, num_px_y, num_px_x, num_features = inputs.shape.as_list()
                num_neurons = data.num_neurons
				
                grid_init = tf.random_uniform_initializer(minval=-init_range, maxval=init_range)
                self.grid = tf.get_variable(
                    'grid',
                    shape=[1, num_neurons, 2],
                    initializer=grid_init)
					
                self.feature_weights = tf.get_variable(
                    'features',
                    shape=[num_neurons, num_features * (self._pool_steps + 1)],
                    initializer=tf.constant_initializer(1/num_features))
                self.mask = tf.get_variable(
                    'mask',
                    shape=self.feature_weights.shape,
                    initializer=tf.ones_initializer)

				
                self.feature_weights *= self.mask
                if positive_feature_weights:
                    self.feature_weights = tf.clip_by_value(self.feature_weights,0,np.infty)
                self.grid = tf.clip_by_value(self.grid, -1, 1)
				
                self.z = tf.reshape(x,[-1,num_px_y,num_px_x,num_features])
                input_shape = tf.shape(inputs)
                grid = tf.tile(self.grid,[input_shape[0]*input_shape[1],1,1])

                pools = [grid_sample(self.z, grid)]
                for i in range(self._pool_steps):
                    self.z = tf.nn.avg_pool(self.z,ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding='VALID',  data_format='NHWC')
                    pools.append(grid_sample(self.z, grid))

                y = tf.concat(pools, axis=-1) # y.shape is B*D,N,C*(P+1)
                y = tf.reshape(y,[input_shape[0],input_shape[1],num_neurons,num_features * (self._pool_steps + 1)])
                y = tf.reduce_sum(y * self.feature_weights, -1)
		
				# add regularization loss for the readout layer
                self.feature_reg = feature_sparsity * tf.reduce_sum(tf.abs(self.feature_weights))
                tf.losses.add_loss(self.feature_reg, tf.GraphKeys.REGULARIZATION_LOSSES)

                # bias and output nonlinearity
                _, responses = data.train()
                if nonlinearity:
                    bias_init = 0.5 * inv_soft_threshold(responses.mean(axis=0))
                else:
                    bias_init = - responses.mean(axis=0)
                self.biases = tf.get_variable(
                    'biases',
                    shape=[num_neurons],
                    initializer=tf.constant_initializer(bias_init))
                if nonlinearity:
                    y = soft_threshold(y + self.biases)
                else:
                    y = y + self.biases
                
                #define the calcium kernel
                ca_kernel_size = 30
                ca_timebase = tf.constant(np.linspace(0, 1, calcium_kernel_size), dtype=tf.float32)
                ca_kernel_tau = tf.get_variable('ca_kernel_tau',shape=2,initializer=tf.constant_initializer([-50,-100]))

                ca_kernel_weight=tf.get_variable('ca_kernel_weights',shape=2,initializer=tf.constant_initializer(0.5))
                calcium_kernel = ca_kernel_weight[0]*tf.exp(tf.scalar_mul(calcium_kernel_tau[0],timebase)) \
                                + ca_kernel_weight[1]*tf.exp(tf.scalar_mul(calcium_kernel_tau[1],timebase))
                calcium_kernel = tf.reverse(calcium_kernel, axis=[0])
                #calcium_kernel = tf.expand_dims(calcium_kernel, axis = [0])
                calcium_kernel = tf.expand_dims(calcium_kernel, axis = [-1])
                calcium_kernel = tf.expand_dims(calcium_kernel, axis = [-1])
                self.calcium_kernel = tf.expand_dims(calcium_kernel, axis = [-1])

                #pad the input
                paddings = tf.constant([[0, 0],
                                        [ca_kernel_size-1, 0],
                                        [0, 0]])
                y = tf.pad(y, paddings, "CONSTANT")
                self.ca_kernel_input = tf.expand_dims(y, axis = [-1])
                ca_kernel_output = tf.nn.convolution(self.ca_kernel_input,
                                                self.calcium_kernel,
                                                name='output',
                                                padding='VALID'
                                               )
                self.output = tf.squeeze(ca_kernel_output,-1)
                
                
                    

class MultiScanReadout:
    def __init__(self,
                 base,
                 data,
                 inputs,
                 readout_type,
                 scope='readout',
                 **kwargs):
        # One readout per scan
        self.readouts = []
        for i in range(len(inputs)):
            self.readouts.append(readout_type(
                base, data.scans[i], inputs[i], reuse=False, scope='{}_{}'.format(scope, i), **kwargs))
        # readout_position = np.zeros(len(inputs)+1,dtype=int)
        # readout_position[1:] = np.cumsum(data.num_rois)
        self.output = tf.concat([r.output for r in self.readouts], axis=2) # VERIFY
