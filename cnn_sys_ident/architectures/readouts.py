import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from .utils import soft_threshold, inv_soft_threshold, sta_init


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
                 mask_sparsity=0.01,
                 feature_sparsity=0.001,
                 init_masks='sta',
                 scope='readout',
                 reuse=False,
                 nonlinearity=True,
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
                self.mask_reg = mask_sparsity * tf.reduce_sum(tf.abs(self.masks))
                self.feature_reg = feature_sparsity * tf.reduce_sum(tf.abs(self.feature_weights))
                self.readout_reg = self.mask_reg + self.feature_reg
                tf.losses.add_loss(self.readout_reg, tf.GraphKeys.REGULARIZATION_LOSSES)

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
                    self.output = tf.identity(soft_threshold(self.h + self.biases), name='output')
                else:
                    self.output = tf.identity(self.h + self.biases, name='output')
                

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



