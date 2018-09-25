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
