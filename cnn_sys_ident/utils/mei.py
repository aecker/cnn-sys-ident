import tensorflow as tf
import numpy as np


@tf.RegisterGradient('gradient_preconditioning')
def _gradient_preconditioning(op, grad):
    '''Spatial whitening by FFT assuming 1/sqrt(F) spectrum'''
    num_px_i, num_px_j = int(grad.shape[1]), int(grad.shape[2])
    grad = tf.transpose(grad, [0, 3, 1, 2])
    grad_fft = tf.fft2d(tf.cast(grad, tf.complex64))
    ti = np.minimum(np.arange(0, num_px_i), np.arange(num_px_i, 0, -1), dtype=np.float32)
    tj = np.minimum(np.arange(0, num_px_j), np.arange(num_px_j, 0, -1), dtype=np.float32)
    t = 1 / np.maximum(1.0, (tj[None,:] ** 2 + ti[:,None] ** 2) ** (1/4))
    F = tf.constant(t / t.mean(), dtype=tf.float32, name='F')
    grad_fft *= tf.cast(F, tf.complex64)
    grad = tf.ifft2d(grad_fft)
    grad = tf.transpose(tf.cast(grad, tf.float32), [0, 2, 3, 1])
    return grad


class ActivityMaximization:
    def __init__(self, graph, checkpoint_file, input_shape, cell_id, smoothness, norm, num_images=1):
        gdef = graph.as_graph_def()
        with graph.as_default():
            var_names = [v.name[:-2] for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            images_raw = tf.get_variable(
                'inputs',
                shape=[num_images, input_shape[0], input_shape[1], 1],
                initializer=tf.random_normal_initializer())
            image_norm = tf.sqrt(tf.reduce_sum(tf.square(images_raw), axis=[1, 2], keep_dims=True))
            self.images = norm * images_raw / image_norm

            # precondition gradient (only spatial whitening)
            with self.graph.gradient_override_map({'Identity': 'gradient_preconditioning'}):
                self.images = tf.identity(self.images, name='Identity')

            # smoothness prior
            lap = tf.constant([[0.25, 0.5, 0.25],
                               [0.5, -3.0, 0.5 ],
                               [0.25, 0.5, 0.25]], shape=[3, 3, 1, 1])
            images_lap = tf.nn.conv2d(self.images, lap, strides=[1, 1, 1, 1], padding='SAME')
            self.smooth_reg = smoothness * tf.reduce_sum(tf.square(images_lap))

            self.predictions, = tf.import_graph_def(
                gdef, input_map={'inputs:0': self.images, 'is_training:0': tf.constant(False)},
                name='net', return_elements=['readout/output:0'])
            self.predictions = self.predictions[:,cell_id]
            self.loss = -tf.reduce_sum(self.predictions) + self.smooth_reg
            var_list = {name: self.graph.get_tensor_by_name('net/{}:0'.format(name)) for name in var_names}
            saver = tf.train.Saver(var_list=var_list)
            self.lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=[images_raw])
            self.session = tf.Session()
            saver.restore(self.session, checkpoint_file)
            self.session.run(tf.global_variables_initializer())

    def __del__(self):
        try:
            if not self.session == None:
                self.session.close()
        except:
            pass

    def maximize(self, learning_rate=1.0, max_iter=1000, patience=100, callback=None, callback_every=100):
        loss = []
        min_loss = 1e10
        not_decreased = 0
        alpha = 0.9
        for i in range(max_iter):
            feed_dict = {self.lr: learning_rate}
            _, loss_i = self.session.run([self.train_step, self.loss], feed_dict=feed_dict)
            loss.append(loss_i)
            loss_ema = alpha * loss_ema + (1 - alpha) * loss_i if i > 0 else loss_i
            if loss_ema < min_loss:
                min_loss = loss_ema
                not_decreased = 0
            else:
                not_decreased += 1
            if not_decreased > patience:
                break
            if callback is not None and not ((i+1) % callback_every):
                callback(self, i)

        images, predictions = self.session.run([self.images, self.predictions])
        return images, predictions, np.array(loss)


class GradientRF:
    def __init__(self, graph, checkpoint_file, input_shape):
        gdef = graph.as_graph_def()
        with graph.as_default():
            var_names = [v.name[:-2] for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.image = tf.get_variable('inputs', shape=input_shape,
                        initializer=tf.constant_initializer(0.0))
            self._image = tf.reshape(self.image, [1, input_shape[0], input_shape[1], 1])
            self.predictions, = tf.import_graph_def(
                        gdef, input_map={'inputs:0': self._image, 'is_training:0': tf.constant(False)},
                        name='net', return_elements=['readout/output:0'])
            var_list = {name: self.graph.get_tensor_by_name('net/{}:0'.format(name)) for name in var_names}
            self.saver = tf.train.Saver(var_list=var_list)
            self.session = tf.Session()
            self.saver.restore(self.session, checkpoint_file)
            self.session.run(tf.global_variables_initializer())

    def __del__(self):
        try:
            if not self.session == None:
                self.session.close()
        except:
            pass

    def gradient(self, cell_id):
        grad = tf.gradients(self.predictions[0,cell_id], self.image)[0]
        return self.session.run(grad)
