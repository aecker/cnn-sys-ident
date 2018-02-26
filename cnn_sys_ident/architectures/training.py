import tensorflow as tf
import numpy as np
from scipy import stats

from .utils import poisson


class Trainer:
    
    def __init__(self, base, model):
        self.base = base
        self.session = base.tf_session.session
        self.graph = base.tf_session.graph
        self.data = base.data
        self.model = model
        with self.graph.as_default():
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            self.poisson = poisson(model.predictions, base.responses)
            self.reg_loss = tf.losses.get_regularization_loss()
            self.total_loss = self.poisson + self.reg_loss
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

    def fit(self,
            max_iter=10000,
            learning_rate=0.001,
            batch_size=256,
            val_steps=100,
            patience=5):
        with self.graph.as_default():
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            inputs_val, res_val = self.data.val()
            val_loss = np.inf
            not_improved = 0
            iter_num = 0
            self.session.run(tf.global_variables_initializer())
            for _ in range(3):
                while iter_num < max_iter:

                    # training step
                    imgs_batch, res_batch = self.data.minibatch(batch_size)
                    feed_dict = {self.base.inputs: imgs_batch,
                                 self.base.responses: res_batch,
                                 self.base.is_training: True,
                                 self.learning_rate: learning_rate}
                    self.session.run([self.train_step, update_ops], feed_dict)
                    iter_num += 1

                    # validate/save periodically
                    if not (iter_num % val_steps):
                        feed_dict_val = {self.base.inputs: inputs_val,
                                         self.base.responses: res_val,
                                         self.base.is_training: False}
                        loss = self.session.run(self.poisson, feed_dict_val)
                        print('{:4d} | Loss: {:.2f}'.format(iter_num, loss))
                        if loss < val_loss:
                            val_loss = loss
                            self.base.tf_session.save()
                            not_improved = 0
                        else:
                            not_improved += 1
                        if not_improved == patience:
                            self.base.tf_session.load()
                            iter_num -= patience * val_steps
                            not_improved = 0
                            break

                learning_rate /= np.sqrt(10)
                print('Reducing learning rate to {:f}'.format(learning_rate))

            test_corr = self.compute_test_corr()
        return iter_num, val_loss, test_corr

    def compute_test_corr(self):
        with self.graph.as_default():
            inputs, responses = self.data.test()
            feed_dict = {self.base.inputs: inputs,
                         self.base.responses: responses,
                         self.base.is_training: False}
            predictions = self.session.run(self.model.predictions, feed_dict)
        rho = np.zeros(self.data.num_neurons)
        for i, (res, pred) in enumerate(zip(responses.T, predictions.T)):
            if np.std(res) > 1e-5 and np.std(pred) > 1e-5:
                rho[i] = stats.pearsonr(res, pred)[0]
        return rho.mean()

