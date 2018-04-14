import tensorflow as tf
import numpy as np
import os
import inspect
import random

class TFSession:
    _saver = None
    
    @property
    def saver(self):
        if self._saver is None:
            with self.graph.as_default():
                self._saver = tf.train.Saver(max_to_keep=1)
        return self._saver
    
    def __init__(self, log_dir=None, log_hash=None):
        log_dir_ = os.path.dirname(os.path.dirname(os.path.dirname(inspect.stack()[0][1])))
        log_dir = os.path.join(log_dir_, 'checkpoints' if log_dir is None else log_dir)
        if log_hash == None: log_hash = '%010x' % random.getrandbits(40)
        self.log_dir = os.path.join(log_dir, log_hash)
        self.log_hash = log_hash
        self.seed = int.from_bytes(log_hash[:4].encode('utf8'), 'big')
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
        self.session = tf.Session(graph=self.graph)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def close(self):
        if self.session is not None:
            self.session.close()

    def save(self):
        with self.graph.as_default():
            self.saver.save(self.session, os.path.join(self.log_dir, 'model.ckpt'))

    def load(self):
        with self.graph.as_default():
            self.saver.restore(self.session, os.path.join(self.log_dir, 'model.ckpt'))


class BaseModel:
    def __init__(self, data, log_dir=None, log_hash=None):
        self.tf_session = TFSession(log_dir=log_dir, log_hash=log_hash)
        self.data = data
        with self.tf_session.graph.as_default():
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.inputs = tf.placeholder(tf.float32, shape=data.input_shape, name='inputs')
            self.responses = tf.placeholder(tf.float32, shape=[None, data.num_neurons], name='responses')

    def evaluate(self, var, *args, **kwargs):
        return self.tf_session.session.run(var, *args, **kwargs)


class CorePlusReadoutModel:
    def __init__(self, base, core, readout):
        self.base = base
        self.core = core
        self.readout = readout
        self.predictions = readout.output

