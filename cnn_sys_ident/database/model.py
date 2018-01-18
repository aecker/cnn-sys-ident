from cnn_sys_ident import architectures
from ..architectures import models, cores, readouts
from .config import Config


class Model(Config):
    """Abstract base class for model definition."""
    
    _type = 'model'

    def build(self, key, data, regularization_parameters):
        type_name = self._type + '_type'
        the_type = (self & key).fetch1(type_name)
        part = getattr(self, the_type)
        return part().build(key, data, regularization_parameters)


class Component(Config):
    """Abstract base class for definition of a type of model components.
    
    For an example, see database/core.py.
    """
    
    def parameters(self, key):
        part = self.instance(key)
        p = (self * part).fetch1()
        return part.decode_params_from_db(p)

    def build(self, key, base, inputs, regularization_parameters):
        parameters = self.parameters(key)
        the_type = parameters.pop(self._type + '_type')
        part = getattr(self, the_type)
        class_name = part().class_name + self._type.title()
        module = getattr(architectures, self._type + 's')
        assert hasattr(module, class_name), (
            '''Cannot find {t} for {name}. '''
            '''It needs to be named "{name}{T} in architectures.{config_type}s'''.format(
                t=self._type, T=self._type.title(), name=class_name))
        the_class = getattr(module, class_name)
        return the_class(base, inputs, **parameters, **regularization_parameters)


class ComponentPart:
    """Abstract base class for definition of a model component."""

    def decode_params_from_db(self, p):
        return p

    @property
    def class_name(self):
        return self.__class__.__name__

    @property
    def content(self):
        raise NotImplementedError(
            'Subclasses have to implement this property to fill the database!')


# TODO: Remove (dummies)
class Data:
    num_neurons = 10
    def train(self):
        return None, np.ones([100, 10])

import tensorflow as tf
class BaseModel:
    is_training = tf.constant(False)
    inputs = tf.constant([100, 20, 20, 1])
    
    def __init__(self, data):
        self.data = data


class CorePlusReadout:
    _core_table = None
    _readout_table = None

    @property
    def definition(self):
        return """
            -> master
            ---
            -> {core}
            -> {readout}
        """.format(core=self._core_table.__name__, readout=self._readout_table.__name__)

    def build(self, key, data, regularization_parameters):
        base = BaseModel(data)
        core_key = (self * self._core_table() & key).fetch1(dj.key)
        core = self._core_table().build(core_key, base, base.inputs, regularization_parameters)
        readout_key = (self * self._readout_table() & key).fetch1(dj.key)
        readout = self._readout_table().build(readout_key, base, core.output, regularization_parameters)
        return CorePlusReadoutModel(base, core, readout)
