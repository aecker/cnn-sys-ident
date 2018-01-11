from collections import OrderedDict, namedtuple
from inspect import isclass

from ..utils.logging import Messager
from ..utils.data import key_hash, to_native

import datajoint as dj
import numpy as np

# from ..architectures import readouts, modulators, shifters
# from .data import MesoNetMultiDataset, MesoNet
from ..architectures import cores, readouts
from cnn_sys_ident import architectures


"""
General-purpose helpers for configurations
"""

class Config(Messager):
    _config_type = None

    @property
    def definition(self):
        return """
        # parameters for {cn}

        {ct}_hash                   : varchar(256) # unique identifier for configuration 
        ---
        {ct}_type                   : varchar(50)  # type
        {ct}_ts=CURRENT_TIMESTAMP   : timestamp    # automatic
        """.format(ct=self._config_type, cn=self.__class__.__name__)

    def fill(self):
        type_name = self._config_type + '_type'
        hash_name = self._config_type + '_hash'
        for rel in [getattr(self, member) for member in dir(self)
                    if isclass(getattr(self, member)) and issubclass(getattr(self, member), dj.Part)]:
            self.msg('Checking', rel.__name__)
            for key in rel().content:
                key[type_name] = rel.__name__
                key[hash_name] = key_hash(key)

                if not key in rel().proj():
                    self.insert1(key, ignore_extra_fields=True)
                    self.msg('Inserting', key, flush=True, depth=1)
                    rel().insert1(key, ignore_extra_fields=True)

    def parameters(self, key):
        type_name = self._config_type + '_type'
        print((self & key))
        key = (self & key).fetch1()  # complete parameters
        part = getattr(self, key[type_name])
        p = (self * part() & key).fetch1()
        p = part().decode_params_from_db(p)
        return p

    def build(self, key, base, inputs, regularization_parameters):
        parameters = self.parameters(key)
        the_type = parameters.pop(self._config_type + '_type')
        part = getattr(self, the_type)
        class_name = part().class_name + self._config_type.title()
        module = getattr(architectures, self._config_type + 's')
        assert hasattr(module, class_name), (
            '''Cannot find {config_type} for {name}. '''
            '''It needs to be named "{name}{Config_type} in architectures.{config_type}s'''.format(
                config_type=self._config_type, 
                Config_type=self._config_type.title(),
                name=class_name))
        the_class = getattr(module, class_name)
        return the_class(base, inputs, **parameters, **regularization_parameters)


class ConfigPart:
    def decode_params_from_db(self, p):
        return p
    
    @property
    def class_name(self):
        return self.__class__.__name__


class RegularizableConfig(ConfigPart):
    _regularization_parameters = None

    @property
    def parameter_names(self):
        p = []
        for par in self._regularization_parameters:
            p.append(par + '_min')
            p.append(par + '_max')
        p += list(self._parameters.keys())    
        return p
            
    @property
    def definition(self):
        assert self._regularization_parameters is not None, 'self._regularization_parameters not set!'
        def_str = """
                    # parameters for {cn}

                    -> master
                    ---""".format(cn=self.__class__.__name__)
        for param in self._regularization_parameters:
            def_str += """
                    {param}_min : float # minimum value for {param}
                    {param}_max : float # maximum value for {param}
                    """.format(param=param)
        for key, val in self._parameters.items():
            def_str += """
                    {key} : {val}""".format(key=key, val=val)
        return def_str

    def decode_params_from_db(self, p):
        for par in self._regularization_parameters:
            p.pop(par + '_min')
            p.pop(par + '_max')
        return p



"""
Helpers for core configurations
"""

class CoreConfig(Config):
    _config_type = 'core'


class StackedConfig(RegularizableConfig):
    _num_layers = None
    _unique_parameters = None
    _stacked_parameters = None

    @property
    def parameter_names(self):
        return super().parameter_names + list(self._stacked_parameters.keys())
    
    @property
    def definition(self):
        assert self._num_layers is not None, 'self._num_layers not set!'
        def_str = super().definition
        for i in range(self._num_layers):
            for key, val in self._stacked_parameters.items():
                def_str += """
                    {key}_{i} : {val} (layer {i})""".format(key=key, i=i, val=val)
        return def_str

    def decode_params_from_db(self, p):
        p = super().decode_params_from_db(p)
        for key in self._stacked_parameters.keys():
            p[key] = np.array([p.pop(key + '_{}'.format(i)) for i in range(self._num_layers)])
        return p

    def encode_params_for_db(self, p):
        for key in self._stacked_parameters.keys():
            for i in range(self._num_layers):
                p[key + '_{}'.format(i)] = p[key][i] if i < self._num_layers - 1 else p.pop(key)[i]
        return p


class StackedConv2dConfig(StackedConfig):
    core_name = 'StackedConv2dCore'
    _regularization_parameters = ['conv_smooth_weight', 'conv_sparse_weight']
    _parameters = OrderedDict()
    _stacked_parameters = OrderedDict([
        ('filter_size',       'tinyint        # filter size'),
        ('num_filters',       'smallint       # number of filters'),
        ('stride',            'tinyint        # stride for filters'),
        ('rate',              'tinyint        # rate for dilated filters'),
        ('padding',           'enum("SAME", "VALID") # type of padding'),
        ('activation_fn',     'enum("none", "relu", "elu", "soft") # activation function'),
        ('rel_smooth_weight', 'float          # relative weight for smoothness regularizer'),
        ('rel_sparse_weight', 'float          # relative weight for sparseness regularizer'),
    ])
    
    @property
    def class_name(self):
        return 'StackedConv2d'


class StackedRotEquiConv2dConfig(StackedConfig):
    _parameters = OrderedDict([
        ('num_rotations',     'tinyint               # number of rotations'),
    ])
    _stacked_parameters = OrderedDict([
        ('filter_size',       'tinyint               # filter size'),
        ('num_filters',       'smallint              # number of filters'),
        ('stride',            'tinyint               # stride for filters'),
        ('rate',              'tinyint               # rate for dilated filters'),
        ('padding',           'enum("SAME", "VALID") # type of padding'),
        ('activation_fn',     'enum("none", "relu", "elu", "soft") # activation function'),
        ('rel_smooth_weight', 'float                 # relative weight for smoothness regularizer'),
        ('rel_sparse_weight', 'float                 # relative weight for sparseness regularizer'),
    ])
        


"""
Helpers for readout configurations
"""

class ReadoutConfig(Config):
    _config_type = 'readout'

