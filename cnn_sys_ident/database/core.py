from collections import OrderedDict
import numpy as np

from .model import Component
from .regularization import RegularizableComponent


class Core(Component):
    _type = 'core'


class Stacked(RegularizableComponent):
    _num_layers = None
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


class StackedConv2d(Stacked):
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


class StackedRotEquiConv2d(Stacked):
    _regularization_parameters = ['conv_smooth_weight', 'conv_sparse_weight']
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

    @property
    def class_name(self):
        return 'StackedRotEquiConv2d'


class StackedRotEquiHermiteConv2d(Stacked):
    _regularization_parameters = ['conv_smooth_weight', 'conv_sparse_weight']
    _parameters = OrderedDict([
        ('num_rotations',     'tinyint               # number of rotations'),
        ('upsampling',        'tinyint               # upsampling factor for filters'),
        ('shared_biases',     'boolean               # share biases across rotations'),
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

    @property
    def class_name(self):
        return 'StackedRotEquiHermiteConv2d'

