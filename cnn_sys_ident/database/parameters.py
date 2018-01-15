from collections import OrderedDict, namedtuple
from inspect import isclass
import hashlib

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

class Component(Messager):
    _type = None

    @property
    def definition(self):
        return """
        # parameters for {cn}

        {ct}_hash                   : varchar(256) # unique identifier for configuration 
        ---
        {ct}_type                   : varchar(50)  # type
        {ct}_ts=CURRENT_TIMESTAMP   : timestamp    # automatic
        """.format(ct=self._type, cn=self.__class__.__name__)

    @property
    def parts(self):
        for member in dir(self):
            if isclass(getattr(self, member)) and issubclass(getattr(self, member), dj.Part):
                yield getattr(self, member)

    def fill(self):
        type_name = self._type + '_type'
        hash_name = self._type + '_hash'
        for rel in self.parts:
            self.msg('Checking', rel.__name__)
            for key in rel().content:
                key[type_name] = rel.__name__
                key[hash_name] = key_hash(key)

                if not key in rel().proj():
                    self.insert1(key, ignore_extra_fields=True)
                    self.msg('Inserting', key, flush=True, depth=1)
                    rel().insert1(key, ignore_extra_fields=True)

    def parameters(self, key):
        type_name = self._type + '_type'
        print((self & key))
        key = (self & key).fetch1()  # complete parameters
        part = getattr(self, key[type_name])
        p = (self * part() & key).fetch1()
        p = part().decode_params_from_db(p)
        return p

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
    def decode_params_from_db(self, p):
        return p
    
    @property
    def class_name(self):
        return self.__class__.__name__


class RegularizableComponent(ComponentPart):
    _regularization_parameters = None
    _parameters = OrderedDict()

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

    def sample_regularization_parameters(self, key, seed):
        reg_params = dict()
        for p in self._regularization_parameters:
            p_min, p_max = (self & key).fetch1(p + '_min', p + '_max')
            p_seed = (seed + int(hashlib.md5(p.encode('utf8')).hexdigest(), 16)) % (2 ** 32)
            state = np.random.RandomState(seed=p_seed)
            rnd = np.exp(state.uniform(np.log(p_min), np.log(p_max)))
            reg_params[p] = rnd
        return reg_params


def regularizable(components):  # components: list [Core, Readout]
    def add_subtables(model):
        setattr(model, '_components', components)
        for c in components:
            for part in c().parts:
                definition = """
                        -> master
                        seed : int # random number generator seed
                        ---
                        """
                for p in part._regularization_parameters:
                    definition += """
                        {param_name} : float # {param_name}""".format(param_name=p)
                setattr(model, part.__name__, type(
                    part.__name__, (dj.Part, ), dict(definition=definition)))
        return model
    return add_subtables


class RegularizableModel(Messager):
    _components = []
    
    def fill(self):
        self.msg('TO DO: Set contents of self!')
        self.msg('TO DO: Set seed in a meaningful way!')
        for c in self._components:
            self.msg('Inserting {}s'.format(c._type))
            type_name = c._type + '_type'
            for key, part_name, num_models in zip(*(self * c()).fetch(dj.key, type_name, 'num_models')):
                c_part = getattr(c, part_name)()
                part = getattr(self, part_name)()
                for seed in range(num_models):
                    reg_params = c_part.sample_regularization_parameters(key, seed)
                    tupl = dict(key, seed=seed, **reg_params)
                    part.insert1(tupl)#, ignore_extra_fields=True)


"""
Helpers for core configurations
"""

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

class Readout(Component):
    _type = 'readout'


class SpatialXFeatureJointL1(RegularizableComponent):
    _regularization_parameters = ['readout_sparsity']
    _parameters = OrderedDict([
        ('positive_feature_weights', 'boolean # enforce positive feature weights?'),
    ])
