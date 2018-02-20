from collections import OrderedDict
from inspect import isclass
import hashlib

from ..utils.logging import Messager

import datajoint as dj
import numpy as np

from cnn_sys_ident import architectures
from .model import ComponentPart


class RegularizableComponent(ComponentPart):
    """Abstract base class for regularizable model components.
    
    Subclass this class to create a model component (e.g. Core, Readout...).
    Subclasses define two properties:
    
        _regularization_parameters (list of strings):
            Names of regularization parameters. These are used to create 
            a regularization path (e.g. random or grid search; see class
            RegPath below).
        _parameters (OrderedDict):
            Definition of regular parameters ('name', 'datatype # comment').
    """
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


def regularizable(components):
    """Decorator for class defining regularization path.
    
    This decorator dynamically adds the class definitions defining the subtables
    for each model component's regularization parameters. Example:
    
    @regularizable([Core, Readout])
    @schema
    class RegPath(dj.Lookup):
        ...
    """
    def add_subtables(model):
        setattr(model, '_regularizable_components', components)
        for c in components:
            for part in c().parts:
                definition = """
                        -> master
                        ---
                        """
                for p in part._regularization_parameters:
                    definition += """
                        {param_name} : float # {param_name}""".format(param_name=p)
                setattr(model, part.__name__, type(
                    part.__name__, (dj.Part, ), dict(definition=definition)))
        return model
    return add_subtables


class RegPath(Messager):
    """Abstract base class for regularization path.
    
    Subclass this class to define the tables storing the regularization path.
    Add the decorator @regularizable (see above) to subclasses to define the
    subtables storing each component's regularization parameters.
    
    Subclasses define the following property:
        _model_table (Model):
            Reference to the class defining models (see database/models)
    """
    _model_table = None
    _regularizable_components = []

    @property
    def definition(self):
        return """
            -> {model}
            reg_seed : int unsigned # regularization seed
            ---
        """.format(model=self._model_table.__name__)

    @property
    def parts(self):
        for member in dir(self):
            if isclass(getattr(self, member)) and issubclass(getattr(self, member), dj.Part):
                yield getattr(self, member)

    def build_model(self, key, base):
        reg_params = dict()
        for part in self.parts:
            p = (part() & key).fetch1()
            reg_params = dict(reg_params, **p)
        return self._model_table().build(key, base, reg_params)

    def _make_tuples(self, key):
        model = self._model_table().instance(key)
        c_key = model.fetch1()
        reg_param_bounds = OrderedDict()
        for c in self._regularizable_components:
            component = c().instance(c_key)
            for param_name in component._regularization_parameters:
                val_min, val_max = component.fetch1(param_name + '_min', param_name + '_max')
                reg_param_bounds[param_name] = (val_min, val_max)

        seed = int(key['model_hash'], 16) % (2 ** 32)
        for p_seed, reg_params in model._reg_path_generator(seed, reg_param_bounds):
            tupl = dict(key, reg_seed=p_seed, **reg_params)
            self.insert1(tupl, ignore_extra_fields=True)
            for c in self._regularizable_components:
                the_type = c().instance_type(c_key)
                part = getattr(self, the_type)
                part().insert1(tupl, ignore_extra_fields=True)


def random_search(seed, reg_param_bounds, num_models=32):
    """Random search over space of regularizaton parameters.
    
    Given hard bounds of the regularization parameters, this function samples
    uniformly in log-transformed parameter space.
    """
    for i in range(num_models):
        reg_params = dict()
        i_seed = (seed + i) % (2 ** 32)
        for key, (val_min, val_max) in reg_param_bounds.items():
            p_seed = (i_seed + int(hashlib.md5(key.encode('utf8')).hexdigest(), 16)) % (2 ** 32)
            state = np.random.RandomState(seed=p_seed)
            rnd = np.exp(state.uniform(np.log(val_min), np.log(val_max)))
            reg_params[key] = rnd
        yield i_seed, reg_params

