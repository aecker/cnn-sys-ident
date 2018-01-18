from collections import OrderedDict

from .model import Component
from .regularization import RegularizableComponent


class Readout(Component):
    _type = 'readout'


class SpatialXFeatureJointL1(RegularizableComponent):
    _regularization_parameters = ['readout_sparsity']
    _parameters = OrderedDict([
        ('positive_feature_weights', 'boolean # enforce positive feature weights?'),
    ])
