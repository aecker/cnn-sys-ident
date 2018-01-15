import datajoint as dj
from itertools import product
from collections import OrderedDict

from ..database import parameters as p


schema = dj.schema('aecker_mesonet_parameters', locals())


@schema
class Core(p.Core, dj.Lookup):

    class ThreeLayerConv2d(p.StackedConv2d, dj.Part):
        _num_layers = 3
        _conv_smooth_min = [0.001]
        _conv_smooth_max = [0.01]
        _conv_sparse_min = [0.001]
        _conv_sparse_max = [0.01]
        _filter_size = [[13, 5, 5]]
        _num_filters = [
            [64, 64, 64],
            [128, 128, 128],
            [256, 256, 256],
            [64, 128, 256],
        ]
        _stride = [[1, 1, 1]]
        _rate = [[1, 1, 1]]
        _padding = [['VALID', 'VALID', 'VALID']]
        _activation_fn= [['soft', 'soft', 'soft']]
        _rel_smooth_weight = [[1, 0, 0]]
        _rel_sparse_weight = [[0, 1, 1]]

        @property
        def content(self):
            for p in product(self._conv_smooth_min, self._conv_smooth_max, self._conv_sparse_min, self._conv_sparse_max,
                             self._filter_size, self._num_filters, self._stride, self._rate, self._padding,
                             self._activation_fn, self._rel_smooth_weight, self._rel_sparse_weight):
                yield self.encode_params_for_db(dict(zip(self.parameter_names, p)))


@schema
class Readout(p.Readout, dj.Lookup):

    class SpatialXFeatureJointL1(p.SpatialXFeatureJointL1, dj.Part):
        _readout_sparsity_min = [0.01]
        _readout_sparsity_max = [0.04]
        _positive_feature_weights = [False, True]

        @property
        def content(self):
            for p in product(self._readout_sparsity_min, self._readout_sparsity_max, self._positive_feature_weights):
                yield(dict(zip(self.parameter_names, p)))


@schema
@p.regularizable([Core, Readout])
class Model(p.RegularizableModel, dj.Lookup):
    definition="""
        -> Core
        -> Readout
        ---
        num_models : integer # number of regularization parameter combinations to try
    """

    # TODO: Store contents here or make function that inserts model combinations
    #       (currently needs to be done by hand)

