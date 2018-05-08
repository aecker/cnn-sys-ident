import datajoint as dj
from itertools import product

from ..database import model, core, readout, regularization as reg, fit
from .data import MultiDataset

schema = dj.schema('aecker_madalex_parameters', locals())


@schema
class Core(core.Core, dj.Lookup):

    class ThreeLayerConv2d(core.StackedConv2d, dj.Part):
        _num_layers = 3
        _conv_smooth_min = [0.001]
        _conv_smooth_max = [0.01]
        _conv_sparse_min = [0.001]
        _conv_sparse_max = [0.01]
        _filter_size = [[13, 5, 5]]
        _num_filters = [
            [64, 64, 64],
            [128, 128, 128],
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
            for p in product(self._conv_smooth_min, self._conv_smooth_max,
                             self._conv_sparse_min, self._conv_sparse_max,
                             self._filter_size, self._num_filters,
                             self._stride, self._rate, self._padding,
                             self._activation_fn, self._rel_smooth_weight,
                             self._rel_sparse_weight):
                yield self.encode_params_for_db(dict(zip(self.parameter_names, p)))


    class ThreeLayerRotEquiHermiteConv2d(core.StackedRotEquiHermiteConv2d, dj.Part):
        _num_layers = 3
        _conv_smooth_min = [0.001]
        _conv_smooth_max = [0.03]
        _conv_sparse_min = [0.001]
        _conv_sparse_max = [0.1]
        _num_rotations = [8]
        _upsampling = [2]
        _filter_size = [[13, 5, 5]]
        _num_filters = [
            [8, 16, 32],
            [16, 16, 16],
        ]
        _stride = [[1, 1, 1]]
        _rate = [[1, 1, 1]]
        _padding = [['VALID', 'VALID', 'VALID']]
        _activation_fn= [
            ['soft', 'soft', 'soft'],
            ['soft', 'soft', 'none'],
        ]
        _rel_smooth_weight = [[1, 0, 0]]
        _rel_sparse_weight = [[0, 1, 1]]

        @property
        def content(self):
            for p in product(self._conv_smooth_min, self._conv_smooth_max,
                             self._conv_sparse_min, self._conv_sparse_max,
                             self._num_rotations, self._upsampling,
                             self._filter_size, self._num_filters,
                             self._stride, self._rate,
                             self._padding, self._activation_fn,
                             self._rel_smooth_weight, self._rel_sparse_weight):
                yield self.encode_params_for_db(dict(zip(self.parameter_names, p)))


@schema
class Readout(readout.Readout, dj.Lookup):

    class SpatialXFeatureJointL1(readout.SpatialXFeatureJointL1, dj.Part):
        _readout_sparsity_min = [0.01]
        _readout_sparsity_max = [0.06]
        _positive_feature_weights = [False, True]
        _init_masks = ['sta', 'rand']

        @property
        def content(self):
            for p in product(self._readout_sparsity_min, self._readout_sparsity_max,
                             self._positive_feature_weights, self._init_masks):
                yield(dict(zip(self.parameter_names, p)))

    class SpatialSparseXFeatureDense(readout.SpatialSparseXFeatureDense, dj.Part):
        _mask_sparsity_min = [0.01]
        _mask_sparsity_max = [0.06]
        _positive_feature_weights = [False]
        _init_masks = ['sta', 'rand']

        @property
        def content(self):
            for p in product(self._mask_sparsity_min, self._mask_sparsity_max,
                             self._positive_feature_weights, self._init_masks):
                yield(dict(zip(self.parameter_names, p)))


@schema
class Model(model.Model, dj.Lookup):

    class CorePlusReadout(model.CorePlusReadout, dj.Part):
        _core_table = Core
        _readout_table = Readout
        _reg_path_generator = lambda self, seed, reg_params: reg.random_search(seed, reg_params, num_models=32)

        @property
        def content(self):
            for core_key, readout_key in product(
                    Core().fetch(dj.key),
                    Readout().fetch(dj.key)):
                yield(dict(core_key, **readout_key))


@schema
@reg.regularizable([Core, Readout])
class RegPath(reg.RegPath, dj.Computed):
    _model_table = Model


@schema
class Fit(fit.Fit, dj.Computed):
    _reg_path_table = RegPath
    _data_table = MultiDataset
    
