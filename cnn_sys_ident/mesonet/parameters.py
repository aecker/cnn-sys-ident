import datajoint as dj
from itertools import product

from ..database import model, core, readout, regularization as reg, fit, cartesian_product
from ..architectures.training import Trainer
from .data import MultiDataset

schema = dj.schema('aecker_mesonet_parameters2', locals())


@schema
class Core(core.Core, dj.Lookup):

    class ThreeLayerConv2d(core.StackedConv2d, dj.Part):
        _num_layers = 3
        _params = {
            'conv_smooth_weight_min': [0.001],
            'conv_smooth_weight_max': [0.03],
            'conv_sparse_weight_min': [0.001],
            'conv_sparse_weight_max': [0.1],
            'filter_size': [[13, 5, 5]],
            'num_filters': [
                [64, 128, 128],
                [64, 128, 256],
                [32, 32, 32],
                [64, 64, 64],
                [128, 128, 128],
                [128, 128, 256],
            ],
            'stride': [[1, 1, 1]],
            'rate': [[1, 1, 1]],
            'padding': [['SAME', 'SAME', 'SAME']],
            'activation_fn': [['soft', 'soft', 'none']],
            'rel_smooth_weight': [[1, 0.5, 0.5]],
            'rel_sparse_weight': [[0, 1, 1]],
        }

        @property
        def content(self):
            for p in cartesian_product(self._params):
                yield self.encode_params_for_db(p)


    class ThreeLayerRotEquiHermiteConv2d(core.StackedRotEquiHermiteConv2d, dj.Part):
        _num_layers = 3
        _params = {
            'conv_smooth_weight_min': [0.001],
            'conv_smooth_weight_max': [0.03],
            'conv_sparse_weight_min': [0.001],
            'conv_sparse_weight_max': [0.1],
            'num_rotations': [8],
            'upsampling': [2],
            'shared_biases': [False, True],
            'filter_size': [[13, 5, 5]],
            'num_filters': [
                [16, 16, 8],
                [16, 16, 10],
                [16, 16, 12],
                [16, 16, 14],
                [16, 16, 16],
                [16, 16, 20],
                [16, 16, 24],
                [16, 16, 28],
                [16, 16, 32],
                [16, 16, 40],
                [16, 16, 48],
            ],
            'stride': [[1, 1, 1]],
            'rate': [[1, 1, 1]],
            'padding': [['SAME', 'SAME', 'SAME']],
            'activation_fn': [['soft', 'soft', 'none']],
            'rel_smooth_weight': [[1, 0.5, 0.5]],
            'rel_sparse_weight': [[0, 1, 1]],
        }

        @property
        def content(self):
            for p in cartesian_product(self._params):
                yield self.encode_params_for_db(p)


@schema
class Readout(readout.Readout, dj.Lookup):

    class SpatialXFeatureJointL1(readout.SpatialXFeatureJointL1, dj.Part):
        _params = {
            'readout_sparsity_min': [0.005],
            'readout_sparsity_max': [0.03],
            'positive_feature_weights': [False, True],
            'init_masks': ['rand'],
        }

        @property
        def content(self):
            return cartesian_product(self._params)

    class SpatialSparseXFeatureDense(readout.SpatialSparseXFeatureDense, dj.Part):
        _params = {
            'mask_sparsity_min': [0.005],
            'mask_sparsity_max': [0.03],
            'positive_feature_weights': [False, True],
            'init_masks': ['rand'],
        }

        @property
        def content(self):
            return cartesian_product(self._params)

    class SpatialXFeatureJointL1Transfer(readout.SpatialXFeatureJointL1Transfer, dj.Part):
        _params = {
            'readout_sparsity_min': [0.005],
            'readout_sparsity_max': [0.03],
            'positive_feature_weights': [False],
            'init_masks': ['rand'],
            'k_transfer': [2, 4, 8, 16, 32],
        }

        @property
        def content(self):
            return cartesian_product(self._params)


@schema
class Model(model.Model, dj.Lookup):

    class CorePlusReadout(model.CorePlusReadout, dj.Part):
        _core_table = Core
        _readout_table = Readout
        _reg_path_generator = lambda self, seed, reg_params: reg.random_search(
            seed, reg_params, num_models=32)

        @property
        def content(self):
            for core_key, readout_key in product(
                    Core.ThreeLayerConv2d().fetch(dj.key),
                    Readout.SpatialXFeatureJointL1().fetch(dj.key)):
                yield(dict(core_key, **readout_key))
            for core_key, readout_key in product(
                    Core.ThreeLayerRotEquiHermiteConv2d().fetch(dj.key),
                    Readout.SpatialXFeatureJointL1().fetch(dj.key)):
                yield(dict(core_key, **readout_key))
            for core_key, readout_key in product(
                    Core.ThreeLayerRotEquiHermiteConv2d().fetch(dj.key),
                    Readout.SpatialSparseXFeatureDense().fetch(dj.key)):
                yield(dict(core_key, **readout_key))
            for core_key, readout_key in product(
                    (Core.ThreeLayerRotEquiHermiteConv2d() \
                         & 'num_filters_2 = 16 AND shared_biases = False').fetch(dj.key),
                    Readout.SpatialXFeatureJointL1Transfer().fetch(dj.key)):
                yield(dict(core_key, **readout_key))


@schema
@reg.regularizable([Core, Readout])
class RegPath(reg.RegPath, dj.Computed):
    _model_table = Model


@schema
class Fit(fit.Fit, dj.Computed):
    _reg_path_table = RegPath
    _data_table = MultiDataset
    
    def _make_tuples(self, key):
        model = self.get_model(key)
        trainer = Trainer(model.base, model)
        tupl = key
        tupl['num_iterations'], tupl['val_loss'], tupl['test_corr'] = trainer.fit(
            val_steps=50, learning_rate=0.002, batch_size=256, patience=5)
        self.insert1(tupl)

    def _make_tuples_from_checkpoints(self, key):
        """ Recover tuples from checkpoints

        This function can replace _make_tuples to recover all tuples
        from the checkpoints on the file system in case the database
        has been lost or deleted.
        """
        key = (self.key_source & key).fetch1(dj.key)
        log_hash = key_hash(key)
        folder = '/gpfs01/bethge/home/aecker/lab/projects/microns/cnn-sys-ident/checkpoints/aecker_mesonet_data'
        if os.path.exists(os.path.join(folder, log_hash)):
            model = self.load_model(key)
            trainer = Trainer(model.base, model)
            tupl = key
            tupl['num_iterations'] = 0
            inputs_val, res_val = trainer.data.val()
            feed_dict_val = {trainer.base.inputs: inputs_val,
                             trainer.base.responses: res_val,
                             trainer.base.is_training: False}
            tupl['val_loss'] = trainer.session.run(trainer.poisson, feed_dict_val)
            tupl['test_corr'] = trainer.compute_test_corr()
            self.insert1(tupl)


