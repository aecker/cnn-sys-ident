import numpy as np
import datajoint as dj
from itertools import product

from .parameters import Fit
from .data import MultiDataset
from . import MODELS
from ..utils.stimuli import GaborSet

schema = dj.schema('aecker_mesonet_insilico', locals())

BATCH_SIZE = 1024


@schema
class GaborParams(dj.Lookup):
    definition = """
        param_id        : tinyint unsigned  # id for parameter set
        ---
        x_start         : tinyint unsigned  # start location in x
        x_end           : tinyint unsigned  # end location in x
        y_start         : tinyint unsigned  # start location in y
        y_end           : tinyint unsigned  # end location in y
        min_size        : float             # minimum size in pixels
        num_sizes       : tinyint unsigned  # number of different sizes
        size_increment  : float             # relative size increments
        min_sf          : float             # minimum spatial frequency
        num_sf          : tinyint unsigned  # number of spatial frequencies (SF)
        sf_increment    : float             # relative SF increments
        min_contrast    : float             # minimum contrast (Michelson)
        num_contrasts   : tinyint unsigned  # number of contrast levels
        contrast_increment : float          # relative contrast increments
        num_orientations : tinyint unsigned # number of orientations
        num_phases      : tinyint unsigned  # number of phases
        """

    contents = [
        [1, 27, 49, 5, 28, 8, 8, 1.25, (1.3**-1), 4, 1.3, 2**-5, 6, 2, 12, 8],
    ]

    def gabor_set(self, key, canvas_size):
        p = (self & key).fetch1()
        center_range = [p['x_start'], p['x_end'], p['y_start'], p['y_end']]
        sizes = p['min_size'] * p['size_increment'] ** np.arange(p['num_sizes'])
        sfs = p['min_sf'] * p['sf_increment'] ** np.arange(p['num_sf'])
        c = p['min_contrast'] * p['contrast_increment'] ** np.arange(p['num_contrasts'])
        g = GaborSet(canvas_size, center_range, sizes, sfs, c,
                     p['num_orientations'], p['num_phases'])
        return g


@schema
class OptimalGabor(dj.Computed):
    definition = """
        -> GaborParams
        -> Fit
        ---
        """

    class Unit(dj.Part):
        definition = """
            -> master
            -> MultiDataset.Unit
            ---
            max_response : float  # response to optimal Gabor
            max_index    : int    # index of optimal Gabor
            """

        def params(self, key):
            gabor_set = GaborParams().gabor_set(key, None)
            idx = (self & key).fetch1('max_index')
            return gabor_set.params_from_idx(idx)


    @property
    def key_source(self):
        data_key = {'data_hash': 'cfcd208495d565ef66e7dff9f98764da'}
        dataset = MultiDataset() & data_key
        num_filters = 16
        model_rel = MODELS['HermiteSparse'] * dataset \
            & 'positive_feature_weights=False AND shared_biases=False' \
            & {'num_filters_2': num_filters}
        print('Recompute best model!!')
        key = (Fit() * model_rel).fetch(dj.key, order_by='val_loss', limit=2)[1]
        return Fit() * GaborParams() & key

    def _make_tuples(self, key):
        model = Fit().load_model(key)
        s = model.base.inputs.shape.as_list()
        canvas_size = [s[2], s[1]]
        g = GaborParams().gabor_set(key, canvas_size)
        max_response, max_idx = 0, 0
        for batch_idx, images in enumerate(g.image_batches(BATCH_SIZE)):
            feed_dict = {model.base.inputs: images[...,None],
                         model.base.is_training: False}
            r = model.base.evaluate(model.predictions, feed_dict=feed_dict)
            max_r = r.max(axis=0)
            max_i = batch_idx * BATCH_SIZE + r.argmax(axis=0)
            new_max = (max_response < max_r)
            max_response = np.maximum(max_response, max_r)
            max_idx = ~new_max * max_idx + new_max * max_i
            if not (batch_idx % 10):
                print(batch_idx, max_response.mean())

        self.insert1(key)
        tuples = [dict(key, unit_id=id, max_response=mr, max_index=mi)
                      for id, mr, mi in zip(np.arange(len(max_idx)),
                                            max_response,
                                            max_idx)]
        self.Unit().insert(tuples)



@schema
class SizeContrastTuningParams(dj.Lookup):
    definition = """
        sc_param_id     : tinyint unsigned  # id for parameter set
        ---
        min_size        : float             # minimum size in pixels
        num_sizes       : tinyint unsigned  # number of different sizes
        size_increment  : float             # relative size increments
        min_contrast    : float             # minimum contrast (Michelson)
        num_contrasts   : tinyint unsigned  # number of contrast levels
        contrast_increment : float          # relative contrast increments
        """

    contents = [
        [1, 8, 12, 1.15, 2**-5.5, 12, np.sqrt(2)]
    ]

    def gabor_set(self, key, canvas_size, loc, spatial_freq, orientation, phase):
        p = (self & key).fetch1()
        center_range = [loc[0], loc[0]+1, loc[1], loc[1]+1]
        sizes = p['min_size'] * p['size_increment'] ** np.arange(p['num_sizes'])
        c = p['min_contrast'] * p['contrast_increment'] ** np.arange(p['num_contrasts'])
        g = GaborSet(canvas_size, center_range, sizes, [spatial_freq], c,
                     [orientation], [phase], relative_sf=False)
        return g


@schema
class SizeContrastTuning(dj.Computed):
    definition = """
        -> OptimalGabor
    """

    class Unit(dj.Part):
        definition = """
            -> master
            -> OptimalGabor.Unit
            ---
            tuning_curve  : blob  # sizes x contrasts
        """

    def _make_tuples(self, key):
        model = Fit().load_model(key)
        s = model.base.inputs.shape.as_list()
        canvas_size = [s[2], s[1]]
        self.insert1(key)
        for key in (OptimalGabor.Unit() & key).fetch(dj.key):
            loc, _, sf, _, ori, ph = OptimalGabor.Unit().params(key)
            g = SizeContrastTuningParams().gabor_set(key, canvas_size, loc, sf, ori, ph)
            feed_dict = {model.base.inputs: g.images()[...,None],
                         model.base.is_training: False}
            tupl = key
            pred = model.base.evaluate(model.predictions, feed_dict=feed_dict)
            tupl['tuning_curve'] = pred[:,key['unit_id']].reshape([len(g.sizes), len(g.contrasts)])
            self.Unit.insert1(tupl)
            print(key['unit_id'])


@schema
class OrthPlaidsContrastParams(dj.Lookup):
    definition = """
        opc_param_id     : tinyint unsigned  # id for parameter set
        ---
        min_contrast    : float             # minimum contrast (Michelson)
        num_contrasts   : tinyint unsigned  # number of contrast levels
        contrast_increment : float          # relative contrast increments
        """

    contents = [
        [1, 2**-5, 9, np.sqrt(2)]
    ]

    def gabor_set(self, key, canvas_size, loc, size, spatial_freq, orientation, phase):
        p = (self & key).fetch1()
        center_range = [loc[0], loc[0]+1, loc[1], loc[1]+1]
        c = p['min_contrast'] * p['contrast_increment'] ** np.arange(p['num_contrasts'])
        c = np.concatenate([np.zeros(1), c], axis=0)
        g = GaborSet(canvas_size, center_range, [size], [spatial_freq], c,
                     [orientation], [phase], relative_sf=False)
        return g


@schema
class OrthPlaidsContrast(dj.Computed):
    definition = """
        -> OptimalGabor.Unit
        ---
        tuning_curve  : blob  # TO DO
    """

    def _make_tuples(self, key):
        model = Fit().load_model(key)
        s = model.base.inputs.shape.as_list()
        canvas_size = [s[2], s[1]]
        loc, sz, sf, _, ori, ph = OptimalGabor.Unit().params(key)
        g = OrthPlaidsContrastParams().gabor_set(key, canvas_size, loc, sz, sf, ori, ph)
        components = g.images()
        plaids = components[None,...] + components[:,None,...]
        plaids = np.reshape(plaids, [-1] + s[1:])
        feed_dict = {model.base.inputs: plaids,
                     model.base.is_training: False}
        tupl = key
        tupl['tuning_curve'] = model.base.evaluate(model.predictions, feed_dict=feed_dict)
        self.insert1(tupl)
