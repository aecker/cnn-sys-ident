import datajoint as dj
from itertools import product

from .parameters import Fit
from .data import MultiDataset
from . import MODELS

schema = dj.schema('aecker_mesonet_insilico', locals())


@schema
class StimulusParams(dj.Lookup):
    definition = """
        param_id        : tinyint unsigned  # id for parameter set
        ---
        x_start         : tinyint unsigned  # start location in x
        x_end           : tinyint unsigned  # end location in x
        y_start         : tinyint unsigned  # start location in y
        y_end           : tinyint unsigned  # end location in y
        min_size        : tinyint unsigned  # minimum size in pixels
        num_sizes       : tinyint unsigned  # number of different sizes
        size_increment  : float             # relative size increments
        num_orientations : tinyint unsigned # number of orientations
        num_phases      : tinyint unsigned  # number of phases
        num_sf          : tinyint unsigned  # number of spatial frequencies (SF)
        sf_increment    : float             # relative SF increments
        """


class InSilicoTuning(dj.Computed):
    definition = """
        -> StimulusParams
        -> Fit
        """

    @property
    def key_source(self):
        raise NotImplementedError('Recompute best model first!')
        
        data_key = {'data_hash': 'cfcd208495d565ef66e7dff9f98764da'}
        dataset = MultiDataset() & data_key
        num_filters = 16
        model_rel = MODELS['HermiteSparse'] * dataset \
            & 'positive_feature_weights=False AND shared_biases=False' \
            & {'num_filters_2': num_filters}
        key = (Fit() * model_rel).fetch(dj.key, order_by='val_loss', limit=1)[0]
        return Fit() * StimulusParams() & key

    def _make_tuples(self, key):
        raise NotImplementedError()

