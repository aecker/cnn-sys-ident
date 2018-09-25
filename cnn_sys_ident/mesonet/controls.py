import datajoint as dj
import numpy as np
import os

from ..architectures.training import Trainer
from ..architectures.models import BaseModel
from ..utils.data import key_hash
from .data import Dataset, MultiDataset
from .parameters import RegPath, Fit
from . import MODELS


schema = dj.schema('aecker_mesonet_controls', locals())

DATA_HASH = 'cfcd208495d565ef66e7dff9f98764da'

@schema
class TrialSubset(dj.Lookup):
    definition = """
        -> MultiDataset
        frac_trials : float # fraction of trials used for training
        ---
    """
    contents = [[DATA_HASH, 0.125],
                [DATA_HASH, 0.25],
                [DATA_HASH, 0.5],
                [DATA_HASH, 1.0]]
    
    class Trials(dj.Part):
        definition = """
            -> master
            trial_num  : int unsigned  # trial number
            ---
        """

    def fill(self):
        for key in self.fetch(dj.key):
            data = (MultiDataset() & key).load_data()
            seed = int.from_bytes(key['data_hash'][:4].encode('utf8'), 'big')
            rnd = np.random.RandomState(seed)
            trials = rnd.permutation(np.arange(data.num_train_samples))
            tuples = []
            for t in trials[:int(key['frac_trials']*len(trials))]:
                tupl = key.copy()
                tupl['trial_num'] = t
                tuples.append(tupl)
            self.Trials().insert(tuples, skip_duplicates=True)

    def load_data(self, key):
        data = (MultiDataset() & key).load_data()
        trials = (self.Trials() & key).fetch('trial_num')
        data.images_train = data.images_train[trials]
        data.responses_train = data.responses_train[trials]
        data.num_train_samples = len(trials)
        return data


@schema
class FitTrialSubset(dj.Computed):
    
    definition = """
        -> RegPath
        -> TrialSubset
        ---
        num_iterations   : int unsigned  # number of iterations
        val_loss         : float         # loss on validation set
        test_corr        : float         # correlation on test set
    """
    
    @property
    def key_source(self):
        rels = [RegPath() * TrialSubset() * MODELS['HermiteSparse'] \
                    & 'positive_feature_weights=False AND shared_biases=False AND num_filters_2=16',
                RegPath() * TrialSubset() * MODELS['CNNSparse'] \
                    & 'positive_feature_weights=False AND num_filters_0=128 AND num_filters_2=128',
               ]
        keys = []
        for rel in rels:
            keys.extend((Fit() & rel).fetch(dj.key, order_by='val_loss'))
        return RegPath() * TrialSubset() & keys

    def _make_tuples(self, key):
        model = self.get_model(key)
        trainer = Trainer(model.base, model)
        tupl = key
        tupl['num_iterations'], tupl['val_loss'], tupl['test_corr'] = trainer.fit(
            val_steps=50, learning_rate=0.002, batch_size=256, patience=5)
        self.insert1(tupl)


    def get_model(self, key):
        key = (self.key_source & key).fetch1(dj.key)
        log_hash = key_hash(key)
        data = TrialSubset().load_data(key)
        log_dir = os.path.join('checkpoints', MultiDataset().database)
        base = BaseModel(data, log_dir=log_dir, log_hash=log_hash)
        model = RegPath().build_model(key, base)
        return model

    def load_model(self, key):
        model = self.get_model(key)
        model.base.tf_session.load()
        return model


@schema
class UnitSubset(dj.Lookup):
    definition = """
        -> MultiDataset
        k_fold : tinyint unsigned # number of folds
        ---
    """
    contents = [[DATA_HASH, 5]]
    
    class Unit(dj.Part):
        definition = """
            -> master
            -> MultiDataset.Unit
            fold_num  : tinyint unsigned  # fold number
            ---
        """

    def fill(self):
        for key in self.fetch(dj.key):
            seed = int.from_bytes(key['data_hash'][:4].encode('utf8'), 'big')
            rnd = np.random.RandomState(seed)
            unit_ids = (MultiDataset.Unit() & key).fetch('unit_id')
            unit_ids = rnd.permutation(unit_ids)
            tuples = []
            for i, unit_id in enumerate(unit_ids):
                tupl = key.copy()
                tupl['fold_num'] = i % key['k_fold']
                tupl['unit_id'] = unit_id
                tuples.append(tupl)
            self.Unit().insert(tuples, skip_duplicates=True)

    def load_data(self, key):
        assert 'fold_num' in key.keys(), 'fold_num must be specified in key.'
        data = (MultiDataset() & key).load_data()
        units = (self.Unit() & key).fetch('unit_id')
        data.images_train = data.images_train[trials]
        data.responses_train = data.responses_train[trials]
        data.num_train_samples = len(trials)
        return data

