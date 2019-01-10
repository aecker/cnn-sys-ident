import datajoint as dj
import os

from ..architectures.training import Trainer
from ..architectures.models import BaseModel
from ..utils.data import key_hash


class Fit:
    _reg_path_table = None
    _data_table = None
    
    @property
    def definition(self):
        return """
            -> {reg_path}
            -> {dataset}
            ---
            num_iterations   : int unsigned  # number of iterations
            val_loss         : float         # loss on validation set
            test_corr        : float         # correlation on test set
        """.format(reg_path=self._reg_path_table.__name__,
                   dataset=self._data_table.__name__)

    def _make_tuples(self, key):
        raise NotImplementedError('To be implemented by child classes.')

    def get_hash(self, key):
        key = (self.key_source & key).fetch1(dj.key)
        return key_hash(key)

    def get_model(self, key):
        log_hash = self.get_hash(key)
        data = (self._data_table() & key).load_data()
        log_dir = os.path.join('checkpoints', self._data_table.database)
        base = BaseModel(data, log_dir=log_dir, log_hash=log_hash)
        model = self._reg_path_table().build_model(key, base)
        return model

    def load_model(self, key):
        model = self.get_model(key)
        model.base.tf_session.load()
        return model
