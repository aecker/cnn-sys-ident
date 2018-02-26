import datajoint as dj

from ..architectures.training import Trainer
from ..architectures.models import TFSession, BaseModel
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
        model = self.get_model(key)
        trainer = Trainer(model.base, model)
        tupl = key
        tupl['num_iterations'], tupl['val_loss'], tupl['test_corr'] = trainer.fit()
        self.insert1(tupl)

    def get_model(self, key):
        log_hash = key_hash(key)
        data = (self._data_table() & key).load_data()
        base = BaseModel(data, log_dir='checkpoints', log_hash=log_hash)
        model = self._reg_path_table().build_model(key, base)
        return model

