import datajoint as dj

from ..architectures.training import Trainer
from ..architectures.models import TFSession, BaseModel


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
        tf_session = TFSession(log_dir='checkpoints', log_hash=key['model_hash'])
        data = (self._data_table() & key).load_data()
        base = BaseModel(tf_session, data)
        model = self._reg_path_table().build_model(key, base)
        trainer = Trainer(base, model)
        tupl = key
        tupl['num_iterations'], tupl['val_loss'], tupl['test_corr'] = trainer.fit()
        self.insert1(tupl)

