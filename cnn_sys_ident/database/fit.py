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
        model = self.get_model(key)
        trainer = Trainer(model.base, model)
        tupl = key
        tupl['num_iterations'], tupl['val_loss'], tupl['test_corr'] = trainer.fit()
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

    def get_model(self, key):
        key = (self.key_source & key).fetch1(dj.key)
        log_hash = key_hash(key)
        data = (self._data_table() & key).load_data()
        log_dir = os.path.join('checkpoints', self._data_table.database)
        base = BaseModel(data, log_dir=log_dir, log_hash=log_hash)
        model = self._reg_path_table().build_model(key, base)
        return model

    def load_model(self, key):
        model = self.get_model(key)
        model.base.tf_session.load()
        return model
