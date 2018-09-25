import datajoint as dj
import os

from .data import MultiDataset
from .parameters import Fit
from ..utils.mei import ActivityMaximization
from . import MODELS

schema = dj.schema('aecker_mesonet_vis', locals())


@schema
class MEIParams(dj.Lookup):
    definition = """  # CNN fitting and results
        param_id      : tinyint unsigned      # parameter id
        ---
        image_norm    : float                 # norm of images
        smoothness    : float                 # penalty for smoothness of images
        """
    contents = [
        [1, 12, 0.005],
        [2, 12, 0.01],
    ]


@schema
class MEIGroup(dj.Lookup):
    definition = """  # Models for which MEIs are computed
        -> Fit
        -> MEIParams
        """
    contents = [
        ['f6226c838e880183fc868c2a7483793a', 1954773337, 'cfcd208495d565ef66e7dff9f98764da', 1],
        ['f6226c838e880183fc868c2a7483793a', 1954773337, 'cfcd208495d565ef66e7dff9f98764da', 2],
        ['f6226c838e880183fc868c2a7483793a', 1954773318, 'cfcd208495d565ef66e7dff9f98764da', 2],
        ['f6226c838e880183fc868c2a7483793a', 1954773314, 'cfcd208495d565ef66e7dff9f98764da', 2],
    ]


@schema
class MEI(dj.Computed):
    definition = """  # MEI for all cells
    -> MEIGroup
    -> MultiDataset.Unit
    ---
    max_rate      : float                 # maximum rate achieved
    max_image     : blob                  # image maximizing rate
    avg_rate      : float                 # average rate achieved
    rates         : blob                  # rates
    images        : mediumblob            # set of images maximizing average rate
    """

    def _make_tuples(self, key):
        net = Fit().get_model(key)
        image_norm, smoothness = (MEIParams() & key).fetch1(
            'image_norm', 'smoothness')
        unit_id = key['unit_id']
        print(unit_id)

        tfs = net.base.tf_session
        checkpoint_file = os.path.join(tfs.log_dir, 'model.ckpt')
        shape = [net.base.data.input_shape[1], net.base.data.input_shape[2]]
        worker = ActivityMaximization(
            tfs.graph, checkpoint_file, shape, unit_id, smoothness, image_norm, num_images=8)
        images, rates, loss = worker.maximize(max_iter=2000, learning_rate=1.0)
        print('Done after {:d} iterations (max rate = {:.2f})'.format(len(loss), rates.max()))

        tupl = key
        tupl['unit_id'] = unit_id
        tupl['max_rate'] = rates.max()
        idx = rates.argmax()
        tupl['max_image'] = images[idx,:,:,0]
        tupl['avg_rate'] = rates.mean()
        tupl['rates'] = rates
        tupl['images'] = images[:,:,:,0]
        self.insert1(tupl)
