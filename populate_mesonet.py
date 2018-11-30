import datajoint as dj
import numpy as np
from cnn_sys_ident.mesonet.parameters import Fit, RegPath, Model, Core, Readout
from cnn_sys_ident.mesonet.data import MultiDataset
from cnn_sys_ident.mesonet import MODELS

'''
RegPath().populate(reserve_jobs=True)

""" Main model selection using
   - three layer rotation-equivariant net
   - filter sizes: 13, 5, 5
   - positive readout weights
   - sparse readout weights
   - no shared biases across rotations
"""
data_key = dict(data_hash='cfcd208495d565ef66e7dff9f98764da')
model_rel = MODELS['HermiteSparse'] * MultiDataset() & data_key & \
    'shared_biases=False AND positive_feature_weights=True'

Fit().populate(model_rel, reserve_jobs=True, suppress_errors=True, order='random')


""" Ablation experiments """
n = (Fit() * model_rel).fetch('num_filters_2', order_by='val_loss', limit=1)[0]
num_filters = 'num_filters_2={:d}'.format(n)

# No sparsity on feature weights
model_rel = MODELS['HermiteDense'] & \
    'shared_biases=False AND positive_feature_weights=True' & num_filters
Fit().populate(model_rel, reserve_jobs=True, suppress_errors=True, order='random')

# No positivity constraint on feature weights
model_rel = MODELS['HermiteSparse'] & \
    'shared_biases=False AND positive_feature_weights=False' & num_filters
Fit().populate(model_rel, reserve_jobs=True, suppress_errors=True, order='random')

# Shared biases across rotations
model_rel = MODELS['HermiteSparse'] & \
    'shared_biases=True AND positive_feature_weights=True' & num_filters
Fit().populate(model_rel, reserve_jobs=True, suppress_errors=True, order='random')

# Regular CNN
model_rel = MODELS['CNNSparse'] & 'positive_feature_weights=True'
Fit().populate(model_rel, reserve_jobs=True, suppress_errors=True, order='random')
'''

'''
data_key = dict(data_hash='cfcd208495d565ef66e7dff9f98764da')
# for k in [48, 32]:
for k in [16]:

    num_filters = 'num_filters_2 = {:d}'.format(k)

    # No sparsity on feature weights
    model_rel = MODELS['HermiteDense'] * MultiDataset() & data_key & \
        'shared_biases=False AND positive_feature_weights=False' & num_filters
    Fit().populate(model_rel, reserve_jobs=True, suppress_errors=True, order='random')

    # No positivity constraint on feature weights
    model_rel = MODELS['HermiteSparse'] * MultiDataset() & data_key & \
        'shared_biases=False AND positive_feature_weights=False' & num_filters
    Fit().populate(model_rel, reserve_jobs=True, suppress_errors=True, order='random')

    # Shared biases across rotations
    model_rel = MODELS['HermiteSparse'] * MultiDataset() & data_key & \
        'shared_biases=True AND positive_feature_weights=False' & num_filters
    Fit().populate(model_rel, reserve_jobs=True, suppress_errors=True, order='random')


# Regular CNN
data_key = dict(data_hash='cfcd208495d565ef66e7dff9f98764da')
n = [128, 128, 256]
key = {'num_filters_{:d}'.format(i): n[i] for i in range(len(n))}
model_rel = MODELS['CNNSparse'] * MultiDataset() & data_key \
    & 'positive_feature_weights=False' & key
Fit().populate(model_rel, reserve_jobs=True, suppress_errors=True, order='random')


RegPath().populate(reserve_jobs=True)

data_key = dict(data_hash='cfcd208495d565ef66e7dff9f98764da')
for k in [8, 16, 32, 12, 24, 40, 10, 14, 20, 28]:
    num_filters = 'num_filters_2 = {:d}'.format(k)
    model_rel = MODELS['HermiteSparse'] * MultiDataset() \
        & data_key & num_filters \
        & 'shared_biases=False AND positive_feature_weights=False'

    Fit().populate(model_rel, reserve_jobs=True, suppress_errors=True, order='random')
'''

'''
data_key = dict(data_hash='cfcd208495d565ef66e7dff9f98764da')
model_rel = MODELS['HermiteTransfer'] * MultiDataset() & data_key
Fit().populate(model_rel, reserve_jobs=True, suppress_errors=True, order='random')
'''

'''
# Control with L2 on feature weights, applied separately from L1 on masks
data_key = dict(data_hash='cfcd208495d565ef66e7dff9f98764da')
model_rel = MODELS['HermiteSparseSeparate'] * MultiDataset() & data_key & 'num_filters_2=16 AND shared_biases=False'
Fit().populate(model_rel, reserve_jobs=True, suppress_errors=True, order='random')
'''

'''
# Control with data split in two halves
from cnn_sys_ident.mesonet.controls import FitDataSplit
data_key = dict(data_hash='cfcd208495d565ef66e7dff9f98764da')
model_rel = MODELS['HermiteSparse'] * MultiDataset() & data_key & 'num_filters_2=16 AND shared_biases=False'
FitDataSplit().populate(model_rel, reserve_jobs=True, suppress_errors=True, order='random')
'''

'''
from cnn_sys_ident.mesonet.controls import FitDataSplit, MEI
unit_ids = np.load('analysis/paper/figures/unit_ids.npy')
keys = [dict(unit_id=id) for id in unit_ids.flatten()]
MEI.populate(keys, reserve_jobs=True, suppress_errors=True, order='random')
'''

from cnn_sys_ident.mesonet.vis import MEI
unit_ids = np.load('analysis/paper/figures/unit_ids.npy')
keys = [dict(unit_id=id) for id in unit_ids.flatten()]
MEI.populate(keys, 'reg_seed=1954773306', reserve_jobs=True, suppress_errors=True, order='random')

