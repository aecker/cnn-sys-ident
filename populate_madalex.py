import datajoint as dj
from cnn_sys_ident.madalex.parameters import Fit, RegPath, Model, Core, Readout
from cnn_sys_ident.madalex.data import MultiDataset
from cnn_sys_ident.madalex import MODELS

""" Main model selection using
   - three layer rotation-equivariant net
   - filter sizes: 13, 5, 5
   - positive readout weights
   - sparse readout weights
   - no shared biases across rotations
"""
model_rel = MODELS['HermiteSparse'] & 'shared_biases=False AND positive_feature_weights=True'
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
