import datajoint as dj
from cnn_sys_ident.mesonet.parameters import Fit, RegPath, Model, Core, Readout
from cnn_sys_ident.mesonet.data import MultiDataset

#model_rel = Model.CorePlusReadout() * \
#    Readout.SpatialXFeatureJointL1() * \
#    Core.ThreeLayerRotEquiConv2d() & \
#    'num_rotations in (8, 12) and num_filters_0 in (8, 16)' & \
#    'init_masks="rand" and positive_feature_weights=0'
#model_rel = Model.CorePlusReadout() * \
#    Readout.SpatialXFeatureJointL1() * \
#    Core.ThreeLayerRotEquiHermiteConv2d() & \
#    'init_masks="rand" and positive_feature_weights=0'
model_rel = Model.CorePlusReadout() * \
    Readout.SpatialSparseXFeatureDense() * \
    Core.ThreeLayerRotEquiHermiteConv2d() & \
    'init_masks="rand" and positive_feature_weights=0'
rel = MultiDataset() * RegPath() * model_rel

key = dict(data_hash='cfcd208495d565ef66e7dff9f98764da')
key = (rel & key).fetch(dj.key)
Fit().populate(key, reserve_jobs=True, suppress_errors=True, order='random')
