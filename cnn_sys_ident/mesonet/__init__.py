from .parameters import Model, Core, Readout

MODELS = {
    'RotEquiSparse':
        Model.CorePlusReadout() * \
        Core.ThreeLayerRotEquiConv2d() * \
        Readout.SpatialXFeatureJointL1(),
    'RotEquiDense':
        Model.CorePlusReadout() * \
        Core.ThreeLayerRotEquiConv2d() * \
        Readout.SpatialSparseXFeatureDense(),
    'HermiteSparse':
        Model.CorePlusReadout() * \
        Core.ThreeLayerRotEquiHermiteConv2d() * \
        Readout.SpatialXFeatureJointL1(),
    'HermiteDense':
        Model.CorePlusReadout() * \
        Core.ThreeLayerRotEquiHermiteConv2d() * \
        Readout.SpatialSparseXFeatureDense(),
}
