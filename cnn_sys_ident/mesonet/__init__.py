from .parameters import Model, Core, Readout

MODELS = {
    'CNNSparse':
        Model.CorePlusReadout() * \
        Core.ThreeLayerConv2d() * \
        Readout.SpatialXFeatureJointL1(),
    'CNNDense':
        Model.CorePlusReadout() * \
        Core.ThreeLayerConv2d() * \
        Readout.SpatialSparseXFeatureDense(),
    'HermiteSparse':
        Model.CorePlusReadout() * \
        Core.ThreeLayerRotEquiHermiteConv2d() * \
        Readout.SpatialXFeatureJointL1(),
    'HermiteSparseSeparate':
        Model.CorePlusReadout() * \
        Core.ThreeLayerRotEquiHermiteConv2d() * \
        Readout.SpatialXFeatureSeparateL1(),
    'HermiteDense':
        Model.CorePlusReadout() * \
        Core.ThreeLayerRotEquiHermiteConv2d() * \
        Readout.SpatialSparseXFeatureDense(),
    'HermiteDenseSeparate':
        Model.CorePlusReadout() * \
        Core.ThreeLayerRotEquiHermiteConv2d() * \
        Readout.SpatialSparseXFeatureDenseSeparate(),
    'HermiteTransfer':
        Model.CorePlusReadout() * \
        Core.ThreeLayerRotEquiHermiteConv2d() * \
        Readout.SpatialXFeatureJointL1Transfer(),
}
