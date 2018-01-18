import tensorflow as tf
from tensorflow.contrib import layers


class CorePlusReadoutModel:
    def __init__(self, base, core, readout):
        self.base = base
        self.core = core
        self.readout = readout
