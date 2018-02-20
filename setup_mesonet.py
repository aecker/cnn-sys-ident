import datajoint as dj
from cnn_sys_ident.mesonet.data import Area, MultiDataset
from cnn_sys_ident.mesonet.parameters import Core, Readout, Model, RegPath

Area().fill()
MultiDataset().make_groups()
Core().fill()
Readout().fill()
Model().fill()
RegPath().populate()
