import datajoint as dj
from cnn_sys_ident.mesonet.parameters import Fit
key = dict(data_hash='cfcd208495d565ef66e7dff9f98764da')
Fit().populate(key, reserve_jobs=True, suppress_errors=True, order='random')
