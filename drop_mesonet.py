import datajoint as dj

schema = dj.schema('aecker_mesonet_parameters', locals())
schema.drop()
schema = dj.schema('aecker_mesonet_data', locals())
schema.drop()
