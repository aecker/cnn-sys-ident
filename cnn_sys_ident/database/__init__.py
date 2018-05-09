from itertools import product

def cartesian_product(params):
    keys, values = [], []
    for key, val in params.items():
        keys.append(key)
        values.append(val)
    dicts = []
    for val in product(*values):
        dicts.append(dict(zip(keys, val)))
    return dicts
