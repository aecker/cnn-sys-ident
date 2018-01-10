import hashlib


def key_hash(key):
    """
    32-byte hash used for lookup of primary keys of jobs
    """
    hashed = hashlib.md5()
    for k, v in sorted(key.items()):
        hashed.update(str(v).encode())
    return hashed.hexdigest()

def to_native(key):
    if isinstance(key, (dict, OrderedDict)):
        for k, v in key.items():
            if hasattr(v, 'dtype'):
                key[k] = v.item()
    else:
        for k, v in enumerate(key):
            if hasattr(v, 'dtype'):
                key[k] = v.item()
    return key
