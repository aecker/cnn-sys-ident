import os
import inspect
import hashlib
import h5py
import datajoint as dj
import numpy as np


schema = dj.schema('aecker_madalex_data', locals())

PATH = os.path.dirname(os.path.dirname(os.path.dirname(inspect.stack()[0][1])))
DATA_PATH = os.path.join(PATH, 'data/madalex/v1')
FILE_NAME_PATTERN = 'animal_id={}_downsample_factor=4_extract_method=2' + \
                    '_remove_overlap=True_return_all=True_scan_idx={}' + \
                    '_session={}_shiftstats=False_spike_method=5' + \
                    '_use_difference=False.npz'
MASK_FILE_NAME_PATTERN = 'v1/masks/AI{}SE{}SI{}.npy'
SCAN_DIMS = {
    12690: (512, 256),  # width, height
    12696: (256, 256),
    11220: (512, 512),
}


class Dataset:
    def __init__(self,
                 images,
                 responses,
                 images_test,
                 responses_test,
                 images_val=None,
                 responses_val=None,
                 seed=8159,
                 train_frac=0.8):
        m = images.mean()
        sd = images.std()
        zscore = lambda img: (img - m) / sd
        if images_val is not None:
            images = np.concatenate((images, images_val), axis=0)
            responses = np.concatenate((responses, responses_val), axis=0)
        self.images = zscore(images)[...,None]
        self.images_test = zscore(images_test)[...,None]
        if responses.ndim > 2:
            responses = responses.sum(axis=1)
            responses_test = responses_test.sum(axis=1)
        sd = responses.std(axis=0)
        sd[sd < (sd.mean() / 100)] = 1
        def rectify_and_normalize(x):
            x[x < 0] = 0
            return x / sd
        self.responses = rectify_and_normalize(responses)
        self.responses_test = rectify_and_normalize(responses_test)
        self.num_neurons = self.responses.shape[1]
        self.num_images = self.responses.shape[0]
        self.px_x = self.images.shape[2]
        self.px_y = self.images.shape[1]
        self.input_shape = [None, self.px_y, self.px_x, 1]

        rnd = np.random.RandomState(seed=seed)
        if images_val is None:
            perm = rnd.permutation(self.num_images)
            self.train_idx = np.sort(perm[:round(self.num_images * train_frac)])
            self.val_idx = np.sort(perm[round(self.num_images * train_frac):])
        else:
            n_train = images.shape[0] - images_val.shape[0]
            self.train_idx = np.arange(n_train)
            self.val_idx = np.arange(n_train, images.shape[0])
        self.num_train_samples = len(self.train_idx)
        self.minibatch_idx = 1e10
        self.train_perm = []

    def val(self):
        return self.images[self.val_idx], self.responses[self.val_idx]

    def train(self):
        return self.images[self.train_idx], self.responses[self.train_idx]

    def test(self, averages=True):
        assert averages == True, 'Raw test responses currently not supported.'
        return self.images_test, self.responses_test

    def minibatch(self, batch_size):
        if self.minibatch_idx + batch_size > len(self.train_perm):
            self.next_epoch()
        idx = self.train_perm[self.minibatch_idx + np.arange(0, batch_size)]
        self.minibatch_idx += batch_size
        return self.images[self.train_idx[idx]], self.responses[self.train_idx[idx]]

    def next_epoch(self):
        self.minibatch_idx = 0
        self.train_perm = np.random.permutation(self.num_train_samples)


@schema
class Scan(dj.Lookup):
    definition = """  # Scan
    animal_id   : int           # animal id
    session_idx : tinyint       # session index
    scan_idx    : tinyint       # scan index
    ---
    comment     : varchar(255)  # comment
    """
    contents = [
        [12690, 1, 13, 'V1, MadAlex01, 150-180um'],
        [12690, 1, 15, 'V1, MadAlex01, 195-225um'],
        [12690, 1, 16, 'V1, MadAlex01, 240-270um'],
        [12696, 2, 6, 'V1, MadAlex01, site 4, 225um depth'],
        [12696, 2, 13, 'V1, MadAlex01, site 5, 225um depth'],
        [12696, 2, 15, 'V1, MadAlex02, site 5, 225um depth'],
        [12696, 2, 17, 'V1, MadAlex03, site 5, 225um depth'],
    ]

    def load_data(self):
        assert len(self) == 1, 'Relation must be scalar.'
        animal_id, session_idx, scan_idx = self.fetch1(
            'animal_id', 'session_idx', 'scan_idx')
        file_name = os.path.join(
            PATH, DATA_PATH, FILE_NAME_PATTERN.format(animal_id, scan_idx, session_idx))
        with np.load(file_name) as data:
            images = np.concatenate((
                data['inputs'],
                data['validation_inputs'],
                data['raw_test_img']), axis=0)
            responses = np.concatenate((
                data['targets'],
                data['validation_targets'],
                data['raw_test_targets']), axis=0)
        return images, responses


@schema
class MultiDataset(dj.Lookup):
    definition = """  # Dataset consisting of multiple scans and brain areas/layers
        data_hash   : char(32) # unique identifier for dataset
        ---
        restriction : varchar(255) # description
        """
    
    _order_members_by = 'animal_id ASC, session_idx ASC, scan_idx ASC'
    
    class Member(dj.Part):
        definition = """ # Scans that are part of this dataset
            -> master
            member_id  : tinyint    # member id
            ---
            -> Scan
            """

    class Unit(dj.Part):
        definition = """ # Scans that are part of this dataset
            -> master
            unit_id  : int    # unit id
            ---
            -> master.Member
            """

    def make_groups(self):
        restrictions = [
            'animal_id=12690',
        ]
        for i, r in enumerate(restrictions):
            data_hash = hashlib.md5((schema.database + r + str(i)).encode()).hexdigest()
            key = dict(data_hash=data_hash, restriction=r)
            if not len(self & key):
                self.insert1(key)
                n = 0
                for j, tupl in enumerate(
                        (self * Scan() & key & r).fetch(
                            dj.key, order_by=self._order_members_by)):
                    tupl['member_id'] = j
                    self.Member().insert1(tupl)
                    _, responses = (Scan() & tupl).load_data()
                    num_neurons = responses.shape[2]
                    units = [{'data_hash': data_hash,
                              'member_id': j,
                              'unit_id': n+k} for k in range(num_neurons)]
                    n += num_neurons
                    self.Unit().insert(units)


    def load_data(self):
        assert len(self) == 1, 'Relation must be scalar.'
        rnd = np.random.RandomState(seed=63475)
        rel = self.Member() * Scan()
        keys = rel.fetch(dj.key, order_by='animal_id ASC, session_idx ASC, scan_idx ASC')
        num_scans = len(keys)
        images_train, images_test = num_scans * [[]], num_scans * [[]]
        responses_train, responses_test = num_scans * [[]], num_scans * [[]]
        lookup = None
        for i, key in enumerate(keys):
            images, responses = (Scan() & key).load_data()

            hashes = [hashlib.md5(im).digest() for im in images]
            if lookup is None:
                lookup = {h: j for h, j in zip(
                    sorted(set(hashes)), range(len(hashes)))}

            ids = np.array([lookup[h] if h in lookup else -1 for h in hashes])
            counts, bins = np.histogram(ids, range(len(lookup) + 2))

            images_train[i] = len(lookup) * [[]]
            responses_train[i] = len(lookup) * [[]]
            images_test[i] = len(lookup) * [[]]
            responses_test[i] = len(lookup) * [[]]
            for j, c in enumerate(counts):
                if c > 0 and c < 10:
                    idx = np.where(ids == j)[0]
                    images_train[i][j] = images[idx[0],:,:,:]
                    img_idx = rnd.choice(idx, 1)[0]
                    responses_train[i][j] = responses[img_idx,:,:]
                elif c > 10:
                    idx = np.where(ids == j)[0]
                    images_test[i][j] = images[idx[0],:,:,:]
                    responses_test[i][j] = responses[idx,:,:].mean(axis=0)

        def _get_images_to_include(images_by_scan, responses_by_scan, lookup):
            num_scans = len(images_by_scan)
            use = np.array([
                sum(
                    [type(images_by_scan[i][j]) is not list for i in range(num_scans)]
                ) for j in range(len(lookup))
            ]) == num_scans
            images = np.concatenate(
                [im[None,:,:,0] for im, u in zip(images_by_scan[0], use) if u],
                axis=0)
            responses = np.concatenate([
                np.concatenate([r[None,...] for r, u in zip(r_sess, use) if u], axis=0)
                for r_sess in responses_by_scan
            ], axis=2)
            return images, responses

        images_train, responses_train = _get_images_to_include(
            images_train, responses_train, lookup)
        images_test, responses_test = _get_images_to_include(
            images_test, responses_test, lookup)
        
        return Dataset(images_train, responses_train, images_test, responses_test, seed=1)

""" OLD code, would have to be adapted...

    def get_trace_ids(self):
        rel = self.Member() * Scan()
        animal_ids, sessions, scan_idxs = rel.fetch(
            'animal_id', 'session_idx', 'scan_idx',
            order_by='animal_id ASC, session_idx ASC, scan_idx ASC')
        trace_ids = []
        for i, (animal_id, session, scan_idx) in enumerate(
                zip(animal_ids, sessions, scan_idxs)):
            file_name = os.path.join(
                PATH, DATA_PATH, FILE_NAME_PATTERN.format(animal_id, scan_idx, session))
            with np.load(file_name) as data:
                trace_ids.append(data['trace_ids'])
        return trace_ids


    def get_locations(self):
        def center_of_mass(idx, val):
            cx, cy = 0.0, 0.0
            for i, v in zip(idx, val):
                cx += np.floor(i / height) * v
                cy += (i % height) * v
            s = sum(val)
            return cx / s, cy / s

        rel = self.Member() * Scan()
        animal_ids, sessions, scan_idxs = rel.fetch(
            'animal_id', 'session_idx', 'scan_idx',
            order_by='animal_id ASC, session_idx ASC, scan_idx ASC')
        trace_ids = self.get_trace_ids()
        coords = []
        scan = []
        for i, (animal_id, session, scan_idx) in enumerate(
                zip(animal_ids, sessions, scan_idxs)):
            file_name = os.path.join(
                PATH, DATA_PATH, MASK_FILE_NAME_PATTERN.format(
                    animal_id, session, scan_idx))
            data = np.load(file_name)
            width, height = SCAN_DIMS[animal_id]
            for _, _, _, _, _, _, trace_id, idx, val in data:
                if trace_id in trace_ids[i]:
                    coords.append(center_of_mass(idx, val))
                    scan.append(scan_idx)
                    
        return np.array(coords), np.array(scan)
"""
