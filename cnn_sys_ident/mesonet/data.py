import os
import inspect
import hashlib
import h5py
import datajoint as dj
import numpy as np


schema = dj.schema('aecker_mesonet_data', locals())

PATH = os.path.dirname(os.path.dirname(os.path.dirname(inspect.stack()[0][1])))
DATA_PATH = os.path.join(PATH, 'data/mesonet')


class Dataset:
    def __init__(self,
                 images_train,
                 responses_train,
                 images_val,
                 responses_val,
                 images_test,
                 responses_test):

        # normalize images (mean=0, SD=1)
        m = images_train.mean()
        sd = images_train.std()
        zscore = lambda img: (img - m) / sd
        self.images_train = zscore(images_train)[...,None]
        self.images_val = zscore(images_val)[...,None]
        self.images_test = zscore(images_test)[...,None]
        
        # normalize responses (SD=1)
        sd = responses_train.std(axis=0)
        sd[sd < (sd.mean() / 100)] = 1
        def rectify_and_normalize(x):
            x[x < 0] = 0
            return x / sd
        self.responses_train = rectify_and_normalize(responses_train)
        self.responses_val = rectify_and_normalize(responses_val)
        self.responses_test = rectify_and_normalize(responses_test)
        
        self.num_neurons = responses_train.shape[1]
        self.num_train_samples = images_train.shape[0]
        self.px_x = images_train.shape[2]
        self.px_y = images_train.shape[1]
        self.input_shape = [None, self.px_y, self.px_x, 1]
        self.next_epoch()

    def val(self):
        return self.images_val, self.responses_val

    def train(self):
        return self.images_train, self.responses_train

    def test(self, averages=True):
        responses = self.responses_test.mean(axis=0) if averages else self.responses_test
        return self.images_test, responses

    def minibatch(self, batch_size):
        if self.minibatch_idx + batch_size > self.num_train_samples:
            self.next_epoch()
        idx = self.train_perm[self.minibatch_idx + np.arange(0, batch_size)]
        self.minibatch_idx += batch_size
        return self.images_train[idx, :, :], self.responses_train[idx, :]

    def next_epoch(self):
        self.minibatch_idx = 0
        self.train_perm = np.random.permutation(self.num_train_samples)


@schema
class Scan(dj.Lookup):
    definition = """ # Scans with mesoscope and ImageNet images
        animal_id   : int           # animal id
        session_idx : tinyint       # session index
        scan_idx    : tinyint       # scan index
        ---
        """
    contents = [
        [11521, 7, 1],
        [11521, 7, 2],
    ]


@schema
class AreaLookup(dj.Lookup):
    definition = """ # brian area and layer lookup
        area_id     : tinyint        # area id
        ---
        brain_area  : varchar(32)    # brain area
        layer       : varchar(32)    # cortical layer 
        """
    contents = [
        [1, 'V1', '2/3'],
        [2, 'V1', '4'],
        [3, 'LM', '2/3'],
        [4, 'LM', '4'],
    ]


@schema
class Area(dj.Lookup):
    definition = """ # Areas scanned
        -> Scan
        -> AreaLookup
        ---
        """

    class File(dj.Part):
        definition = """ # H5 data files
            -> master
            split     : varchar(5)  # train|val|test
            ---
            file_name : varchar(35) # file name
            """
        
        def file(self):
            file_name = os.path.join(DATA_PATH, self.fetch1('file_name'))
            file = h5py.File(file_name, 'r')
            return file
        
        def load(self):
            file = self.file()
            images = file['inputs'][:,0,:,:]
            responses = file['responses'][:]
            n_images, n_neurons = responses.shape
            if len(self & 'split="test"'):
                image_ids = file['image_ids'][:]
                n_unique = len(np.unique(image_ids))
                n_repeats = n_images // n_unique
                idx = np.argsort(image_ids)
                images = images[idx][::n_repeats]
                responses = np.reshape(responses[idx].T, [n_neurons, n_unique, n_repeats])
                responses = np.transpose(responses, [2, 1, 0])
            return images, responses

        def info(self):
            file = self.file()
            return dict(file['info'])

    def num_neurons(self):
        return len((self.File() & self & 'split="train"').info()['unit_ids'])

    def fill(self):
        files = [
            ['V1', 'd5b9327523db5cf25641137d779a7add.h5', '4',   11521, 7, 1, 'train'],
            ['V1', '221d7a789c5f59e2cd1e3c853171c805.h5', '2/3', 11521, 7, 1, 'train'],
            ['LM', 'f200f225359957adaf95a6fe249a228e.h5', '4',   11521, 7, 1, 'train'],
            ['LM', 'fcbcd9807328258a2946a59ada86a4e8.h5', '2/3', 11521, 7, 1, 'train'],
            ['V1', '2b4c3bd0291767806e5e785368395066.h5', '4',   11521, 7, 2, 'train'],
            ['V1', 'd0c13e27600e27a363b93749b20353cd.h5', '2/3', 11521, 7, 2, 'train'],
            ['LM', 'bc51c9bcbdbb91dd2c602281edd5ea63.h5', '4',   11521, 7, 2, 'train'],
            ['LM', '5ff8e603a3e11fd890b2da62d68b9190.h5', '2/3', 11521, 7, 2, 'train'],
            ['V1', '47fd6bfbba778b3c7a82cf1e4f9958d7.h5', '4',   11521, 7, 1, 'test'],
            ['V1', '7efbcaf468d00df669dbd7db4eedb512.h5', '2/3', 11521, 7, 1, 'test'],
            ['LM', 'd51218bbc7dc48fb84f3c717d0e8d926.h5', '4',   11521, 7, 1, 'test'],
            ['LM', '6036f4f296c793630de1c7de3685cf68.h5', '2/3', 11521, 7, 1, 'test'],
            ['V1', '809d9e7c73930c774878b3149216dff2.h5', '4',   11521, 7, 2, 'test'],
            ['V1', '2a85148525ebe9668479f0e7eee03016.h5', '2/3', 11521, 7, 2, 'test'],
            ['LM', '5f72310758aa8134030f459096efb674.h5', '4',   11521, 7, 2, 'test'],
            ['LM', '4bb8dc76e1706ceb189dbe2f9a6e7feb.h5', '2/3', 11521, 7, 2, 'test'],
            ['V1', '88e87381440f6ac12f2417b8f0e79bd8.h5', '4',   11521, 7, 1, 'val'],
            ['V1', '53aafae5651fd317229b73d30678f9f2.h5', '2/3', 11521, 7, 1, 'val'],
            ['LM', '2b98b819016cc118e9a7787f41aa9183.h5', '4',   11521, 7, 1, 'val'],
            ['LM', '16de863369789c27a3b6e65cd9923300.h5', '2/3', 11521, 7, 1, 'val'],
            ['V1', 'eba43ce40c23f6001d39b4789597c6e9.h5', '4',   11521, 7, 2, 'val'],
            ['V1', 'df08d8319b108f92a82058133d3fb2f3.h5', '2/3', 11521, 7, 2, 'val'],
            ['LM', 'e6a7e46353aa9762d151dc456d9bc20c.h5', '4',   11521, 7, 2, 'val'],
            ['LM', 'ed767de3d4b32a935852f2c14c00134c.h5', '2/3', 11521, 7, 2, 'val'],
        ]
        for area, file_name, layer, animal_id, session_idx, scan_idx, split in files:
            area_id = (AreaLookup() & dict(brain_area=area, layer=layer)).fetch1('area_id')
            key = dict(animal_id=animal_id,
                       session_idx=session_idx,
                       scan_idx=scan_idx,
                       area_id=area_id)
            self.insert1(key, skip_duplicates=True)
            self.File().insert1(dict(animal_id=animal_id,
                                     session_idx=session_idx,
                                     scan_idx=scan_idx,
                                     area_id=area_id,
                                     split=split,
                                     file_name=file_name), skip_duplicates=True)

    def load_files(self):
        assert len(self) == 1, 'Relation must be scalar!'
        load = lambda split: (self.File() & self & 'split="{}"'.format(split)).load()
        return load('train') + load('val') + load('test')


@schema
class MultiDataset(dj.Lookup):
    definition = """  # Dataset consisting of multiple scans and brain areas/layers
        data_hash   : char(32) # unique identifier for dataset
        ---
        restriction : varchar(255) # description
        """
    
    _order_members_by = 'animal_id ASC, session_idx ASC, scan_idx ASC, area_id ASC'
    
    class Member(dj.Part):
        definition = """ # Scans that are part of this dataset
            -> master
            member_id  : tinyint    # member id
            ---
            -> Area
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
            'animal_id=11521 and brain_area="V1"',
            'animal_id=11521 and brain_area="V1" and scan_idx=1 and area_id=1',
        ]
        for i, r in enumerate(restrictions):
            data_hash = hashlib.md5(str(i).encode()).hexdigest()
            key = dict(data_hash=data_hash, restriction=r)
            if not len(self & key):
                self.insert1(key)
                n = 0
                for j, tupl in enumerate(
                        ((self * Area() * AreaLookup()) & key & r).fetch(
                            dj.key, order_by=self._order_members_by)):
                    tupl['member_id'] = j
                    self.Member().insert1(tupl)
                    num_neurons = (Area() & tupl).num_neurons()
                    units = [{'data_hash': data_hash, 'member_id': j, 'unit_id': n+k} for k in range(num_neurons)]
                    n += num_neurons
                    self.Unit().insert(units)


    def load_data(self):
        assert len(self) == 1, 'Relation must be scalar.'
        
        data = []
        for key in (self * self.Member()).fetch(dj.key, order_by='member_id'):
            data.append((Area() & (self.Member() & key)).load_files())

        def merge(k):
            if k % 2:
                return np.concatenate([d[k] for d in data], axis=-1)
            else:
                return data[0][k]

        return Dataset(*[merge(i) for i in range(6)])
