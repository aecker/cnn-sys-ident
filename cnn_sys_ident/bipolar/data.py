import os
import inspect
import hashlib
import h5py
import datajoint as dj
import numpy as np


schema = dj.schema('dklindt_bipolar_data', locals())

PATH = os.path.dirname(os.path.dirname(os.path.dirname(inspect.stack()[0][1])))
DATA_PATH = os.path.join(PATH, 'data/bipolar')

NUM_CLIPS = 108
NUM_VAL_CLIPS = 15
rnd = np.random.RandomState(seed=2364782)
VAL_CLIP_IDX = set(rnd.choice(NUM_CLIPS, NUM_VAL_CLIPS, replace=False))
TRAIN_CLIP_IDX = set(range(NUM_CLIPS)) - VAL_CLIP_IDX

class Dataset:
    def __init__(self,
                 movie_train,
                 movie_test,
                 movie_ordering,
                 movie_trigger_times,
                 responses,
                 response_trigger_times):

        # preprocess images (mean=0, SD=1)
        m = images_train.mean()
        sd = images_train.std()
        zscore = lambda img: (img - m) / sd
        self.images_train = zscore(images_train)[...,None]
        self.images_val = zscore(images_val)[...,None]
        self.images_test = zscore(images_test)[...,None]
        
        # preprocess responses (SD=1)
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
    definition = """ # Bipolar cells with iGluSnfr
        animal_id   : int             # animal id
        retina      : ENUM("L", "R")  # retina side
        scan_idx    : tinyint         # scan index
        ---
        folder      : varchar(255)  # folder name
        """

    contents = [
        [1, 'R', 2, 'pilot1'],
        [1, 'R', 3, 'pilot1'],
        [1, 'R', 4, 'pilot1'],
        [2, 'L', 0, 'pilot2/LeftRetina/IPL0'],
        [2, 'R', 0, 'pilot2/rightretina/IPL0'],
        [2, 'R', 1, 'pilot2/rightretina/IPL1'],
        [2, 'R', 2, 'pilot2/rightretina/IPL2'],
        [2, 'R', 3, 'pilot2/rightretina/IPL3'],
        [2, 'R', 4, 'pilot2/rightretina/IPL4'],
    ]


@schema
class StimulusLookup(dj.Lookup):
    definition = """ # Stimulus types
        stimulus_id   : tinyint unsigned  # stimulus id
        ---
        stimulus_name : varchar(255)      # name of stimulus
    """

    contents = [
        [1, 'step'],
        [2, 'local chirp'],
        [3, 'global chirp'],
        [4, 'natural movies raw'],
        [5, 'natural movies equalized'],
        [6, 'dense noise'],
    ]


@schema
class NaturalMovies(dj.Lookup):
    definition = """
        -> StimulusLookup
        ---
        train_movie_file   : varchar(255)  # file containing movie frames
        test_movie_file    : varchar(255)  # file containing movie frames
    """

    contents = [
        [4, 'movies_train.tiff', 'movies_test.tiff'],
        [5, 'train.tiff', 'test.tiff'],
    ]


@schema
class MovieSplits(dj.Lookup):
    definition = """
        clip_idx  : int unsigned           # clip index
        ---
        split     : ENUM("train", "val")   # train or validation set?
    """
    
    @property
    def contents(self):
        contents = []
        for clip_idx in range(NUM_CLIPS):
            contents += [[float(clip_idx),
                         'val' if clip_idx in VAL_CLIP_IDX else 'train']]
        return contents
                

@schema
class Stimulus(dj.Lookup):
    definition = """ # Stimuli shown
        -> Scan
        -> StimulusLookup
        ---
        data_file     : varchar(255)    # name of data file
    """
    
    @property
    def contents(self):
        contents = [
            [1, 'R', 2, 4, 'SMP_M1_RR_IPL2_NM.h5'],
            [1, 'R', 2, 6, 'SMP_M1_RR_IPL2_DN.h5'],
            [1, 'R', 3, 4, 'SMP_M1_RR_IPL3_NM.h5'],
            [1, 'R', 3, 6, 'SMP_M1_RR_IPL3_DN.h5'],
            [1, 'R', 4, 4, 'SMP_M1_RR_IPL4_NM.h5'],
            [1, 'R', 4, 6, 'SMP_M1_RR_IPL4_DN.h5'],
            [2, 'L', 0, 1, 'SMP_M1_LR_IPL0_1s.h5'],
            [2, 'L', 0, 2, 'SMP_M1_LR_IPL0_lChirp.h5'],
            [2, 'L', 0, 3, 'SMP_M1_LR_IPL0_Chirp.h5'],
            [2, 'L', 0, 5, 'SMP_M1_LR_IPL0_NM.h5'],
            [2, 'L', 0, 6, 'SMP_M1_LR_IPL0_DN.h5'],
        ]
        for i in range(5):
            contents += [
                [2, 'R', i, 1, 'SMP_M1_RR_IPL{}_1s.h5'.format(i)],
                [2, 'R', i, 2, 'SMP_M1_RR_IPL{}_lChirp.h5'.format(i)],
                [2, 'R', i, 3, 'SMP_M1_RR_IPL{}_Chirp.h5'.format(i)],
                [2, 'R', i, 5, 'SMP_M1_RR_IPL{}_NM.h5'.format(i)],
                [2, 'R', i, 6, 'SMP_M1_RR_IPL{}_DN.h5'.format(i)],
            ] 
        return contents

    def num_rois(self):
        file_name = os.path.join(DATA_PATH, self.fetch1('data_file'))
        file = h5py.File(file_name, 'r')
        return len(list(file['Traces0_raw'])[0])
    
    def load_data(self):
        file_name = self.fetch1('data_file')
        # TODO: load files
        return Dataset(movie_train,
                       movie_test,
                       movie_ordering,
                       movie_trigger_times,
                       responses,
                       response_trigger_times)


@schema
class MultiDataset(dj.Lookup):
    definition = """  # Dataset consisting of multiple scans
        data_hash   : char(32) # unique identifier for dataset
        ---
        restriction : varchar(255) # description
        """
    
    _order_members_by = 'animal_id ASC, retina ASC, scan_idx ASC'
    
    class Member(dj.Part):
        definition = """ # Scans and stimuli that are part of this dataset
            -> master
            member_id  : tinyint unsigned    # member id
            ---
            -> Stimulus
            """

    class Roi(dj.Part):
        definition = """ # Scans that are part of this dataset
            -> master
            roi_id  : int unsigned   # ROI id
            ---
            -> master.Member
            """

    def fill(self):
        restrictions = [
            'animal_id=1 and stimulus_id=4',
            'animal_id=2 and stimulus_id=5',
        ]
        for i, r in enumerate(restrictions):
            data_hash = hashlib.md5(str(i).encode()).hexdigest()
            key = dict(data_hash=data_hash, restriction=r)
            if not len(self & key):
                self.insert1(key)
                n = 0
                for j, tupl in enumerate(
                        (self * Stimulus() & key & r).fetch(
                            dj.key, order_by=self._order_members_by)):
                    tupl['member_id'] = j
                    self.Member().insert1(tupl)
                    num_rois = (Stimulus() & tupl).num_rois()
                    rois = [{'data_hash': data_hash, 'member_id': j, 'unit_id': n+k} for k in range(num_rois)]
                    n += num_rois
                    self.Roi().insert(rois)


    def load_data(self):
        assert len(self) == 1, 'Relation must be scalar.'
        data = []
        for key in (self * self.Member).fetch(dj.key):
            data.append((Stimulus() & key).load_data())
        return data
