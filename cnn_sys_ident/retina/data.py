## Add the end: create dataset object and save it with a new hash
# and add entry to Multiscan table in old schema = dj.schema('dklindt_bipolar_data', locals())


import os
import inspect
import hashlib
import h5py
import numpy as np
from PIL import Image,ImageSequence
import sys
from datetime import datetime 
import tensorflow as tf
from scipy import stats
from scipy.signal import butter, filtfilt

STIMULUS_PATH = '/gpfs01/euler/data/Resources/Stimulus/natural_movies/'

NUM_CLIPS = 108
NUM_VAL_CLIPS = 15
rnd = np.random.RandomState(seed=2364782)
VAL_CLIP_IDX = set(rnd.choice(NUM_CLIPS, NUM_VAL_CLIPS, replace=False))
TRAIN_CLIP_IDX = list(set(range(NUM_CLIPS)) - VAL_CLIP_IDX)
VAL_CLIP_IDX = list(VAL_CLIP_IDX)
NUM_NOISE_CLIPS = 40
NUM_NOISE_VAL_CLIPS = 6
NOISE_VAL_CLIP_IDX = set(rnd.choice(NUM_NOISE_CLIPS, NUM_NOISE_VAL_CLIPS, replace=False))
NOISE_TRAIN_CLIP_IDX = list(set(range(NUM_NOISE_CLIPS)) - NOISE_VAL_CLIP_IDX)
NOISE_VAL_CLIP_IDX = list(NOISE_VAL_CLIP_IDX)
    
## go through all clips and just match responses
def load_stimuli(train_movie_file,
                 test_movie_file,
                 random_sequences_file = 'RandomSequences.npy',
                 downsample_this = True,
                 downsample_size = 32,
                 mouse_cam = True,
                 STIMULUS_PATH = STIMULUS_PATH):
    #mouse_cam = train_movie_file[:5] == 'mouse'
#     mouse_cam = True
    # train movie
    train_movie_file = os.path.join(STIMULUS_PATH, train_movie_file)
    Train = Image.open(train_movie_file)
    Train.load()
    N = Train.n_frames
    original_size = Train.size[0]
    i=0
    if mouse_cam:
        movie_train = np.zeros([N, original_size, original_size, 2])
        for frame in ImageSequence.Iterator(Train):
            movie_train[i] = np.array(frame)[:,:,:2]
            i+=1
    else:
        movie_train = np.zeros([N, original_size, original_size, 1])
        for frame in ImageSequence.Iterator(Train):
            movie_train[i,:,:,0] = np.array(frame)
            i+=1

    # test movie
    test_movie_file = os.path.join(STIMULUS_PATH, test_movie_file)
    Test = Image.open(test_movie_file)
    Test.load()
    N = Test.n_frames
    i=0
    if mouse_cam:
        movie_test = np.zeros([N, original_size, original_size, 2])
        for frame in ImageSequence.Iterator(Test):
            movie_test[i] = np.array(frame)[:,:,:2]
            i+=1
    else:
        movie_test = np.zeros([N, original_size, original_size, 1])
        for frame in ImageSequence.Iterator(Test):
            movie_test[i,:,:,0] = np.array(frame)
            i+=1

    # downsample
    if downsample_this and (original_size != downsample_size):
        movie_train = downsample(movie_train, downsample_size, verbose = True)
        movie_test = downsample(movie_test, downsample_size)

    # preprocess images (mean=0, SD=1)
    m = movie_train.mean().reshape((-1,1))
    sd = movie_train.std().reshape((-1,1))
    zscore = lambda img: (img - m) / sd
    movie_train = zscore(movie_train)
    m = movie_test.mean().reshape((-1, 1))
    sd = movie_test.std().reshape((-1, 1))
    zscore = lambda img: (img - m) / sd
    movie_test = zscore(movie_test)

    # load ordering of random sequences
    random_sequences = np.load(os.path.join(STIMULUS_PATH, random_sequences_file))

    return movie_train, movie_test, random_sequences

def downsample(images, out_size, verbose = False):
    assert images.shape[1]==images.shape[2]
    if verbose:
        print("resizing images from {} to {} (square)".format(images.shape[1], out_size))
    with tf.Graph().as_default():
        with tf.Session() as sess:
            return sess.run(tf.image.resize_images(images, (out_size, out_size)))

class Dataset:
    def __init__(self,
                 movie_train,
                 movie_test,
                 random_sequences,
                 seq_idx,
                 responses, # ND
                 adapt = 0, # how many frames to discard from each clip / beginning of test to ignore adaptation
                 snr_tresh = - np.inf,
                 luminance_paths_train=[],
                 luminance_paths_test=[],
                 contrast_paths_train=[],
                 contrast_paths_test=[]
                ):
        #some init variabels
        if len(random_sequences)==0:
            noise = True
            train_clip_idx = NOISE_TRAIN_CLIP_IDX
            val_clip_idx = NOISE_VAL_CLIP_IDX
            num_clips = NUM_NOISE_CLIPS
        else:
            noise = False
            train_clip_idx = TRAIN_CLIP_IDX
            val_clip_idx = VAL_CLIP_IDX
            num_clips = NUM_CLIPS
        # movie_ordering = random_sequences[:, seq_idx]###
        self.num_train_samples, self.px_y, self.px_x, self.channels = movie_train.shape
        if noise:
            self.clip_length = 90
            self.num_clips = 40
            movie_ordering = np.arange(self.num_clips)
        else:
            movie_ordering = random_sequences[:, seq_idx]
            self.clip_length = 150
            self.num_clips = 108
        self.adapt = adapt
        self.num_neurons = responses.shape[0]
        n_preprocessed_stim_versions = len(luminance_paths_train)
        if n_preprocessed_stim_versions > 0:
            self.channels += 4*n_preprocessed_stim_versions #(luminance + contrast) * color_channels = 4
        #split into train and (averaged) test responses
        if noise:
            self.responses_test = responses[:,:10*self.clip_length].T
            self.responses_train = responses[:,10*self.clip_length:].T
        else:
            self.responses_test = np.zeros((5*self.clip_length,self.num_neurons))#DN
            self.responses_train = np.zeros((self.num_clips*self.clip_length,self.num_neurons))#DN
            self.test_responses_by_trial = []
            for roi in range(self.num_neurons):
                tmp = np.vstack((responses[roi,:5*self.clip_length],
                                 responses[roi,59*self.clip_length:64*self.clip_length],
                                 responses[roi,118*self.clip_length:]))
                self.test_responses_by_trial.append(tmp)#calculated below after z-scoring
                self.responses_test[:,roi] = np.mean(tmp,0)
                self.responses_train[:,roi] = np.concatenate((responses[roi,5*self.clip_length:59*self.clip_length],
                                                              responses[roi,64*self.clip_length:118*self.clip_length]))
            self.test_responses_by_trial = np.asarray(self.test_responses_by_trial)
            # measure oracle (test set correlation each with remaining n-1, averaged)
            self.test_responses_for_oracle = np.zeros_like(self.test_responses_by_trial)
            self.oracle = np.zeros((self.num_neurons, 3))
            for i in range(3):
                for n in range(self.num_neurons):
                    others = list({0,1,2} - {i})
                    others = .5 * (self.test_responses_by_trial[n, others[0]] +
                                   self.test_responses_by_trial[n, others[1]])
                    self.test_responses_for_oracle[n, i] = others
                    self.oracle[n, i] = stats.pearsonr(self.test_responses_by_trial[n, i], others)[0]
        # measure signal variance per ROI
        #self.total_variance = np.mean(np.var(self.test_responses_by_trial,axis=2),1)
        #self.noise_variance = np.mean(np.var(self.test_responses_by_trial,axis=1),1)
        #self.signal_variance = self.total_variance - self.noise_variance
        if not noise:
            self.signal_variance = np.var(self.responses_test,0)
            self.noise_variance = np.mean(np.var(self.test_responses_by_trial,1)/3,1)#divided by three repeats
            self.total_variance = self.signal_variance + self.noise_variance
            self.SNR = self.signal_variance / self.noise_variance
        else:
            self.total_variance = np.var(self.responses_test,0)

        if noise:
            self.snr_ind = np.full(self.num_neurons,True)
            self.input_shape = [None, None, self.px_y, self.px_x, self.channels]
        else:
            # filter out too low SNR
            snr_ind = self.SNR > snr_tresh
            self.num_neurons = np.sum(snr_ind)
            self.input_shape = [None, None, self.px_y, self.px_x, self.channels]
            self.output_shape = [None, None, self.num_neurons]
            self.responses_train = self.responses_train[:, snr_ind]
            self.responses_test = self.responses_test[:, snr_ind]
            self.test_responses_for_oracle = self.test_responses_for_oracle[snr_ind]
            self.test_responses_by_trial = self.test_responses_by_trial[snr_ind]
            self.total_variance = self.total_variance[snr_ind]
            self.noise_variance = self.noise_variance[snr_ind]
            self.signal_variance = self.signal_variance[snr_ind]
            self.SNR = self.SNR[snr_ind]
            self.snr_ind = snr_ind

        if noise:
            self.movie_train = movie_train
            self.movie_test = movie_test
        else:
            # CC_norm (Schoppe &al, 2016)
            self.SP = (np.var(np.sum(self.test_responses_by_trial, 1), 1) -
                       np.sum(np.var(self.test_responses_by_trial, 2), 1)) / (
                      3 * (3 - 1))
            self.SP = self.SP.clip(min=1e-8)
            # order train movies
            if n_preprocessed_stim_versions > 0:
                tmp_movies_shape = list(movie_train.shape)
                tmp_movies_shape[-1] += 4*n_preprocessed_stim_versions
                tmp_movies = np.zeros(tmp_movies_shape)
                for i,ind in enumerate(movie_ordering):
                    tmp_movies[i*self.clip_length:(i+1)*self.clip_length, :, :, :2] = \
                        movie_train[ind*self.clip_length:(ind+1)*self.clip_length]
                for i, path in enumerate(luminance_paths_train):
                    tmp_movies[:, :, :, 2*i+2:2*i+4] = \
                        np.load(path)[seq_idx, :]
                tmp_idx = 2*i+4
                for i, path in enumerate(contrast_paths_train):
                    tmp_movies[:, :, :, 2*i+tmp_idx:2*i+tmp_idx+2] = \
                        np.load(path)[seq_idx,:]

                test_shape = list(movie_test.shape)
                test_shape[-1] += 4*n_preprocessed_stim_versions
                test_shape[0] = 2*test_shape[0]
                tmp_test_movies = np.zeros(test_shape)
                tmp_test_movies[:movie_test.shape[0], :, :, :2] = movie_test
                tmp_test_movies[movie_test.shape[0]:, :, :, :2] = movie_test

                for i, path in enumerate(luminance_paths_test):
                    tmp_test_movies[:, :, :, 2*i+2:2*i+4] = \
                        np.load(path)[seq_idx, :]
                tmp_idx = 2 * i + 4
                for i, path in enumerate(contrast_paths_test):
                    tmp_test_movies[:, :, :, 2 * i + tmp_idx:2 * i + tmp_idx + 2] = \
                        np.load(path)[seq_idx, :]
                self.movie_test = np.reshape(tmp_test_movies,
                                             [2,
                                             self.clip_length * 5,
                                             self.px_y,
                                             self.px_x,
                                             self.channels])  # BDHWC
            else:
                tmp_movies = np.zeros_like(movie_train)
                for i,ind in enumerate(movie_ordering):
                    tmp_movies[i*self.clip_length:(i+1)*self.clip_length] = movie_train[ind*self.clip_length:(ind+1)*self.clip_length]

                self.movie_test = np.reshape(movie_test,
                                             [1,self.clip_length*5,self.px_y,
                                              self.px_x, self.channels])#BDHWC
            self.movie_train = tmp_movies

        # calculate STA
        # self.sta_space, self.sta_time = self.STA(self.movie_train,self.responses_train)

        # validation split
        if noise:
            self.movie_val = np.zeros((NUM_NOISE_VAL_CLIPS, self.clip_length, self.px_y, self.px_x, self.channels))#BDHW
            self.responses_val = np.zeros([NUM_NOISE_VAL_CLIPS,self.clip_length,self.num_neurons])#BDN
            for i,ind in enumerate(val_clip_idx):
                self.movie_val[i] = self.movie_train[ind*self.clip_length:(ind+1)*self.clip_length]
                self.responses_val[i] = self.responses_train[ind*self.clip_length:(ind+1)*self.clip_length,:]

            # val and test set in right shapes
            #self.movie_test = np.reshape(tmp_test_movies,[2,self.clip_length*5,self.px_y,self.px_x, self.channels])#BDHWC
            self.movie_val = np.reshape(self.movie_val,[NUM_NOISE_VAL_CLIPS,self.clip_length,self.px_y,self.px_x, self.channels])#BDHWC
            self.responses_test = np.reshape(self.responses_test,[1,-1,self.num_neurons])#BDN
        else:
            self.movie_val = np.zeros((NUM_VAL_CLIPS, self.clip_length, self.px_y, self.px_x, self.channels))#BDHW
            self.responses_val = np.zeros([NUM_VAL_CLIPS,self.clip_length,self.num_neurons])#BDN
            inv_order = np.argsort(movie_ordering)
            for i,ind1 in enumerate(val_clip_idx):
                ind2 = inv_order[ind1]
                self.movie_val[i] = self.movie_train[ind2*self.clip_length:(ind2+1)*self.clip_length]
                self.responses_val[i] = self.responses_train[ind2*self.clip_length:(ind2+1)*self.clip_length,:]

            # val and test set in right shapes
            #self.movie_test = np.reshape(tmp_test_movies,[2,self.clip_length*5,self.px_y,self.px_x, self.channels])#BDHWC
            self.movie_val = np.reshape(self.movie_val,[NUM_VAL_CLIPS,self.clip_length,self.px_y,self.px_x, self.channels])#BDHWC
            self.responses_test = np.reshape(self.responses_test,[1,-1,self.num_neurons])#BDN

        # for train: find sequences of clips (uniterrupted by val clips)
        start_idx = 0
        current_idx = 0
        self.seq_start_idx = []
        self.seq_length = []
        for ind in movie_ordering: # over clips
            if ind in val_clip_idx:
                length = current_idx - start_idx
                if length > 0:
                    self.seq_start_idx.append(start_idx)
                    self.seq_length.append(length)
                start_idx = current_idx + self.clip_length
            current_idx += self.clip_length
        length = current_idx - start_idx
        if length > 0:
            self.seq_start_idx.append(start_idx)
            self.seq_length.append(length)

        # taking away X seconds of each clip to ignore adaptation
        if self.adapt > 0:
            # train
            start_idx = self.adapt
            length = self.clip_length - self.adapt
            self.seq_start_idx = []
            self.seq_length = []
            for ind in movie_ordering: # over clips
                if ind not in val_clip_idx:
                    self.seq_start_idx.append(start_idx)
                    self.seq_length.append(length)
                start_idx += self.clip_length
            # test and val
            self.movie_test = self.movie_test[:, self.adapt:]
            self.movie_val = self.movie_val[:, self.adapt:]
            self.responses_test = self.responses_test[:, self.adapt:]
            self.responses_val = self.responses_val[:, self.adapt:]

        self.current_chunk_idx = 1e10
        self.chunk_start_idx = []

    def val(self):
        return self.movie_val, self.responses_val

    def train(self):
        return self.movie_train, self.responses_train

    def test(self):
        return self.movie_test, self.responses_test

    def minibatch(self, batch_size, chunk_size=50):# return one batch
        if self.current_chunk_idx + batch_size > len(self.chunk_start_idx): # self.num_chunks:
            self.next_epoch(chunk_size)
        movies, responses = [], []
        for i in range(batch_size):
            idx = self.chunk_start_idx[self.current_chunk_idx + i]
            movies.append(np.reshape(self.movie_train[idx:idx+chunk_size],[chunk_size,self.px_y,self.px_x, self.channels]))#DHWC    ## [AE] shouldn't this already have the correct shape?
            responses.append(self.responses_train[idx:idx+chunk_size,:])#DN
        self.current_chunk_idx += batch_size
        return np.stack(movies, axis=0), np.stack(responses, axis=0)#BDHWC and BDN

    def next_epoch(self, chunk_size=50):
        shift = np.random.randint(0, np.min([chunk_size, self.clip_length - self.adapt - chunk_size]))
        chunk_start_idx = []
        for start, length in zip(self.seq_start_idx, self.seq_length):
            idx = np.arange(start + shift, start + shift + length - chunk_size + 1, chunk_size)
            chunk_start_idx += list(idx[:-1])
        self.chunk_start_idx = np.random.permutation(chunk_start_idx)
        # self.num_chunks = len(chunk_start_idx)
        self.current_chunk_idx = 0#beginning of epoch

    def STA(self, X,Y): # bad, correlated, dont use
        hist=10#more makes the localization diffuser
        future=5#gives also info because of spatiotemporal correlations in natural movie
        sta_space = np.zeros((self.num_neurons,self.px_y,self.px_x))#NHW
        sta_time = np.zeros((self.num_neurons,hist+future))#NT
        weights=np.zeros([self.num_neurons,hist+future,self.px_y,self.px_x])
        for i in range(hist,self.num_train_samples-future):
            weights += np.outer(Y[i,:],X[i-hist:i+future]).reshape([self.num_neurons,hist+future,self.px_y,self.px_x])
        for roi in range(self.num_neurons):
            S,V,D = np.linalg.svd(weights[roi].reshape(-1,self.px_y*self.px_x))
            #sign flip (STA just preceding stimulus gives polarity of ROI)
            if D[0,:].flatten().dot(weights[roi,-future-1].flatten())<0:
                S*=-1
                D*=-1
            sta_space[roi] = D[0,:].reshape([self.px_y,self.px_x])
            sta_time[roi] = S[:,0]
        return sta_space, sta_time

class MultiDataset:
    def __init__(self,
                 responses, 
                 num_rois, 
                 num_scans, 
                 scan_sequence_ind, 
                 scan_keys,
                 restriction,
                 depths,
                 adapt = 0,
                 movies = None,
                 luminance_paths_train = [],
                 luminance_paths_test = [],
                 contrast_paths_train = [],
                 contrast_paths_test = [],
                 snr_thresh = -np.inf,
                 scan_in_batch = False,
                 group = False,
                 downsample_size = 32):
        self.responses = responses
        self.num_rois = num_rois
        self.num_scans = num_scans
        self.scan_sequence_ind = scan_sequence_ind
        self.scan_keys = scan_keys
        self.restriction = restriction
        self.depths = depths
        self.snr_thresh = snr_thresh
        self.scan_in_batch = scan_in_batch

        if movies == None:
            if scan_keys[0]['stim_id'] == 4: # Hollywood movies
                self.movie_train, self.movie_test, self.random_sequences = load_stimuli(
                    'hollywood_train.tiff','hollywood_test.tiff', downsample_size=downsample_size)
            elif scan_keys[0]['stim_id'] == 5: # Mouse cam movies
                self.movie_train, self.movie_test, self.random_sequences = load_stimuli(
                    'mousecam_train.tiff','mousecam_test.tiff', downsample_size=downsample_size)
        else:
            self.movie_train, self.movie_test, self.random_sequences = movies

        if group:
            # generate component datasets which saw same random sequence
            self.grouped_responses = []
            self.grouped_num_rois = []
            self.grouped_scan_keys = []
            self.grouped_depths = []
            self.scans = [] # grouped
            for rand_seq in range(20):
                self.grouped_responses.append([])
                self.grouped_num_rois.append([])
                self.grouped_scan_keys.append([])
                self.grouped_depths.append([])
                for i, scan in enumerate(self.scan_sequence_ind):
                    if scan == rand_seq:
                        self.grouped_responses[-1].append(self.responses[i])
                        self.grouped_num_rois[-1].append(self.num_rois[i])
                        self.grouped_scan_keys[-1].append(self.scan_keys[i])
                        self.grouped_depths[-1].append(self.depths[i])
                if self.grouped_responses[-1] != []:
                    print('Random sequence {}, number of ROIs {}'.format(rand_seq, self.grouped_num_rois[-1]))
                    self.grouped_responses[-1] = np.concatenate(self.grouped_responses[-1], 0)
                    self.scans.append(Dataset(self.movie_train, self.movie_test,
                                              self.random_sequences[:, rand_seq],
                                              self.grouped_responses[-1],
                                              adapt, snr_thresh))
                    self.grouped_depths[-1] = np.concatenate(self.grouped_depths[-1])[self.scans[-1].snr_ind]
            self.depths = np.concatenate(self.grouped_depths)
        else: # One feed per scan (required for scan specific correction)
            self.scans = []
            for i, scan in enumerate(self.scan_sequence_ind):
                print('Random sequence {}, number of ROIs {}'.format(scan, self.num_rois[i]))
                print(self.scan_keys[i])
                self.scans.append(Dataset(self.movie_train, self.movie_test,
                                          self.random_sequences,
                                          scan,
                                          self.responses[i],
                                          adapt, snr_thresh,
                                          luminance_paths_train,
                                          luminance_paths_test,
                                          contrast_paths_train,
                                          contrast_paths_test))
                self.depths[i] = self.depths[i][self.scans[i].snr_ind]
            self.depths = np.concatenate(self.depths)

        #same between scans
        S = len(self.scans)
        self.movie_val = np.tile(self.scans[0].movie_val,[S,1,1,1,1,1])#SBDHWC
        self.movie_test = np.tile(self.scans[0].movie_test,[S,1,1,1,1,1])#SBDHWC
        self.num_train_samples = self.scans[0].num_train_samples#T
        self.px_x = self.scans[0].px_x#W
        self.px_y = self.scans[0].px_y#H
        self.clip_length = self.scans[0].clip_length
        self.num_clips = self.scans[0].num_clips
        #different between scans
        self.num_rois = []
        self.total_variance = []
        self.movie_train = []
        self.responses_train = []
        self.responses_test = []
        self.responses_val = []
        #self.sta_space = []
        #self.sta_time = []
        if len(self.random_sequences)>0:
            self.noise_variance = []
            self.signal_variance = []
            self.SNR = []
            self.SP = []
            self.oracle = []
            self.test_responses_by_trial = []
            self.test_responses_for_oracle =[]
        for scan in self.scans:
            self.num_rois.append(scan.num_neurons)
            self.total_variance.append(scan.total_variance)
            self.movie_train.append(scan.movie_train)
            self.responses_train.append(scan.responses_train)
            self.responses_test.append(scan.responses_test)
            self.responses_val.append(scan.responses_val)
            if len(self.random_sequences)>0:
                self.noise_variance.append(scan.noise_variance)
                self.signal_variance.append(scan.signal_variance)
                self.SNR.append(scan.SNR)
                self.SP.append(scan.SP)
                self.oracle.append(scan.oracle)
                self.test_responses_by_trial.append(scan.test_responses_by_trial)
                self.test_responses_for_oracle.append(scan.test_responses_for_oracle)
        #    self.sta_space.append(scan.sta_space)
        #    self.sta_time.append(scan.sta_time)
        self.num_neurons = np.sum(self.num_rois)
        self.output_shape = [None,None,np.sum(self.num_neurons)]
        self.total_variance = np.concatenate(self.total_variance)
        if len(self.random_sequences)>0:
            self.noise_variance = np.concatenate(self.noise_variance)
            self.signal_variance = np.concatenate(self.signal_variance)
            self.SNR = np.concatenate(self.SNR)
            self.SP = np.concatenate(self.SP)
            self.oracle = np.concatenate(self.oracle)
            self.test_responses_by_trial = np.concatenate(self.test_responses_by_trial, axis=0)#NDR
            self.test_responses_for_oracle = np.concatenate(self.test_responses_for_oracle, axis=0)#NDR
        self.movie_train = np.stack(self.movie_train, axis=0)#STHWC
        self.responses_train = np.concatenate(self.responses_train, axis=1)#TN
        self.responses_test = np.concatenate(self.responses_test, axis=2)#TBN
        self.responses_val = np.concatenate(self.responses_val, axis=2)#TBN

        #self.sta_space = np.concatenate(self.sta_space,0)#NWH
        #self.sta_time = np.concatenate(self.sta_time,0)#NT

        # change by hand if scan in batch
        if not self.scan_in_batch:
            self.input_shape = [len(self.scans)] + self.scans[0].input_shape
        else:
            self.input_shape = self.scans[0].input_shape

    def next_epoch(self, chunk_size=50):
        for scan in self.scans:
            scan.next_epoch(chunk_size)

    def minibatch(self, batch_size, chunk_size=50):
        batch_x = []
        batch_y = []
        for scan in self.scans:
            x, y = scan.minibatch(batch_size, chunk_size)
            batch_x.append(x)
            batch_y.append(y)
        if not self.scan_in_batch:
            return np.stack(batch_x, axis=0), np.concatenate(batch_y, axis=2)#SBDHWC and BDN
        else:
            return np.concatenate(batch_x, axis=0), np.concatenate(batch_y, axis=2)#(S*B)DHWC and BDN

    def val(self):
        if not self.scan_in_batch:
            return self.movie_val, self.responses_val
        else:
            return self.scan2batch(self.movie_val), self.responses_val

    def train(self):
        return self.movie_train, self.responses_train

    def test(self):
        if not self.scan_in_batch:
            return self.movie_test, self.responses_test
        else:
            return self.scan2batch(self.movie_test), self.responses_test

    def scan2batch(self, X):
        return np.concatenate([i for i in X], 0)   ## [AE] shouldn't np.reshape do the job?
