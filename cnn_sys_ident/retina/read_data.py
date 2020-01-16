# %%
import os
import sys
import numpy as np
import re
from copy import deepcopy
from scipy.interpolate import interp1d
import datajoint as dj
import warnings
import c2s

# Specify repository directory
repo_directory = "/gpfs01/berens/user/cbehrens/RGC_DNN/datajoint_imaging_V2/"
# Add repository directory to %PYTHONPATH
sys.path.insert(0, repo_directory)
# Load configuration for user
dj.config.load(repo_directory + "conf/dj_conf_cbehrens.json")
from schema.imaging_schema import *
from schema.stimulus_schema import MovieQI, DetrendTraces, ChirpQI, OsDsIndexes,\
    DetrendParams, MouseCamMovieFiltParams, MouseCamMovieFiltering, Stimulus, Calcium2Spikes
from cnn_sys_ident.retina.data import *
from schema.stimulus_schema import MovieQI


# %%
class MultiDatasetWrapper:
    def __init__(self, experimenter, date, exp_num, stim_path, stim_id,
                 field_id=[]):
        """
        generates a list of lists of keys, the outer lists for experiments, the inner list for fields within experiments
        """
        
        self.stim_path = stim_path
        if type(experimenter) == list:
            #several experiments
            keys = []
            all_params = [date, exp_num, stim_id, field_id]
            n_exps = len(experimenter)
            sid = stim_id[0]
            assert all(len(el) == n_exps for el in all_params), \
                "You have not passed the same number of parameters for each experiment"
            assert all(id == sid for id in stim_id), \
                "Stimuli are not the same for all experiments"
            for params in zip(experimenter, date, exp_num, stim_id, field_id):
                keys.append(self.gen_key(*params))
            self.key = keys
            self.n_exps = n_exps
        else:
            #only one experiment
            self.key = [self.gen_key(
                experimenter, date, exp_num, stim_id, field_id
            )]
            self.n_exps = 1

    def gen_key(self, experimenter, date, exp_num, stim_id, field_id):
        """
        generates a list of keys, 1 key per field
        """
        key = []
        if len(field_id) > 0: # if the fields are specified, add only those
            for fid in field_id:
                key.append(dict(experimenter=experimenter,
                                date=date,
                                exp_num=exp_num,
                                stim_id=stim_id,
                                field_id=fid))
        else: #otherwise add all fields
            temp = dict(experimenter=experimenter,
                                date=date,
                                exp_num=exp_num,
                                stim_id=stim_id,
                        )
            field_id = (Presentation() & temp).fetch("field_id")
            for fid in field_id:
                temp.update(dict(field_id=fid))
                key.append(deepcopy(temp))
        return key


    def interpolate_weights(self, orig_times, orig_data, new_times):
        data_interp = interp1d(
            orig_times.flatten(),
            orig_data.flatten(),
            kind="linear"
        )(new_times)

        return data_interp

    def generate_dataset(self,
                         detrend_traces=True,
                         detrend_param_set_id=1,
                         quality_threshold_movie=0,
                         quality_threshold_chirp=0,
                         quality_threshold_ds=0,
                         downsample_size=32,
                         mouse_cam_filt_params=[],
                         color_channels=True,
                         adapt=0,
                         target_fs=15,
                         spikes=False,
                         ):

        if detrend_traces:
            detrend_param_key = 'detrend_param_set_id = {}'.format(
                detrend_param_set_id
            )
            window_length = \
                (DetrendParams() & detrend_param_key).fetch1('window_length')
            poly_order = \
                (DetrendParams() & detrend_param_key).fetch1('poly_order')
            non_negative = \
                (DetrendParams() & detrend_param_key).fetch1('non_negative')
            subtract_baseline = \
                (DetrendParams() & detrend_param_key).fetch1('subtract_baseline')
            standardize = \
                (DetrendParams() & detrend_param_key).fetch1('standardize')

            print('Loading traces preprocessed with the following settings: '
                  'Window length for SavGol filter : {}, polynomial order for '
                  'SavGol filter: {}, non negative: {}'
                  'baseline subtracted: {}, standardized: {}'.format(
                   window_length, poly_order, non_negative, subtract_baseline,
                   standardize
                  )
                  )
        else:
            warnings.warn("You are retrieving raw traces, but quality indexes "
                          "are returned based on traces preprocessed with "
                          "settings detrend_param_set_id = {}".format(
                           detrend_param_set_id
                          ))
            detrend_param_key = 'detrend_param_set_id = 1'

        key = self.key[0][0]
        stim_path = self.stim_path
        projname = (ExpInfo() & key).fetch1('projname')
        if projname.find("RGC") >= 0:
            hn = (Experiment() & key).fetch1("headername")
            eye = re.search("__(.+?).ini", hn).group(1)
            stim_path = stim_path + eye + "/"
        elif projname.find("BC") >= 0:
            stim_path = stim_path
        elif projname == "ret2dlgn":
            stim_path = stim_path
        else:
            raise Exception('Cannot identify stimulus location '
                            'for experiment from project ' + projname)
        luminance_paths_train = []
        luminance_paths_test = []
        contrast_paths_train = []
        contrast_paths_test = []
        for param in mouse_cam_filt_params:
            mc_stim_path = (
                    MouseCamMovieFiltParams() &
                    'mouse_cam_filt_params = {}'.format(param)
            ).fetch1("mouse_cam_movie_stim_path")
            #assert stim_path == mc_stim_path, "Conflicting stimulus paths"
            mc_downsample_size = (
                MouseCamMovieFiltParams() &
                'mouse_cam_filt_params = {}'.format(param)
            ).fetch1("downsample_size")
            assert mc_downsample_size == downsample_size, "Conflicting downsampling sizes"
            luminance_path_train = (
                    MouseCamMovieFiltering() &
                    'mouse_cam_filt_params = {}'.format(param)
            ).fetch1("luminance_path_train")
            luminance_path_test = (
                    MouseCamMovieFiltering() &
                    'mouse_cam_filt_params = {}'.format(param)
            ).fetch1("luminance_path_test")
            contrast_path_train = (
                    MouseCamMovieFiltering() &
                    'mouse_cam_filt_params = {}'.format(param)
            ).fetch1("contrast_path_train")
            contrast_path_test = (
                    MouseCamMovieFiltering() &
                    'mouse_cam_filt_params = {}'.format(param)
            ).fetch1("contrast_path_test")
            luminance_path_train = stim_path + luminance_path_train
            luminance_path_test = stim_path + luminance_path_test
            contrast_path_train = stim_path + contrast_path_train
            contrast_path_test = stim_path + contrast_path_test
            luminance_paths_train.append(luminance_path_train)
            luminance_paths_test.append(luminance_path_test)
            contrast_paths_train.append(contrast_path_train)
            contrast_paths_test.append(contrast_path_test)
        if key["stim_id"] == 5:
            #movie_train shape: (16200, 56, 56, 2)
            #movie_test shape: (750, 56, 56, 2)
            movie_train, movie_test, random_sequences = \
                load_stimuli("Train_joined.tif",
                             "Test_joined.tif",
                             STIMULUS_PATH=stim_path,
                             downsample_size=downsample_size,
                             mouse_cam=color_channels)
        elif key["stim_id"]==4:
            movie_train, movie_test, random_sequences = \
                load_stimuli("movies_train.tif",
                             "movies_test.tif",
                             STIMULUS_PATH=stim_path,
                             downsample_size=downsample_size,
                             mouse_cam=color_channels)
        elif key["stim_id"] == 0:
            stim_framerate = (Stimulus() & 'stim_id = {}'.format(
                key["stim_id"])
                    ).fetch1("framerate")
            assert target_fs % stim_framerate == 0, "target frame rate is not an integer multiple of native stimulus framerate"
            up_factor = int(target_fs // stim_framerate)  # determine by how much the trigger should be upsampled
            stim = (Stimulus() & 'stim_id = {}'.format(
                key["stim_id"])
                    ).fetch1("stimulus_trace") #shape (15, 20, 1500)
            stim = stim.transpose(-1, 0, 1)
            #repeat every stimulus frame up_factor times
            reshaped = stim.reshape(stim.shape[0], -1)
            repeated = np.repeat(reshaped, [up_factor]*stim.shape[0], axis=0)
            stim = repeated.reshape(repeated.shape[0], 15, 20)
            stim -= np.mean(stim) # set stimululs mean to zero
            #split into 20 % test, 80 % train
            movie_train = stim[300*up_factor:, :, :]
            movie_test = stim[:300*up_factor, :, :]
            #insert channel dimension
            movie_train = np.expand_dims(movie_train, -1)
            movie_test = np.expand_dims(movie_test, -1)
            random_sequences = []

        roi_ids_final = [[] for _ in range(self.n_exps)]
        movie_qis_final = [[] for _ in range(self.n_exps)]
        responses_final = [[] for _ in range(self.n_exps)]
        num_rois_final = [[] for _ in range(self.n_exps)]
        restriction_final = [[] for _ in range(self.n_exps)]
        keys_field_final = [[] for _ in range(self.n_exps)]
        scan_sequence_final = [[] for _ in range(self.n_exps)]
        depths_final = [[] for _ in range(self.n_exps)]
        roi_masks_final = [[] for _ in range(self.n_exps)]
        n_scans = 0
        for exp, keys in enumerate(self.key): #go through different experiments
            n_fields = len(keys)
            roi_ids_all = [[] for _ in range(n_fields)]
            movie_qis_all = [[] for _ in range(n_fields)]
            responses_all = [[] for _ in range(n_fields)]
            num_rois_all = [[] for _ in range(n_fields)]
            restriction = [[] for _ in range(n_fields)]
            keys_field = [{} for _ in range(n_fields)]
            roi_masks = [[] for _ in range(n_fields)]
            depths = [[] for _ in range(n_fields)]
            scan_sequence_idxs = np.zeros(n_fields, dtype=int)
            for i, field_key in enumerate(keys): # go through different fields per experiment
                restriction[i] = field_key
                roi_masks[i] = (Field().RoiMask() & field_key).fetch1("roi_mask")
                header_path = (Presentation() & field_key).fetch1('h5_header')
                scan_frequency = (Presentation() & field_key).fetch1('scan_frequency')
                n_scans += 1
                if field_key["stim_id"] == 5:
                    filename = header_path.split('/')[-1]
                    scan_sequence_idxs[i] = \
                        int(re.search("MC(.+?).h5", filename).group(1))
                elif field_key["stim_id"] == 4:
                    filename = header_path.split('/')[-1]
                    #TODO: find out how to search case-insensitive
                    try:
                        scan_sequence_idxs[i] = \
                            int(re.search("nm(.+?).h5", filename).group(1))
                    except:
                        scan_sequence_idxs[i] = \
                            int(re.search("NM(.+?).h5", filename).group(1))
                if detrend_traces and not(spikes):
                    traces = \
                        (DetrendTraces() * Presentation() &
                         field_key & detrend_param_key).fetch("detrend_traces")
                    raw_traces = \
                        (Traces() * Presentation() &
                         field_key).fetch("traces")
                    assert len(traces) == len(raw_traces), \
                        "Number of ROIs returned for raw traces is not the " \
                        "same as number of ROIs returned for detrend traces. You " \
                        "need to populate DetrendTraces() with the desired detrend " \
                        "settings."\

                elif (not(detrend_traces) and not(spikes)):
                    traces = \
                        (Traces() * Presentation() & field_key).fetch("traces")
                else: 
                    traces = (
                            Calcium2Spikes() * Presentation() & field_key
                    ).fetch("spikes")
                tracestimes = \
                    (Traces() * Presentation() & field_key).fetch("traces_times")
                triggertimes = \
                    (Presentation() & field_key).fetch1("triggertimes")
                if (key["stim_id"] == 5) or (key["stim_id"] == 4):
                    #if movie, upsample triggertimes to get 1 trigger per frame, (instead of just 1 trigger per sequence)
                    upsampled_triggertimes = \
                        [np.linspace(t, t + 4.9666667, 5 * 30)
                         for t in triggertimes]
                    upsampled_triggertimes = np.concatenate(upsampled_triggertimes)
                elif key["stim_id"] == 0:
                    up_factor = target_fs/stim_framerate # determine by how much the trigger should be upsampled
                    ifi = 1/stim_framerate #interframe interval
                    upsampled_triggertimes = \
                        [np.linspace(t, t + ifi, up_factor, endpoint=False)
                         for t in triggertimes]
                    upsampled_triggertimes = np.concatenate(upsampled_triggertimes)
                num_neurons = len(traces)
                roi_ids = (Roi() & field_key).fetch("roi_id")
                #perform all quality checks -> fetch quality indexes from DJ
                qual_idxs_movie = \
                    (MovieQI() & field_key &
                     detrend_param_key).fetch("movie_qi")
                # if quality_threshold_movie > 0:
                #     qual_idxs_movie = \
                #         (MovieQI() & field_key &
                #          detrend_param_key).fetch("movie_qi")
                # else:
                #     qual_idxs_movie = np.ones(num_neurons, dtype=bool)
                temp_key = deepcopy(field_key)
                temp_key.pop("stim_id")
                if quality_threshold_chirp > 0:
                    qual_idxs_chirp = (
                            ChirpQI() & temp_key & detrend_param_key
                    ).fetch("chirp_qi")
                else:
                    qual_idxs_chirp = np.ones(num_neurons, dtype=bool)
                if quality_threshold_ds > 0:
                    qual_idxs_ds = (
                            OsDsIndexes() & temp_key & detrend_param_key
                    ).fetch("d_qi")
                else:
                    qual_idxs_ds = np.ones(num_neurons, dtype=bool)

                if quality_threshold_movie > 0:
                    assert num_neurons == len(qual_idxs_movie), \
                        "Number of neurons and movie quality indexes not the same"
                if quality_threshold_chirp > 0:
                    assert num_neurons == len(qual_idxs_chirp), \
                        "Number of neurons and chirp quality indexes not the same"
                if quality_threshold_ds > 0:
                    assert num_neurons == len(qual_idxs_ds), \
                        "Number of neurons and ds quality indexes not the same"
                #create Boolean mask which is True for cells that pass quality checks, False otherwise

                quality_mask = np.logical_and(
                   (qual_idxs_movie > quality_threshold_movie),
                   np.logical_and((qual_idxs_chirp > quality_threshold_chirp),
                                  (qual_idxs_ds > quality_threshold_ds)))

                if (key["stim_id"] == 5) or (key["stim_id"]==4):
                    responses = np.zeros((num_neurons, 150 * 123))
                elif key["stim_id"] == 0:
                    responses = np.zeros((num_neurons, stim.shape[0]))
                #do c2s here

                for n in range(num_neurons):
                    responses[n, :] = \
                        self.interpolate_weights(tracestimes[n],
                                                 traces[n],
                                                 upsampled_triggertimes)

                responses = responses / np.std(responses, axis=1, keepdims=True)  # normalize response std

                responses_all[i] = responses[quality_mask]
                roi_ids_all[i] = roi_ids[quality_mask]
                movie_qis_all[i] = qual_idxs_movie[quality_mask]
                num_rois_all[i] = len(qual_idxs_movie[quality_mask])
                depths[i] = np.zeros(num_rois_all[i])
                movies = movie_train, movie_test, random_sequences

            roi_ids_final[exp] = roi_ids_all
            movie_qis_final[exp] = movie_qis_all
            responses_final[exp] = responses_all
            num_rois_final[exp] = num_rois_all
            restriction_final[exp] = restriction
            keys_field_final[exp] = keys_field
            scan_sequence_final[exp] = scan_sequence_idxs
            depths_final[exp] = depths
            roi_masks_final[exp] = roi_masks
        roi_ids_final = [el for sublist in roi_ids_final for el in sublist]
        movie_qis_final = [el for sublist in movie_qis_final for el in sublist]
        responses_final = [el for sublist in responses_final for el in sublist]
        num_rois_final = [el for sublist in num_rois_final for el in sublist]
        restriction_final = [el for sublist in restriction_final for el in sublist]
        keys_field_final =[el for sublist in keys_field_final for el in sublist]
        scan_sequence_final = [el for sublist in scan_sequence_final for el in sublist]
        depths_final = [el for sublist in depths_final for el in sublist]
        roi_masks_final = [el for sublist in roi_masks_final for el in sublist]
        multi_dataset = MultiDataset(responses_final,
                                     num_rois_final,
                                     n_scans,
                                     scan_sequence_final,
                                     restriction_final,
                                     restriction_final,
                                     depths_final,
                                     movies=movies,
                                     luminance_paths_train=luminance_paths_train,
                                     luminance_paths_test=luminance_paths_test,
                                     contrast_paths_train=contrast_paths_train,
                                     contrast_paths_test=contrast_paths_test,
                                     group=False,
                                     downsample_size=downsample_size,
                                     adapt=adapt)
        self.multi_dataset = multi_dataset
        self.roi_masks = roi_masks_final
        self.roi_ids = roi_ids_final
        self.movie_qis = movie_qis_final


# %%
