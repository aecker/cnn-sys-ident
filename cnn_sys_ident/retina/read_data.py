# %%
import os
import sys
import numpy as np
import re
from copy import deepcopy
from scipy.interpolate import interp1d
import datajoint as dj
import warnings

# Specify repository directory
repo_directory = "/gpfs01/berens/user/cbehrens/RGC_DNN/datajoint_imaging_V2/"
# Add repository directory to %PYTHONPATH
sys.path.insert(0, repo_directory)
# Load configuration for user
dj.config.load(repo_directory + "conf/dj_conf_cbehrens.json")
from schema.imaging_schema import *
from schema.stimulus_schema import MovieQI, DetrendTraces, ChirpQI, OsDsIndexes,\
    DetrendParams, MouseCamMovieFiltParams, MouseCamMovieFiltering, Stimulus
from cnn_sys_ident.retina.data import *
from schema.stimulus_schema import MovieQI


# %%
class MultiDatasetWrapper:
    def __init__(self, experimenter, date, exp_num, stim_path, stim_id):
        self.stim_path = stim_path
        self.key = dict(experimenter=experimenter,
                        date=date,
                        exp_num=exp_num,
                        stim_id=stim_id)

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
                         target_fs=15
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
        key = self.key
        roi_masks = (Field().RoiMask() & key).fetch("roi_mask")
        stim_path = self.stim_path
        projname = (ExpInfo() & key).fetch1('projname')
        if projname.find("RGC") >= 0:
            hn = (Experiment() & key).fetch1("headername")
            eye = re.search("__(.+?).ini", hn).group(1)
            stim_path = stim_path + eye + "/"
        elif projname.find("BC") >= 0:
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
        if self.key["stim_id"] == 5:
            #movie_train shape: (16200, 56, 56, 2)
            #movie_test shape: (750, 56, 56, 2)
            movie_train, movie_test, random_sequences = \
                load_stimuli("Train_joined.tif",
                             "Test_joined.tif",
                             STIMULUS_PATH=stim_path,
                             downsample_size=downsample_size,
                             downsample_this=False,
                             mouse_cam=color_channels)
        elif self.key["stim_id"] == 0:
            stim_framerate = (Stimulus() & 'stim_id = {}'.format(
                self.key["stim_id"])
                    ).fetch1("framerate")
            assert target_fs % stim_framerate == 0, "target frame rate is not an integer multiple of native stimulus framerate"
            up_factor = int(target_fs // stim_framerate)  # determine by how much the trigger should be upsampled
            stim = (Stimulus() & 'stim_id = {}'.format(
                self.key["stim_id"])
                    ).fetch1("stimulus_trace") #shape (15, 20, 1500)
            stim = stim.transpose(-1, 0, 1)
            #repeat every stimulus frame up_factor times
            reshaped = stim.reshape(stim.shape[0], -1)
            repeated = np.repeat(reshaped, [up_factor]*stim.shape[0], axis=0)
            stim = repeated.reshape(repeated.shape[0], 15, 20)
            #split into 10 % test, 80 % train, 10 % test
            movie_train = stim[150*up_factor:-150*up_factor]
            movie_test = np.vstack(
                (stim[:150*up_factor, :, :], stim[-150*up_factor:, :, :]))
            #insert channel dimension
            movie_train = np.expand_dims(movie_train, -1)
            movie_test = np.expand_dims(movie_test, -1)
            random_sequences = []

        header_paths = (Presentation() & key).fetch('h5_header')
        n_scans = len(header_paths)
        scan_sequence_idxs = np.zeros(n_scans, dtype=int)
        if self.key["stim_id"] == 5:
            for i, h in enumerate(header_paths):
                filename = h.split('/')[-1]
                scan_sequence_idxs[i] = \
                    int(re.search("MC(.+?).h5", filename).group(1))

        fields = (Presentation() & key).fetch("field_id")
        roi_ids_all = [[] for _ in fields]
        responses_all = [[] for _ in fields]
        num_rois_all = [[] for _ in fields]
        restriction = [[] for _ in fields]
        keys_field = [{} for _ in fields]
        for i, f in enumerate(fields):
            keys_field[i].update(key)
            keys_field[i].update(dict(field_id=f))
            if detrend_traces:
                traces = \
                    (DetrendTraces() * Presentation() &
                     keys_field[i] & detrend_param_key).fetch("detrend_traces")
                raw_traces = \
                    (Traces() * Presentation() &
                     keys_field[i]).fetch("traces")
                assert len(traces) == len(raw_traces), \
                    "Number of ROIs returned for raw traces is not the " \
                    "same as number of ROIs returned for detrend traces. You " \
                    "need to populate DetrendTraces() with the desired detrend " \
                    "settings."\

            else:
                traces = \
                    (Traces() * Presentation() & keys_field[i]).fetch("traces")
            tracestimes = \
                (Traces() * Presentation() & keys_field[i]).fetch("traces_times")
            triggertimes = \
                (Presentation() & keys_field[i]).fetch1("triggertimes")
            if self.key["stim_id"] == 5:
                #if movie, upsample triggertimes to get 1 trigger per frame, (instead of just 1 trigger per sequence)
                upsampled_triggertimes = \
                    [np.linspace(t, t + 4.9666667, 5 * 30)
                     for t in triggertimes]
                upsampled_triggertimes = np.concatenate(upsampled_triggertimes)
            elif self.key["stim_id"] == 0:
                up_factor = target_fs/stim_framerate # determine by how much the trigger should be upsampled
                ifi = 1/stim_framerate #interframe interval
                upsampled_triggertimes = \
                    [np.linspace(t, t + ifi, up_factor, endpoint=False)
                     for t in triggertimes]
                upsampled_triggertimes = np.concatenate(upsampled_triggertimes)
            num_neurons = len(traces)
            roi_ids = (Roi() & keys_field[i]).fetch("roi_id")
            if quality_threshold_movie > 0:
                qual_idxs_movie = \
                    (MovieQI() & keys_field[i] &
                     detrend_param_key).fetch("movie_qi")
            else:
                qual_idxs_movie = np.ones(num_neurons, dtype=bool)
            temp_key = deepcopy(keys_field[i])
            temp_key.pop("stim_id")
            if quality_threshold_chirp > 0:
                qual_idxs_chirp = (
                        ChirpQI() & temp_key & 'detrend_param_set_id=1'
                ).fetch("chirp_qi")
            else:
                qual_idxs_chirp = np.ones(num_neurons, dtype=bool)
            if quality_threshold_ds > 0:
                qual_idxs_ds = (
                        OsDsIndexes() & temp_key & 'detrend_param_set_id=1'
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

            quality_mask = np.logical_and(
               (qual_idxs_movie > quality_threshold_movie),
               np.logical_and((qual_idxs_chirp > quality_threshold_chirp),
                              (qual_idxs_ds > quality_threshold_ds)))

            if self.key["stim_id"] == 5:
                responses = np.zeros((num_neurons, 150 * 123))
                for n in range(num_neurons):
                    responses[n, :] = \
                        self.interpolate_weights(tracestimes[n],
                                                 traces[n],
                                                 upsampled_triggertimes)
            elif self.key["stim_id"] == 0:
                responses = np.zeros((num_neurons, stim.shape[0]))
                for n in range(num_neurons):
                    responses[n, :] = \
                        self.interpolate_weights(tracestimes[n],
                                                 traces[n],
                                                 upsampled_triggertimes)

            responses_all[i] = responses[quality_mask]
            roi_ids_all[i] = roi_ids[quality_mask]
            num_rois_all[i] = len(qual_idxs_movie[quality_mask])
            depths = [np.zeros(num_rois_all[i]) for i in range(len(fields))]
            movies = movie_train, movie_test, random_sequences

        multi_dataset = MultiDataset(responses_all,
                                     num_rois_all,
                                     n_scans,
                                     scan_sequence_idxs,
                                     keys_field,
                                     restriction,
                                     depths,
                                     movies=movies,
                                     luminance_paths_train=luminance_paths_train,
                                     luminance_paths_test=luminance_paths_test,
                                     contrast_paths_train=contrast_paths_train,
                                     contrast_paths_test=contrast_paths_test,
                                     group=False,
                                     downsample_size=downsample_size,
                                     adapt=adapt)
        self.multi_dataset = multi_dataset
        self.roi_masks = roi_masks
        self.roi_ids = roi_ids_all


# %%
