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
repo_directory = "/gpfs01/euler/User/lhoefling/GitHub/datajoint_imaging_V2/"
# Add repository directory to %PYTHONPATH
sys.path.insert(0, repo_directory)
# Load configuration for user
dj.config.load(repo_directory + "conf/dj_conf_lhoefling.json")
from schema.imaging_schema import *
from schema.stimulus_schema import MovieQI, PreprocTraces, ChirpQI, OsDsIndexes,\
    PreprocParams
from cnn_sys_ident.retina.data import *
from schema.stimulus_schema import MovieQI


# %%
class MultiDatasetWrapper:
    def __init__(self, experimenter, date, exp_num, stim_path):
        self.stim_path = stim_path
        self.key = dict(experimenter=experimenter,
                        date=date,
                        exp_num=exp_num,
                        stim_id=5)

    def interpolate_weights(self, orig_times, orig_data, new_times):
        data_interp = interp1d(
            orig_times.flatten(),
            orig_data.flatten(),
            kind="linear"
        )(new_times)

        return data_interp

    def generate_dataset(self,
                         filter_traces=True,
                         preproc_param_set_id=2,
                         quality_threshold_movie=0,
                         quality_threshold_chirp=0,
                         quality_threshold_ds=0):
        if filter_traces:
            preproc_param_key = 'preproc_param_set_id = {}'.format(
                preproc_param_set_id
            )
            ff_ = (PreprocParams() & preproc_param_key).fetch1('cutoff')
            nn_ = (PreprocParams() & preproc_param_key).fetch1('non_negative')
            bs_ = (PreprocParams() & preproc_param_key).fetch1('subtract_baseline')
            bd_ = (PreprocParams() & preproc_param_key).fetch1('standardize')
            print('Loading traces preprocessed with the following settings: '
                  'Filter frequencey : {}, non-negative: {}, '
                  'baseline subtracted: {}, standardized: {}'.format(
                ff_, nn_, bs_, bd_
            ))
        else:
            warnings.warn("You are retrieving raw traces, but quality indexes "
                          "are returned based on traces preprocessed with "
                          "settings preproc_param_set_id = {}".format(
                preproc_param_set_id
            ))
            preproc_param_key = 'preproc_param_set_id = 1'
        key = self.key
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
        movie_train, movie_test, random_sequences = \
            load_stimuli("Train_joined.tif",
                         "Test_joined.tif",
                         STIMULUS_PATH=stim_path)

        header_paths = (Presentation() & key).fetch('h5_header')
        n_scans = len(header_paths)
        scan_sequence_idxs = np.zeros(n_scans, dtype=int)
        for i, h in enumerate(header_paths):
            filename = h.split('/')[-1]
            scan_sequence_idxs[i] = \
                int(re.search("MC(.+?).h5", filename).group(1))

        fields = (Presentation() & key).fetch("field_id")
        responses_all = [[] for _ in fields]
        num_rois_all = [[] for _ in fields]
        restriction = [[] for _ in fields]
        keys_field = [{} for _ in fields]
        for i, f in enumerate(fields):
            keys_field[i].update(key)
            keys_field[i].update(dict(field_id=f))
            qual_idxs_movie = \
                (MovieQI() & keys_field[i] &
                 preproc_param_key).fetch("movie_qi")
            temp_key = deepcopy(keys_field[i])
            temp_key.pop("stim_id")
            qual_idxs_chirp = \
                (ChirpQI() & temp_key & 'preproc_param_set_id=1').fetch("chirp_qi")
            qual_idxs_ds = \
                (OsDsIndexes() & temp_key & 'preproc_param_set_id=1').fetch("d_qi")
            if filter_traces:
                traces = \
                    (PreprocTraces() * Presentation() &
                     keys_field[i] & preproc_param_key).fetch("preproc_traces")
                raw_traces = \
                    (Traces() * Presentation() &
                     keys_field[i]).fetch("traces")
                assert len(traces) == len(raw_traces), \
                    "Number of ROIs returned for raw traces is not the " \
                    "same as number of ROIs returned for preproc traces. You " \
                    "need to populate PreprocTraces() with the desired preproc " \
                    "settings."\

            else:
                traces = \
                    (Traces() * Presentation() & keys_field[i]).fetch("traces")
            tracestimes = \
                (Traces() * Presentation() & keys_field[i]).fetch("traces_times")
            triggertimes = \
                (Presentation() & keys_field[i]).fetch1("triggertimes")
            upsampled_triggertimes = \
                [np.linspace(t, t + 4.9666667, 5 * 30) for t in triggertimes]
            upsampled_triggertimes = np.concatenate(upsampled_triggertimes)
            num_neurons = len(traces)
            assert num_neurons == len(qual_idxs_movie), \
                "Number of neurons and movie quality indexes not the same"
            assert num_neurons == len(qual_idxs_chirp), \
                "Number of neurons and chirp quality indexes not the same"
            assert num_neurons == len(qual_idxs_ds), \
                "Number of neurons and ds quality indexes not the same"
            responses = np.zeros((num_neurons, 150 * 123))
            quality_mask = np.logical_and(
                (qual_idxs_movie > quality_threshold_movie),
                np.logical_and((qual_idxs_chirp > quality_threshold_chirp),
                               (qual_idxs_ds > quality_threshold_ds)))
            for n in range(num_neurons):
                responses[n, :] = \
                    self.interpolate_weights(tracestimes[n],
                                             traces[n],
                                             upsampled_triggertimes)
            responses_all[i] = responses[quality_mask]
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
                                     group=False)
        self.multi_dataset = multi_dataset


# %%
