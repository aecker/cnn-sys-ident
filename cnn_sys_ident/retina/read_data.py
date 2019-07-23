# %%
import os
import sys
import numpy as np
import re
from scipy.interpolate import interp1d
import datajoint as dj

# Specify repository directory
repo_directory = "/gpfs01/berens/user/cbehrens/RGC_DNN/datajoint_imaging_V2/"
# Add repository directory to %PYTHONPATH
sys.path.insert(0, repo_directory)
# Load configuration for user
dj.config.load(repo_directory + "conf/dj_conf_cbehrens.json")
from schema.imaging_schema import *
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

    def generate_dataset(self, filter_traces=False, quality_threshold=0):
        key = self.key
        stim_path = self.stim_path
        if key["experimenter"] == "Franke":
            hn = (Experiment() & key).fetch1("headername")
            eye = re.search("__(.+?).ini", hn).group(1)
            stim_path = stim_path + eye + "/"

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
            qual_idxs = \
                (MovieQI() & keys_field[i]).fetch("movie_qi")
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
            responses = np.zeros((num_neurons, 150 * 123))
            quality_mask = qual_idxs > quality_threshold
            for n in range(num_neurons):
                responses[n, :] = \
                    self.interpolate_weights(tracestimes[n],
                                             traces[n],
                                             upsampled_triggertimes)
            responses_all[i] = responses[quality_mask]
            num_rois_all[i] = len(qual_idxs[quality_mask])
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
                                     group=False,
                                     filter_traces=filter_traces)
        self.multi_dataset = multi_dataset


# %%
