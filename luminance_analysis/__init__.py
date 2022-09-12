import flammkuchen as fl
import numpy as np
from pathlib import Path
import json
from numba import jit
from luminance_analysis.loading import load_into_hdf5_python
from math import ceil
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import interp1d
from luminance_analysis.roi_display import overimpose_shade
from luminance_analysis.roi_display import overimpose_shade
from luminance_analysis.utilities import reliability

from skimage.filters import threshold_otsu


class Data(object):
    """Abstract class with attributes and methods that are used for both
    single fish and pooled data.
    """
    def __init__(self, path):
        self.path = Path(path)

        self._anatomy = None
        self._roi_stack = None
        self._roi_stack_morphed = None
        self._traces = None
        self._stim_params = None
        self._metadata = None
        self._stimarray = None
        self._stimarray_rep = None
        self._resampled_stim = None
        self._roi_map = None

        self._time_im = None
        self._time_im_rep = None
        self.dt_im = self.time_im[1]

        # Definition of stimulus times depending of protocol.
        # This is a bit hardcoded, but the stimulus was always the same so there
        # is nooo need to be flexible. The try-except takes care of corrupted metadata
        # lacking the protocol_params key.

        # Correct corrupted metadata:
        if not "protocol_params" in self.metadata["stimulus"].keys():
            if self.metadata["stimulus"]["log"][-1]["t_stop"] < 314:
                name = "luminance_flashes"
                self.metadata["stimulus"]["protocol_params"] = dict(name=name,
                                                                    shuffled_reps=6)
            else:
                name = "luminance_steps"
                self.metadata["stimulus"]["protocol_params"] = dict(name=name,
                                                                    shuffled_reps=6)

        if self.metadata["stimulus"]["protocol_params"]["name"] == "luminance_flashes":
            # Flashes start and stop times in seconds,
            # from the protocol definition:
            stim_limits_s = {'f1': (7, 10), 'f2': (17, 24), 'f3': (31, 52)}
            self.rep_dur = 52

        elif self.metadata["stimulus"]["protocol_params"]["name"] == "luminance_steps":
            # Steps start and stop times in seconds,
            # from the protocol definition:
            stim_limits_s = {'s1': (7, 32), 's2': (39, 68), 's3': (75, 106)}
            self.rep_dur = 106


        # Offset: since the pause after the last flash/step serves as the
        # pause before the first one of the new sweep, we crop trace
        # starting with an offset to avoid that the trial starts with the
        # off of the last stim.
        self.offset_s = 5

        # padding around the stimulus beginning and end:
        self.pad_val_pre = 2
        self.pad_val_post = 5

        self.stim_limits = dict()
        for k, v in stim_limits_s.items():
            self.stim_limits[k] = slice(
                self.frame_from_t(v[0] - self.offset_s - self.pad_val_pre),
                self.frame_from_t(v[1] - self.offset_s + self.pad_val_post))

    @property
    def anatomy(self):
        return None

    @property
    def time_im(self):
        return None

    @property
    def time_im_rep(self):
        return None

    @property
    def resampled_stim(self):
        return None

    @property
    def roi_stack(self):
        return None

    @property
    def roi_stack_morphed(self):
        return None

    @property
    def traces(self):
        return None

    @property
    def stim_params(self):
        return None

    @property
    def stimarray(self):
        return None

    def frame_from_t(self, t):
        """
        Find closest frame before given time:
        :param t: time, in seconds
        :return: frame index
        """
        if type(t) == list:
            return [np.argwhere(self.time_im - t_i <= 0)[-1][0] for t_i in t]
        else:
            return np.argwhere(self.time_im - t <= 0)[-1][0]




class PooledData(Data):
    """Class to handle pooled data across several fish.
    """
    def __init__(self, path, *args, resp_threshold=None,
                 nanfraction_thr=None, **kwargs):
        path = Path(path)
        self.fishdata = []
        for f in list(path.glob("*_f[0-9]")):
            self.fishdata.append(FishData(f, resp_threshold=resp_threshold,
                                          nanfraction_thr=nanfraction_thr))
            # print("loaded {}".format(f))

        print(self.fishdata)
        super().__init__(path, *args, **kwargs)
        self.rois_fish_map = None

    @property
    def traces(self):
        if self._traces is None:
            # Load traces from all fish to concatenate them:
            traces_list = [f.traces for f in self.fishdata]

            # Find length on the reps axis for all traces,
            # to find cell with maximum n of repetitions for the shape of the array:
            shapes_reps = [t.shape[2] for t in traces_list]

            self._traces = np.empty((self.roi_map.shape[1],
                                    traces_list[0].shape[1],
                                    max(shapes_reps))) * np.nan
            for i, t in enumerate(traces_list):
                self._traces[self.roi_map[0,:] == i, :, :t.shape[2]] = t

        return self._traces

    @property
    def roi_map(self):
        if self._roi_map is None:
            map_list = [f.roi_map for f in self.fishdata]

            # Make an array with length=total ROIs number that tells for each
            #  ROI the index of the fish from which it comes:
            rois_fish_map = np.concatenate([[n, ] * t.shape[0] for n, t in
                                           enumerate(map_list)])
            rois_roi_map = np.concatenate(map_list)

            self._roi_map = np.array([rois_fish_map, rois_roi_map])

        return self._roi_map

    @property
    def metadata(self):
        # We just grab metadata from the first fish:
        if self._metadata is None:
            self._metadata = self.fishdata[0].metadata

        return self._metadata

    @property
    def stimarray_rep(self):
        if self._stimarray_rep is None:
            self._stimarray_rep = self.fishdata[0].stimarray_rep

        return self._stimarray_rep

    @property
    def stimarray(self):
        # We also grab stimulus array from the first fish:
        if self._stimarray is None:
            self._stimarray = self.fishdata[0].stimarray

        return self._stimarray

    @property
    def time_im(self):
        # We just grab metadata from the first fish:
        if self._time_im is None:
            self._time_im = self.fishdata[0].time_im

        return self._time_im

    @property
    def time_im_rep(self):
        # We just grab metadata from the first fish:
        if self._time_im_rep is None:
            self._time_im_rep = self.fishdata[0].time_im_rep

        return self._time_im_rep

    @property
    def resampled_stim(self):
        # We just grab metadata from the first fish:
        if self._resampled_stim is None:
            self._resampled_stim = self.fishdata[0].resampled_stim

        return self._resampled_stim

    def get_roi_anatomy(self, roi_id, crop_around=30):
        return self.fishdata[self.roi_map[0, roi_id]].get_roi_anatomy(self.roi_map[1, roi_id], crop_around=crop_around)

    def get_roi_anatomy_stacks(self, roi_id, crop_around=30):
        return self.fishdata[self.roi_map[0, roi_id]].get_roi_anatomy_stacks(self.roi_map[1, roi_id], crop_around=crop_around)

    def get_cluster_anatomy(self):
        pass

    def get_cluster_anatomy(self, roi_ids, morphed=True):
        anatomies_list = []

        for f_id in np.unique(roi_ids[0, :]):
            anatomies_list.append(self.fishdata[f_id].get_cluster_anatomy(
                roi_ids[1, roi_ids[0, :] == f_id], morphed=morphed))

        return anatomies_list


class FishData(Data):
    def __init__(self, path, *args, resp_threshold=None, nanfraction_thr=None):
        path = Path(path)
        self.fish_id = path.name
        self.resp_threshold = resp_threshold
        self.nanfraction_thr = nanfraction_thr

        try:
            self.data_dict_file = next(path.glob("*python_data*"))
        except StopIteration:
            try:
                self.data_dict_file = next(path.glob("*data_dict*"))
            except StopIteration:
                load_into_hdf5_python(path)
                self.data_dict_file = next(path.glob("*python_data*"))

        super().__init__(path, *args)

    @property
    def metadata(self):
        if self._metadata is None:
            with open(str(next(self.path.glob("*be*/*.json"))), "r") as f:
                self._metadata = json.load(f)
        return self._metadata

    @property
    def stimarray(self):
        if self._stimarray is None:
            try:
                dynamic_log = fl.load(str(next(self.path.glob("*be*/*dynamic_log.hdf5"))))
            except StopIteration:
                dynamic_log = fl.load(
                    str(next(self.path.glob("*be*/*stimulus_log.hdf5"))))
            time = dynamic_log['data']['t'].values
            lum = dynamic_log['data']['flash_luminance'].values
            # Real luminance values as measured in 2P projector
            # (fractions of max)
            lum[lum == 0.5] = 0.05
            lum[lum == 0.65] = 0.2
            
            self._stimarray = np.array([time, lum])         
            
        return self._stimarray

    @property
    def stimarray_rep(self):
        if self._stimarray_rep is None:
            self._stimarray_rep = np.concatenate([self.time_im_rep[:, np.newaxis],
                                                  self.resampled_stim[:, np.newaxis]], 1)

        return self._stimarray_rep

    @property
    def anatomy(self):
        if self._anatomy is None:
            self._anatomy = fl.load(str(self.data_dict_file), "/anatomy")
        return self._anatomy

    @property
    def time_im(self):
        if self._time_im is None:
            self._time_im = fl.load(str(self.data_dict_file),
                                      "/time_im") / 1000
            self._time_im -= self._time_im[0]
        return self._time_im

    @property
    def time_im_rep(self):
        if self._time_im_rep is None:
            self._time_im_rep = np.arange(self.traces.shape[1]) * self.dt_im
        return self._time_im_rep

    @property
    def roi_stack(self):
        if self._roi_stack is None:
            self._roi_stack = fl.load(str(self.data_dict_file), "/roi_stack")
        return self._roi_stack

    @property
    def roi_stack_morphed(self):
        if self._roi_stack_morphed is None:
            self._roi_stack_morphed = fl.load(str(self.data_dict_file), "/roi_stack_morphed")
        return self._roi_stack_morphed

    @property
    def roi_map(self):
        if self._roi_map is None:
            try:
                self._roi_map = np.array(fl.load(str(self.data_dict_file), "/roi_sel_list"))
            except ValueError:
                self._roi_map = np.arange(np.max(self.roi_stack))
        return self._roi_map

    @property
    def traces(self):
        if self._traces is None:
            traces = fl.load(str(self.data_dict_file), "/traces")
            traces = traces[self.roi_map, :, :]
            # Here we put the traces in a nicer form. Since in every plane there
            # are multiple repetitions of the stimulus, we crop and concatenate
            # them to convert the (roi_n x plane_timesteps x plane_reps) array
            # into a (roi_n x sweep_timesteps x (plane_reps*sweep_per_plane)) one.

            # Calculate sweep duration in number of frames (imaging points):
            sweep_duration_s = self.metadata["stimulus"]["log"][0]["duration"] / \
                               self.metadata["stimulus"]["protocol_params"]["shuffled_reps"]
            sweep_duration_pts = ceil(sweep_duration_s / self.dt_im)

            # Find start time of all sweeps:
            sweep_start_t_s = np.arange(6) * sweep_duration_s + self.offset_s
            sweep_start_pts = [self.frame_from_t(t) for t in sweep_start_t_s]

            # Crop and concatenate traces along the "repetition" dimension:
            traces = np.concatenate([traces[:, t:t + sweep_duration_pts, :]
                                     for t in sweep_start_pts], 2)

            if self.resp_threshold is not None or self.nanfraction_thr is not None:
                rel = reliability(traces, self.nanfraction_thr)
                if self.resp_threshold == "otsu":
                    rel_thr = threshold_otsu(rel[~np.isnan(rel)])
                    selection = rel > rel_thr
                else:
                    selection = rel > self.resp_threshold
                traces = traces[selection, :]

                self._roi_map = self._roi_map[selection]
            self._traces = traces
        return self._traces

    @property
    def resampled_stim(self):
        if self._resampled_stim is None:
            lum = self.stimarray[1, :]
            t = self.stimarray[0, :]
            f = interp1d(t, lum)
            self._resampled_stim = f(self.time_im_rep + self.offset_s)

        return self._resampled_stim

    def get_roi_planes(self, roi_id):
        return np.argwhere((self.roi_stack == roi_id).any((1, 2)))[0]

    def get_roi_anatomy(self, roi_id, crop_around=30):
        cell_center = np.array(center_of_mass(self.roi_stack == roi_id), dtype=np.int)

        anatomy_img = self.anatomy[cell_center[0],
                                   cell_center[1]-crop_around:cell_center[1]+crop_around,
                                   cell_center[2]-crop_around:cell_center[2]+crop_around]
        roi_img = (self.roi_stack == roi_id)[cell_center[0],
                                   cell_center[1]-crop_around:cell_center[1]+crop_around,
                                   cell_center[2]-crop_around:cell_center[2]+crop_around]

        return overimpose_shade(anatomy_img, roi_img)

    def get_roi_anatomy_stacks(self, roi_id, crop_around=30):
        roi_planes = np.unique(np.argwhere(self.roi_stack == roi_id)[:, 0])
        roi_center = np.array(center_of_mass(self.roi_stack == roi_id), dtype=np.int)

        anatomy_img = self.anatomy[roi_planes,
                      roi_center[1] - crop_around:roi_center[1] + crop_around,
                      roi_center[2] - crop_around:roi_center[2] + crop_around]

        roi_img = (self.roi_stack == roi_id)[roi_planes,
                  roi_center[1] - crop_around:roi_center[1] + crop_around,
                  roi_center[2] - crop_around:roi_center[2] + crop_around]

        return anatomy_img, roi_img



    def get_cluster_anatomy(self, roi_idxs, morphed=True):
        if morphed:
            return _find_cluster_rois(self.roi_stack_morphed.astype(np.float),
                                      list(roi_idxs))
        else:
            return _find_cluster_rois(self.roi_stack.astype(np.float),
                                      list(roi_idxs))


@jit(nopython=True)
def _find_cluster_rois(stack, roi_list):
    cluster_map = np.zeros(stack.shape)
    for n in range(stack.shape[0]):
        for j in range(stack.shape[1]):
            for i in range(stack.shape[2]):
                if stack[n, j, i] in roi_list:
                    cluster_map[n, j, i] = 1

    return cluster_map


def traces_stim_from_path(path, resp_threshold=None, nanfraction_thr=None, return_pooled_data=False):
    # Create class and load data:
    pooled_data = PooledData(path, resp_threshold=resp_threshold,
                             nanfraction_thr=nanfraction_thr)
    stim = pooled_data.stimarray_rep  # get stimulus array
    traces = pooled_data.traces  # get raw traces

    # Calculate mean responses and normalize:
    meanresps = np.nanmean(traces, 2)
    meanresps = (
    (meanresps.T - np.nanmean(meanresps, 1)) / np.nanstd(meanresps, 1)).T

    if return_pooled_data:
        return stim, traces, meanresps, pooled_data
    else:
        return stim, traces, meanresps

def get_meanresp_during_interval(path, timepoints, start_after, post_int_s, resp_threshold=None, nanfraction_thr=None,):

    # Create class and load data:
    pooled_data = PooledData(path, resp_threshold=resp_threshold,
                             nanfraction_thr=nanfraction_thr)
    stim = pooled_data.stimarray_rep  # get stimulus array
    traces = pooled_data.traces  # get raw traces

    # Calculate mean responses and normalize:
    meanresps = np.nanmean(traces, 2)
    meanresps = (
    (meanresps.T - np.nanmean(meanresps, 1)) / np.nanstd(meanresps, 1)).T

    mean_interval_resps = np.array([np.nanmean(meanresps[:, slice(*pooled_data.frame_from_t([t + start_after, t + post_int_s]))], 1)
              for t in timepoints])

    return mean_interval_resps