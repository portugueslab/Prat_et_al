import flammkuchen as fl
import numpy as np
from scipy import io
from pathlib import Path
from skimage import io as skio
from ast import literal_eval
from scipy.stats import mode


def load_into_f5(path):
    path = Path(path)
    try:
        traces = io.loadmat(str(path / 'traces3d.mat'))['traces_z']
        roi_stack = io.loadmat(str(path / 'rois3d.mat'))['rois']
    except:
        traces = io.loadmat(str(path / 'tracesmanual.mat'))['traces_z']
        roi_stack = io.loadmat(str(path / 'roismanual.mat'))['rois']
    roi_stack -= 1  # subtract 1 for convention on ROIs counting from 0

    stim = io.loadmat(str(path / 'behavior.mat'))['behavior']
    anatomy = io.loadmat(str(path / 'anatomy.mat'))['anatomy']

    try:
        # Load morphed stack and subtract 1 as there the convention is
        # 0es outside.
        roi_morphed = skio.imread(str(path / "morphed_rois.tif")).astype(np.int16) - 1
    except FileNotFoundError:
        print("morphed ROIs not found!")
        # If no morphed file found, use normal ROI stack instead:
        roi_morphed = np.moveaxis(np.int16(roi_stack), 2, 0)

    try:
        with open(next(path.glob("*rois*.txt")), "r") as f:
            roi_sel_list = [literal_eval(line) for line in f]
        roi_sel_list = np.array(roi_sel_list[0])
    except StopIteration:
        print("no mask found for these data!")
        roi_sel_list = np.arange(traces.shape[0])

    # Create dict with all data:
    data_dict = dict()

    data_dict['roi_stack'] = np.moveaxis(np.int16(roi_stack), 2, 0)  # Positions of the ROIs
    data_dict['roi_stack_morphed'] = roi_morphed  # Positions of the morphed ROIs
    data_dict['anatomy'] = np.moveaxis(anatomy, 2, 0)  # Fish anatomy
    data_dict['time_im'] = stim['time_im'][0][0][0]  # Time of imaging
    data_dict['time_be'] = stim['time_be'][0][0][0]  # Time of stimulus
    data_dict['roi_sel_list'] = roi_sel_list  # Selected ROIs from masking

    # Here we create numpy matrix to store efficiently the rois.
    # Find maximum number of repetitions for a ROI:
    max_reps = 0
    for t in traces:
        max_reps = max(max_reps, t[0].shape[0])

    # Create array of n_rois x timepoints x max_reps shape:
    traces_mat = np.empty(
        (traces.shape[0], traces[0][0].shape[1], max_reps)) * np.nan

    # Now store ROI traces here:
    for i, t in enumerate(traces):
        traces_mat[i, :, :t[0].shape[0]] = t[0].T

    data_dict['traces'] = traces_mat

    filename = path / (path.name + '_data_dict.h5')
    fl.save(str(filename), data_dict)

def load_into_hdf5_python(path, stim_protocol=None):
    path = Path(path)
    print(path)
    if stim_protocol is None:
        stim_protocol = path.parts[-3]
    roi_data = fl.load(str(path / "python_rois.hdf5"))
    roi_stack = roi_data["stack"]
    roi_morphed = np.int16(roi_stack)

    stim = io.loadmat(str(path / 'behavior.mat'))['behavior']
    anatomy = skio.imread(str(path / 'anatomy.tif'))

    roi_sel_list = np.arange(roi_data["traces"].shape[0])

    # Create dict with all data:
    data_dict = dict()

    data_dict['roi_stack'] = np.int16(roi_stack)  # Positions of the ROIs
    data_dict['roi_stack_morphed'] = roi_morphed  # Positions of the morphed ROIs
    data_dict['anatomy'] = np.swapaxes(anatomy, 1, 2)  # Fish anatomy
    data_dict['time_im'] = stim['time_im'][0][0][0]  # Time of imaging
    data_dict['time_be'] = stim['time_be'][0][0][0]  # Time of stimulus
    data_dict['roi_sel_list'] = roi_sel_list  # Selected ROIs from masking

    ##Simplify traces matrix to reduce size of third dimension
    raw_traces = roi_data["traces"]

    ##Ugly fix to account for some change in fimpy pipeline
    if raw_traces.shape[-1] > 500: #If some ROI spans more than 500 planes, we assume this dimension correspond to # of ROIs
        raw_traces = np.swapaxes(raw_traces, 1, 2)
    else:
        pass

    ##Define value used to fill empty reps
    traces_mode = mode(raw_traces[0, :, :], axis=None)
    mode_value = traces_mode.mode[0]

    #Calculate maximum amount of planes in which a single ROI is found
    max_planes_list = []

    for roi in range(raw_traces.shape[0]):
        if mode_value == 0:
            roi_plane_count = np.sum(~np.all(raw_traces[roi, :, :] == 0, axis=0))
            max_planes_list.append(roi_plane_count)
        else:
            roi_plane_count = np.sum(~np.all(np.isnan(raw_traces[roi, :, :]), axis=0))
            max_planes_list.append(roi_plane_count)

    planes_dim = np.max(max_planes_list)

    #Create new traces matrix with reduced size
    traces = np.empty((raw_traces.shape[0], raw_traces.shape[1], planes_dim))
    traces[:] = np.nan

    for roi in range(raw_traces.shape[0]):
        if mode_value == 0:
            roi_planes_mask = ~np.all(raw_traces[roi, :, :] == 0, axis=0)
            traces[roi, :, :np.sum(roi_planes_mask)] = raw_traces[roi, :, roi_planes_mask].transpose()
        else:
            roi_planes_mask = ~np.all(np.isnan(raw_traces[roi, :, :]), axis=0)
            traces[roi, :, :np.sum(roi_planes_mask)] = raw_traces[roi, :, roi_planes_mask].transpose()

    data_dict['traces'] = traces

    filename = path / (path.name + 'python_data.h5')
    fl.save(str(filename), data_dict)


def yxz_to_zxy(stack):
    np.swapaxes(stack, 1, 2)
