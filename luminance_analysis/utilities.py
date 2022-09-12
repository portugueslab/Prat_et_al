

import numpy as np
from numba import jit
import pandas as pd
from math import factorial
from scipy import signal

def nanzscore(array, axis=0):
    return (array - np.nanmean(array, axis=axis))/np.nanstd(array, axis=axis)



@jit(nopython=True)
def get_ROI_coords_3D_planewise(stack, rois: np.ndarray):
    """ A function to efficiently extract ROI data, 3D ROIs

    :param stack: imaging stack
    :param rois: image where each ROI is labeled by the same integer
    :param max_rois: number of ROIs per stack
    :return: coords, areas, traces
    """
    coords = np.zeros(len(rois.shape))
    area = 0
    for i in range(rois.shape[0]):
        for j in range(rois.shape[1]):
            for k in range(rois.shape[2]):
                roi_id = rois[i, j, k]
                if roi_id > -1:
                    area += 1
                    coords[roi_id, 0] += i
                    coords[roi_id, 1] += j
                    coords[roi_id, 2] += k

    coords /= area

    return coords


def find_transitions(stim, time=None):
    if time is None:
        time = np.arange(stim.shape[0])

    return _find_transitions(stim.copy(), time.copy())


@jit(nopython=True)
def _find_transitions(stim, time):
    # Calculate change between each stimulus timepoint
    stim_borders = np.diff(stim)

    # Get timepoints at which stim array changes and value of change
    lum_transitions = [(time[0], stim[0])]
    for i in range(stim_borders.shape[0]):
        if stim_borders[i] != 0.:
            lum_transitions.append((time[i + 1], stim_borders[
                i]))  # +1 to account for np.diff loosing first point
    lum_transitions.append((time[-1], stim[-1]))

    return lum_transitions


def fill_nan(traces):
    return pd.DataFrame(traces).interpolate().as_matrix()


def reliability(traces, nanfraction_thr=0.15):
    """ Function to calculate reliability of cell responses.
    Reliability is defined as the average of the across-trials correlation.
    This measure seems to generally converge after a number of 7-8 repetitions, so it
    is advisable to have such a repetition number to use it.
    :param traces:
    :return:
    """
    reliability = np.zeros(traces.shape[0])
    for i in range(len(reliability)):
        trace = traces[i, :, :]
        sel_trace = trace[:, np.sum(np.isnan(trace), 0) != trace.shape[0]]

        if nanfraction_thr is not None:
            if np.sum(np.isnan(sel_trace)) > nanfraction_thr * sel_trace.size:
                reliability[i] = 0
            else:
                corr = np.corrcoef(sel_trace.T)
                np.fill_diagonal(corr, np.nan)
                reliability[i] = np.nanmean(corr)

    return reliability


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """ Apply Savitzky-Golay filter to a 1-d arrays

    :param data: matrix of ROI 1-d arrays to filter
    :param window_size: length of the filter window. Must be a positive & odd integer
    :param order: order of the polynomial to fit the data points
    :param deriv: order of the derivative to compute
    :return: filtered_data

    https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except (ValueError, msg):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    y_filtered = np.convolve( m[::-1], y, mode='valid')

    return y_filtered


def smooth_data(data, window_size, order, deriv=0, rate=1):
    """ Apply Savitzky-Golay filter to 1-d array or matrix of 1-d arrays

    :param data: matrix of ROI 1-d arrays to filter
    :param window_size: length of the filter window. Must be a positive & odd integer
    :param order: order of the polynomial to fit the data points
    :param deriv: order of the derivative to compute
    :return: filtered_data

    https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    """
    if data.ndim == 1:
        smoothed_data = savitzky_golay(data, window_size, order, deriv, rate)
    elif data.ndim == 2:
        smoothed_data = np.empty_like(data)
        for roi in range(data.shape[0]):
            smoothed_trace = savitzky_golay(data[roi, :], window_size, order, deriv, rate)
            smoothed_data[roi, :] = smoothed_trace

    return smoothed_data


def pearson_regressors(traces, regressors):
    """ Gives the pearson correlation coefficient

    :param traces: the traces, with time in rows
    :param regressors: the regressors, with time in rows
    :return: the pearson correlation coefficient
    """
    # two versions, depending whether there is one or multiple regressors
    X = traces
    Y = regressors
    if len(Y.shape) == 1:
        numerator = np.dot(X.T, Y) - X.shape[0] * np.nanmean(X, 0) * np.nanmean(Y)
        denominator = (X.shape[0] - 1) * np.nanstd(X, 0) * np.nanstd(Y)
        result = numerator / denominator
    else:
        numerator = np.dot(X.T, Y) - X.shape[0] * np.outer(np.nanmean(X, 0),
                                                           np.nanmean(Y, 0))
        denominator = (X.shape[0] - 1) * np.outer(np.nanstd(X, 0),
                                                  np.nanstd(Y, 0))
        result = (numerator / denominator).T

    return result


def pixelwise_correlation(stack, regressors):
    traces = stack.reshape((stack.shape[0], -1))

    coefs = pearson_regressors(traces, regressors)

    coefs[np.isnan(coefs)] = 0
    return coefs.reshape(avg.shape[1:])


def sample_cluster_rois(labels, cluster, sample_size):
    cluster_rois = [roi[0] for roi in np.argwhere(labels == cluster)]
    print('Cluster {} has {} ROIs'.format(cluster, len(cluster_rois)))
    sampled_rois = np.random.choice(cluster_rois, sample_size)
    return sampled_rois


def get_kernel(tau=7, ker_len=30, delay=0, off=0.000001):
    return np.insert(np.exp(-np.arange(0, ker_len) / tau), 0, np.ones(delay)*off)


def deconv_resamp_norm_trace(trace, trace_time, ref_time, tau, ker_len,
                             smooth_wnd=4,
                             normalization="zscore"):
    # Smooth:
    trace_sm = pd.DataFrame(trace.T).rolling(smooth_wnd, center=True,
                                             min_periods=1).mean().values

    # Deconvolve:
    if tau is not None:
        extended = np.append(trace_sm, np.zeros(ker_len - 1))
        deconvolved, _ = signal.deconvolve(extended, get_kernel(tau=tau, ker_len=ker_len))
        deconvolved[0] = deconvolved[1]  # ugly fix for artifact at the beginning
    else:
        deconvolved = trace_sm[:, 0]

    # Resample:
    resampled = np.interp(ref_time, trace_time, deconvolved)

    # Normalize:
    if normalization == "none":
        normalized = resampled  # nanzscore(resampled)
    elif normalization == "zscore":
        normalized = nanzscore(resampled)
    elif normalization == "minmax":
        trace_min = np.nanmin(resampled)
        normalized = resampled - trace_min

        trace_max = np.nanmax(normalized)
        normalized = normalized / trace_max
    elif normalization == "integral":
        trace_min = np.nanmin(resampled)
        normalized = resampled - trace_min

        trace_sum = np.nansum(normalized)
        normalized = normalized / trace_sum
    return normalized


def crop_intervals_from_mat(data_mat, timepoints, dt, window, pre_trigger=0):
    cropped_mat = np.empty((data_mat.shape[0] * len(timepoints), int((window + pre_trigger) / dt), data_mat.shape[2]))

    if pre_trigger != 0:
        timepoints = [(int(trigger[0] - (pre_trigger / dt)), trigger[1] - pre_trigger) for trigger in
                      timepoints]

    for fish in range(data_mat.shape[2]):
        for stim_rep in range(data_mat.shape[0]):
            for timepoint in range(len(timepoints)):
                if timepoints[timepoint][0] < 0:
                    cropped_interval = np.concatenate((data_mat[stim_rep, timepoints[timepoint][0]:, fish],
                                                       data_mat[stim_rep,
                                                       :cropped_mat.shape[1] + timepoints[timepoint][0], fish]))
                else:
                    cropped_interval = data_mat[stim_rep,
                                       timepoints[timepoint][0]:timepoints[timepoint][0] + int((window + pre_trigger) /
                                                                                               dt), fish]
                cropped_mat[timepoint + len(timepoints) * stim_rep, :, fish] = cropped_interval
    return (cropped_mat)


def smooth_df(df, window, omit_col=None):
    extended_df = pd.DataFrame(np.concatenate((df.values[-window // 2:, :], df.values, df.values[:window // 2, :])))
    smoothed_extended_df = extended_df.rolling(window + 1, center=True).mean()
    smoothed_df = pd.DataFrame(smoothed_extended_df.values[window // 2:-window // 2, :], index=df.index,
                               columns=df.columns)

    if omit_col is not None:
        smoothed_df[omit_col] = df[omit_col]


# This function is a beauty:
def smooth_traces(traces, win=4, method="mean"):
    if method == "mean":
        return pd.DataFrame(
            np.concatenate([traces[:, -win//2:], traces, traces[:, :win//2]], 1).T
        ).rolling(5, center=True).mean().values[win//2:-win//2, :].T
    elif method == "median":
        return pd.DataFrame(
            np.concatenate(
                [traces[:, -win // 2:], traces, traces[:, :win // 2]], 1).T
        ).rolling(5, center=True).median().values[win // 2:-win // 2, :].T



def butter_highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y



def get_mn_and_error(traces, normalize="zscore"):
    # Find number of repetitions for each cell:
    repetition_n = traces.shape[2] - np.isnan(traces).all(1).sum(1)

    mn = np.nanmean(traces, 2)  # mean
    error = (np.nanstd(traces, 2).T / np.sqrt(repetition_n)).T  # error

    if normalize == "zscore":
        error = ((error.T - np.nanmean(mn, 1)) / np.nanstd(mn, 1)).T
        mn = ((mn.T - np.nanmean(mn, 1)) / np.nanstd(mn, 1)).T

    return mn, error


def train_test_split(traces):
    """ Split a matrix of traces in a train and a test block of equal number
    of repetitions.
    """
    train = np.ones(traces.shape[:2] + (traces.shape[2] // 2,)) * np.nan
    test = np.ones(traces.shape[:2] + (traces.shape[2] // 2,)) * np.nan
    for i in range(traces.shape[0]):
        valid_idxs = np.argwhere(~np.isnan(traces[i, :, :]).all(0))[:, 0]
        # for some reason this indexing require transposing back
        train[i, :, :len(valid_idxs) // 2] = traces[i, :, valid_idxs[:len(valid_idxs) // 2]].T
        test[i, :, :len(valid_idxs) // 2] = traces[i, :, valid_idxs[len(valid_idxs) // 2:]].T

    return train, test