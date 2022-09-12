import numpy as np
from luminance_analysis import PooledData
from luminance_analysis.utilities import reliability


def load_pooled_dataset(path, resp_threshold=None, nanfraction_thr=0.15):
    pooled_data = PooledData(path)

    # Load stimulation arrays and traces
    stim = pooled_data.stimarray_rep
    traces = pooled_data.traces
    meanresps = np.nanmean(traces, 2)

    if resp_threshold is not None:
        rel = reliability(traces, nanfraction_thr)
        selection = rel > resp_threshold
        traces = traces[selection, :]
        meanresps = meanresps[selection, :]
    return pooled_data, stim, traces, meanresps
