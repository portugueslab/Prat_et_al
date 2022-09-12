import numpy as np
from numba import jit


@jit(nopython=True)
def time_since_flash(stim_sequence, th=0.01, dt=0):
    t_since = np.full(stim_sequence.shape[0], np.nan)
    t_in_flash = 0
    prev_stim = 0
    in_flash = True
    for i, curr_stim in enumerate(stim_sequence):
        if curr_stim > th and prev_stim <= th:
            in_flash = True
        if curr_stim <= th  and prev_stim > th:
            t_in_flash = 0
            in_flash =False
        if in_flash:
            t_in_flash += dt
            t_since[i] = t_in_flash
        prev_stim = curr_stim
    return t_since

@jit(nopython=True)
def get_valid_periods_steps(luminance_steps, n_skip=3):
    valid = np.zeros(len(luminance_steps), np.uint8)
    ls_lum = -1
    i_in = 0
    for i, l in enumerate(luminance_steps):
        if l != ls_lum:
            i_in = 0
        if i_in>=n_skip:
            valid[i] = True
        i_in += 1
        ls_lum = l
    return valid