from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from pylab import *
from scipy import optimize
from pyprind import prog_percent
import numba
from numba import jit
import random

from luminance_analysis.utilities import *
from luminance_analysis.plotting import *


@jit(nopython=True)
def sigmoid(stim, a, b, c):
    """Sigmoidal transformation to introduce response nonlinearities"""

    nl_transform = c / (1 + np.exp(-(stim-a)/b))
    nl_transform = nl_transform - nl_transform[0] #In case we want normalized transformation (shifted down to 0 in y axis)
    return(nl_transform)


@jit(nopython=True)
def numba_diff(arr):
    """Numba-compatiblle diff() function"""

    arr_diff = np.zeros_like(arr)
    for i in range(1, arr.shape[0]-1):
        arr_diff[i] = arr[i] - arr[i-1]
    return(arr_diff)


@jit(nopython=True)
def leaky_integrator(input, leak_rate):
    output = np.zeros_like(input)
    for i in range(1, input.shape[0]):
        output[i] = leak_rate*output[i-1] + (1-leak_rate)*input[i]
    return(output)


@jit(nopython=True)
#def GC_model_regressors_v2(model_regressors, exc_sh, exc_nl1, exc_nl2, inh_sh, inh_nl1, inh_nl2, onset_nl1, onset_nl2,
#                           offset_nl1, offset_nl2, w_exc, w_inh, w_onset, w_offset, leak_rate):
def GC_model_regressors_v2(model_regressors, frametime, exc_sh, exc_nl1, exc_nl2, inh_sh, inh_nl1, inh_nl2, onset_nl1, onset_nl2,
                           offset_nl1, offset_nl2, w_exc, w_inh, w_onset, w_offset, leak_rate):


    """Regressor-based model for fitting GC and IO responses"""

    # Excitatory input with nonlinearity
    stim_exc = sigmoid(model_regressors[0] - exc_sh * model_regressors[3], exc_nl1, exc_nl2, 1)

    # Inhibitory input with nonlinearity
    stim_inh = -sigmoid(model_regressors[0] - inh_sh * model_regressors[3], inh_nl1, inh_nl2, 1)

    # Onset input with nonlinearity
    stim_onset = sigmoid(model_regressors[1], onset_nl1, onset_nl2, 1)

    # Offset input with nonlinearity
    stim_offset = sigmoid(model_regressors[2], offset_nl1, offset_nl2, 1)

    # Calculate neuron output
    R = np.zeros_like(model_regressors[0])
    for i in range(R.shape[0]):
        R[i] = w_exc * stim_exc[i] + w_inh * stim_inh[i] + w_onset * stim_onset[i] + w_offset * stim_offset[i]

    #Filter response with a leaky integrator filter to introduce temporal integration
    output = leaky_integrator(R, leak_rate)

    # Convolve output with Ca2+ kernel
    #filter_t = np.arange(0, 1, 0.1)
    #tau = 0.1
    filter_t = np.arange(0, 10, frametime)
    tau = 2.2

    ca_kernel = np.exp(-filter_t / tau)
    GC_resp = np.convolve(output, ca_kernel)[:model_regressors[0].shape[0]]

    return (GC_resp)


@jit(nopython=True)
def create_regressors(stim):
    """Create necessary regressors to run the regressor-based model"""

    stim_reg = stim

    onset_reg = numba_diff(stim)
    onset_reg[np.where(onset_reg < 0)] = 0

    offset_reg = -numba_diff(stim)
    offset_reg[np.where(offset_reg < 0)] = 0
    

    sh_reg = np.zeros_like(stim)
    lum_changes = np.zeros(numba_diff(stim).nonzero()[0].shape[0] + 2, dtype=np.int64)
    lum_changes[1:-1] = numba_diff(stim).nonzero()[0]
    lum_changes[-1] = stim.shape[0]

    interval_bins = np.digitize(np.arange(0, stim.shape[0]), lum_changes)
    for timepoint, interval in zip(range(stim.shape[0]), interval_bins):
        senshist_val = np.nanmean(stim[lum_changes[interval - 2]: lum_changes[interval - 1]])
        sh_reg[timepoint] = senshist_val
        sh_reg[np.isnan(sh_reg)] = 0

    return (stim_reg, onset_reg, offset_reg, sh_reg)


def l1_cost_func(fit_params, model, model_regressors, frametime, roi_resp):
    """MAE cost function for minimization"""
    return (np.sum(np.abs(roi_resp - model(model_regressors, frametime, *fit_params)))) / roi_resp.shape[0]


def l2_cost_func(fit_params, model, model_regressors, frametime, roi_resp):
    """MSE cost function for minimization"""
    return (np.sum((roi_resp - model(model_regressors, frametime, *fit_params))**2)) / (2*roi_resp.shape[0])


def l1_reg_func(fit_params, reg_coef=0):
    """Regularization term for Lasso regularization"""
    return reg_coef*np.sum(np.abs(fit_params))


def l2_reg_func(fit_params, reg_coef=0):
    """Regularization term for Ridge regularization"""
    return reg_coef*np.sum((fit_params)**2)


def minimization_func(fit_params, cost_func, reg_func, model, model_regressors, frametime, roi_resp, reg_coef):
    """Full function to minimize, including cost and regularization terms"""
    return cost_func(fit_params, model, model_regressors, frametime, roi_resp) + reg_func(fit_params, reg_coef)


@jit(nopython=True)
def set_initial_guesses(guess_n, lower_bounds, upper_bounds, method):
    """Set initial guesses values randomly-sampled from the parametric space"""
    initial_guesses = np.empty((guess_n, len(lower_bounds)))

    if method == 'random':
        for j in range(guess_n):
            rand_guess = np.empty((len(lower_bounds)))
            for i in range(len(rand_guess)):
                rand_guess[i] = np.random.uniform(lower_bounds[i], upper_bounds[i])
            initial_guesses[j] = rand_guess
    elif method == 'midrange':
        mid_guess = np.empty((len(lower_bounds)))
        for i in range(len(mid_guess)):
            mid_guess[i] = (lower_bounds[i] + upper_bounds[i]) / 2
        for j in range(guess_n):
            initial_guesses[j] = mid_guess
    else:
        print("This is not a valid method to define initial guesses")

    return initial_guesses


def fit_rois_reps_regmodel(stim, traces, roi_list, frametime, model, params0, lower_bounds, upper_bounds, cost_func, reg_func,
                           goodness_func, reg_coef=0, minimization_method='SLSQP'):
    """Function to perform response fitting USING THE REGRESSOR-BASED MODEL.
    Uses half of the repetitions for training the model, and the other half for testing it"""

    # Create list with upper and lower boundaries for all parameters
    bounds = []
    for i in range(len(lower_bounds)):
        bounds.append((lower_bounds[i], upper_bounds[i]))

    # Create regressors on which the model is based
    model_regressors = create_regressors(stim)

    ##Fitting##
    model_fitting = {roi: {'params': [], 'fit_goodness': []} for roi in roi_list}

    for roi in prog_percent(roi_list):
        # Select columns without NaNs
        trace_reps = traces[roi, :, :][:, np.sum(np.isnan(traces[roi, :, :]), 0) != traces.shape[1]]

        # Check number of repetitions and randomly split into fitting and testing repetitions
        roi_reps = np.arange(trace_reps.shape[1])
        train_reps_num = int(np.ceil(trace_reps.shape[1] / 2))

        random.shuffle(roi_reps)
        train_reps = roi_reps[:train_reps_num]
        test_reps = roi_reps[train_reps_num:]

        # Concatenate all training traces (one after another) that will be used for fitting parameters
        train_rep_resps = np.concatenate([trace_reps[:, rep] for rep in train_reps])

        # Do the same for the regressors
        train_model_regressors = [np.tile(reg, train_reps_num) for reg in model_regressors]

        # Fit parameters
        res = optimize.minimize(minimization_func, method=minimization_method, args=(cost_func, reg_func, model,
                                                                                     train_model_regressors,
                                                                                     frametime, train_rep_resps,
                                                                                     reg_coef),
                                x0=params0, bounds=bounds)
        model_fitting[roi]['params'] = res.x

        # Concatenate all testing repetitions in order to test the goodness of the fit
        test_rep_resps = np.concatenate([trace_reps[:, rep] for rep in test_reps])

        # Do the same for the regressor
        test_model_regressors = [np.tile(reg, roi_reps.shape[0] - train_reps_num) for reg in model_regressors]

        # Calculate goodness of fit for test repetitions concatenated
        goodness_fit = goodness_func(model_fitting[roi]['params'], model, test_model_regressors, frametime,
                                     test_rep_resps)
        model_fitting[roi]['fit_goodness'] = goodness_fit

    return (model_fitting)