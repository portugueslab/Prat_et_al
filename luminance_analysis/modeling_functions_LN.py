from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec/home/otprat/Documents/Neuroscience/Master/2017/MasterLab/Repos/luminance_analysis
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
def roll_arr(arr, n_roll):
    """Numba-compatible function to roll arrays"""
    rolled = np.empty_like(arr)
    rolled[:n_roll] = arr[-n_roll:]
    rolled[n_roll:] = arr[:-n_roll]
    return rolled

@jit(nopython=True)
def GC_model_full(stim, frametime, u1l1, u1l2, u1l3, u1nl1, u1nl2, u2l1, u2l2, u2l3, u2nl1, u2nl2, u3l1, u3l2, u3l3, u3nl1,
             u3nl2, u3nl3):
    #Model for GC responses
    ##Get shifted stimulation arrays
    stim_1 = roll_arr(stim, 1)
    stim_2 = roll_arr(stim, 2)

    ##Calculate number of frames required to encompass an interval of a certain duration
    t_interval = 5
    frames_interval = int(np.floor(t_interval / frametime))

    ##Calculate stim(t)-mean.stim(t[-5sec] - t)
    stim_history = np.empty(stim.shape)

    for i in range(stim_history.shape[0]):
        if i == 0 or i == 1:
            stim_history[i] = 0
        elif i <= frames_interval:
            stim_history[i] = stim[i] - stim[0:i - 1].mean()
        else:
            stim_history[i] = stim[i] - stim[i - frames_interval:i - 1].mean()

    ##Unit 1
    u1_l_filter = u1l1 * stim_1 + u1l2 * stim_2 + u1l3 * stim_history  # Linear filter for unit 1
    u1_output = sigmoid(u1_l_filter, u1nl1, u1nl2, 1)  # Apply nonlinear transformation (sigmoidal) to get unit1 output

    ##Unit 2
    u2_l_filter = u2l1 * stim_1 + u2l2 * stim_2 + u2l3 * stim_history  # Linear filter for unit 2
    u2_output = sigmoid(u2_l_filter, u2nl1, u2nl2, 1)  # Apply nonlinear transformation (sigmoidal) to get unit2 output

    ##Unit 3
    u3_l_filter = np.empty(stim.shape[0]) #Linear filter for unit 3

    for i in range(u3_l_filter.shape[0]):
        if i == 0:
            u3_l_filter[i] = u3l1*u1_output[i] + u3l2*u2_output[i]
        else:
            u3_l_filter[i] = u3l1*u1_output[i] + u3l2*u2_output[i] + u3l3*u3_l_filter[i-1]

    u3_output = sigmoid(u3_l_filter, u3nl1, u3nl2, u3nl3) #Apply nonlinear transformation (sigmoidal) to get unit3 output

    #Convolve unit3 output with Ca2+ kernel
    filter_t = np.arange(-1, 1, 0.1)
    tau = 0.1
    ca_kernel = np.exp(-filter_t / tau)

    GC_resp = np.convolve(u3_output, ca_kernel / 100)[:stim.shape[0]]

    return (GC_resp)

@jit(nopython=True)
def GC_model_nosenshist(stim, frametime, u1l1, u1l2, u1nl1, u1nl2, u2l1, u2l2, u2nl1, u2nl2, u3l1, u3l2, u3l3, u3nl1,
             u3nl2, u3nl3):
    #Model for GC responses
    ##Get shifted stimulation arrays
    stim_1 = roll_arr(stim, 1)
    stim_2 = roll_arr(stim, 2)

    ##Unit 1
    u1_l_filter = u1l1 * stim_1 + u1l2 * stim_2  # Linear filter for unit 1
    u1_output = sigmoid(u1_l_filter, u1nl1, u1nl2, 1)  # Apply nonlinear transformation (sigmoidal) to get unit1 output

    ##Unit 2
    u2_l_filter = u2l1 * stim_1 + u2l2 * stim_2  # Linear filter for unit 2
    u2_output = sigmoid(u2_l_filter, u2nl1, u2nl2, 1)  # Apply nonlinear transformation (sigmoidal) to get unit2 output

    ##Unit 3
    u3_l_filter = np.empty(stim.shape[0]) #Linear filter for unit 3

    for i in range(u3_l_filter.shape[0]):
        if i == 0:
            u3_l_filter[i] = u3l1*u1_output[i] + u3l2*u2_output[i]
        else:
            u3_l_filter[i] = u3l1*u1_output[i] + u3l2*u2_output[i] + u3l3*u3_l_filter[i-1]

    u3_output = sigmoid(u3_l_filter, u3nl1, u3nl2, u3nl3) #Apply nonlinear transformation (sigmoidal) to get unit3 output

    #Convolve unit3 output with Ca2+ kernel
    filter_t = np.arange(-1, 1, 0.1)
    tau = 0.1
    ca_kernel = np.exp(-filter_t / tau)

    GC_resp = np.convolve(u3_output, ca_kernel / 100)[:stim.shape[0]]

    return (GC_resp)


def l1_cost_func(fit_params, model, stim, frametime, roi_resp):
    """MAE cost function for minimization"""

    return (np.sum(np.abs(roi_resp - model(stim, frametime, *fit_params)))) / roi_resp.shape[0]


def l2_cost_func(fit_params, model, stim, frametime, roi_resp):
    """MSE cost function for minimization"""

    return (np.sum((roi_resp - model(stim, frametime, *fit_params))**2)) / (2*roi_resp.shape[0])


def l1_reg_func(fit_params, reg_coef=0):
    """Regularization term for Lasso regularization"""

    return reg_coef*np.sum(np.abs(fit_params))


def l2_reg_func(fit_params, reg_coef=0):
    """Regularization term for Ridge regularization"""

    return reg_coef*np.sum((fit_params)**2)


def minimization_func(fit_params, model, cost_func, reg_func, stim, frametime, roi_resp, reg_coef):
    """Full function to minimize, including cost and regularization terms"""

    return cost_func(fit_params, model, stim, frametime, roi_resp) + reg_func(fit_params, reg_coef)


def fit_rois_reps(stim, traces, roi_list, frametime, model, params0, lower_bounds, upper_bounds, cost_func, reg_func,
                  goodness_func, reg_coef=0, minimization_method='SLSQP'):
    """Function to perform response fitting.
    Uses half of the repetitions for training the model,
    and the other half for testing it"""

    # Create list with upper and lower boundaries for all parameters
    bounds = []
    for i in range(len(lower_bounds)):
        bounds.append((lower_bounds[i], upper_bounds[i]))

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

        # Perform fitting for all training traces, concatenated one after the other
        train_rep_resps = np.concatenate([trace_reps[:, rep] for rep in train_reps])

        res = optimize.minimize(minimization_func, method=minimization_method, args=(model, cost_func, reg_func,
                                                                                     np.tile(stim, train_reps_num),
                                                                                     frametime, train_rep_resps,
                                                                                     reg_coef),
                                    x0=params0, bounds=bounds)

        model_fitting[roi]['params'] = res.x

        # Calculate goodness of fit for test repetitions concatenated one after the other
        test_rep_resps = np.concatenate([trace_reps[:, rep] for rep in test_reps])

        goodness_fit = goodness_func(model_fitting[roi]['params'], model, np.tile(stim, roi_reps.shape[0] - train_reps_num),
                                     frametime, test_rep_resps)

        model_fitting[roi]['fit_goodness'] = goodness_fit

    return (model_fitting)


@jit(nopython=True)
def set_initial_guesses(guess_n, lower_bounds, upper_bounds):
    """Set initial guesses values randomly-sampled from the parametric space"""
    initial_guesses = np.empty((guess_n, len(lower_bounds)))

    for j in range(guess_n):
        rand_guess = np.empty((len(lower_bounds)))
        for i in range(len(rand_guess)):
            rand_guess[i] = np.random.uniform(lower_bounds[i], upper_bounds[i])
        initial_guesses[j] = rand_guess

    return initial_guesses


@jit(nopython=True)
def get_unit_outputs(stim, frametime, u1l1, u1l2, u1l3, u1nl1, u1nl2, u2l1, u2l2, u2l3, u2nl1, u2nl2, u3l1, u3l2, u3l3,
                     u3nl1, u3nl2, u3nl3):
    """Get outputs from input units"""
    ##Get shifted stimulation arrays
    stim_1 = roll_arr(stim, 1)
    stim_2 = roll_arr(stim, 2)

    ##Calculate number of frames required to encompass an interval of a certain duration
    t_interval = 5
    frames_interval = int(np.floor(t_interval / frametime))

    ##Calculate stim(t)-mean.stim(t[-5sec] - t)
    stim_history = np.empty(stim.shape)

    ##Unit 1
    u1_l_filter = u1l1 * stim_1 + u1l2 * stim_2 + u1l3 * stim_history  # Linear filter for unit 1
    u1_output = sigmoid(u1_l_filter, u1nl1, u1nl2)  # Apply nonlinear tranformation (sigmoidal) to get unit1 output

    ##Unit 2
    u2_l_filter = u2l1 * stim_1 + u2l2 * stim_2 + u2l3 * stim_history  # Linear filter for unit 2
    u2_output = sigmoid(u2_l_filter, u2nl1, u2nl2)  # Apply nonlinear tranformation (sigmoidal) to get unit2 output

    return (u3l1 * u1_output, u3l2 * u2_output)


def plot_fit(stim, traces, stim_t, frametime, roi_list, model_fitting):
    for roi in roi_list:
        # Get ROI resp (normalized)
        roi_resp = np.nanmean(traces[roi, :, :], 1)

        # Estimate ROI response based on model fit
        GC_resp = GC_model(stim, frametime, *model_fitting[roi]['params'])

        ###Plot fitting###
        u1_output, u2_output = get_unit_outputs(stim, frametime, *model_fitting[roi]['params'])

        f = plt.figure(figsize=(7, 5))
        gs = gridspec.GridSpec(21, 15)
        gs.update(wspace=0.25, hspace=0)
        ax1 = plt.subplot(gs[:, :10])
        ax2 = plt.subplot(gs[1:10, 10:])
        ax3 = plt.subplot(gs[12:, 10:], sharex=ax2, sharey=ax2)

        ax1.plot(stim_t, roi_resp, label='ROI resp')
        ax1.plot(stim_t, GC_resp, ls='--', label='fit')
        ax1.legend()
        shade_plot((stim_t, stim), ax=ax1, shade_range=(0.5, 1))
        ax1.set_xlim(min(stim_t), max(stim_t))

        ax2.plot(u1_output, c='red', label='Unit1')
        ax2.plot(stim, alpha=0.25, c='black')
        ax2.set_title('Input 1 (w={:.2f})'.format(model_fitting[roi]['params'][10]), fontsize=9)
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3.plot(u2_output, c='green', label='Unit1')
        ax3.plot(stim, alpha=0.25, c='black')
        ax3.set_title('Input 2 (w={:.2f})'.format(model_fitting[roi]['params'][11]), fontsize=9)
        ax3.set_xticks([])
        ax3.set_yticks([])

        f.suptitle('ROI_{}_fitting'.format(roi))