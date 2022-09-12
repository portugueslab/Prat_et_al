import numpy as np
from scipy import optimize
#from pyprind import prog_percent
import sys

import numba
from numba import jit

from luminance_analysis.plotting import *


@jit(nopython=True)
def resp_from_contributions(contribution_coefs, gc_resp, io_resp):
    pc_resp = contribution_coefs[0]*gc_resp + contribution_coefs[1]*io_resp
    return pc_resp


def l1_cost_func(contribution_coefs, pc_resp, gc_resp, io_resp):
    """MAE cost function for minimization"""
    return (np.sum(np.abs(pc_resp - resp_from_contributions(contribution_coefs, gc_resp, io_resp)))) / pc_resp.shape[0]


def l2_cost_func(contribution_coefs, pc_resp, gc_resp, io_resp, power=2):
    """MSE cost function for minimization"""
    return (np.sum((pc_resp - resp_from_contributions(contribution_coefs, gc_resp, io_resp))**power)) / pc_resp.shape[0]


def l1_reg_func(contribution_coefs, reg_coef=0):
    """Regularization term for Lasso regularization"""
    return reg_coef*np.sum(np.abs(contribution_coefs))


def l2_reg_func(contribution_coefs, reg_coef=0):
    """Regularization term for Ridge regularization"""
    return reg_coef*np.sum(contribution_coefs**2)


def minimization_func(contribution_coefs, cost_func, reg_func, pc_resp, gc_resp, io_resp, reg_coef):
    """Full function to minimize, including cost and regularization terms"""
    return cost_func(contribution_coefs, pc_resp, gc_resp, io_resp) + reg_func(contribution_coefs, reg_coef)


def fit_contributions(resp_mat_fit, resp_mat_test, cluster_resps, cluster_combos, cost_func, reg_func, error_func, x0, bounds, reg_coef=0,
                      minimization_method='SLSQP'):

    if resp_mat_fit.shape[0] != resp_mat_test.shape[0]:
        sys.exit('Fitting and testing arrays have different shapes')

    #Create arrays where fitting results will be stored
    res_mat = np.empty((resp_mat_fit.shape[0], len(cluster_combos), 2))
    err_mat = np.empty((resp_mat_test.shape[0], len(cluster_combos)))

    for roi in prog_percent(range(resp_mat_fit.shape[0])):
        #Create arrays where fitting results for single ROIs will be stored
        res_arr = []
        err_arr = []
        pc_resp_fit = resp_mat_fit[roi, :]
        pc_resp_test = resp_mat_test[roi, :]

        #Iterate through all GC&IO cluster combinations
        for combo in cluster_combos:
            gc_resp = cluster_resps['GC'][[combo[0]], :]
            io_resp = cluster_resps['IO'][[combo[1]], :]

            res = optimize.minimize(minimization_func, method=minimization_method, args=(cost_func, reg_func,
                                                                                         pc_resp_fit, gc_resp, io_resp,
                                                                                         reg_coef),
                                    x0=x0, bounds=bounds)
            error = error_func(res.x, pc_resp_test, gc_resp, io_resp)

            res_arr.append(res.x)
            err_arr.append(error)

        res_mat[roi, :, :] = res_arr
        err_mat[roi, :] = err_arr

    return res_mat, err_mat


# Code for fitting multiple (3) clusters of any type
@jit(nopython=True)
def resp_from_contributions_m(contribution_coefs, resp_0, resp_1, resp_2):
    pc_resp = contribution_coefs[0]*resp_0 + contribution_coefs[1]*resp_1 + contribution_coefs[2]*resp_2
    return pc_resp


def l1_cost_func_m(contribution_coefs, pc_resp, resp_0, resp_1, resp_2):
    """MAE cost function for minimization"""
    return (np.sum(np.abs(pc_resp - resp_from_contributions_m(contribution_coefs, resp_0, resp_1, resp_2)))) / pc_resp.shape[0]


def l2_cost_func_m(contribution_coefs, pc_resp, resp_0, resp_1, resp_2, power=2):
    """MSE cost function for minimization"""
    return (np.sum((pc_resp - resp_from_contributions_m(contribution_coefs, resp_0, resp_1, resp_2))**power)) / pc_resp.shape[0]

def minimization_func_m(contribution_coefs, cost_func, reg_func, pc_resp, resp_0, resp_1, resp_2, reg_coef):
    """Full function to minimize, including cost and regularization terms"""
    return cost_func(contribution_coefs, pc_resp, resp_0, resp_1, resp_2) + reg_func(contribution_coefs, reg_coef)

def fit_contributions_m(resp_mat_fit, resp_mat_test, cluster_resps, cluster_combos, cost_func, reg_func, error_func, x0, bounds, reg_coef=0,
                      minimization_method='SLSQP'):

    if resp_mat_fit.shape[0] != resp_mat_test.shape[0]:
        sys.exit('Fitting and testing arrays have different shapes')

    #Create arrays where fitting results will be stored
    res_mat = np.empty((resp_mat_fit.shape[0], len(cluster_combos), len(cluster_combos[0])))
    err_mat = np.empty((resp_mat_test.shape[0], len(cluster_combos)))

    for roi in prog_percent(range(resp_mat_fit.shape[0])):
        #Create arrays where fitting results for single ROIs will be stored
        res_arr = []
        err_arr = []
        pc_resp_fit = resp_mat_fit[roi, :]
        pc_resp_test = resp_mat_test[roi, :]

        #Iterate through all GC&IO cluster combinations
        for combo in cluster_combos:
            resp_0 = cluster_resps[combo[0], :]
            resp_1 = cluster_resps[combo[1], :]
            resp_2 = cluster_resps[combo[2], :]

            res = optimize.minimize(minimization_func_m, method=minimization_method, args=(cost_func, reg_func,
                                                                                         pc_resp_fit, resp_0, resp_1, resp_2,
                                                                                         reg_coef),
                                    x0=x0, bounds=bounds)
            error = error_func(res.x, pc_resp_test, resp_0, resp_1, resp_2)

            res_arr.append(res.x)
            err_arr.append(error)

        res_mat[roi, :, :] = res_arr
        err_mat[roi, :] = err_arr

    return res_mat, err_mat