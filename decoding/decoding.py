from numba import jit
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from collections import namedtuple
from sklearn.linear_model import LogisticRegression


@jit(nopython=True)
def test_train_sets(population, n_test):
    n_cells, n_t, n_reps = population.shape
    n_train = n_reps - n_test
    train_set = np.empty((n_cells, n_train*n_t))
    test_set = np.empty((n_cells, n_test*n_t))
    for i_cell in range(n_cells):
        ids_test = np.sort(np.random.choice(n_reps, n_test, replace=False))
        i_test = 0
        i_train = 0
        for i_rep in range(n_reps):
            if i_rep == ids_test[i_test]:
                test_set[i_cell, i_test*n_t:(i_test+1)*n_t] = population[i_cell, :, i_rep]
                i_test += 1
            else:
                train_set[i_cell, i_train*n_t:(i_train+1)*n_t] = population[i_cell, :, i_rep]
                i_train += 1
    return train_set, test_set


TrainData = namedtuple("TrainData", "train_set test_train_set cross_vals train_gt")


def get_population(full_dict, cell_type="GC", protocol="flashes", n_rep=12, max_rois_incl='all'):
    traces = full_dict["{}_{}".format(cell_type, protocol)]["clean_traces"]
    n_valid_traces = np.sum(np.logical_not(np.all(np.isnan(traces), axis=1)), axis=1)
    sel_cells = np.where(n_valid_traces >= n_rep)[0]
    sel_traces = traces[sel_cells, :, :n_rep]
    
    if max_rois_incl=='all':
        return sel_traces
    else:
        return sel_traces[np.random.choice(np.arange(sel_traces.shape[0]), max_rois_incl), :, :]
    
def prepare_population(population, stim, n_test):
    n_pop, n_t_decode, n_rep = population.shape
    n_train = n_rep - n_test
    train_set, test_set = test_train_sets(population, n_test)
    cross_vals = [(np.concatenate(
        [np.arange(i_tt * n_t_decode, (i_tt + 1) * n_t_decode) for i_tt in range(n_train) if i_tt != i_t]),
                   np.arange(i_t * n_t_decode, (i_t + 1) * n_t_decode)) for i_t in range(n_train)]
    train_gt = np.tile(stim, n_train)
    return TrainData(train_set, test_set, cross_vals, train_gt)

def decode_from_population(population, stim, n_test=2, model=None, hyperparams=None, 
                           probabilities=False):
    if hyperparams is None:
        hyperparams = dict(alpha=10.0 ** np.arange(-3, 5))
    if model is None:
        model = Ridge()
    train_set, test_set, cross_vals, train_gt = prepare_population(population, stim, n_test)
    sm = GridSearchCV(model, hyperparams, cv=cross_vals)
    sm.fit(train_set.T, train_gt)
    if not probabilities:
        return sm, sm.predict(test_set.T)
    else:
        return sm, sm.predict_proba(test_set.T)
    
    
def confmat(pred_bins, pred_gt):
    """ Calculate the confusion matrix by averaging the probaility
    distribution of the decoded category for each appearence of that
    category (indexed by pred_gt)
    """
    pred_gt = pred_gt.astype(np.int32)
    n_bins = pred_bins.shape[1]
    confmat = np.zeros((n_bins, n_bins))
    n_avg = np.zeros(n_bins, dtype=np.uint16)
    for i in range(len(pred_gt)):
        confmat[pred_gt[i], :] += pred_bins[i, :]
        n_avg[pred_gt[i]] += 1
    return confmat/n_avg[:, None]
