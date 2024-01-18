from skimage import color
import numpy as np
from matplotlib import pyplot as plt
from luminance_analysis import Data
from luminance_analysis.utilities import find_transitions, nanzscore
from scipy.cluster.hierarchy import dendrogram, cut_tree, set_link_color_palette
from luminance_analysis.clustering import find_trunc_dendro_clusters
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from luminance_analysis.pc_contributions import resp_from_contributions
from luminance_analysis.utilities import crop_intervals_from_mat
from scipy import stats


from matplotlib import cm


def shade_plot(stim, ax=None, gamma=1/6, shade_range=(0.6, 0.9)):
    if type(stim) == list:  # these would be transitions
        _shade_plot(stim, ax=ax, gamma=gamma, shade_range=shade_range)

    elif type(stim) == Data:  # fish data
        transitions = find_transitions(Data.resampled_stim, Data.time_im_rep)
        _shade_plot(transitions, ax=ax, gamma=gamma, shade_range=shade_range)

    elif type(stim) == np.ndarray:  # stimulus array
        transitions = find_transitions(stim[:,1], stim[:,0])
        _shade_plot(transitions, ax=ax, gamma=gamma, shade_range=shade_range)

    elif type(stim) == tuple:  # time, lum tuple
        transitions = find_transitions(stim[1], stim[0])
        _shade_plot(transitions, ax=ax, gamma=gamma, shade_range=shade_range)


def _shade_plot(lum_transitions, ax=None, gamma=1/6, shade_range=(0.6, 0.9)):
    """ Plot stimulus as shades of gray in the background.
    :param lum_transitions:
    :param ax:
    :param shade_range:
    :return:
    """
    if ax is None:
        ax = plt.gca()
    shade = lum_transitions[0][1]
    for i in range(len(lum_transitions)-1):
        shade = shade + lum_transitions[i][1]
        new_shade = shade_range[0] + np.power(np.abs(shade), gamma) * (shade_range[1] - shade_range[0])
        ax.axvspan(lum_transitions[i][0], lum_transitions[i+1][0], color=(new_shade, )*3)


def add_offset_axes(figure, ax_d, frame=None, **kwargs):
    """ Little function to add to a figure a set of axes scaled to a
    specified frame.
    """
    if frame is None:
        ax = figure.add_axes(ax_d, **kwargs)
    else:
        ax = figure.add_axes((ax_d[0] * frame[2] + frame[0],
                              ax_d[1] * frame[3] + frame[1],
                              ax_d[2] * frame[2], ax_d[3] * frame[3]), **kwargs)
    return ax


def get_yg_custom_cmap(n=100):
    """Create a matplotlib color map for the traces customizing
    a seaborn color map.
    """
    cols = sns.diverging_palette(47, 150, l=45, s=75, n=9, sep=1,
                                 center="light")
    return LinearSegmentedColormap.from_list("clusters",
                                             [cols[0], cols[1], cols[2],
                                              cols[-3], cols[-2], cols[-1]],
                                              N=n)


def make_bar(axis, bounds, label=None, orientation="horizontal", lw=1):
    """ Replace axis in a plot with a bar
    :param axis:
    :param bounds:
    :param label:
    :param orientation:
    :param remove_ticks:
    :return:
    """
    if label is None:
        label = abs(bounds[0] - bounds[1])
    if orientation == "horizontal":
        axis.spines['bottom'].set_bounds(bounds[0], bounds[1])
        axis.spines['bottom'].set_linewidth(lw)
        axis.set_xticks([np.mean(bounds)])
        axis.set_xticklabels([label])
        axis.tick_params("x", length=0)
    elif orientation == "vertical":
        axis.spines['left'].set_bounds(bounds[0], bounds[1])
        axis.spines['left'].set_linewidth(lw)
        axis.set_yticks([np.mean(bounds)])
        axis.set_yticklabels([label], rotation='vertical', va='center')
        axis.tick_params("y", length=0)
    # if remove_ticks:


def _find_thr(linked, n_clust):
    interval = [0, 2000]
    new_height = np.mean(interval)
    clust = 0
    n_clust = n_clust
    while clust != n_clust:
        new_height = np.mean(interval)
        clust = cut_tree(linked, height=new_height).max()
        if clust > n_clust:
            interval[0] = new_height
        elif clust < n_clust:
            interval[1] = new_height


    return new_height


def find_plot_thr(linked, n_clust):
    min_thr = _find_thr(linked, n_clust - 1)
    # max_thr = _find_thr(linked, n_clust - 1)

    return min_thr  #np.mean((min_thr, max_thr))


def cluster_cols():
    return ["#bf3f76", "#577b34", "#9d6620", "#c54238", "#925b84", "#546dae",
             "#976a61", "#397b74", "#5981a3"]*3


def plot_clusters_dendro(traces, stim, linkage_mat, labels, prefix="",
                         figure=None, w=1, h=1, w_p=0, h_p=0,
                         dendrolims=(900, 30), colorbar=True,
                         thr=None, f_lim=1.5, gamma=1, spacing=3):

    if figure is None:
        figure = plt.figure(figsize=(7.5, 4.5))
    hexac = cluster_cols()

    n_clust = labels.max() + 1
    hexac = hexac[:n_clust]

    ##################
    ### Dendrogram ###
    # Compute and plot first dendrogram.
    if thr is None:
        thr = find_plot_thr(linkage_mat, n_clust)
    ax_dendro = figure.add_axes([0.1*w + w_p, 0.2*h + h_p, 0.1*w, 0.5*h])

    set_link_color_palette(hexac[::-1])
    panel_dendro = dendrogram(linkage_mat,
                              color_threshold=thr,
                              orientation='left',
                              distance_sort='descending',
                              show_leaf_counts=False,
                              no_labels=True,
                              above_threshold_color='#%02x%02x%02x' % (
                              120, 120, 120))
    ax_dendro.axvline(thr, linewidth=0.7, color="k")
    # ax_dendro.set_xscale('log')
    ax_dendro.axis("off")
    ax_dendro.set_xlim(dendrolims)

    ##################
    ### Traces mat ###
    # Plot traces matrix.
    axmatrix = figure.add_axes([0.2*w + w_p, 0.2*h + h_p, 0.3*w, 0.5*h])
    im = axmatrix.imshow(traces[panel_dendro["leaves"], :],
                         aspect='auto', origin='lower', cmap=cm.RdBu_r,
                         vmin=-f_lim, vmax=f_lim, interpolation='none')
    axmatrix.axes.spines['left'].set_visible(False)
    axmatrix.set_yticks([])

    # Time bar:
    dt = stim[1, 0]
    barlength = 10
    bounds = np.array([traces.shape[1] - barlength / dt,
                       traces.shape[1]])
    make_bar(axmatrix, bounds, label="{} s".format(barlength))

    # Colorbar:
    if colorbar:
        axcolor = figure.add_axes([0.2*w + w_p, 0.17*h + h_p, 0.05*w,
                                      0.015*h])
        cbar = plt.colorbar(im, cax=axcolor, orientation="horizontal")
        cbar.set_ticks([-f_lim, f_lim])
        cbar.set_label("dF/F")
        cbar.ax.tick_params(length=3)


    ##################
    # Cluster sizes ##
    # Calculate size of each defined cluster to put colored labels on the side.
    # Find intervals spanned by each cluster in the sorted traces matrix.
    # Add percentages spanned by each cluster.
    sizes = np.cumsum(np.array([np.sum(labels == i) for i in range(np.max(labels) + 1)[::-1]]))
    print(sizes)
    intervals = np.insert(sizes, 0, 0)

    axlabelcols = figure.add_axes([0.501*w + w_p, 0.2*h + h_p, 0.005*w,
                                      0.5*h])
    ticks = []
    ticks_labels = []
    for i in range(len(intervals) - 1):
        axlabelcols.axhspan(intervals[i + 1], intervals[i],
                            color=hexac[::-1][i])  # - sign to reverse order
        ticks.append(np.mean((intervals[i + 1], intervals[i])))
        size = 100 * (intervals[i + 1] - intervals[i]) / intervals[-1]
        ticks_labels.append("{}{} ({}%)".format(prefix, n_clust-i, int(round(size))))
    axlabelcols.set_ylim(0, intervals[-1])
    [axlabelcols.axes.spines[s].set_visible(False) for s in
     ["left", "right", "top", "bottom"]]
    axlabelcols.set_xticks([])
    axlabelcols.set_yticks(ticks)
    axlabelcols.set_yticklabels(ticks_labels)
    axlabelcols.yaxis.tick_right()

    for ytick, color in zip(axlabelcols.get_yticklabels(), hexac[::-1]):
        ytick.set_color(color)
    axlabelcols.tick_params(length=0)

    ##################
    # Cluster means ##
    axtraces = figure.add_axes([0.6*w + w_p, 0.2*h + h_p, 0.3*w, 0.5*h])

    for i in range(n_clust):
        axtraces.plot(stim[:, 0],
                      np.nanmean(traces[labels == i, :], 0) -
                      i * spacing, label=i, color=hexac[i])
    axtraces.axes.spines['left'].set_visible(False)
    shade_plot(stim, shade_range=(0.7, 0.90))
    axtraces.set_yticks([])
    axtraces.set_xlim(stim[0, 0], stim[-1, 0])

    barlength = 10
    make_bar(axtraces, [stim[-1, 0] - barlength, stim[-1, 0]],
             label="{} s".format(barlength))

    ##################
    ### Luminance ####
    for xpos in [0.2, 0.6]:
        ax_lum = figure.add_axes([xpos*w + w_p, 0.7*h + h_p, 0.3*w, 0.05*h])
        ax_lum.plot(stim[:, 0], np.power(stim[:, 1], gamma), color="k")
        ax_lum.set_xlim(stim[0, 0], stim[-1, 0])
        ax_lum.axis("off")

    return figure


from numba import jit

@jit(nopython=True)
def color_stack(stack, colors):
    """ Function to color a stack of ROIs.
    :param stack: Stack with ROIs. Value for each ROI should correspond to index
     on the first dimension of the colors matrix of the color to be assigned to
     that ROI.
    :param colors: list of colors.
    :return:
    """
    out_stack = np.zeros(stack.shape + (3,), dtype=np.float32)

    for i in range(stack.shape[0]):
        for j in range(stack.shape[1]):
            for k in range(stack.shape[2]):
                if stack[i, j, k] != 0:
                    out_stack[i, j, k, :] = colors[stack[i, j, k], :3]

    return out_stack


def re_histogram(rel_idxs, rel_thr, figure=None, w=1, h=1, w_p=0, h_p=0,
                 color=None):
    if color is None:
        color = sns.color_palette()[0]

    if figure is None:
        figure = plt.figure(figsize=(2.5, 1.5))

    ax = figure.add_axes([0.3*w + w_p, 0.3*h + h_p, 0.7*w, 0.7*h])
    a = ax.hist(rel_idxs[~np.isnan(rel_idxs)], np.arange(-0.2, 1, 0.02),
                     color=color)

    ax.axvline(rel_thr, color=(0.3,) * 3)
    ax.text(1.1, 0.8*a[0].max(), "{:2.1f}%\n({} ROIs)".format(
        100 * sum(rel_idxs > rel_thr) / len(rel_idxs), sum(rel_idxs > rel_thr)),
             fontsize=7,  color=(0.3,)*3, fontdict=dict(name="Arial"), ha="right")
    ax.set_ylabel("Count")
    ax.set_xlabel("Correlation")
    plt.tight_layout()

    return figure


# def plot_fish_contribution(data_dict, figure=None, frame=None):
#     if figure is None:
#         figure = plt.figure(figsize=(9, 3))
#
#     barWidth = 0.85
#     n_fish = 5
#     colors = (sns.color_palette("deep", 10))
#     fish_contribution = {brain_region: {} for brain_region in brain_regions_list}
#
#     for i, brain_region in enumerate(brain_regions_list):
#         ax_hist = add_offset_axes(figure, (0.1 + 0.32 * i, 0.15, .25, .7), frame=frame)




def stim_plot(stim, xlims, gamma, figure=None, frame=None):
    if figure is None:
        figure = plt.figure(figsize=(3., 1.8))

    shade_ax = add_offset_axes(figure, [0.1, 0.67, 0.6, 0.2], frame=frame)
    plot_ax = add_offset_axes(figure, [0.1, 0.3, 0.6, 0.35], frame=frame)


    stim[0], stim[1] = [s[stim[1] < xlims[1]] for s in stim]
    shade_plot((stim[0], stim[1]), shade_ax, shade_range=(0.7, 0.90))
    shade_ax.set_xlim(*xlims)
    shade_ax.axis("off")

    plot_ax.plot(stim[0], np.power(stim[1], gamma))
    plot_ax.set_xticks(np.arange(0, 101, 25))
    plot_ax.set_xlim(*xlims)
    plot_ax.set_ylabel("Luminance")
    plot_ax.set_xlabel("Time (s)")
    plt.show()

    return (figure)

def TPI_plot(TPI_dict, reliability_dict, brain_area, v_min, v_max, subsample=False):
    #Define colormap
    if brain_area == 'GC_flashes':
        end_col = sns.color_palette()[0] + (1,)
    elif brain_area == 'IO_flashes':
        end_col = sns.color_palette()[1] + (1,)
    elif brain_area == 'PC_flashes':
        end_col = sns.color_palette()[2] + (1,)
    else:
        print('this is not a valid brain region!')

    colors = [[0.9, 0.9, 0.9], end_col]
    rel_cmap = LinearSegmentedColormap.from_list('rel_map', colors, N=100)

    #Plot
    fig_TPI, axes = plt.subplots(1, 3, figsize=(9,4), sharey=True)
    for flash, ax in zip(range(3), axes):
        if subsample is False:
            rois = range(TPI_dict[flash].shape[0])
            mean_tpi = np.nanmean(TPI_dict[flash], 1)
            points = ax.scatter(rois, mean_tpi, c=reliability_dict[flash]['reliability'], cmap=rel_cmap, vmin=v_min, vmax=v_max)
        else:
            rois = subsample
            mean_tpi = np.nanmean(TPI_dict[flash], 1)[subsample]
            points = ax.scatter(rois, mean_tpi, c=reliability_dict[flash]['reliability'][subsample], cmap=rel_cmap,
                                vmin=v_min, vmax=v_max)

        ax.set_xlim(0, TPI_dict[flash].shape[0])
        ax.set_title('Flash {}'.format(flash+1))
        ax.set_xlabel('ROI (sorted by C.O.M. time)')

    #Colorbar
    axcolor = fig_TPI.add_axes([.95, .33, .015, .35])
    cbar = plt.colorbar(points, cax=axcolor, shrink=.5)
    cbar.set_ticks([v_min, v_max])
    cbar.set_label('Reliability coef.', labelpad=-37)
    cbar.ax.tick_params(length=3)

    axes[0].set_ylabel('TPI')

    return(fig_TPI)

def plot_roi_reps(stim, traces, roi, color='blue', rep_alpha=0.25, plot_stim=True, zscored=True):
    roi_traces = np.empty_like(traces[roi, :, :])
    for rep in range(traces[roi, :, :].shape[1]):
        if zscored == True:
            roi_traces[:, rep] = nanzscore(traces[roi, :, rep])
        else:
            roi_traces[:, rep] = traces[roi, :, rep]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title('ROI {}'.format(roi))
    ax.plot(stim[:, 0], roi_traces, alpha=rep_alpha, c = color)
    ax.plot(stim[:, 0], np.nanmean(roi_traces, 1), c = color)
    ax.set_xlim([stim[:, 0].min(), stim[:, 0].max()])
    ax.set_xlabel('Time [s.]')
    ax.get_yaxis().set_visible(False)

    if plot_stim == True:
        shade_plot(stim, ax=ax)
    else:
        pass

    return(fig)


def plot_contributions_fit(roi, resps_mat, stim, best_fits_coefs, best_fits_clusters, best_fits_errors, clust_resps):
    gc_resp = clust_resps['GC'][best_fits_clusters[roi][0], :]
    io_resp = clust_resps['IO'][best_fits_clusters[roi][1], :]

    fig_roi_fit = plt.figure()
    gs = GridSpec(6, 12)
    ax1 = fig_roi_fit.add_subplot(gs[:, :8])
    ax2 = fig_roi_fit.add_subplot(gs[:3, 8:])
    ax3 = fig_roi_fit.add_subplot(gs[3:, 8:])

    ax1.plot(stim[:, 0], resps_mat[roi, :], c='green', label='PC resp')
    ax1.plot(stim[:, 0], resp_from_contributions(best_fits_coefs[roi], gc_resp, io_resp),
             c='black', alpha=.7, ls='--', label='prediction')

    ax1.legend()
    ax1.set_xlim(stim[0, 0], stim[-1, 0])
    ax1.set_ylabel('Mean resp')
    ax1.set_xlabel('Time [s.]')
    ax1.set_title('PC prediction (error = {:.4f})'.format(best_fits_errors[roi]))

    ax2.plot(stim[:, 0], gc_resp, c=sns.color_palette()[0])
    ax3.plot(stim[:, 0], io_resp, c=sns.color_palette()[1])

    for ax in [ax1, ax2, ax3]:
        shade_plot(stim, ax=ax)

    for ax in [ax2, ax3]:
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)

    ax2.set_title('GC cluster: {}'.format(best_fits_clusters[roi][0]))
    ax3.set_title('IO cluster: {}'.format(best_fits_clusters[roi][1]))

    ax2.text(.63, .93, r'$\alpha$ = {:.4f}'.format(best_fits_coefs[roi][0]),
             fontsize=7, transform=ax2.transAxes)

    ax3.text(.63, .93, r'$\beta$ = {:.4f}'.format(best_fits_coefs[roi][1]),
             fontsize=7, transform=ax3.transAxes)

    ax3.set_xlabel('Time [s.]')

    fig_roi_fit.suptitle('ROI {} fit'.format(roi))
    fig_roi_fit.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig_roi_fit


def plot_contributions_fit_arbitrarygroup(cluster_n, data_dict, stim, best_fits_coefs,
                                          best_fits_clusters, best_fits_errors):
    resps_separation_idx = data_dict["GC"].shape[0]
    all_fitted_clust = np.concatenate([data_dict[k]for k in ["GC", "IO"]])

    best_fit_resp = []
    fittes_is_gc = []
    for i in range(2):
        best_fit_resp.append(all_fitted_clust[best_fits_clusters[cluster_n][i], :])
        fittes_is_gc.append(best_fits_clusters[i] < resps_separation_idx)

    fig_roi_fit = plt.figure(figsize=(8,3))
    gs = GridSpec(6, 12)
    axs = [fig_roi_fit.add_subplot(g for g in [gs[:, :8], gs[:3, 8:], gs[3:, 8:]])]

    axs[0].plot(stim[:, 0], data_dict["PC"][cluster_n, :], c='green', label='PC resp')
    axs[0].plot(stim[:, 0], resp_from_contributions(best_fits_coefs[cluster_n], *best_fit_resp),
             c='black', alpha=.7, ls='--', label='prediction')

    axs[0].legend()
    axs[0].set_xlim(stim[0, 0], stim[-1, 0])
    axs[0].set_ylabel('Mean resp')
    axs[0].set_xlabel('Time [s.]')
    axs[0].set_title('PC prediction (error = {:.4f})'.format(best_fits_errors[cluster_n]))

    for j, i in enumerate(np.argsort(best_fits_clusters)):
        axs[j+1].plot(stim[:, 0], best_fit_resp[i], c=sns.color_palette()[int(fittes_is_gc[i])])
    # ax3.plot(stim[:, 0], io_resp, c=sns.color_palette()[1])

    for ax in axs:
        shade_plot(stim, ax=ax)

    for ax in axs[1:]:
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)

    axs[1].set_title('GC cluster: {}'.format(best_fits_clusters[cluster_n][0]))
    axs[2].set_title('IO cluster: {}'.format(best_fits_clusters[cluster_n][1]))

    axs[1].text(.63, .93, r'$\alpha$ = {:.4f}'.format(best_fits_coefs[cluster_n][0]),
             fontsize=7, transform=axs[1].transAxes)

    axs[1].text(.63, .93, r'$\beta$ = {:.4f}'.format(best_fits_coefs[cluster_n][1]),
             fontsize=7, transform=axs[2].transAxes)

    axs[2].set_xlabel('Time [s.]')

    fig_roi_fit.suptitle('Cluster {} fit'.format(cluster_n))
    fig_roi_fit.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig_roi_fit


def plot_bout_prob_window(dataset_dict, trigger_dict, trigger, dt, t_window, pre_trigger=0, norm_baseline=None,
                          use_mat='norm_prob_mat'):
    fig = plt.figure()
    bout_prob_window_dict = {}
    for dataset, color in zip(dataset_dict.keys(), [sns.color_palette()[4], sns.color_palette()[3]]):
        interval_probabilities = crop_intervals_from_mat(dataset_dict[dataset][use_mat], trigger_dict[trigger],
                                                         dt, t_window, pre_trigger=pre_trigger)
        bout_prob_window_mat = np.empty((interval_probabilities.shape[2], interval_probabilities.shape[1]))

        if norm_baseline is not None:
            interval_probabilities_norm = np.empty_like(interval_probabilities)
            for fish in range(interval_probabilities.shape[2]):
                for rep in range(interval_probabilities.shape[0]):
                    interval_probabilities_norm[rep, :, fish] = interval_probabilities[rep, :, fish] \
                                                                - np.nanmean(interval_probabilities[rep,
                                                                             :int(norm_baseline/dt), fish])
            interval_probabilities = interval_probabilities_norm

        for fish in range(interval_probabilities.shape[2]):
            fish_bout_prob_window = np.nanmean(interval_probabilities[:, :, fish], 0)
            plt.plot(np.arange(-pre_trigger, t_window, dt), fish_bout_prob_window, c=color,
                     alpha=.1)
            bout_prob_window_mat[fish, :] = fish_bout_prob_window

        plt.plot(np.arange(-pre_trigger, t_window, dt), np.nanmean(bout_prob_window_mat, 0), c=color,
                 linewidth=2,
                 label=dataset)

        bout_prob_window_dict[dataset] = bout_prob_window_mat

    if pre_trigger != 0:
        plt.axvline(0, ls=':', c='red')
    plt.legend()
    plt.xlabel('Time [s.]')
    plt.xlim(-pre_trigger, t_window)
    plt.ylabel('Avrg. bout prob.')
    plt.title('Bout probability after {}'.format(trigger))

    return (fig, bout_prob_window_dict)


def plot_bout_prob_evol(dataset_dict, trigger_dict, trigger, dt, t_window, trigger_substract=None,
                        use_mat='norm_prob_mat'):
    fig = plt.figure()
    mean_bout_probs_dict = {}
    for dataset, color in zip(dataset_dict.keys(), [sns.color_palette()[4], sns.color_palette()[3]]):
        interval_probabilities = crop_intervals_from_mat(dataset_dict[dataset][use_mat], trigger_dict[trigger],
                                                         dt, t_window)
        mean_bout_prob = np.empty((interval_probabilities.shape[2], dataset_dict[dataset][use_mat].shape[0]))

        if trigger_substract is not None:
            subs_interval_probabilities = crop_intervals_from_mat(dataset_dict[dataset][use_mat],
                                                                  trigger_dict[trigger_substract], dt, t_window)

        for fish in range(interval_probabilities.shape[2]):
            mean_interval_boutprob_bystimrep = np.empty(
                (int(interval_probabilities.shape[0] / 5), interval_probabilities.shape[1]))
            if trigger_substract is not None:
                mean_subs_interval_boutprob_bystimrep = np.empty(
                    (int(subs_interval_probabilities.shape[0] / 5), subs_interval_probabilities.shape[1]))
            for stim_rep in range(dataset_dict[dataset][use_mat].shape[0]):
                mean_interval_boutprob_bystimrep[stim_rep, :] = np.nanmean(
                    interval_probabilities[5 * stim_rep:5 * stim_rep + 5, :, fish], 0)
                if trigger_substract is not None:
                    mean_subs_interval_boutprob_bystimrep[stim_rep, :] = np.nanmean(
                        subs_interval_probabilities[5 * stim_rep:5 * stim_rep + 5, :, fish], 0)

            if trigger_substract is None:
                mean_bout_prob[fish, :] = np.nanmean(mean_interval_boutprob_bystimrep, 1)
                plt.plot(mean_bout_prob[fish, :], c=color, alpha=.2)

            else:
                fish_diff = np.nanmean(mean_interval_boutprob_bystimrep, 1) - np.nanmean(
                    mean_subs_interval_boutprob_bystimrep, 1)
                mean_bout_prob[fish, :] = fish_diff
                plt.plot(fish_diff, c=color, alpha=.2)

        plt.plot(np.nanmean(mean_bout_prob, 0), c=color, linewidth=2, label=dataset)

        mean_bout_probs_dict[dataset] = mean_bout_prob

    plt.legend()
    plt.xlabel('Stimulus repetition')
    if trigger_substract is None:
        plt.title('Avrg. bout prob. upon {}'.format(trigger))
        plt.ylabel('Bout probability')
    else:
        plt.title('Avrg. bout prob. difference ({} - {})'.format(trigger, trigger_substract))
        plt.ylabel('Bout probab. difference')


    return (fig, mean_bout_probs_dict)


def plot_bout_prob_evol_boxplot(evol_dict, fused_timepoints=5, p_val=0.05):
    fig = plt.figure()
    ax = plt.gca()

    # Plot boxplot
    for dataset, color in zip(evol_dict.keys(), [sns.color_palette()[4], sns.color_palette()[3]]):
        array = evol_dict[dataset]
        array[np.isnan(array)] = 0

        intervals = np.arange(0, array.shape[1], fused_timepoints)
        intervals_arr = np.empty((array.shape[0] * fused_timepoints, intervals.shape[0]))

        for interval_idx, interval in enumerate(intervals):
            intervals_arr[:, interval_idx] = array[:, interval:interval + fused_timepoints].ravel()

        style_dict = dict(markerfacecolor='none', linestyle='none', markeredgecolor=color)
        if dataset == 'control_v2':
            bp = plt.boxplot(intervals_arr, patch_artist=True, flierprops=style_dict, widths=.25,
                             positions=np.arange(len(intervals)) - .15)
        elif dataset == 'PC_ablated_v2':
            bp = plt.boxplot(intervals_arr, patch_artist=True, flierprops=style_dict, widths=.25,
                             positions=np.arange(len(intervals)) + .15)

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=color)

        for patch in bp['boxes']:
            patch.set(facecolor='None')

    # Perform T-test between groups
    ymin, ymax = ax.get_ylim()
    for interval_idx, interval in enumerate(intervals):
        control_arr = evol_dict['control_v2'][:, interval:interval + fused_timepoints].ravel()
        ablated_arr = evol_dict['PC_ablated_v2'][:, interval:interval + fused_timepoints].ravel()
        t, pval = stats.ttest_ind(control_arr, ablated_arr)

        plt.hlines(ymax, interval_idx - .25, interval_idx + .25)
        if pval < p_val:
            plt.text(interval_idx, ymax, '*', ha='center')
        else:
            plt.text(interval_idx, ymax + .02, 'n.s.', fontsize=7, ha='center')

    # Perform start vs. end T-test
    for dataset in evol_dict.keys():
        start_arr = evol_dict[dataset][:, intervals[0]:intervals[0] + fused_timepoints].ravel()
        end_arr = evol_dict[dataset][:, intervals[-1]:intervals[-1] + fused_timepoints].ravel()
        t, pval = stats.ttest_ind(start_arr, end_arr)

        if dataset == 'control_v2':
            color = sns.color_palette()[4]
            h = ymin
            xmin = intervals[0] - .15
            xmax = intervals.shape[0] - 1 - .15
            plt.hlines(h, xmin, xmax, color=color)
            if pval < p_val:
                plt.text((xmax - xmin) / 2, h, '*', va='top', color=color)
            else:
                plt.text((xmax - xmin) / 2, h, 'ns', va='top', fontsize=7, color=color)

        elif dataset == 'PC_ablated_v2':
            color = sns.color_palette()[3]
            h = ymin - 0.075
            xmin = intervals[0] + .15
            xmax = intervals.shape[0] - 1 + .15
            plt.hlines(h, xmin, xmax, color=color)
            if pval < p_val:
                plt.text((xmax - xmin) / 2, h, '*', va='top', color=color)
            else:
                plt.text((xmax - xmin) / 2, h, 'ns', va='top', fontsize=7, color=color)

    plt.xticks(np.arange(len(intervals)), ['{}-{}'.format(i, i + 5) for i in intervals])
    plt.ylabel('Avg. bout prob.')
    plt.xlabel('Repetitions')

    return (fig)