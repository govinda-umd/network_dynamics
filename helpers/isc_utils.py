import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import (norm, zscore, permutation_test)
from itertools import combinations

# ISC
from brainiak.isc import (
    isc, isfc, bootstrap_isc, compute_summary_statistic, squareform_isfc, compute_correlation,
    _check_timeseries_input, _check_targets_input, _threshold_nans
)
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import squareform

# plotting
import matplotlib.pyplot as plt
plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 14
plt.rcParams["errorbar.capsize"] = 0.5

import cmasher as cmr #CITE ITS PAPER IN YOUR MANUSCRIPT

# DATA TIME SERIES
# -----------------
def get_block_time_series(args, X, ):
    # step 1. sort trials based on number of nan values present
    for label in args.LABELS:
        for idx in np.arange(len(X[label])):
            x = X[label][idx]
            num_nans_trial = np.squeeze(
                np.apply_over_axes(
                    np.sum, 
                    np.isnan(x), 
                    axes=(1, 2)
                )
            )
            num_nans_trial_idxs = np.argsort(num_nans_trial)
            x = x[num_nans_trial_idxs, :, :]
            X[label][idx] = x

    # step 2. create time series
    '''
    find minimum number of trials across subjects
    '''
    min_trials = []
    for label in args.LABELS:
        min_trials += [x.shape[0] for x in X[label]]
    min_trials = min(min_trials)
    print(f"minimum number of trials = {min_trials}")

    '''
    time series of early and late periods
    '''
    args.PERIODS = ['early', 'late']
    # time periods
    args.TR = 1.25 #seconds
    EARLY = np.arange(2.5, 8.75+args.TR, args.TR) // args.TR
    LATE = np.arange(10.0, 16.25+args.TR, args.TR) // args.TR
    EARLY = EARLY.astype(int)
    LATE = LATE.astype(int)
    args.PERIOD_TRS = [EARLY, LATE]

    ts = {}
    for label, name in zip(args.LABELS, args.NAMES):
        for idx_period, (period, TRs) in enumerate(zip(args.PERIODS, args.PERIOD_TRS)):
            ts[f"{name}_{period}"] = []
            for x in X[label]:
                x = x[:, args.PERIOD_TRS[idx_period], :]
                trl, t, r = x.shape
                x = np.reshape(x[:min_trials, ...], (min_trials*t, r))
                ts[f"{name}_{period}"] += [zscore(x, axis=0, nan_policy='omit')]

    for block in ts.keys():
        ts[block] = np.dstack(ts[block])
    
    return ts

# ISC/ISFC MATRICES
# -----------------
def get_isfcs(args, ts):
    corrs = {}; bootstraps = {}; rois = {}

    for block in ts.keys():
        # if block == 'safe_early': continue
        isfcs, iscs = isfc(
            ts[block], 
            pairwise=args.pairwise, 
            summary_statistic=None,
            vectorize_isfcs=args.vectorize_isfcs
        )
        corrs[block] = {'isfcs':isfcs, 'iscs':iscs}

        bootstraps[block] = {}
        rois[block] = {}
        for corr_name in args.CORR_NAMES:
            
            # permutation test
            observed, ci, p, distribution = bootstrap_isc(
                corrs[block][corr_name], 
                pairwise=args.pairwise, 
                summary_statistic='median',
                n_bootstraps=args.n_bootstraps,
                ci_percentile=95, 
                side='two-sided',
                random_state=args.SEED
            )
            bootstraps[block][corr_name] = observed, ci, p, distribution

            # surviving rois
            # rois[block][corr_name] = q[block][corr_name] < 0.05
            rois[block][corr_name] = bootstraps[block][corr_name][2] < 0.05 # p < 0.05

            # correlations only for surviving rois
            # corrs[block][corr_name] *= rois[block][corr_name]

            print(
                (
                    f"condition {block} and correlation {corr_name[:-1]} : " 
                    f"{100. * np.sum(rois[block][corr_name]) / len(rois[block][corr_name])} %"
                    f"significant roi(-pairs)"
                )
            )

    return corrs, bootstraps, rois

def get_squareform_matrices(args, bootstraps, rois):
    observed_isfcs = {}; observed_p_vals = {}; 
    significant_rois = {}; conf_intervals = {}
    for block in bootstraps.keys():
        # if block == 'safe_early': continue
        observed_isfcs[block] = squareform_isfc(
            bootstraps[block]['isfcs'][0], 
            bootstraps[block]['iscs'][0]
        )
        observed_p_vals[block] = squareform_isfc(
            bootstraps[block]['isfcs'][2], 
            bootstraps[block]['iscs'][2]
        )
        significant_rois[block] = squareform_isfc(
            rois[block]['isfcs'], 
            rois[block]['iscs']
        )
        conf_intervals[block] = (
            squareform_isfc(
                bootstraps[block]['isfcs'][1][0],
                bootstraps[block]['iscs'][1][0]
            ),
            squareform_isfc(
                bootstraps[block]['isfcs'][1][1],
                bootstraps[block]['iscs'][1][1]
            )
        )

        observed_isfcs[block] *= significant_rois[block]
    
    return observed_isfcs, observed_p_vals, significant_rois, conf_intervals

def get_min_max(d):
    '''
    min and max values of the matrices:
    used in plotting the matrices
    '''
    vals = []
    for block in d.keys():
        vals.append(d[block])
    vals = np.concatenate(vals, axis=0).flatten()
    vmin = np.nanquantile(vals, q=0.05)
    vmax = np.nanquantile(vals, q=0.95)
    return -max(-vmin, vmax), max(-vmin, vmax)

def plot_isfcs(args, isfcs, rois): 
    vmin, vmax = get_min_max(isfcs)

    nrows, ncols = len(args.LABELS), len(args.PERIODS)
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(5*ncols, 4*nrows), 
        sharex=False, 
        sharey=False, 
        dpi=120
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=None, hspace=0.5
    )

    for label, name in zip(args.LABELS, args.NAMES):
        for idx_period, period in enumerate(args.PERIODS):
            ax = axs[label, idx_period]
            block = f"{name}_{period}"

            # if block == 'safe_early': continue

            im = ax.imshow(
                isfcs[block], #* rois[block], 
                cmap=cmr.iceburn, vmin=vmin, vmax=vmax
            )
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if label == 0: ax.set_title(f"{period}")
            if idx_period == 0: ax.set_ylabel(f"{name}", size='large')
            
            ax.set_yticks(args.major_ticks, args.major_tick_labels, rotation=0, va='center')
            ax.set_xticks(args.major_ticks, args.major_tick_labels, rotation=90, ha='center')

            ax.set_yticks(args.minor_ticks-0.5, minor=True)
            ax.set_xticks(args.minor_ticks-0.5, minor=True)
            ax.tick_params(
                which='major', direction='out', length=5.5, 
                # grid_color='white', grid_linewidth='1.5',
                labelsize=10,
            )
            ax.grid(which='minor', color='w', linestyle='-', linewidth=1.5)

    return None

# COMPARISONS BETWEEN ISC MATRICES
# --------------------------
def get_comparison_stats(args, corrs,):
    def statistic(obs1, obs2, axis):
        return (
            + compute_summary_statistic(obs1, summary_statistic='median', axis=axis) 
            - compute_summary_statistic(obs2, summary_statistic='median', axis=axis)
        ) 

    stats_results = {}
    for (block1, block2) in tqdm(combinations(corrs.keys(), 2)):
        obs1 = np.concatenate([corrs[block1]['isfcs'], corrs[block1]['iscs']], axis=-1)
        obs2 = np.concatenate([corrs[block2]['isfcs'], corrs[block2]['iscs']], axis=-1)
        
        stats_results[(block1, block2)] = permutation_test(
            data=(obs1, obs2),
            statistic=statistic,
            permutation_type='samples',
            vectorized=True,
            n_resamples=args.n_bootstraps,
            batch=5,
            alternative='two-sided',
            axis=0,
            random_state=args.SEED
        )
    
    return stats_results

def get_diff_isfcs(args, stats_results, significant_rois):
    diff_isfcs = {}; diff_pvals = {}
    for (block1, block2) in stats_results.keys():

        diff_isfcs[(block1, block2)] = squareform_isfc(
            isfcs=stats_results[(block1, block2)].statistic[:-args.num_rois],
            iscs=stats_results[(block1, block2)].statistic[-args.num_rois:]
        )

        diff_pvals[(block1, block2)] = squareform_isfc(
            isfcs=stats_results[(block1, block2)].pvalue[:-args.num_rois],
            iscs=stats_results[(block1, block2)].pvalue[-args.num_rois:],
        )
        diff_pvals[(block1, block2)] = diff_pvals[(block1, block2)] < 0.05
        # FDR correction
        # coming soon...

        # keep isfc values only for significant roi pairs
        diff_isfcs[(block1, block2)] *= diff_pvals[(block1, block2)]
        diff_isfcs[(block1, block2)] *= significant_rois[block1]
        diff_isfcs[(block1, block2)] *= significant_rois[block2]

    return diff_isfcs, diff_pvals

def plot_isfc_comparisons(args, corrs, diff_isfcs,):
    vmin, vmax = get_min_max(diff_isfcs)

    nrows, ncols = [len(corrs.keys())]*2
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(4*ncols, 4*nrows), 
        sharex=False, 
        sharey=False, 
        dpi=120
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=0.45, hspace=None
    )

    for idx_blk1, block1 in enumerate(corrs.keys()):
        for idx_blk2, block2 in enumerate(corrs.keys()):

            if idx_blk1 <= idx_blk2: 
                axs[idx_blk1, idx_blk2].remove()
                continue

            ax = axs[idx_blk1, idx_blk2]

            im = ax.imshow(
                diff_isfcs[(block2, block1)], 
                cmap=cmr.iceburn, vmin=vmin, vmax=vmax
            )
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if idx_blk1 == idx_blk2+1: ax.set_title(f"{block2}")
            if idx_blk2 == 0: ax.set_ylabel(f"{block1}", size='large')

            ax.set_yticks(args.major_ticks, args.major_tick_labels, rotation=0, va='center')
            ax.set_xticks(args.major_ticks, args.major_tick_labels, rotation=90, ha='center')

            ax.set_yticks(args.minor_ticks-0.5, minor=True)
            ax.set_xticks(args.minor_ticks-0.5, minor=True)
            ax.tick_params(
                which='major', direction='out', length=7, 
                # grid_color='white', grid_linewidth='1.5',
                labelsize=10,
            )
            ax.grid(which='minor', color='w', linestyle='-', linewidth=1.5)