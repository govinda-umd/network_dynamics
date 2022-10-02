from math import floor
import os 
import sys
from os.path import join as pjoin
import numpy as np
import scipy as sp
from scipy import signal
import pandas as pd
from tqdm.notebook import tqdm
import pickle, random
import csv
from copy import deepcopy

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 14
plt.rcParams["errorbar.capsize"] = 0.5

import cmasher as cmr #CITE ITS PAPER IN YOUR MANUSCRIPT

# ROI MASKS
# ---------

def get_max_rois(args):
    args.rois_df = pd.read_csv(
        f"/home/govindas/parcellations/MAX_85_ROI_masks/README_MAX_ROIs_final_gm_85.txt",
        delimiter='\t'
    )
    args.rois_df = args.rois_df.sort_values(by=['Schaefer_network'])
    args.roi_idxs = args.rois_df.index.values
    args.roi_names = (args.rois_df['Hemi'] + ' ' + args.rois_df['ROI']).values
    args.nw_names = (args.rois_df['Schaefer_network']).values
    args.num_rois = len(args.roi_names)

    return args

def get_mashid_rois(args):
    args.rois_df = pd.read_csv(
        f"{args.proj_dir}/data/rois/mashid/roi_set_mashid.csv",
        delimiter=','
    )
    # args.rois_df = args.rois_df.sort_values(by=['network'])
    args.roi_idxs = args.rois_df.index.values
    args.roi_names = (args.rois_df['roi_name']).values
    args.nw_names = (args.rois_df['network']).values
    args.num_rois = len(args.roi_names)

    return args

def get_mashid_plot_tick_labels(args):
    '''
    plotting tick labels
    '''
    ticks = []
    for nw in np.unique(args.nw_names):
        ticks.append(np.where(args.nw_names == nw)[0].shape[0])
    args.ticks = np.array(ticks)[[1, 0, 3, 2]]
    print(args.ticks)

    minor_ticks = np.cumsum(args.ticks)
    args.major_ticks = minor_ticks - args.ticks // 2
    args.minor_ticks = minor_ticks[:-1]
    print(args.minor_ticks)
    print(args.major_ticks)

    args.major_tick_labels = np.unique(args.nw_names)[[1, 0, 3, 2]]
    print(args.major_tick_labels)

    return args

# COMMON
# ------

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

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

# FC
# --

def get_fcs(args, ts):
    fcs = {}; summary_fcs = {}
    for block in ts.keys():
        fcs[block] = []
        for idx_subj in range(ts[block].shape[0]):
            fcs[block].append(np.corrcoef(ts[block][idx_subj].T))
        fcs[block] = np.stack(fcs[block], axis=0)
        
        summary_fcs[block] = np.nanmedian(fcs[block], axis=0)
    return fcs, summary_fcs

# MAX DATASET
# -----------

def get_max_response_data(args, response_file):
    response_cols = open(response_file, 'r').readlines()[7].split('"')[1].split(';')
    none_col_idxs = [
        idx 
        for idx, col in enumerate(response_cols)
        if 'none' in col
    ]
    response_txt = np.loadtxt(response_file)[:, none_col_idxs]
    return response_txt

def get_max_design_matrix(args, design_mat_file):
    raw_cols = open(design_mat_file, 'r').readlines()[3].split('"')[1].split(';')
    raw_cols = [raw_col.strip() for raw_col in raw_cols]

    design_mat_df = pd.DataFrame(
        data=np.loadtxt(design_mat_file),
        columns=raw_cols
    )
    # print(design_mat_df.shape[1], len(raw_cols))

    # keep only stimulus columns
    raw_cols = [
        col 
        for col in raw_cols
        if 'Run' not in col 
        if 'Motion' not in col
    ]
    design_mat_df = design_mat_df[raw_cols]
    # print(design_mat_df.shape[1], len(raw_cols))

    # check if a TR is censored, i.e. that col. will be all 0's.
    used_cols = []
    for col, col_data in design_mat_df.iteritems():
        if col_data.sum() == 0: continue
        used_cols.append(col)
    # print(len(used_cols))

    return design_mat_df, raw_cols, used_cols

def get_max_cond_ts(args, response_data, label_name='FNS#'):
    cols = [
        col 
        for col in response_data 
        if label_name in col 
        if 'r_' not in col
    ]
    cond_ts = np.stack(
        np.split(
            response_data[cols].to_numpy().astype(np.float32).T, 
            len(cols) // args.TRIAL_LEN, 
            axis=0
        ),
        axis=0
    )
    return cond_ts

def get_max_trial_level_responses(args, ):
    X = {}
    for name in args.NAMES:
        X[name] = []

    for subj in tqdm(args.explor_subjects):

        data_dir = f"{args.main_data_dir}/{subj}"

        # response file
        response_file = f"{data_dir}/{subj}_Main_block_Deconv_bucket.1D"
        response_txt = get_max_response_data(args, response_file)
        # print(response_txt.shape)

        # design matrix
        design_mat_file = f"{data_dir}/{subj}_Main_block_Deconv.x1D"
        design_mat_df, raw_cols, used_cols = get_max_design_matrix(args, design_mat_file)

        # columns in the response text file should 
        # correspond to the used columns in design matrix
        assert (response_txt.shape[1] == len(used_cols))

        # organize responses with their TRs.
        response_data = pd.DataFrame(columns=raw_cols)
        response_data[used_cols] = response_txt

        # segregate responses for each condition and trial
        for label_name, name in zip(args.LABEL_NAMES, args.NAMES):
            X[name].append(get_max_cond_ts(args, response_data, label_name))

    return X #raw_cols, used_cols, response_txt, response_data

def get_max_block_time_series(args, X, trial_option='concat'):
    def get_min_max_trials(ts):
        num_trials = []
        for block in ts.keys():
            for x in ts[block]:
                num_trials.append(x.shape[0])
        return np.min(num_trials), np.max(num_trials)

    def get_usable_trials(ts_block):
        # nan trials
        nan_trials = []
        for idx_subj in range(len(ts_block)):
            x = ts_block[idx_subj]
            nan_trials += list(np.where(
                np.squeeze(
                    np.apply_over_axes(
                        np.sum, 
                        np.isnan(x[:, :, 0]),
                        axes=[1]
                    )
                ) # list trials having nans
            )[0])
        nan_trials = set(nan_trials)
        
        # usable trials
        min_trials, max_trials = get_min_max_trials(ts_block)
        usable_trials = set(range(min_trials))
        usable_trials -= nan_trials
        return list(usable_trials)

    # collect responses for each block
    ts = {}
    for label, name in enumerate(args.NAMES):
        for idx_period, (period, trs) in enumerate(args.PERIOD_TRS.items()):
            block = f"{name}_{period}"
            ts[block] = []
            
            for x in X[name]:
                ts[block].append(
                    x[:, trs, :]
                )
    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    min_trials, _ = get_min_max_trials(ts)
    for block in ts.keys():
        for idx_subj in range(len(ts[block])):
            x = ts[block][idx_subj]
            x = np.nan_to_num(x)[:min_trials, :, :]

            if trial_option == 'concat':
                ts[block][idx_subj] = np.concatenate(x, axis=0)
            elif trial_option == 'mean':
                ts[block][idx_subj] = np.nanmean(x, axis=1)
        
        ts[block] = np.stack(ts[block], axis=0)

    '''
    # remove trials with nans, 
    # use same trials for all subjs
    for block in ts.keys():
        usable_trials = get_usable_trials(ts[block])
        for idx_subj in range(len(ts[block])):
            ts[block][idx_subj] = np.concatenate(
                ts[block][idx_subj][usable_trials, :, :],
                axis=0
            )
        print(block, usable_trials)
        ts[block] = np.stack(ts[block], axis=0)
    '''
    return ts

def plot_max_responses(args, X):
    num_rois = X[list(X.keys())[0]][0].shape[-1]
    nrows, ncols = int(np.ceil(num_rois / 5)), 5
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

    X_blocks = {}
    stats_blocks = {}
    for label, name in enumerate(X.keys()):
        X_blocks[name] = np.concatenate(X[name], axis=0)
        stats_blocks[name] = {}
        stats_blocks[name]['mean'] = np.nanmean(X_blocks[name], axis=0)
        stats_blocks[name]['ste'] = 1.96 * np.nanstd(X_blocks[name], axis=0) / np.sqrt(X_blocks[name].shape[0])

    for idx_roi, roi in enumerate(args.roi_names):
        ax = axs[idx_roi // ncols, idx_roi % ncols]

        for label, name in enumerate(stats_blocks.keys()):
            m = stats_blocks[name]['mean'][:, idx_roi]
            s = stats_blocks[name]['ste'][:, idx_roi]
            ax.plot(
                m, 
                color=args.plot_colors[name], 
                linewidth=3,
                label=name
            )

            ax.fill_between(
                np.arange(args.TRIAL_LEN),
                (m + s),
                (m - s),
                color=args.plot_colors[name],
                alpha=0.3,
            )

            for period in args.PERIOD_TRS.keys():
                ax.fill_betweenx(
                    np.array([np.max(m+s), np.min(m-s)]),
                    x1=args.PERIOD_TRS[period][0],
                    x2=args.PERIOD_TRS[period][-1],
                    color='lightgrey',
                    alpha=0.3,
                )

            ax.set_title(f"{roi}")
            if idx_roi % ncols == 0: ax.set_ylabel('responses')
            ax.set_xlabel('TRs')
            ax.legend()
            ax.grid(True)

def plot_max_fcs(args, fcs, cmap=cmr.iceburn): 
    vmin, vmax = get_min_max(fcs)

    nrows, ncols = len(args.NAMES), len(args.PERIOD_TRS)
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
        wspace=0.65, hspace=0.65
    )

    for label, name in enumerate(args.NAMES[::-1]):
        for idx_period, period in enumerate(list(args.PERIOD_TRS.keys())[::-1]):
            ax = axs[label, idx_period]
            block = f"{name}_{period}"

            # if block == 'safe_early': continue

            im = ax.imshow(
                fcs[block], #* rois[block], 
                cmap=cmap, 
                # vmin=vmin, vmax=vmax
                vmin=0.0, vmax=vmax
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

# ABA DATASET
# -----------

def get_aba_response_data(args, response_file):
    response_cols = open(response_file, 'r').readlines()[7].split('"')[1].split(';')
    none_col_idxs = [
        idx 
        for idx, col in enumerate(response_cols)
        if 'none' in col
    ]
    response_txt = np.loadtxt(response_file)[:, none_col_idxs]
    return response_txt

def get_aba_design_mat(args, design_mat_file):
    raw_cols = open(design_mat_file, 'r').readlines()[3].split('"')[1].split(';')
    raw_cols = [raw_col.strip() for raw_col in raw_cols]

    design_mat_df = pd.DataFrame(
        data=np.loadtxt(design_mat_file),
        columns=raw_cols
    )

    # keep only stimulus columns
    raw_cols = [
        col 
        for col in raw_cols
        if 'Run' not in col
        if 'Motion' not in col
    ]

    design_mat_df = design_mat_df[raw_cols]
    # all columns are usable. Songtao did not include 
    # trials with censored TRs in the first level analysis.
    used_cols = deepcopy(raw_cols)
    return design_mat_df, used_cols

def get_aba_cond_ts(args, response_data, label_name='highT'):
    cols = [
        col
        for col in response_data
        if f"PLAY_{label_name}" in col
        if 'FEED' not in col
    ]
    cond_ts = np.stack(
        np.split(
            response_data[cols].to_numpy().astype(np.float32).T,
            len(cols) // args.TRIAL_LEN,
            axis=0
        ),
        axis=0
    )
    return cond_ts

def get_aba_trial_level_responses(args, ):
    X = {}
    for name in args.NAMES:
        X[name] = []

    for subj in tqdm(args.explor_subjects):

        data_dir = f"{args.main_data_dir}/{subj}"

        # response file
        response_file = f"{data_dir}/{subj}_bucket_REML.1D"
        if not os.path.exists(response_file): continue
        response_txt = get_aba_response_data(args, response_file)

        # design matrix
        design_mat_file = f"{data_dir}/{subj}_Main_block_deconv.x1D"
        if not os.path.exists(design_mat_file): continue
        design_mat_df, used_cols = get_aba_design_mat(args, design_mat_file)

        # columns in the response text file should 
        # correspond to the used columns in design matrix
        assert (response_txt.shape[1] == len(used_cols))

        # organize responses with their TRs.
        response_data = pd.DataFrame(
            data=response_txt,
            columns=used_cols
        )

        # segregate responses for each condition and trial
        for label_name, name in zip(args.LABEL_NAMES, args.NAMES):
            X[name].append(get_aba_cond_ts(args, response_data, label_name))

    return X

def get_aba_block_time_series(args, X, trial_option='concat'):
    def get_min_max_trials(ts):
        num_trials = []
        for block in ts.keys():
            for x in ts[block]:
                num_trials.append(x.shape[0])
        return np.min(num_trials), np.max(num_trials)
    
    ts = {}
    for name in args.NAMES:
        ts[name] = []
        for x in X[name]:
            ts[name].append(
                x[:, args.PERIOD_TRS['late'], :]
            )
    min_trials, _ = get_min_max_trials(ts)
    for block in ts.keys():
        for idx_subj in range(len(ts[block])):
            x = ts[block][idx_subj]
            x = x[:min_trials, :, :]

            if trial_option == 'concat':
                ts[block][idx_subj] = np.concatenate(x, axis=0)
            elif trial_option == 'mean':
                ts[block][idx_subj] = np.nanmean(x, axis=0)
        
        ts[block] = np.stack(ts[block], axis=0) # (subj, time, roi)

    return ts

def plot_aba_responses(args, X):
    num_rois = X[list(X.keys())[0]][0].shape[-1]
    nrows, ncols = int(np.ceil(num_rois / 5)), 5
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

    X_blocks = {}
    stats_blocks = {}
    for label, name in enumerate(X.keys()):
        X_blocks[name] = np.concatenate(X[name], axis=0)
        stats_blocks[name] = {}
        stats_blocks[name]['mean'] = np.nanmean(X_blocks[name], axis=0)
        stats_blocks[name]['ste'] = 1.96 * np.nanstd(X_blocks[name], axis=0) / np.sqrt(X_blocks[name].shape[0])

    for idx_roi, roi in enumerate(args.roi_names):
        ax = axs[idx_roi // ncols, idx_roi % ncols]

        for label, name in enumerate(stats_blocks.keys()):
            m = stats_blocks[name]['mean'][:, idx_roi]
            s = stats_blocks[name]['ste'][:, idx_roi]
            ax.plot(
                m, 
                color=args.plot_colors[name], 
                linewidth=3,
                # label=name
            )

            ax.fill_between(
                np.arange(args.TRIAL_LEN),
                (m + s),
                (m - s),
                color=args.plot_colors[name],
                alpha=0.3,
            )

            for period in args.PERIOD_TRS.keys():
                ax.fill_betweenx(
                    np.array([np.max(m+s), np.min(m-s)]),
                    x1=args.PERIOD_TRS[period][0],
                    x2=args.PERIOD_TRS[period][-1],
                    color='lightgrey',
                    alpha=0.3,
                )

            ax.set_title(f"{roi}")
            if idx_roi % ncols == 0: ax.set_ylabel('responses')
            ax.set_xlabel('TRs')
            # ax.legend()
            ax.grid(True)

def plot_aba_fcs(args, fcs, cmap=cmr.iceburn): 
    vmin, vmax = get_min_max(fcs)

    nrows, ncols = len(args.NAMES) // 2, 2
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
        wspace=0.65, hspace=0.65
    )

    for idx_block, block in enumerate(fcs.keys()):
        ax = axs[idx_block // 2, idx_block % 2]

        # if block == 'safe_early': continue

        im = ax.imshow(
            fcs[block], #* rois[block], 
            cmap=cmap, 
            # vmin=vmin, vmax=vmax
            vmin=0.0, vmax=vmax
        )
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if idx_block // 2 == 0: ax.set_title(f"{block.split('_')[0]}")
        if idx_block % 2 == 0: ax.set_ylabel(f"{block.split('_')[1]}", size='large')
        
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

# EMOPROX2 DATASET
# ----------------

def get_emo2_ts_single_subj(args, subj, motion_df, X):
    subj_motion_df = motion_df.loc[motion_df.pid.isin([subj])]
    proximity = np.hstack(
        subj_motion_df['proximity'].to_list()
    ).T
    direction = np.hstack(
        subj_motion_df['direction'].to_list()
    ).T
    ts = np.loadtxt(
        f"{args.main_data_dir}/CON{subj}/CON{subj}_resids_REML.1D"
    ).T

    assert (proximity.shape[0] == ts.shape[0])
    assert (direction.shape[0] == ts.shape[0])
    assert (np.sum(np.isnan(ts)) == 0)

    # censor proximity values 
    censor_TRs = ts[:,0] == 0
    proximity[censor_TRs] = 0.0

    # near misses
    peaks, props = signal.find_peaks(
        proximity, 
        height=args.near_miss_thresh, 
        width=args.near_miss_width
    )

    near_miss_windows = np.round(
        np.stack(
            [props['left_ips'], props['right_ips']], 
            axis=-1
            )
    ).astype(int)

    near_misses = np.zeros_like(proximity)
    for idx in range(near_miss_windows.shape[0]):
        near_misses[near_miss_windows[idx, 0] : near_miss_windows[idx, 1]+1] = 1.0
        # +1 because we need the last TR also

    # appr and retr segments
    for idx_label, (label, name) in enumerate(zip(args.LABELS, args.LABEL_NAMES)):
        trials = contiguous_regions((direction == label) * (near_misses))
        ts_list = [
            ts[trial[0]+args.hemo_lag: trial[1]+args.hemo_lag, :]
            for trial in trials
        ]
        # contiguous_regions takes care of the last element, so no need to add 1 in the indices.
        X[name].append(ts_list)

    return X

def get_emo2_trial_level_responses(args, motion_df):
    X = {}
    for label, name in enumerate(args.NAMES):
        X[name] = []

    for subj in tqdm(args.explor_subjects):
        X = get_emo2_ts_single_subj(args, subj, motion_df, X)

    for name in X.keys():
        # find max TRs
        max_TRs = []
        for x in X[name]:
            max_TRs += [len(x)]
        args.MIN_TRS[name] = int(floor(np.nanquantile(max_TRs, q=0.05)))
        args.TRIAL_LEN[name] = int(round(np.nanquantile(max_TRs, q=0.95))) #np.max(max_TRs)

        # stack trials 
        for idx_x, xs in enumerate(X[name]):
            x = np.zeros((len(xs), args.TRIAL_LEN[name], args.num_rois))
            for trl in range(len(xs)):
                t, _ = xs[trl].shape
                if name == 'APPR':
                    # x[trl, -t:, :] = xs[trl]
                    x[trl, :t, :] = xs[trl]
                elif name == 'RETR':
                    x[trl, :t, :] = xs[trl]
            
            X[name][idx_x] = x

    return X

def get_emo2_block_time_series(args, X):
    ts = {}
    for name in args.NAMES:
        ts[name] = []
        for x in X[name]:
            ts[name].append(
                np.nanmean(x, axis=0)
            )
        ts[name] = np.stack(ts[name], axis=0)
    return ts

def plot_emo2_responses(args, X):
    nrows, ncols = int(np.ceil(args.num_rois / 5)), 5
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

    X_blocks = {}
    stats_blocks = {}
    for label, name in enumerate(X.keys()):
        X_blocks[name] = np.concatenate(X[name], axis=0)
        stats_blocks[name] = {}
        stats_blocks[name]['mean'] = np.nanmean(X_blocks[name], axis=0)
        stats_blocks[name]['ste'] = (
            1.96 * np.nanstd(X_blocks[name], axis=0) / np.sqrt(X_blocks[name].shape[0])
        )

    for idx_roi, roi in enumerate(args.roi_names):
        ax = axs[idx_roi // ncols, idx_roi % ncols]

        for label, name in enumerate(stats_blocks.keys()):
            m = stats_blocks[name]['mean'][:, idx_roi]
            s = stats_blocks[name]['ste'][:, idx_roi]
            ax.plot(
                m, 
                color=args.plot_colors[name], 
                linewidth=3,
                label=name
            )

            ax.fill_between(
                np.arange(args.TRIAL_LEN[name]),
                (m + s),
                (m - s),
                color=args.plot_colors[name],
                alpha=0.3,
            )

            ax.set_title(f"{roi}")
            if idx_roi % ncols == 0: ax.set_ylabel('responses')
            ax.set_xlabel('TRs')
            ax.legend()
            ax.grid(True)

def plot_emo2_fcs(args, fcs, cmap=cmr.iceburn): 
    vmin, vmax = get_min_max(fcs)

    nrows, ncols = 1, len(args.NAMES)
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
        wspace=0.65, hspace=0.65
    )

    for idx_block, block in enumerate(fcs.keys()):
        ax = axs[idx_block]

        # if block == 'safe_early': continue

        im = ax.imshow(
            fcs[block], #* rois[block], 
            cmap=cmap, 
            # vmin=vmin, vmax=vmax
            vmin=0.0, vmax=vmax
        )
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(f"{block}")
        
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