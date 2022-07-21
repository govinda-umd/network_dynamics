import numpy as np
import pandas as pd
from tqdm import tqdm

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

def get_max_data_trials(args, data_df, subj_idx_list):
    X = {
        0: [], #safe
        1: [], #threat
    } # (label, subj, trial, time, roi)
    for idx_row in tqdm(subj_idx_list):
        subj, ts, targets = data_df.iloc[idx_row]

        for label in args.LABELS:
            x = []
            for region in contiguous_regions(targets == label):
                x.append(ts[region[0]: region[1], args.roi_idxs])
            x = np.stack(x, axis=0)
            X[label].append(x)
    
    return X

# ----------------------------------

def get_columns_design_matrix(design_mat_path):
    raw_cols = open(design_mat_path, 'r').readlines()[3].split('"')[1].split(';')
    raw_cols = [raw_col.strip() for raw_col in raw_cols]

    design_mat = np.loadtxt(design_mat_path)
    design_mat = pd.DataFrame(design_mat, columns=raw_cols)

    raw_cols = [
        raw_col 
        for raw_col in raw_cols 
        if "Run" not in raw_col 
        if "Motion" not in raw_col
    ]
    design_mat = design_mat[raw_cols]

    used_cols = []
    for col, col_data in design_mat.iteritems():
        if col_data.sum() == 0: continue
        used_cols.append(col)
    
    # display(design_mat[used_cols])

    return raw_cols, used_cols


def get_cond_ts(args, response_data, label='FNS#'):
    cols = [
        col 
        for col in response_data 
        if label in col 
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


def get_max_trial_level_responses(args, main_data_dir, subjs):
    X = {}
    for label in args.LABELS:
        X[label] = []

    for subj in tqdm(subjs):

        data_dir = f"{main_data_dir}/{subj}"

        design_mat_path = f"{data_dir}/{subj}_Main_block_Deconv.x1D"
        response_file_path = f"{data_dir}/{subj}_Main_block_Deconv_bucket.1D"

        raw_cols, used_cols = get_columns_design_matrix(design_mat_path)
        response_data = pd.DataFrame(columns=raw_cols)
        response_data[used_cols] = np.loadtxt(response_file_path)[:, 1::2]

        for label, name in zip(args.LABELS, args.LABEL_NAMES):
            X[label].append(
                get_cond_ts(args, response_data, name).astype(np.float32)
            )
    
    return X

# ----------------------------------

def sim_data_additive_white_noise(args, X, y):
    '''
    white-noise 
    -----------
    1. std's for each roi and each tp
    2. simulated data with i.i.d. normal noise (white noise) around mean time series
    '''
    X_, y_ = [], []
    for label in args.LABELS:
        idx = y[:, 0] == label
        X_ += [np.random.normal(
            loc=np.mean(X[idx], axis=0), 
            scale=args.noise_level*np.std(X[idx], axis=0), 
            size=X[idx].shape
        )]
        y_ += [np.ones(shape=(X[idx].shape[:-1])) * label]

    X_ = np.concatenate(X_, axis=0)
    y_ = np.concatenate(y_, axis=0)

    perm = np.random.permutation(y_.shape[0])
    X_ = X_[perm]
    y_ = y_[perm]

    return X_, y_