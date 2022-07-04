#!/usr/bin/env python
# coding: utf-8

# # June 24, 2022: create exploratory dataset from the MAX dataset
# randomly sample 25\% of subjects and run all analysis on them. once we explore and settle on some analyses, we can report the results on the other (bigger) dataset enruring generalizability of our findings.

# In[1]:


import os
import sys
from os.path import join as pjoin

import numpy as np
import pandas as pd
import pickle, random
from tqdm import tqdm
from scipy.stats import zscore

# main dirs
proj_dir = pjoin(os.environ['HOME'], 'network_dynamics')
results_dir = f"{proj_dir}/results"
month_dir = f"{proj_dir}/nb/jun22"

# folders
sys.path.insert(0, proj_dir)
import helpers.dataset_utils as dataset_utils
 


# In[2]:


'''
dataframe
'''
max_data_path = f"/home/govindas/explainable-ai/data/max/data_df.pkl"
with open(max_data_path, 'rb') as f:
    max_data_df = pickle.load(f)

'''
exploratory data
'''
class ARGS(): pass
args = ARGS()

args.SEED = 74
args.LABELS = [0, 1]
args.names = ['safe', 'threat']
args.MASK = -100

num_rois = 85
args.roi_idxs = np.arange(num_rois)

np.random.seed(args.SEED)

args.num_subjects = len(max_data_df)
args.num_explor = round(0.25 * args.num_subjects)

subject_idx_list = np.arange(args.num_subjects)
random.Random(args.SEED).shuffle(subject_idx_list)

explor_list = subject_idx_list[:args.num_explor]
X = dataset_utils.get_max_data_trials(args, max_data_df, explor_list)

with open(f"{proj_dir}/data/max/exploratory_data.pkl", 'wb') as f:
    pickle.dump(X, f)

