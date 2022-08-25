#!/usr/bin/env python
# coding: utf-8

# # August 10, 2022: generate trial level responses of MAX data for ABA rois

# In[1]:


import os 
import sys
from os.path import join as pjoin
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import pickle, random

from nltools import mask
from nilearn import image, masking

# plotting
import matplotlib.pyplot as plt
plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 14
plt.rcParams["errorbar.capsize"] = 0.5

import cmasher as cmr #CITE ITS PAPER IN YOUR MANUSCRIPT

# main dirs
proj_dir = pjoin(os.environ['HOME'], 'network_dynamics')
results_dir = f"{proj_dir}/results"
month_dir = f"{proj_dir}/nb/aug22"

# folders
sys.path.insert(0, proj_dir)
import helpers.dataset_utils as dataset_utils


# ## MAX rois

# In[2]:


max_roi_name_file = (
    f"{os.environ['HOME']}/parcellations/MAX_85_ROI_masks/ROI_names.txt"
)
max_roi_names = pd.read_csv(max_roi_name_file, names=['roi_name']).values.squeeze()
max_roi_names = list(max_roi_names)
max_roi_names


# In[3]:


class ARGS(): pass
args = ARGS()

with open(f"{proj_dir}/data/max/exploratory_data_roi_indices.pkl", 'rb') as f:
    args.up_roi_idxs, args.down_roi_idxs, args.roi_idxs = pickle.load(f)
    args.num_rois = len(args.roi_idxs)


# ## ABA rois

# In[4]:


set_name = 'aba'
aba_roi_name_file = (
    f"/home/govindas/vscode-BSWIFT-mnt/ABA/ROI_mask/ABA_36ROIs_gm.txt"
)
aba_roi_names =  pd.read_csv(
    aba_roi_name_file,
    delimiter='\t'
)[['sHemi', 'ROI']]
aba_roi_names = (aba_roi_names['sHemi'] + ' ' + aba_roi_names['ROI']).values
aba_roi_names = list(aba_roi_names)
aba_roi_names


# 34 out of 36 ABA rois are taken from the MAX rois, and two rois are added: Pulvinar and VTA-SNc.

# ## ABA to MAX

# In[5]:


'''
map aba onto max
'''
# for 33 rois
aba_to_max_idxs = {
    aba_idx:max_roi_names.index(aba_roi) 
    for aba_idx, aba_roi in enumerate(aba_roi_names) 
    if aba_roi in max_roi_names
}
# for the two additional rois
for aba_roi_idx in range(30, 34):
    aba_to_max_idxs[aba_roi_idx] = None
# for vmPFC
aba_to_max_idxs[34] = 4
aba_to_max_idxs[35] = 5

display(aba_to_max_idxs)

'''
map max to aba
'''
max_to_aba_idxs = {
    v:k for k, v in aba_to_max_idxs.items()
}
display(max_to_aba_idxs)


# In[6]:


for max_roi in args.up_roi_idxs:
    if max_roi not in max_to_aba_idxs.keys(): continue
    print(max_to_aba_idxs[max_roi])


# In[7]:


'''
group aba rois separately from max and save the grouping
'''
args.up_roi_idxs = np.array(
    [2, 3] + # ACC
    list(range(4, 8)) + # MCC
    [12, 13] + # IFG-5
    list(range(14, 20)) + # insula
    [20, 21] + # BST
    [30, 31] # Pulvinar
)

args.down_roi_idxs = np.array(
    [8, 9] + # PCC
    [26, 27] + # ant. Hippocampus
    [34, 35] # vmPFC
)

args.zero_roi_idxs = np.array(
    [0, 1] + # Hypothalamus
    [10, 11] + # ventral striatum
    list(range(22, 26)) + # amygdala
    [28, 29] + # PAG
    [32, 33] # VTA-SNc
)

args.roi_idxs = np.concatenate([args.up_roi_idxs, args.zero_roi_idxs, args.down_roi_idxs])


# In[9]:


len(args.roi_idxs)


# In[8]:


with open(f"{proj_dir}/data/max/exploratory_data_aba_roi_indices.pkl", 'wb') as f:
    pickle.dump([args.up_roi_idxs, args.down_roi_idxs, args.zero_roi_idxs, args.roi_idxs], f)

