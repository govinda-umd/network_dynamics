#!/usr/bin/env python
# coding: utf-8

# # August 17, 2022: organize MAX ROIs into the 7 networks defined in Schaefer parcellation

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


# ## Schaefer and MAX parcellations

# In[2]:


class ARGS(): pass
args = ARGS()

args.n_parcels = 1000
schaefer_main_path = f"{proj_dir}/data/schaefer_parcellations/n_parcels_{args.n_parcels}"


# In[3]:


'''
network order information
of Schaefer ROIs
'''
nw_order = pd.read_csv(
    f"{schaefer_main_path}/Schaefer2018_1000Parcels_7Networks_order_info.txt",
    header=None,
).iloc[0::2]
nw_order = nw_order.reset_index(drop=True)
nw_order = nw_order[0].apply(lambda s: s.split('_')[2]).to_numpy()
nw_order


# In[4]:


'''
schaefer parcellation
'''
schaefer_parcel = image.load_img(
    f"{schaefer_main_path}/Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.nii"
).get_fdata()

schaefer_parcel.shape


# In[5]:


'''
MAX parcellation
'''
max_parcel = image.load_img(
    f"/home/govindas/parcellations/MAX_85_ROI_masks/MAX_ROIs_final_gm_85.nii.gz"
).get_fdata()

max_parcel.shape


# ## associate MAX rois to the 7 networks + subcortical network

# In[6]:


'''
associate MAX rois to the 7 networks + subcortical network
'''
def get_most_freq_nw(nws):
    if len(nws) == 0: return 'Subcort'
    nws, cts = np.unique(nws, return_counts=True)
    return nws[np.argmax(cts)]

max_rois = np.unique(max_parcel)[1:]
max_order = np.empty(shape=(max_rois.shape[0]), dtype=object)
for idx, roi in enumerate(max_rois):
    roi_mask = (max_parcel == roi)
    schaefer_region = schaefer_parcel * roi_mask
    nws = nw_order[np.unique(schaefer_region)[1:].astype(int) - 1]
    print(
        roi, nws, get_most_freq_nw(nws)
    )
    max_order[idx] = get_most_freq_nw(nws) 


# In[7]:


'''
save in the MAX README file
'''
max_readme_file = f"/home/govindas/parcellations/MAX_85_ROI_masks/README_MAX_ROIs_final_gm_85.txt"
max_readme_df = pd.read_csv(
    max_readme_file,
    sep='\t',
)
max_readme_df['Schaefer_network'] = max_order
display(max_readme_df)

max_readme_df.to_csv(
    max_readme_file,
    sep='\t',
    index=False
)


# In[8]:


# max_readme_df = max_readme_df.drop(['Unnamed: 0'], axis=1)
# max_readme_df.to_csv(
#     max_readme_file,
#     sep='\t',
#     index=False
# )


# ## ROI ordering: group according to networks

# In[10]:


'''
network grouping: sort ROIs
'''
roi_ordering = np.argsort(max_order)
nw_names = np.unique(np.sort(max_order))
display(nw_names)

roi_ordering

