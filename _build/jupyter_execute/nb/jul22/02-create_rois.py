#!/usr/bin/env python
# coding: utf-8

# # July 28, August 1,3, 2022: create new set of rois from center coordinates

# The previous code did not get saved, so doing again on August 3, 2022.

# In[1]:


import os 
import sys
from os.path import join as pjoin
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from nltools import mask
from nilearn import image

# main dirs
proj_dir = pjoin(os.environ['HOME'], 'network_dynamics')
results_dir = f"{proj_dir}/results"
month_dir = f"{proj_dir}/nb/jul22"


# In[2]:


set_name = 'mashid'
mask_dir = f"{proj_dir}/data/rois/{set_name}/individual_nifti_files"
roi_set_file = f"{proj_dir}/data/rois/{set_name}/roi_set_{set_name}.csv"

roi_set_df = pd.read_csv(roi_set_file)
def str_to_list(s): 
    a = s.strip('][').split(', ')
    if len(a) == 3:
        return list(map(int, a))
    else:
        return []
roi_set_df['coordinates'] = roi_set_df['coordinates'].apply(str_to_list)
roi_set_df


# In[3]:


radius = 5 # mm
final_mask_file = f"{mask_dir}/../final_mask.nii.gz"
if not os.path.exists(final_mask_file):
    for idx, row in roi_set_df.iterrows():
        if len(row['coordinates']) != 3: continue

        prefix = f"{mask_dir}/{row['roi_name']}.nii.gz"
        if idx == 0:
            final_mask = mask.create_sphere(row['coordinates'], radius=radius)
        else:
            roi_mask = mask.create_sphere(row['coordinates'], radius=radius)
            final_mask = image.math_img(
                f'img1 + {idx+1}*img2',
                img1=final_mask,
                img2=roi_mask
            )
    final_mask.to_filename(final_mask_file)
    

