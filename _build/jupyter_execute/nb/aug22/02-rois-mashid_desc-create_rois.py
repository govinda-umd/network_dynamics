#!/usr/bin/env python
# coding: utf-8

# # July 28, August 1,3,8,15-16, 2022: create new set of rois from center coordinates
# 
# - August 15, 2022: rename Mashid rois according to MAX rois
# - August 16, 2022: add subcortical rois in the roi mask and generate trial level responses

# The previous code did not get saved, so doing again on August 3, 2022.

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


# ## set of rois

# ### roi names file

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


# In[3]:


'''
Salience
'''
roi_set_df.at[0,  'roi_name'] = 'L mid/post Insula'
roi_set_df.at[1, 'roi_name'] = 'R mid/post Insula'
roi_set_df.at[2, 'roi_name'] = 'R post. MCC'
roi_set_df.at[3, 'roi_name'] = 'L TPJ'
roi_set_df.at[4, 'roi_name'] = 'R TPJ'
roi_set_df.at[5, 'roi_name'] = 'L ITC'
roi_set_df.at[6, 'roi_name'] = 'R ITC'
roi_set_df.at[7, 'roi_name'] = 'L Precentral'
roi_set_df.at[8, 'roi_name'] = 'R Precentral'
roi_set_df.at[9, 'roi_name'] = 'L dlPFC'
roi_set_df.at[10, 'roi_name'] = 'R dlPFC'
roi_set_df.at[11, 'roi_name'] = 'L IFG-1'
roi_set_df.at[12, 'roi_name'] = 'R IFG-6'

'''
Executive
'''
roi_set_df.at[13, 'roi_name'] = 'L orbit. front. Insula'
roi_set_df.at[14, 'roi_name'] = 'R MFG-1'
roi_set_df.at[15, 'roi_name'] = 'L MFG-1'
roi_set_df.at[16, 'roi_name'] = 'R MOG'
roi_set_df.at[17, 'roi_name'] = 'L MOG'
roi_set_df.at[18, 'roi_name'] = 'R IFG-6'
roi_set_df.at[19, 'roi_name'] = 'R MFG-2'
roi_set_df.at[20, 'roi_name'] = 'L MFG-2'
roi_set_df.at[21, 'roi_name'] = 'dmPFC'
roi_set_df.at[22, 'roi_name'] = 'R lat. PC'
roi_set_df.at[23, 'roi_name'] = 'L lat. PC'
roi_set_df.at[24, 'roi_name'] = 'R ITG'

'''
Task-negative
'''
roi_set_df.at[25, 'roi_name'] = 'L post. MCC'
roi_set_df.at[26, 'roi_name'] = 'Retro-splenial'
roi_set_df.at[27, 'roi_name'] = 'L Angular Gyrus'
roi_set_df.at[28, 'roi_name'] = 'R Angular Gyrus'
roi_set_df.at[29, 'roi_name'] = 'M vmPFC1'
roi_set_df.at[30, 'roi_name'] = 'L SMG'
roi_set_df.at[31, 'roi_name'] = 'L SFG'
roi_set_df.at[32, 'roi_name'] = 'R SFG'
roi_set_df.at[33, 'roi_name'] = 'L ITG'
roi_set_df.at[34, 'roi_name'] = 'R MTG'
roi_set_df.at[35, 'roi_name'] = 'L para Hippocampus'
roi_set_df.at[36, 'roi_name'] = 'R para Hippocampus'

'''
Subcortical
'''
roi_set_df.at[37, 'roi_name'] = 'L CeMe Amygdala'
roi_set_df.at[38, 'roi_name'] = 'R CeMe Aymgdala'
roi_set_df.at[39, 'roi_name'] = 'L BLBM Amygdala'
roi_set_df.at[40, 'roi_name'] = 'R BLBM Amygdala'
roi_set_df.at[41, 'roi_name'] = 'L PAG'
roi_set_df.at[42, 'roi_name'] = 'R PAG'
roi_set_df.at[43, 'roi_name'] = 'L Habenula'
roi_set_df.at[44, 'roi_name'] = 'R Habenula'
roi_set_df.at[45, 'roi_name'] = 'L BST'
roi_set_df.at[46, 'roi_name'] = 'R BST'


# In[4]:


'''
Salience
'''
roi_set_df.at[0,  'in_MAX'] = True
roi_set_df.at[1, 'in_MAX'] = True
roi_set_df.at[2, 'in_MAX'] = True
roi_set_df.at[3, 'in_MAX'] = False
roi_set_df.at[4, 'in_MAX'] = False
roi_set_df.at[5, 'in_MAX'] = False
roi_set_df.at[6, 'in_MAX'] = False
roi_set_df.at[7, 'in_MAX'] = False
roi_set_df.at[8, 'in_MAX'] = False
roi_set_df.at[9, 'in_MAX'] = True
roi_set_df.at[10, 'in_MAX'] = True
roi_set_df.at[11, 'in_MAX'] = True
roi_set_df.at[12, 'in_MAX'] = True

'''
Executive
'''
roi_set_df.at[13, 'in_MAX'] = False
roi_set_df.at[14, 'in_MAX'] = False
roi_set_df.at[15, 'in_MAX'] = False
roi_set_df.at[16, 'in_MAX'] = False
roi_set_df.at[17, 'in_MAX'] = False
roi_set_df.at[18, 'in_MAX'] = True
roi_set_df.at[19, 'in_MAX'] = False
roi_set_df.at[20, 'in_MAX'] = False
roi_set_df.at[21, 'in_MAX'] = False
roi_set_df.at[22, 'in_MAX'] = False
roi_set_df.at[23, 'in_MAX'] = False
roi_set_df.at[24, 'in_MAX'] = False

'''
Task-negative
'''
roi_set_df.at[25, 'in_MAX'] = True
roi_set_df.at[26, 'in_MAX'] = False
roi_set_df.at[27, 'in_MAX'] = False
roi_set_df.at[28, 'in_MAX'] = False
roi_set_df.at[29, 'in_MAX'] = True
roi_set_df.at[30, 'in_MAX'] = False
roi_set_df.at[31, 'in_MAX'] = False
roi_set_df.at[32, 'in_MAX'] = False
roi_set_df.at[33, 'in_MAX'] = False
roi_set_df.at[34, 'in_MAX'] = False
roi_set_df.at[35, 'in_MAX'] = False
roi_set_df.at[36, 'in_MAX'] = False

'''
Subcortical
'''
roi_set_df.at[37, 'in_MAX'] = True
roi_set_df.at[38, 'in_MAX'] = True
roi_set_df.at[39, 'in_MAX'] = True
roi_set_df.at[40, 'in_MAX'] = True
roi_set_df.at[41, 'in_MAX'] = True
roi_set_df.at[42, 'in_MAX'] = True
roi_set_df.at[43, 'in_MAX'] = False
roi_set_df.at[44, 'in_MAX'] = False
roi_set_df.at[45, 'in_MAX'] = True
roi_set_df.at[46, 'in_MAX'] = True


# In[5]:


roi_set_df.to_csv(roi_set_file, index=False)


# In[6]:


display(roi_set_df)


# ### creating rois mask

# In[10]:


'''
cortical rois
'''
radius = 5 # mm
final_mask_file = f"{mask_dir}/../final_mask_cortical.nii.gz"
cortical_mask_file = final_mask_file
if not os.path.exists(final_mask_file):
    for idx, row in tqdm(roi_set_df.iterrows()):
        if len(row['coordinates']) != 3: continue

        prefix = f"{mask_dir}/{row['roi_name']}.nii.gz"
        if idx == 0:
            final_mask = mask.create_sphere(row['coordinates'], radius=radius)
        else:
            roi_mask = mask.create_sphere(row['coordinates'], radius=radius)

            # step1: find intersection
            intersect_mask = masking.intersect_masks(
                [image.binarize_img(final_mask), roi_mask],
                threshold=1,
                connected=False
            )

            # step2: remove intersection from the roi_mask
            roi_mask = image.math_img(
                f'img1 - img2',
                img1=roi_mask,
                img2=intersect_mask
            )

            # step3: then include the roi into final_mask
            final_mask = image.math_img(
                f'img1 + {idx+1}*img2',
                img1=final_mask,
                img2=roi_mask
            )
    final_mask.to_filename(final_mask_file)

else:
    final_mask = image.load_img(final_mask_file)


# In[11]:


'''
subcortical:
taken from MAX rois
'''
max_rois_img = image.load_img(
    f"/home/govindas/parcellations/MAX_85_ROI_masks/MAX_ROIs_final_gm_85.nii.gz"
)
max_rois = max_rois_img.get_fdata()

# these are numbers in the mask, not indices, 
# so directly use them.
subcort_roi_max_idxs = [
    57, 56, # L, R CeMe Amygdala
    59, 58, # L, R BLBM Amygdala
    81, 80, # L, R PAG
    None, None, # L, R Habenula
    55, 54 # L, R BST
]

# roi indices
subcort_roi_mashid_idxs = list(range(37, 47))

'''
roi masks
'''
final_mask_file = f"{mask_dir}/../final_mask_subcortical.nii.gz"
subcortical_mask_file = final_mask_file
if not os.path.exists(final_mask_file):
    for i, mashid_idx in tqdm(enumerate(subcort_roi_mashid_idxs)):
        max_idx = subcort_roi_max_idxs[i]
        if max_idx is None: continue
        roi_mask = (max_rois == max_idx) * (mashid_idx+1)

        if i == 0: 
            final_mask = image.new_img_like(
                ref_niimg=final_mask,
                data=roi_mask,
                copy_header=False
            )
        else:
            roi_mask = image.new_img_like(
                ref_niimg=final_mask,
                data=roi_mask,
                copy_header=False
            )
            
            intersect_mask = masking.intersect_masks(
                [image.binarize_img(final_mask), image.binarize_img(roi_mask)],
                threshold=1,
                connected=False
            )

            roi_mask = image.math_img(
                f'img1 - {mashid_idx+1}*img2',
                img1=roi_mask,
                img2=intersect_mask
            )

            final_mask = image.math_img(
                f'img1 + img2',
                img1=final_mask,
                img2=roi_mask
            )
    final_mask.to_filename(final_mask_file)
else:
    final_mask = image.load_img(final_mask_file)


# In[12]:


'''
combine cortical and subcortical masks
'''
final_mask = image.math_img(
    f'img1 + img2',
    img1=cortical_mask_file,
    img2=subcortical_mask_file,
)


# In[13]:


plt.plot(np.unique(final_mask.get_fdata()))

np.unique(final_mask.get_fdata())


# individual rois may intersect if we simply add them to create a `final_mask`.
# we avoid this by subtracting the intersection from the newly added roi and then adding the remaining roi in the `final_mask`.
# 
# the plot verifies that `final_mask` contains numbers only upto 37; number of rois, and individual rois do not intersect in the `final_mask`. 

# In[14]:


'''
good voxels

basic step to be taken from the script:
/home/govindas/vscode-BSWIFT-mnt/MAX/scripts/Murty_Final/ROI_analysis/trial_level/FNSandFNT/MAX_fMRI_Analysis_neutral_deconv_reducedRuns.sh
'''

main_mask_path = (
    f"/home/govindas/vscode-BSWIFT-mnt/MAX"
    f"/dataset/preproc/masksAndCensors"
)

for subj in tqdm(os.listdir(main_mask_path)):
    mask_path = (
        f"{main_mask_path}/{subj}"
    )

    mask_goodVoxels_file = f"{mask_path}/mask_{set_name}_goodVoxels.nii.gz"
    # if os.path.exists(mask_goodVoxels_file): continue
    mask_goodVoxels = image.math_img(
        f'img1 * img2 * img3',
        img1=f"{mask_path}/goodVoxelsMask.nii.gz",
        img2=f"{mask_path}/commonVoxelsMask.nii.gz",
        img3=f"{final_mask_file}"
    )    
    mask_goodVoxels.to_filename(mask_goodVoxels_file)


# ## trial level analysis responses

# In[ ]:


'''
run in terminal:
bash runDeconvolve.sh
'''


# In[6]:


main_data_dir = (
    f"/home/govindas/network_dynamics/data/max"
    f"/neutral_runs_trial_level_FNSandFNT/{set_name}"
)

class ARGS(): pass
args = ARGS()

args.TRIAL_LEN = 14
args.LABELS = [0, 1] #safe, threat
args.LABEL_NAMES = ['FNS#', 'FNT#']

args.SEED = 74
np.random.seed(args.SEED)

args.subjects = os.listdir(main_data_dir)
random.Random(args.SEED).shuffle(args.subjects)

'''
exploratory dataset
'''
args.explor_subjects = args.subjects[ : round(0.25 * len(args.subjects))]
X, _, _ = dataset_utils.get_max_trial_level_responses(
    args, 
    main_data_dir,
    args.explor_subjects
)

with open(f"{proj_dir}/data/max/desc-exploratory_data_trial_level_responses_rois-{set_name}.pkl", 'wb') as f:
    pickle.dump(X, f)


# In[7]:


'''
plot the activations
'''
num_rois = X[0][0].shape[2]
args.roi_idxs = np.arange(num_rois)

def plot_roi_time_series(args, X, savefig=False, fig_file=None):
    X_conds = {}
    X_ = {}
    for label in args.LABELS:
        X_[label] = np.concatenate(X[label], axis=0)
        X_conds[f"{label}_m"] = np.nanmean(X_[label], axis=0)
        X_conds[f"{label}_s"] = 1.96 * np.nanstd(X_[label], axis=0) / np.sqrt(X_[label].shape[0])

    roi_name_file = (
        f"{proj_dir}/data/rois/{set_name}/roi_set_{set_name}.csv"
    )
    roi_names =  pd.read_csv(
        roi_name_file
    )['roi_name']

    time = np.arange(X_[0].shape[1])
    names = ['safe', 'threat']
    colors = {0:'royalblue', 1:'firebrick'}
    nrows, ncols = int(np.ceil(len(args.roi_idxs)/5)), 5

    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(5*ncols, 4*nrows), 
        sharex=False, 
        sharey=True, 
        dpi=150
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=None, hspace=0.5
    )

    for idx, roi in enumerate(args.roi_idxs):
        roi_name = roi_names[roi]
        if nrows > 1:
            ax = axs[idx//ncols, np.mod(idx,ncols)]
        else:
            ax = axs[idx]

        ax.set_title(f"{roi} {roi_name}")
        for label in args.LABELS:
            ts_mean = X_conds[f"{label}_m"][:, idx]
            ts_std = X_conds[f"{label}_s"][:, idx]

            ax.plot(ts_mean, color=colors[label], label=names[label])

            ax.fill_between(
                time, 
                (ts_mean - ts_std), 
                (ts_mean + ts_std),
                alpha=0.3, color=colors[label],
            )
        ax.set_xlabel(f"time")
        ax.set_ylabel(f"roi resp.")
        ax.grid(True)
        ax.legend()
        ax.set_ylim(-0.25, 0.25)

    if savefig:
        fig.savefig(
            fig_file,
            dpi=150,
            format='png',
            bbox_inches='tight',
            transparent=False
        )


# In[ ]:


plot_roi_time_series(args, X)


# In[8]:


args.up_roi_idxs = np.array(
    list(range(0, 5)) +
    list(range(7, 14)) +
    [22, 23]
)

args.zero_roi_idxs = np.array(
    list(range(14, 22)) +
    list(range(33, 37))
)

args.down_roi_idxs = np.array(
    [5, 6] +
    list(range(24, 33))
)

args.roi_idxs = np.concatenate([args.up_roi_idxs, args.zero_roi_idxs, args.down_roi_idxs])


# In[9]:


len(args.roi_idxs)


# In[10]:


with open(f"{proj_dir}/data/max/exploratory_data_{set_name}_roi_indices.pkl", 'wb') as f:
    pickle.dump([args.up_roi_idxs, args.zero_roi_idxs, args.down_roi_idxs, args.roi_idxs], f)

