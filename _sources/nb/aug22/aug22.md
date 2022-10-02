Summary
===============================

Summary of progress in August 2022

- **Progress** Network analysis is constrained by its resolution in terms of number of nodes and edges. To analyse (and compare) ISC matrices at different resolutions, I am collecting data in two more sets of ROIs.
    1. [ROIs from Mashid's and Brenton's paper](./02-rois-mashid_desc-create_rois.ipynb) Brenton McMenamin and Mashid Najafi had defined 36+ ROIs as 5mm radius spheres around center coordinates and categorized them into three networks: Saliency, Executive, and Task-negative along with 10+ Subcortical ROIs. I generated trial level responses of the MAX paradigm for these ROIs, computed ISC matrices, and plotted with (1.) [the networks](./03-data-max_trial_level_rois-mashid_order-nw-model-isfc_desc-early_vs_late.ipynb), and (2.) [selected up-zero-down ROIs](./03-data-max_trial_level_rois-mashid_order-uzd-model-isfc_desc-early_vs_late.ipynb), based on their activations.

    2. [ROIs used in ABA paradigm](./02-rois-aba_desc-create_rois.ipynb) These are a subset of MAX ROIs with two additional subcortical ROIs, Pulvinar and VTA-SNc. I generated trial level responses and computed [ISC matrices](./03-data-max_trial_level_rois-aba_model-isfc_desc-early_vs_late.ipynb).

    3. [MAX ROIs to resting state networks](./04-rois-max_desc-schaefer_network_organization.ipynb) Selecting ROIs based on their activations will lead to circularity in analysing threat conditions (because I was looking at responses in threat blocks). To avoid this scenario I organized the MAX 85 ROIs into 7 resting state networks + a Subcortical network. Thomas Yeo's lab had parcellated cortex into 1000 parcels and grouped them into 7 resting state networks. These are used as standard parcellations. So for each MAX ROI I found all the intersecting parcels and assigned the ROI to the network to which most of those intersecting parcels belong to. 

    I obtained the trial level responses and computed ISC matrices. Next task is to analyse them further in terms of these networks. 

    4. [ABA dataset](./05-rois-max_data-aba_desc-create_rois_trial_level_responses.ipynb) Similar to the MAX paradigm, I collected trial level responses for the ABA paradigm. The responses are of the 85 MAX ROIs for play periods of each trial. Each trial block is a 13 TR window aligned to the end of the play period. Each block/trial has time range $[-11.25, 6.25]$ seconds with end of that play period as $0.0$ seconds.

I was down with Covid-19 in between for 10 days of this month.
