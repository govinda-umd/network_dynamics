Summary
===============================

Summary of progress in July 2022

- **Progress**
    - **MAX paradigm** 
        1. The MAX paradigm has stimulus blocks of either `safe` or `threat` valence. Each block is 14TRs long (1TR = 1.25 seconds, meaning each block is 16 seconds long). Previous analysis found that `early` and `late` periods within each block has different ROI activations. Therefore, they analyzed MAX data in four conditions: `safe_early`, `safe_late`, `threat_early`, and `threat_late`.
        
        2. [Trial level responses](./00-max_dataset_trial_level_responses.ipynb) Another key aspect of fMRI data is correctly extracting responses from the contaminated signals. The fMRI time series dataset may be contanimated with 1. head motion artifacts, 2. hemodynamic lags of the previous events. More explicitly, if you convolve a boxcar of a trial you will see that the response from the rating period spills over into the next trial, so the response is mixed. If the responses are mixed, you're not evaluating functional connectivity that is related to the condition only but it's mixed. It is desirable to remove noise from the signal and get responses at each trial (a block of a condition). This trial level analysis for all ROIs was done by V.P.S. Murty, and I extracted the responses from his analysis.

    - [**ISC matrices**](./01-data-max_trial_level_model-isfc_desc-early_vs_late.ipynb) Once I got the correct data I followed the procedure:
        1. Compute leave-one-out (LOO) ISC values for the four conditions. In LOO method, the mean time series of the remaining subjects converges to the common signal across subjects.

        2. Bootstrap hypothesis testing, ([Chen et al.2016](https://doi.org/10.1016/j.neuroimage.2016.05.023): Statistical significance was assessed by a nonparametric bootstrap hypothesis test resampling left-out subjects (Leavo-one-out (LOO)). 

    - **Statistical testing between two conditions** *pairwise non parametric permutation test*. We perform a non parametric pairwise permutation test suggested by [Chen et al.2016](https://doi.org/10.1016/j.neuroimage.2016.05.023) to compare ISC matrices between any two pairs of conditions. We use statistic in the permutation test as the difference between median ISC matrices between two conditions. We plot the statistic between every pair of conditions only for the roi-pairs that differentiate significantly between the conditions.

    - [**Selecting ROIs**](./00-data-max_desc-roi_ordering_and_colormap.ipynb) We can understand patterns in the ISC matrices in terms of pre-defined groups of ROIs. i.e. if we organize ROIs into disjoint sets, or `networks`, we may analyze better the ISC matrices. e.g. if group the ROIs into the 7 canonical resting state networks we might make inferences in terms of the networks. As a start in that direction, Luiz and I selected a few ROIs based on their activations in the threat condition. We made two classes: ROIs whose activations go up, and those whose activations go down. I did ISC analysis on these selected ROIs this month.   
