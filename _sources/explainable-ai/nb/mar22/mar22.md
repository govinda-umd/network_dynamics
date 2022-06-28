Summary
===============================

Summary of progress in March 2022

- Item 1
- Item 2
- Emoprox2 full dataset:
    I got the full data of emoprox2 paradigm for the 300 ROI Schaefer parcellation, (not the MAX paradigm’s 85 ROIs). Each participant goes through 6 runs, each with 2 blocks of stimuli. A block may have shock events. So the data is naturally partitioned into continuous segments between
    start of block to shock onset, 
    shock offset to next shock onset, and
    shock offset to the end of the block. 
    I used these segments as my data samples and trained the GRU model. The data contains time series from all the 300 ROIs.

    The model’s accuracy is 72% and is above chance.

    Then I did lesion analysis. As the ROIs were categorized into 7 networks, I lesioned these networks and their sub networks and found that no single network drops the accuracy more than the second decimal point. So I went on to lesion combinations of networks; 7 choose 1, 7 choose 2, and 7 choose 3. I do this in two ways:
    I lesioned a set of networks by masking time series of those networks and keeping others intact and saw which set of networks decrease the accuracy the most. The more drop in accuracy, the better.
    I only kept time series of a set of networks and zeroed out all other ROIs and saw what is the contribution of that particular set of networks in isolation from others. Here, the more the accuracy, the better. 
    In both ways I found that the model’s performance depends upon time series from Visual, Somato motor, and Ventral attention networks.

    Report:
    https://govinda-umd.github.io/explainable-ai/nb/mar22/06-ROIwise_analysis_emoprox2_full_dataset.html
- ...
