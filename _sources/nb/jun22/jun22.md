Summary
===============================

Summary of progress in June 2022

- **Intersubject Correlation (ISC) analysis** In terms of analysing data as dynamic networks, Luiz suggested to use intersubject correlation (ISC) analysis. 
    - One of the widespread types of fMRI analysis is the investigation of functional connectivity (FC), temporal covariance of brain activity across ROIs, and how it changes with stimuli or internal goals. Covariance of brain activity between a pair of ROIs in the FC matrix may be caused by various factors other than the provided stimulus; like physiological factors (respiration, heart beat, etc.), indirect influence through another ROI, motion confounds, etc. 
    
    - It becomes essential to isolating stimulus-driven covariances shared across participants. A new variant of FC, ISC, achieves this objective. ISC is a model agnostic way of capturing shared responses to stimuli across a group of participants. It does so by modeling a participant's brain activity in terms of others' brain activities, thereby capturing common activity across the group. This analysis provides *complementary* insights to traditional GLM analysis having a pre-defined response model. 

    - **Definition** I formally introduce ISC now. Brain activity of a ROI of a subject can be interpreted as a mixture of three components:
        1. *common* signal: stimulus-driven signal consistent across participants,
        2. *idiosyncratic* signal: stimulus-driven signal particular to that participant, and independent of the same signal in other participants,
        3. *spontaneous* signal: spontaneous activity unrelated to the stimulus, thus independent of all other signals.
    
    ISC matrix is computed as follows:
        1. Leave out a participant's responses and average responses of the remaining participants.
        2. Compute Pearson correlation coefficient between the participant's responses and the averaged responses.This is the ISC matrix of that participant.
        3. Repeat the procedure for all the participants. 
        4. Define group ISC matrix as average of all the obtained ISC matrices.

    - **Group level inference** Once we obtain ISC matrix for a group of participants for a stimulus, we need to **statistically validate** the correlation values. There are two main ways of computing null distribution:
        1. On the response time series:
            1. `timeshift`: as responses are time locked to the stimulus, shifting them in time by random amounts will distupt the locking and diminish ISC values. This preserves autocorrelation structure.
            2. `phaseshift`:  take Fourier transform of the responses, randomize phase of each Fourier component, and invert to obtain randomized responses. This preserves power spectrum of the responses, but disrupts temporal alignment.
        2. On the obtained participant-wise ISC matrices: (non parametric approaches)
            1. `bootstrap` for one-sample test: at each iteration of bootstrap, randomly sample with replacement N participants and compute group-wise ISC matrix. Generate a distribution of such values, and compare the actual ISC with this distribution.
            2. `permutation` for two-sample test: randomly permute group assignments at each iteration.  

- **Progress** 
    1. I found a toolbox, called [BrainIAK](https://github.com/snastase/isc-tutorial) (abbreviaiton for Brain Imaging Analysis Kit) and accompanying [review paper](https://academic.oup.com/scan/article/14/6/667/5489905) that provides functions for computing ISC, and statistics over them. I went through the entire toolbox, familiarized myself with the important functions.