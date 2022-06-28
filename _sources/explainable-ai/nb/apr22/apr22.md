Summary
===============================

Summary of progress in April 2022

- **Emoprox2 full dataset**:
    I got the full data of emoprox2 paradigm for the 300 ROI Schaefer parcellation, the MAX paradigm’s 85 ROIs parcellation. 
    The paradigm stimulus was given as follows: each participant goes through 6 runs, each with 2 blocks of stimuli. A block may have shock events. So the data is naturally partitioned into continuous segments /windows between start of block to shock onset, shock offset to next shock onset, and shock offset to the end of the block. 
    I used these segments as my data samples. The data contains time series from all the ROIs - 300 or 85 depending on the parcellation.
    In order to incorporate hemodynamic lag in the fMRI responses w.r.t. to the stimulus, I selected time series by shifting each continuous segment by 2TRs and 3TRs.
    With these datasets I trained the **GRU model**, and checked whether their performances are above chance.

    1. Although model's accuracy is ~0.7, this single value is not representative of how the model *sees* the input time series. i.e. information from which time points does the model use more(less) to classify the segment correctly?

    2. Since we have the stimulus timing files, we should understand how the stimulus and its following fMRI response vary with time. Can we clearly separate approach segments from retreat segments? Ans. No! *The paradigm was designed to study how threat anticipation (approach) and relaxation (retreat) phases interact, and how does one phase transition into the other.* **What are the dynamic patterns of roi activities that encode such stimuli?**

    3. The purpose is to predict the class labels for whole time series of unseen participants.

- **Observations from emoprox2 stimuli**. [](./02-understanding_emoprox2_stimulus.ipynb)
    1. The plots show that the (canonical) fMRI response peaks at around 8.75TRs. And the information about the stimulus is present from around 5(or 6)TRs. So we should take segments of the fMRI signal shifted by 5(6)TRs from the onset of stimulus. 

    2. We also observe that the responses for the two classes (appr/ retr) oscillate. *We can define a segment as time points between the intersections of the response curves.* This is because at intersection influence of both the stimuli is similar, and at other time points one of them dominates. Thus such a segment would have least *contamination* from response for the other stimulus.

    3. Since it may be difficult to determine which class the fMRI responses at the intersection time points belong to, we can introduce a third target label: **Don't care, or indistinguishable, or unidentifiable**. So the time points between intersections can belong to either of the two classes and the points around intersections can belong to the new third class. 
        - ~~A drawback may be that the segments will be only 2-3 TRs long.~~ A simpler model than RNN may classify the data better. ~~We may not be learning any temporal patterns in the data.~~
        - Many segments are 6 to 7 TRs long. 

- **Ideas for setting up training procedure**
    1. Introduce a **third label**.

    2. Use a **Temporal Convolutional Network (TCN)** and classify a segment into the two classes. TCN takes as input an image, does temporal convolution, and predicts a label for the image. In our case, input image can be the time series of all rois within a segment: a time x roi matrix. TCN will do temporal convolution and may find patterns of approach and retreat in the data. By computing importance scores, we can build intuition as to which rois at which time points are important for classification. 

    3. **Soft labels**. Why associate a time segment with a hard label? We can assign for each time point a vector as a label. Each element of this vector label will represent the degree of belongingness to each class. We can then provide a longer segment as input and let the model learn temopral dependencies. This may not be a classification set up, it may become a regression set up. We can use the canonical responses of each stimulus, [as in here](./02-understanding_emoprox2_stimulus.ipynb) as the soft labels. We will neither need the third *don't care* label, nor any alternative simpler model. The only issue will be of finding saliency methods for regression set ups.  


- **Progress**
    1. I created all stimulus files and their convolved regressors from scratch.
    These files in bswift were too old and had undergone many changes. It was difficult to track them without proper documentation. 
    And since I am creating dataset freshly, I wanted to create stimulus files from the basic paradigm files. 

    2. I will implement the 3 ideas in May...
