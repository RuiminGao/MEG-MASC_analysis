# MEG-MASC_analysis

Dataset was downloaded from: https://osf.io/ag3kj/

Assoicated paper: https://www.nature.com/articles/s41597-023-02752-5

# Data Preprocessing

Codes derived from Dr. Jon Brennan's codes: https://github.com/cnllab/lpp-l2-eeg

Steps:

- Manual bad channel exclusion

- Epoch + resample MEG data to 200 Hz and 0.1 Hz highpass.

    - Epoching is suggested to be done before resampling to avoid jittering, check [this tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html).

- ICA: [1, 100] Hz pre-ICA bandpass filtering + extended INFOMAX

    - EOG and ECG are conservatively annotated.

- Auto Rejection

# Source Reconstruction

Steps: 

- 'fsaverage' template is used for all subject. Prior to alignment, scaling is done by *details*.

# Stimuli Feature Extraction

Calculate surprisal features: *filename*

Calculate surprisal metadata for each subject: *filename*

# Analysis

Codes derived from Dr. Jon Brennan's codes: https://github.com/cnllab/lpp-l2-eeg

## Evoked

## rERP Sensor

## rERP Source
