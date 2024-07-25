# Dataset

Dataset was downloaded from: https://osf.io/ag3kj/

Assoicated paper: https://www.nature.com/articles/s41597-023-02752-5

# Processing Pipeline

## Preprocessing

- Manual bad channel exclusion

- ICA: [1, 100] Hz pre-ICA bandpass filtering + extended INFOMAX

- Epoching and resampling to 200 Hz.

- Auto Rejection

## Stimuli Feature Extraction

Calculate surprisal dump: extract_surprisal.ipynb

Tidy surprisals into table format: tidy_surprisal.ipynb

## Forward and Inverse Modeling

Alignment is done with scaled fsaverage template.

## Analysis

- Evoked (sensor and source space)

- rERF (sensor and source space)