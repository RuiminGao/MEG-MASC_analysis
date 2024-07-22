import utils
import os
import pandas as pd
import numpy as np
import mne
from mne.io import read_raw_kit, read_raw
import ast
import time
from matplotlib import pyplot as plt
from autoreject import AutoReject


config = utils.load_config()

def preprocess(subject, session, task, preICA, redo=False):
    print(f"{'Pre-ICA' if preICA else 'Post-ICA'}: subject {subject}, session {session}, task {task}")

    subject_data_dir = os.path.join(config['directories']['source_dir'], f"sub-{subject}", f"ses-{session}", "meg")
    derivative_dir = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg")

    if not redo and os.path.exists(os.path.join(derivative_dir, f"sub-{subject}_ses-{session}_task-{task}_desc-{'PreICA' if preICA else 'PostICA'}_epo.fif")):
        print(f"{'Pre-ICA' if preICA else 'Post-ICA'} file for subject {subject}, session {session}, task {task} already exists")
        return read_raw(os.path.join(derivative_dir, f"sub-{subject}_ses-{session}_task-{task}_desc-{'PreICA' if preICA else 'PostICA'}_epo.fif")).load_data()

    # Create mne report
    report = mne.Report(verbose=True, title=f"Processing report for subject {subject}, session {session}, task {task}")

    # Load raw data
    elp = os.path.join(subject_data_dir, f"sub-{subject}_ses-{session}_acq-ELP_headshape.pos")
    hsp = os.path.join(subject_data_dir, f"sub-{subject}_ses-{session}_acq-HSP_headshape.pos")
    elp = utils.read_pos(elp)
    hsp = utils.read_pos(hsp)
    raw = read_raw_kit(os.path.join(subject_data_dir, f"sub-{subject}_ses-{session}_task-{task}_meg.con"),
                       mrk = os.path.join(subject_data_dir, f"sub-{subject}_ses-{session}_task-{task}_markers.mrk"),
                       elp = elp / 1000, hsp = hsp / 1000).load_data()

    # Bad channel handling
    bad_channels_file = config['preprocessing']['bad_channels_file']
    bad_channels = pd.read_csv(bad_channels_file)
    bad_channels = bad_channels.loc[(bad_channels['SubjectID'] == 'sub-' + subject) & (bad_channels['session'] == int(session)) & (bad_channels['task'] == int(task))]
    assert len(bad_channels) == 1, f"Bad channels information for subject {subject}, session {session}, task {task} not found"
    bad_channels = bad_channels.iloc[0]
    raw.info['bads'] = ast.literal_eval(bad_channels['bad_channels'])
    report.add_raw(raw, title='Raw data')

    current_sfreq = raw.info["sfreq"]
    desired_sfreq = config.getfloat('preprocessing', 'resample')
    decim = np.round(current_sfreq / desired_sfreq).astype(int)
    obtained_sfreq = current_sfreq / decim
    resample_lowpass_freq = obtained_sfreq / 3.0
    raw.filter(config.getfloat('preprocessing', 'highpass') if not preICA else config.getfloat('ica', 'pre_ica_highpass'), 
               resample_lowpass_freq if not preICA else config.getfloat('ica', 'pre_ica_lowpass'))

    report.save(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_desc-{'PreICA' if preICA else 'PostICA'}_report.html"), overwrite=redo)

    prec_path = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_desc-{'PreICA' if preICA else 'PostICA'}_epo.fif")
    raw.save(prec_path, overwrite=redo)
    
    return raw

    
def trial_ICA(subject, session, task, redo=False):
    trial_dir = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg")
    ica_file = os.path.join(trial_dir, f"sub-{subject}_ses-{session}_task-{task}_ica.fif")

    if not redo:
        assert not os.path.exists(ica_file), f"ICA file {ica_file} already exists"

    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)
        print(f"Created directory {trial_dir}")

    preprocessed = preprocess(subject, session, task, preICA=True)

    ica = mne.preprocessing.ICA(
        n_components = config.getfloat('ica', 'n_components'),
        method = config['ica']['method'],
        verbose = True,
        fit_params=dict(extended=True),
        random_state = config.getint('ica', 'random_state')
    )
    ica.fit(preprocessed)
    ica.save(ica_file, overwrite=redo)


def annotate_ICA(subject, session, task):
    preprocessed = preprocess(subject, session, task, preICA=False)

    report = mne.Report(verbose=True, title=f"ICA report for subject {subject}, session {session}, task {task}")

    # Load ICA file
    ica_file = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_ica.fif")
    ica = mne.preprocessing.read_ica(ica_file)

    mne.set_config('MNE_BROWSER_BACKEND', 'qt')
    ica.plot_sources(inst=preprocessed, show=True, start=0, stop=10, title="ICA sources", show_scrollbars=True, block=False)
    ica.plot_components(show=True, title="ICA components")

    print(ica.exclude)

    report.add_ica(
        ica=ica,
        title="ICA cleaning",
        inst=preprocessed,
        n_jobs=8
    )

    report.save(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_ica_report.html"), overwrite=True)

    # Save excluded ICA
    ica_file = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_ica.fif")
    ica.save(ica_file, overwrite=True)


def trial_postICA(subject, session, task, redo=False):
    preprocessed = preprocess(subject, session, task, preICA=False)
    report = mne.Report(verbose=True, title=f"Post-ICA report for subject {subject}, session {session}, task {task}")

    # Load ICA file
    ica_file = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_ica.fif")
    ica = mne.preprocessing.read_ica(ica_file)

    report.add_raw(preprocessed, title='Raw data')

    ica.apply(preprocessed)

    report.add_raw(preprocessed, title='Post-ICA raw data')

    meta = utils.read_annotations(subject, session, task, config)
    meta = meta.query('kind=="word"')
    events = np.c_[meta.onset * preprocessed.info["sfreq"], np.ones((len(meta), 2))].astype(int)
    current_sfreq = preprocessed.info["sfreq"]
    desired_sfreq = config.getfloat('preprocessing', 'resample')
    decim = np.round(current_sfreq / desired_sfreq).astype(int)
    epochs = mne.Epochs(preprocessed, events, decim=decim, event_repeated='drop',
                        tmin=config.getfloat('preprocessing', 'tmin'), tmax=config.getfloat('preprocessing', 'tmax'))
    epochs.resample(desired_sfreq)  # Filtering, epoching, and resampling follows the best practice suggested in https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html
    report.add_epochs(epochs, title='Epochs after ICA')

    ar = AutoReject(n_jobs=8, random_state=config.getint('autoreject', 'random_state'))
    epochs_clean = ar.fit_transform(epochs)

    report.add_epochs(epochs_clean, title='Epochs after AutoReject')
    report.save(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_AR_report.html"), overwrite=True)

    epochs_clean.save(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_desc-PostICA_epo.fif"), overwrite=True)


if __name__ == '__main__':
    # # Time execution
    # start = time.time()
    # trial_ICA('01', '0', '0', redo=True)
    # end = time.time()
    # print(f"Elapsed time: {end - start} seconds")

    # annotate_ICA('01', '0', '0')

    trial_postICA('01', '0', '0', redo=True)