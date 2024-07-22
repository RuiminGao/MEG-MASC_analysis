import utils
import os
import pandas as pd
import numpy as np
import mne
from mne.io import read_raw_kit, read_raw, write_info, read_info
import ast
import time
from autoreject import AutoReject
from mne.coreg import Coregistration, scale_mri


config = utils.load_config()


############################################
# Preprocessing pipeline
############################################

def preprocess(subject, session, task, preICA, redo=False):
    print(f"{'Pre-ICA' if preICA else 'Post-ICA'}: subject {subject}, session {session}, task {task}")

    subject_data_dir = os.path.join(config['directories']['source_dir'], f"sub-{subject}", f"ses-{session}", "meg")
    derivative_dir = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg")

    if not redo and os.path.exists(os.path.join(derivative_dir, f"sub-{subject}_ses-{session}_task-{task}_desc-{'PreICA' if preICA else 'PostICA'}_meg.fif")):
        print(f"{'Pre-ICA' if preICA else 'Post-ICA'} file for subject {subject}, session {session}, task {task} already exists")
        return read_raw(os.path.join(derivative_dir, f"sub-{subject}_ses-{session}_task-{task}_desc-{'PreICA' if preICA else 'PostICA'}_meg.fif")).load_data()

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

    prec_path = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_desc-{'PreICA' if preICA else 'PostICA'}_meg.fif")
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


############################################
# Evoked
############################################

def trial_evoked(subject):
    # Read participants information
    participants_info = pd.read_csv(config['preprocessing']['participants_info_file'])
    participant = participants_info.loc[participants_info['participant_id'] == 'sub-' + subject]
    assert len(participant) == 1, f"Participant {subject} not found"
    participant = participant.iloc[0]
    n_sessions = participant['n_sessions']

    report = mne.Report(verbose=True, title=f"Evoked report for subject {subject}")

    evoked = list()
    for session in range(n_sessions):
        for task in range(4):
            epochs = mne.read_epochs(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_desc-PostICA_epo.fif"))
            evoked.append(epochs.average())

    evoked_grand = mne.grand_average(evoked)
    evoked_grand.plot_joint()

    report.add_evokeds(evoked, title='Evoked responses')
    report.save(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"sub-{subject}_evoked_report.html"), overwrite=True)

    return evoked_grand


def trial_evoked_source(subject):
    return NotImplementedError


############################################
# Decoding
############################################


############################################
# Source localization pipeline
############################################

def trial_coregister(subject, session):
    subject_data_dir = os.path.join(config['directories']['source_dir'], f"sub-{subject}", f"ses-{session}", "meg")
    derivative_dir = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg")

    if not os.path.exists(derivative_dir):
        os.makedirs(derivative_dir)
        print(f"Created directory {derivative_dir}")

    # Create mne report
    report = mne.Report(verbose=True, title=f"Coregistration report for subject {subject}, session {session}")

    info_path = os.path.join(derivative_dir, f"sub-{subject}_ses-{session}_info.fif")
    if not os.path.exists(info_path):
        print(f"Creating info file for subject {subject}, session {session}")
        # Load raw data
        elp = os.path.join(subject_data_dir, f"sub-{subject}_ses-{session}_acq-ELP_headshape.pos")
        hsp = os.path.join(subject_data_dir, f"sub-{subject}_ses-{session}_acq-HSP_headshape.pos")
        elp = utils.read_pos(elp)
        hsp = utils.read_pos(hsp)
        raw = read_raw_kit(os.path.join(subject_data_dir, f"sub-{subject}_ses-{session}_task-0_meg.con"),
                        mrk = os.path.join(subject_data_dir, f"sub-{subject}_ses-{session}_task-0_markers.mrk"),
                        elp = elp / 1000, hsp = hsp / 1000).load_data()
        info = raw.info
        write_info(info_path, info)
    else:
        print(f"Loading info file for subject {subject}, session {session}")
        info = mne.io.read_info(info_path)
    
    # Coregistration
    coreg = Coregistration(info, 'fsaverage', subjects_dir=config['directories']['freesurfer_subjects_dir'], fiducials="estimated")
    coreg.set_scale_mode('uniform').set_fid_match('matched')
    coreg.fit_fiducials()
    coreg.set_scale_mode('3-axis')
    for _ in range(4): 
        coreg.fit_icp(nasion_weight=1)
    trans_path = os.path.join(derivative_dir, f"sub-{subject}_ses-{session}_trans.fif")
    mne.write_trans(trans_path, coreg.trans, overwrite=True)
    new_mri_name = f"MEG-MASC_sub-{subject}_ses-{session}"
    scale_mri('fsaverage', new_mri_name, subjects_dir=config['directories']['freesurfer_subjects_dir'], scale=coreg.scale, overwrite=True, annot=True)

    report.add_trans(
        trans=trans_path,
        info=info,
        subject=new_mri_name,
        subjects_dir=config['directories']['freesurfer_subjects_dir'],
        alpha=0.7,
        title="Coregistration"
    )

    # Source space
    src = mne.setup_source_space(f"MEG-MASC_sub-{subject}_ses-{session}", spacing="oct6", add_dist="patch", subjects_dir=config['directories']['freesurfer_subjects_dir'])
    mne.write_source_spaces(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_src.fif"), src, overwrite=True)
    # BEM
    bem_model = mne.make_bem_model(subject=f"MEG-MASC_sub-{subject}_ses-{session}", 
                                   ico=4, conductivity=(0.3,), subjects_dir=config['directories']['freesurfer_subjects_dir'])
    bem = mne.make_bem_solution(bem_model)
    mne.write_bem_solution(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_bem.fif"), bem, overwrite=True)

    report.add_bem(subject=f"MEG-MASC_sub-{subject}_ses-{session}", subjects_dir=config['directories']['freesurfer_subjects_dir'], title="BEM")

    report.save(os.path.join(derivative_dir, f"sub-{subject}_ses-{session}_coreg_report.html"), overwrite=True)


def trial_forward(subject, session, task):
    report = mne.Report(verbose=True, title=f"Forward solution report for subject {subject}, session {session}, task {task}")

    info = read_info(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_info.fif"))
    bad_channels = pd.read_csv(config['preprocessing']['bad_channels_file'])
    bad_channels = bad_channels.loc[(bad_channels['SubjectID'] == 'sub-' + subject) & (bad_channels['session'] == int(session)) & (bad_channels['task'] == int(task))]
    assert len(bad_channels) == 1, f"Bad channels information for subject {subject}, session {session}, task {task} not found"
    bad_channels = bad_channels.iloc[0]
    info['bads'] = ast.literal_eval(bad_channels['bad_channels'])

    src = mne.read_source_spaces(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_src.fif"))
    bem = mne.read_bem_solution(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_bem.fif"))

    trans = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_trans.fif")
    fwd = mne.make_forward_solution(info, trans, src, bem, ignore_ref = True)
    mne.write_forward_solution(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_fwd.fif"), fwd, overwrite=True)

    report.add_forward(fwd, title="Forward solution")
    report.save(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_fwd_report.html"), overwrite=True)


def trial_inverse(subject, session, task):
    report = mne.Report(verbose=True, title=f"Inverse solution report for subject {subject}, session {session}, task {task}")

    # Read epochs
    epochs = mne.read_epochs(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_desc-PostICA_epo.fif"))

    info = read_info(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_info.fif"))
    bad_channels = pd.read_csv(config['preprocessing']['bad_channels_file'])
    bad_channels = bad_channels.loc[(bad_channels['SubjectID'] == 'sub-' + subject) & (bad_channels['session'] == int(session)) & (bad_channels['task'] == int(task))]
    assert len(bad_channels) == 1, f"Bad channels information for subject {subject}, session {session}, task {task} not found"
    bad_channels = bad_channels.iloc[0]
    info['bads'] = ast.literal_eval(bad_channels['bad_channels'])

    epochs.info = info

    # Compute covariance
    cov = mne.compute_covariance(epochs, tmax=0, method='auto')
    mne.write_cov(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_cov.fif"), cov, overwrite=True)
    
    report.add_covariance(cov, info = epochs.info, title="Covariance")

    # Read forward solution
    fwd = mne.read_forward_solution(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_fwd.fif"))
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, loose=1.)
    mne.minimum_norm.write_inverse_operator(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_inv.fif"), inv)
    
    report.add_inverse_operator(inv, title="Inverse operator")
    report.save(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_inv_report.html"), overwrite=True)

############################################
# Regression analysis pipeline
############################################

def get_surprisal_predictors():
    # Load epoch time information

    # Load surprisal data

    # Calculate surprisal with time-based window

    raise NotImplementedError

def rERPs(subject, session, task):
    # TODO: need to check epochs number & metadata alignment first
    raise NotImplementedError


def rERP_source(subject, session, task):
    raise NotImplementedError
 

if __name__ == '__main__':
    # Time execution
    start = time.time()

    trial_ICA('01', '1', '0')
    # annotate_ICA('01', '0', '2')
    # trial_postICA('01', '0', '1')
    # trial_coregister('01', '1')
    # trial_forward('01', '0', '0')
    # trial_inverse('01', '0', '0')

    end = time.time()
    print(f"Elapsed time: {end - start} seconds")