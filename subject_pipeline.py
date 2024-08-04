import utils
import os
import pandas as pd
import numpy as np
import mne
from mne.io import read_raw_kit, read_raw, write_info, read_info
import ast
import time
from autoreject import AutoReject, get_rejection_threshold, read_reject_log
from mne.coreg import Coregistration, scale_mri
from wordfreq import zipf_frequency
import itertools
import argparse
from tqdm import tqdm


config = utils.load_config()


############################################
# Preprocessing pipeline
############################################

def trial_get_raw(subject, session, task):
    subject_data_dir = os.path.join(config['directories']['source_dir'], f"sub-{subject}", f"ses-{session}", "meg")
    
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

    return raw

def get_epochs(subject, session, task, raw, preICA):
    current_sfreq = raw.info["sfreq"]
    desired_sfreq = config.getfloat('preprocessing', 'resample')
    decim = np.round(current_sfreq / desired_sfreq).astype(int)
    obtained_sfreq = current_sfreq / decim
    resample_lowpass_freq = obtained_sfreq / 3.0
    raw.filter(config.getfloat('preprocessing', 'highpass') if not preICA else config.getfloat('ica', 'pre_ica_highpass'), 
               resample_lowpass_freq if not preICA else config.getfloat('ica', 'pre_ica_lowpass'))
    
    # Epoching
    meta = utils.read_annotations(subject, session, task, config)
    meta = meta.query('kind=="word"')
    events = np.c_[meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))].astype(int)
    epochs = mne.Epochs(raw, events, decim=decim, event_repeated='drop',
                        tmin=config.getfloat('preprocessing', 'tmin'), tmax=config.getfloat('preprocessing', 'tmax'),
                        baseline = None if preICA else (None, 0))
    return epochs
    

def preprocess(subject, session, task, preICA):
    print(f"{'Pre-ICA' if preICA else 'Post-ICA'}: subject {subject}, session {session}, task {task}")

    # Create mne report
    report = mne.Report(title=f"Processing report for subject {subject}, session {session}, task {task}")

    # Load raw data
    raw = trial_get_raw(subject, session, task)
    report.add_raw(raw, title='Raw data')

    if not preICA:
        ica_file = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_ica.fif")
        ica = mne.preprocessing.read_ica(ica_file)

        ica.apply(raw)
        report.add_raw(raw, title='Post-ICA raw data')

    epochs = get_epochs(subject, session, task, raw, preICA)
    report.add_epochs(epochs, title='Resampled epochs')
    
    # AutoReject: global thresholding
    if preICA:
        threshold = get_rejection_threshold(epochs, random_state=config.getint('autoreject', 'random_state'))
        epochs = epochs.drop_bad(reject=threshold)
        report.add_epochs(epochs, title='Epochs after AutoReject')

    else:
        ar = AutoReject(n_jobs=8, random_state=config.getint('autoreject', 'random_state'))
        epochs = ar.fit_transform(epochs)

        report.add_epochs(epochs, title='Epochs after AutoReject')
        prec_path = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_desc-PostAR_epo.fif")
        epochs.save(prec_path, overwrite=True)

        rej_log = ar.get_reject_log(epochs)
        rej_log.save(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_reject_log.npz"))

    report.save(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_desc-{'PreICA' if preICA else 'PostICA'}_report.html"), overwrite=True, open_browser=False)
    
    return epochs

    
def trial_ICA(subject, session, task, redo=False):
    trial_dir = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg")
    ica_file = os.path.join(trial_dir, f"sub-{subject}_ses-{session}_task-{task}_ica.fif")

    if not redo and os.path.exists(ica_file):
        print(f"ICA file {ica_file} already exists")
        return

    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)
        print(f"Created directory {trial_dir}")

    epochs = preprocess(subject, session, task, preICA=True)

    ica = mne.preprocessing.ICA(
        n_components = config.getint('ica', 'n_components'),
        method = config['ica']['method'],
        verbose = True,
        fit_params=dict(extended=True),
        random_state = config.getint('ica', 'random_state')
    )
    ica.fit(epochs)
    ica.save(ica_file, overwrite=redo)


def annotate_ICA(subject, session, task, redo=False):
    if not redo and os.path.exists(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_desc-PostAR_epo.fif")):
        print(f"Post-AR epochs for subject {subject}, session {session}, task {task} already exist")
        return

    raw = trial_get_raw(subject, session, task)

    # Load ICA file
    ica_file = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_ica.fif")
    ica = mne.preprocessing.read_ica(ica_file)

    mne.set_config('MNE_BROWSER_BACKEND', 'qt')
    ica.plot_sources(inst=raw, show=True, start=0, stop=10, title="ICA sources", show_scrollbars=True, block=False)
    ica.plot_components(show=True, title="ICA components")

    print(ica.exclude)

    # Save excluded ICA
    ica_file = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_ica.fif")
    ica.save(ica_file, overwrite=True)


############################################
# Evoked
############################################

def trial_evoked(subject, session, task):
    report = mne.Report(title=f"Evoked report for subject {subject} session {session} task {task}")

    epochs = mne.read_epochs(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_desc-PostAR_epo.fif"))
    evoked = epochs.average()
    report.add_evokeds(evoked)
    
    inv = mne.minimum_norm.read_inverse_operator(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_inv.fif"))
    stc = mne.minimum_norm.apply_inverse(evoked, inv)
    report.add_stc(stc, title="Evoked source")

    report.save(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_evoked_report.html"), overwrite=True, open_browser=False)


############################################
# Source localization pipeline
############################################

def trial_coregister(subject, session, task):
    subject_data_dir = os.path.join(config['directories']['source_dir'], f"sub-{subject}", f"ses-{session}", "meg")
    derivative_dir = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg")

    if not os.path.exists(derivative_dir):
        os.makedirs(derivative_dir)
        print(f"Created directory {derivative_dir}")

    # Create mne report
    report = mne.Report(title=f"Coregistration report for subject {subject}, session {session}, task {task}")

    info_path = os.path.join(derivative_dir, f"sub-{subject}_ses-{session}_task-{task}_info.fif")
    if not os.path.exists(info_path):
        print(f"Creating info file for subject {subject}, session {session}, task {task}")
        # Load raw data
        elp = os.path.join(subject_data_dir, f"sub-{subject}_ses-{session}_acq-ELP_headshape.pos")
        hsp = os.path.join(subject_data_dir, f"sub-{subject}_ses-{session}_acq-HSP_headshape.pos")
        elp = utils.read_pos(elp)
        hsp = utils.read_pos(hsp)
        raw = read_raw_kit(os.path.join(subject_data_dir, f"sub-{subject}_ses-{session}_task-{task}_meg.con"),
                        mrk = os.path.join(subject_data_dir, f"sub-{subject}_ses-{session}_task-{task}_markers.mrk"),
                        elp = elp / 1000, hsp = hsp / 1000).load_data()
        info = raw.info
        write_info(info_path, info)
    else:
        print(f"Loading info file for subject {subject}, session {session} task {task}")
        info = mne.io.read_info(info_path)
    
    # Coregistration
    if not os.path.exists(os.path.join(config['directories']['freesurfer_subjects_dir'], f"MEG-MASC_sub-{subject}")):
        print(f"Creating MRI for subject {subject}")
        coreg = Coregistration(info, 'fsaverage', subjects_dir=config['directories']['freesurfer_subjects_dir'], fiducials="estimated")
        coreg.set_scale_mode('uniform').set_fid_match('matched')
        coreg.fit_fiducials()
        coreg.set_scale_mode('3-axis')
        for _ in range(4): 
            coreg.fit_icp(nasion_weight=1)
        trans_path = os.path.join(derivative_dir, f"sub-{subject}_ses-{session}_task-{task}_trans.fif")
        mne.write_trans(trans_path, coreg.trans, overwrite=True)
        new_mri_name = f"MEG-MASC_sub-{subject}"
        scale_mri('fsaverage', new_mri_name, subjects_dir=config['directories']['freesurfer_subjects_dir'], scale=coreg.scale, overwrite=True, annot=True)
    else:
        print(f"MRI for subject {subject} already exists")
        coreg = Coregistration(info, f'MEG-MASC_sub-{subject}', subjects_dir=config['directories']['freesurfer_subjects_dir'], fiducials="estimated")
        coreg.fit_fiducials()
        for _ in range(4): 
            coreg.fit_icp(nasion_weight=1)
        trans_path = os.path.join(derivative_dir, f"sub-{subject}_ses-{session}_task-{task}_trans.fif")
        mne.write_trans(trans_path, coreg.trans, overwrite=True)
        
    report.add_trans(
        trans=trans_path,
        info=info,
        subject=f"MEG-MASC_sub-{subject}",
        subjects_dir=config['directories']['freesurfer_subjects_dir'],
        alpha=0.7,
        title="Coregistration"
    )

    # Source space
    src = mne.setup_source_space(f"MEG-MASC_sub-{subject}", spacing="oct6", add_dist="patch", subjects_dir=config['directories']['freesurfer_subjects_dir'])
    mne.write_source_spaces(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}-task-{task}_src.fif"), src, overwrite=True)
    # BEM
    bem_model = mne.make_bem_model(subject=f"MEG-MASC_sub-{subject}", 
                                   ico=4, conductivity=(0.3,), subjects_dir=config['directories']['freesurfer_subjects_dir'])
    bem = mne.make_bem_solution(bem_model)
    mne.write_bem_solution(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_bem.fif"), bem, overwrite=True)

    report.add_bem(subject=f"MEG-MASC_sub-{subject}", subjects_dir=config['directories']['freesurfer_subjects_dir'], title="BEM")

    report.save(os.path.join(derivative_dir, f"sub-{subject}_ses-{session}_task-{task}_coreg_report.html"), overwrite=True, open_browser=False)


def trial_forward(subject, session, task):
    report = mne.Report(title=f"Forward solution report for subject {subject}, session {session}, task {task}")

    info = read_info(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_info.fif"))
    bad_channels = pd.read_csv(config['preprocessing']['bad_channels_file'])
    bad_channels = bad_channels.loc[(bad_channels['SubjectID'] == 'sub-' + subject) & (bad_channels['session'] == int(session)) & (bad_channels['task'] == int(task))]
    assert len(bad_channels) == 1, f"Bad channels information for subject {subject}, session {session}, task {task} not found"
    bad_channels = bad_channels.iloc[0]
    info['bads'] = ast.literal_eval(bad_channels['bad_channels'])

    src = mne.read_source_spaces(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}-task-{task}_src.fif"))
    bem = mne.read_bem_solution(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_bem.fif"))

    trans = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_trans.fif")
    fwd = mne.make_forward_solution(info, trans, src, bem, ignore_ref = True)
    # fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
    mne.write_forward_solution(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_fwd.fif"), fwd, overwrite=True)

    report.add_forward(fwd, title="Forward solution")
    report.save(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_fwd_report.html"), overwrite=True, open_browser=False)


def trial_inverse(subject, session, task):
    report = mne.Report(title=f"Inverse solution report for subject {subject}, session {session}, task {task}")

    # Read epochs
    epochs = mne.read_epochs(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_desc-PostAR_epo.fif"))

    info = read_info(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_info.fif"))
    bad_channels = pd.read_csv(config['preprocessing']['bad_channels_file'])
    bad_channels = bad_channels.loc[(bad_channels['SubjectID'] == 'sub-' + subject) & (bad_channels['session'] == int(session)) & (bad_channels['task'] == int(task))]
    assert len(bad_channels) == 1, f"Bad channels information for subject {subject}, session {session}, task {task} not found"
    bad_channels = bad_channels.iloc[0]
    info['bads'] = ast.literal_eval(bad_channels['bad_channels'])

    epochs.info = info

    # Compute covariance
    if os.path.exists(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_cov.fif")):
        print(f"Reading covariance for subject {subject}, session {session}, task {task}")
        cov = mne.read_cov(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_cov.fif"))
    else:
        print(f"Computing covariance for subject {subject}, session {session}, task {task}")
        cov = mne.compute_covariance(epochs, tmax=0, method='auto')
        mne.write_cov(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_cov.fif"), cov, overwrite=True)
        
        report.add_covariance(cov, info = epochs.info, title="Covariance")

    # Read forward solution
    fwd = mne.read_forward_solution(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_fwd.fif"))
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, loose=1.0)
    mne.minimum_norm.write_inverse_operator(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_inv.fif"), inv, overwrite=True)
    
    report.add_inverse_operator(inv, title="Inverse operator")
    report.save(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_inv_report.html"), overwrite=True, open_browser=False)


############################################
# Regression analysis pipeline
############################################

def get_surprisal_predictors(story_name, time_table, surprisal_time_windows):
    surprisal_predictor = np.zeros((len(time_table), len(surprisal_time_windows)))

    # Load surprisal data
    surp, word2tok = utils.load_surp(story_name, config)

    meta_word_start = np.zeros(len(time_table), dtype=int)
    meta_word_end = np.zeros(len(time_table), dtype=int)
    meta_i = 0

    for i, word in enumerate(word2tok['word']):
        clean_word_i = utils.clean_text(word)
        clean_meta_i = utils.clean_text(time_table['word'][meta_i])
        j = i + 1
        while len(clean_word_i) < len(clean_meta_i):
            clean_word_i = clean_word_i + utils.clean_text(word2tok['word'][j])
            j += 1
        if clean_word_i == clean_meta_i:
            meta_word_start[meta_i] = i
            meta_word_end[meta_i] = j
            meta_i += 1
            if meta_i == len(time_table):
                break

    # Calculate surprisal with time-based window
    for i, time in enumerate(time_table.iterrows()):
        for j, window in enumerate(surprisal_time_windows):
            window_end = time_table['onset'][i] + time_table['duration'][i]
            window_start = window_end - window
            # locate the last word with onset < window_start
            start_idx = np.where(time_table['onset'] < window_start)[0][-1] if np.where(time_table['onset'] < window_start)[0].size > 0 else -1
            if start_idx != -1:
                start_idx = word2tok['end'][meta_word_end[start_idx]]
            ntok = word2tok['start'][meta_word_start[i]] - start_idx
            if ntok <= 0: surprisal_predictor[i, j] = np.nan
            else:
                if ntok > 1024: ntok = 1024
                while np.any(np.isnan(surp[f"surp_{ntok}"][meta_word_start[i]:meta_word_end[i]])):
                    ntok -= 1
                    if ntok == 0:
                        raise ValueError(f"Surprisal data for word {time_table['word'][i]} is missing")
                surprisal_predictor[i, j] = surp[f"surp_{ntok}"][meta_word_start[i]:meta_word_end[i]].sum()

    return surprisal_predictor


def trial_rERF_source(subject, session, task, surp_win = [1, 300]):
    # subject to change: trial evoked regression -> source localization 
    report = mne.Report(title=f"rERF source report for subject {subject}")
    
    rerf_predictors = ["intercept", "zipf_frequency"]
    for win in surp_win:
        rerf_predictors.append(f"surprisal_{win}s")
    design = np.ones((0, len(rerf_predictors)))

    epochs = mne.read_epochs(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_desc-PostAR_epo.fif"))
    meta = utils.read_annotations(subject, str(session), str(task), config)
    meta = meta.query('kind=="word"')
    meta = meta.reset_index(drop=True)
    # read bad epochs and remove them
    bad_epochs = read_reject_log(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_reject_log.npz")).bad_epochs
    keep_epochs = np.logical_not(bad_epochs)
    meta = meta.loc[keep_epochs]

    intercept = np.ones((len(meta), 1))
    zipf_freq = np.array([zipf_frequency(word, 'en') for word in meta['word']])
    surps_design = get_surprisal_predictors(meta['story'][0], meta, surp_win)
    valid_epo_idx = np.where(~np.isnan(surps_design).any(axis=1))[0]

    print("Valid: {} out of {}, {:.2f}%".format(len(valid_epo_idx), len(surps_design), len(valid_epo_idx) / len(surps_design) * 100))
          
    intercept = intercept[valid_epo_idx]
    zipf_freq = zipf_freq[valid_epo_idx]
    surps_design = surps_design[valid_epo_idx]
    epochs = epochs[valid_epo_idx]

    design = np.vstack((design, np.c_[intercept, zipf_freq, surps_design]))
    # picks only good chanels
    epochs.pick_types(meg=True, eeg=False, eog=False, ecg=False, stim=False, exclude='bads')
    rerf = mne.stats.linear_regression(epochs, design_matrix=design, names=rerf_predictors)

    inv = mne.minimum_norm.read_inverse_operator(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_inv.fif"))
    # Plot and save rERF
    for pred in rerf_predictors:
        stc = mne.minimum_norm.apply_inverse(rerf[pred].beta, inv)
        report.add_stc(stc, title=f"rERF source beta for {pred}")

    report.save(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_rerf_report.html"), overwrite=True, open_browser=False)



if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--print', action='store_true', help='Print processing info')
    parser.add_argument('--preICA', action='store_true', help='Run pre-ICA pipeline')
    parser.add_argument('--annotateICA', action='store_true', help='Annotate ICA components')
    parser.add_argument('--postICA', action='store_true', help='Run post-ICA pipeline')
    parser.add_argument('--calcInv', action='store_true', help='Run inverse solution pipeline')
    parser.add_argument('--calcEvoked', action='store_true', help='Run evoked pipeline')
    parser.add_argument('--calcRegression', action='store_true', help='Run regression analysis pipeline')
    parser.add_argument('--exclude_subjects', nargs='+', help='List of subjects to exclude')
    parser.add_argument('--include_subjects', nargs='+', help='List of subjects to include')
    parser.add_argument('--exclude_tasks', nargs='+', help='List of tasks to exclude')
    parser.add_argument('--include_tasks', nargs='+', help='List of tasks to include')
    parser.add_argument('--redo', action='store_true', help='Redo processing')
    args = parser.parse_args()

    participants_info = pd.read_csv(config['preprocessing']['participants_info_file'], sep='\t' if config['preprocessing']['participants_info_file'].endswith('tsv') else ',')

    if args.include_subjects is None and args.exclude_subjects is None:
        args.include_subjects = [subject.split('-')[1] for subject in participants_info['participant_id']]
    elif args.exclude_subjects is not None:
        args.include_subjects = [subject.split('-')[1] for subject in participants_info['participant_id'] if subject.split('-')[1] not in args.exclude_subjects]

    if args.include_tasks is None and args.exclude_tasks is None:
        args.include_tasks = range(4)    
    if args.exclude_tasks is not None:
        args.include_tasks = [task for task in range(4) if str(task) not in args.exclude_tasks]

    print(f"Include subjects: {args.include_subjects}")
    print(f"Include tasks: {args.include_tasks}")

    if args.print:
        utils.print_processing_info(config)
        exit()

    # Time execution
    start = time.time()

    participants_info = participants_info.loc[participants_info['participant_id'].apply(lambda x: x.split('-')[1] in args.include_subjects)]
    
    for subject in tqdm(participants_info['participant_id']):
        subject = subject.split('-')[1]
        n_sessions = participants_info.loc[participants_info['participant_id'] == 'sub-' + subject, 'n_sessions'].iloc[0]
        for session in range(n_sessions):
            for task in args.include_tasks:
                if args.preICA: trial_ICA(subject, str(session), str(task), redo=args.redo)
                if args.annotateICA: annotate_ICA(subject, str(session), str(task), redo=args.redo)
                if args.postICA: 
                    if not args.redo and os.path.exists(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_desc-PostAR_epo.fif")):
                        print(f"Post-AR epochs for subject {subject}, session {session}, task {task} already exist")
                    else:
                        preprocess(subject, str(session), str(task), preICA=False)
                if args.calcInv: 
                    if not args.redo and os.path.exists(os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg", f"sub-{subject}_ses-{session}_task-{task}_inv.fif")):
                        print(f"Inverse solution for subject {subject}, session {session}, task {task} already exists")
                    else:
                        trial_coregister(subject, str(session), str(task))
                        trial_forward(subject, str(session), str(task))
                        trial_inverse(subject, str(session), str(task))
                if args.calcEvoked:
                    trial_evoked(subject, str(session), str(task))

                if args.calcRegression:
                    trial_rERF_source(subject, str(session), str(task))
                

    end = time.time()
    print(f"Elapsed time: {end - start} seconds")