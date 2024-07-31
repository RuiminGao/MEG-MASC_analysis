import configparser
import os
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
import string
import glob


def load_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    
    source_dir = config['directories']['source_dir']
    derivative_dir = config['directories']['derivative_dir']
    freesurfer_subjects_dir = config['directories']['freesurfer_subjects_dir']

    assert os.path.exists(source_dir), f"Source directory {source_dir} does not exist"
    if not os.path.exists(derivative_dir):
        os.makedirs(derivative_dir)
        print(f"Created derivative directory {derivative_dir}")
    assert os.path.exists(freesurfer_subjects_dir), f"Freesurfer subjects directory {freesurfer_subjects_dir} does not exist"

    bad_channels_file = config['preprocessing']['bad_channels_file']
    assert os.path.exists(bad_channels_file), f"Bad channels file {bad_channels_file} does not exist"

    mne.set_log_level(config['logging']['mne_log_level'])

    return config


def read_pos(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    pos = np.zeros((0, 3))
    for i, line in enumerate(lines):
        if line.startswith('%'):
            continue
        pos = np.vstack((pos, np.array([float(x) for x in line.split()])))
    return pos


def read_annotations(subject, session, task, config):
    # to time
    annotation_path = BIDSPath(subject=subject, session=session, task=task, datatype='meg', root=config['directories']['source_dir'], suffix='events', extension='.tsv').match()
    annotations = pd.read_csv(annotation_path[0], sep='\t').to_dict('records')
    meta = list()
    for annot in annotations:
        d = eval(annot.pop("trial_type"))
        for k, v in annot.items():
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    return meta


def concat_generators(*gens):
    for gen in gens:
        yield from gen


def load_surp(story_name, config):
    surp = pd.read_csv(os.path.join(config['directories']['stimuli_dir'], f'surprisals_{story_name}.csv'))
    word2tok = pd.read_csv(os.path.join(config['directories']['stimuli_dir'], f'word2tokens_{story_name}.csv'))
    return surp, word2tok


def clean_text(s):
  s = ''.join(s.split()).lower()
  s = s.translate(str.maketrans('', '', string.punctuation))
  return s


def print_processing_info(config, export=True):
    print("Processing info:")
    participants_info = pd.read_csv(os.path.join(config['directories']['source_dir'], 'participants.tsv'), sep='\t')
    processing_info = pd.DataFrame(columns=['subject', 'session', 'task', 'ICA', 'AR', 'add_predictors', 'forward', 'inverse', 'rERF_sensor', 'rERF_source'])
    for subject in participants_info['participant_id']:
        subject = subject.split('-')[1]
        n_sessions = participants_info.loc[participants_info['participant_id'] == 'sub-' + subject, 'n_sessions'].iloc[0]
        for session in range(n_sessions):
            for task in range(4):
                trial_dir = os.path.join(config['directories']['derivative_dir'], f"sub-{subject}", f"ses-{session}", "meg")
                row = {'subject': subject, 'session': session, 'task': task}
                row['ICA'] = os.path.exists(os.path.join(trial_dir, f"sub-{subject}_ses-{session}_task-{task}_ica.fif"))
                row['AR'] = os.path.exists(os.path.join(trial_dir, f"sub-{subject}_ses-{session}_task-{task}_desc-PostAR_epo.fif"))
                row['add_predictors'] = os.path.exists(os.path.join(trial_dir, f"sub-{subject}_ses-{session}_task-{task}_predictors.csv"))
                row['forward'] = os.path.exists(os.path.join(trial_dir, f"sub-{subject}_ses-{session}_task-{task}_fwd.fif"))
                row['inverse'] = os.path.exists(os.path.join(trial_dir, f"sub-{subject}_ses-{session}_task-{task}_inv.fif"))

                template = os.path.join(trial_dir, f"sub-{subject}_ses-{session}_task-{task}_desc-rERF-*_evoked.fif")  
                rERF_files = glob.glob(template)
                rERF_files = [os.path.basename(f).split('_')[3].split('-')[2] for f in rERF_files]
                row['rERF_sensor'] = rERF_files

                template = os.path.join(trial_dir, f"sub-{subject}_ses-{session}_task-{task}_desc-rERF-*-lh.*")  
                rERF_files = glob.glob(template)
                rERF_files = [os.path.basename(f).split('_')[3].split('-')[2] for f in rERF_files]
                row['rERF_source'] = rERF_files

                processing_info = processing_info.append(row, ignore_index=True)

    print(processing_info)
    if export:
        processing_info.to_csv(os.path.join(config['directories']['derivative_dir'], 'processing_info.csv'), index=False)
                