import configparser
import os
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath


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


def generate_config_template(config_file='config_template.ini'):
    raise NotImplementedError


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


# TODO: parallel processing support

