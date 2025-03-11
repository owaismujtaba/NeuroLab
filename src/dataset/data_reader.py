import numpy as np

import config as config

import pdb


class DataReader:
    def __init__(self, subject_id):
        self.subject_id = subject_id
        self.audio = None
        self.eeg = None
        self.words = None
        self.channels = None

        self._read_data()

        
    def _read_data(self):
        pdb.set_trace()
        dir = config.DATA_DIR
        self.eeg = np.load('audio.npy')
        self.audio = np.load('audio.npy')
        self.words = np.load('audio.npy')
        self.channels = np.load('audio.npy')