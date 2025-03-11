import numpy as np
from pathlib import Path

from src.utils.graphics import styled_print
import config as config

import pdb


class DataReader:
    def __init__(self, subject_id):
        styled_print('', 'Iniatilizing DataReader Class ', 'cyan', panel=True)
        styled_print('', f'Suhject:{subject_id}', 'green')
        self.subject_id = subject_id
        self.audio = None
        self.eeg = None
        self.words = None
        self.channels = None

        self._read_data()
        self._clean_eeg()
        self._electrode_shaft_referencing()
        
    def _read_data(self): 
        dir = config.DATA_DIR
        self.eeg = np.load(Path(dir, f'P{self.subject_id}_sEEG.npy'))
        self.audio = np.load(Path(dir, f'P{self.subject_id}_audio.npy'))
        self.words = np.load(Path(dir, f'P{self.subject_id}_stimuli.npy'))
        self.channels = np.load(Path(dir, f'P{self.subject_id}_channels.npy'))
        

    def _electrode_shaft_referencing(self):
        """
        Perform electrode shaft referencing by computing the mean signal
        for each shaft and subtracting it from the corresponding channels.
        """
        styled_print('', 'Electrode Shaft Rerefrencing sEEG', 'green')
        data_esr = np.zeros_like(self.eeg)

        shafts = {}
        for i, chan in enumerate(self.channels):
            shaft_name = chan[0].rstrip('0123456789')
            shafts.setdefault(shaft_name, []).append(i)

        shaft_averages = {
            shaft: np.mean(self.eeg[:, indices], axis=1, keepdims=True)
            for shaft, indices in shafts.items()
        }
        
        for i, chan in enumerate(self.channels):
            shaft_name = chan[0].rstrip('0123456789')
            data_esr[:, i] = self.eeg[:, i] - shaft_averages[shaft_name].squeeze()

        self.eeg = data_esr


    def _clean_eeg(self):
        styled_print('', 'Cleaning sEEG', 'green')
        exclude_prefixes = ('+', 'E', 'el')  # Unwanted channel prefixes
        valid_indices = [
            i for i, ch in enumerate(self.channels[:, 0])
            if not any(ch.startswith(prefix) for prefix in exclude_prefixes)
        ]
        clean_data = self.eeg[:, valid_indices]
        clean_channels = self.channels[valid_indices]

        self.eeg = clean_data.astype(np.float64)
        self.channels = clean_channels
        
        
