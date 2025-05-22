
import numpy as np
from pathlib import Path
import config as config
import os
import glob
from src.utils.graphics import styled_print

class SpectrogramFoldLoader:
    def __init__(self, subject_id, model='LR'):
        styled_print("📊", "Initializing Spectrogram Loader Class", "yellow", panel=True)
        self.subject_id = subject_id
        self.model = model
        self.spectrogram_dir = Path(config.CUR_DIR, 'Predictions', str(self.subject_id), model)
        
        actual_filepaths = sorted(glob.glob(os.path.join(self.spectrogram_dir, 'Actual*.npy')))
        actual = [np.load(fp) for fp in actual_filepaths]
        self.actual_spectrograms = np.concatenate(actual, axis=0)

        predicted_filepaths = sorted(glob.glob(os.path.join(self.spectrogram_dir, 'Predictions*.npy')))
        predicted = [np.load(fp) for fp in predicted_filepaths]
        self.predicted_spectrograms = np.concatenate(predicted, axis=0)





