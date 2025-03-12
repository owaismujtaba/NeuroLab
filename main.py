
from src.dataset.data_reader import DataReader

import pdb

reader = DataReader(subject_id='21')
eeg_features = reader.extract_eeg_and_audio_features()