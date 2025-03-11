
from src.dataset.data_reader import DataReader

import pdb

reader = DataReader(subject_id='11')
pdb.set_trace()
eeg_features = reader.extract_freq_band_envelope()
audio_features = reader._audio_processor()