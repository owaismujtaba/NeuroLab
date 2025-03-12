
from src.dataset.data_reader import DataReader

import pdb

reader = DataReader(subject_id='21')

eeg_features, audio_features, feature_names, target, channels = reader.extract_eeg_and_audio_features()

pdb.set_trace()