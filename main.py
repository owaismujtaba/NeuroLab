
from src.dataset.data_reader import DataReader


reader = DataReader(subject_id='11')

feat = reader.extract_freq_band_envelope()