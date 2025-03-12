from src.utils.graphics import styled_print



class DataLoader:
    def __init__(self, data_reader):
        styled_print('', 'Initializing DataLoader Class', 'cyan', panel=True)
        self.eeg_features = data_reader.eeg_features
        self.audio_features  = data_reader.audio_features
        self.feature_names = data_reader.feature_names
        self.target  = data_reader.target
        self.channels = data_reader.channels

    def get_audio_and_eeg_features(self):
        return self.audio_features, self.eeg_features



    

        