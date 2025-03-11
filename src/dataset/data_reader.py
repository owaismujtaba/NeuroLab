import numpy as np
from pathlib import Path
import numpy as np
import scipy.signal
from scipy import fftpack
from scipy.signal import  hilbert, decimate
from mne.filter import filter_data

from src.utils.graphics import styled_print
from src.utils.mel_filterbank import MelFilterBank
import config as config

import pdb

hilbert3 = lambda x: hilbert(x, fftpack.next_fast_len(len(x)),axis=0)[:len(x)]


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
        
    def extract_freq_band_envelope(self):
        styled_print('', 'Extracting EEG Features', 'green')
        sr = config.EEG_SR
        data = scipy.signal.detrend(self.eeg, axis=0)

        data = filter_data(data.T, sr, 70, 170, method="iir").T
        data = filter_data(data.T, sr, 102, 98, method="iir").T
        data = filter_data(data.T, sr, 152, 148, method="iir").T

        data = np.abs(hilbert3(data))

        num_windows = int(np.floor((data.shape[0] - config.WIN_LENGHT * sr) / (config.FRAME_SHIFT * sr))) + 1
        feat = np.zeros((num_windows, data.shape[1]))

        for win in range(num_windows):
            start = int(np.floor(win * config.FRAME_SHIFT * sr))
            stop = int(start + config.WIN_LENGHT * sr)
            feat[win, :] = np.mean(data[start:stop, :], axis=0)
        self.eeg_freq_band_features = feat
        

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

    def _audio_processor(self):
        self.target_sr = 16000
        audio = decimate(self.audio,int(config.AUDIO_SR / self.target_sr))
        self.audio = np.int16(audio/np.max(np.abs(audio)) * 32767)  
    
    def extract_mel_spectrograms(self,audio):    
        num_windows=int(np.floor((audio.shape[0]-config.WIN_LENGHT*self.target_sr)/(config.FRAME_SHIFT*self.target_sr)))
        win = np.hanning(np.floor(config.WIN_LENGHT*self.target_sr + 1))[:-1]
        spectrogram = np.zeros((num_windows, int(np.floor(config.WIN_LENGHT*self.target_sr / 2 + 1))),dtype='complex')
        for w in range(num_windows):
            start_audio = int(np.floor((w*config.FRAME_SHIFT)*self.target_sr))
            stop_audio = int(np.floor(start_audio+config.WIN_LENGHT*self.target_sr))
            a = audio[start_audio:stop_audio]
            spec = np.fft.rfft(win*a)
            spectrogram[w,:] = spec
        mfb = MelFilterBank(spectrogram.shape[1], config.N_FILTERS, self.target_sr)
        spectrogram = np.abs(spectrogram)
        spectrogram = (mfb.toLogMels(spectrogram)).astype('float')

        '''
        if spectrogram.shape[0] != self.eeg_features.shape[0]:
            t_len = np.min([spectrogram.shape[0], self.eeg_features.shape[0]])
            spectrogram = spectrogram[:t_len, :]
            self.eeg_features = self.eeg_features[:t_len, :]
            target = target[:t_len]
        '''

        self.audio_spectrograms = spectrogram
    
    def stack_features(self, features):
        feat_stacked = np.zeros((
            features.shape[0] - (2 * config.MODEL_ORDER * config.STEP_SIZE),
            (2 * config.MODEL_ORDER + 1) * features.shape[1]
        ))

        for f_num, i in enumerate(
            range(config.MODEL_ORDER * config.STEP_SIZE, features.shape[0] - config.MODEL_ORDER * config.STEP_SIZE)
        ):
            ef = features[i - config.MODEL_ORDER * config.STEP_SIZE : i + config.MODEL_ORDER * config.STEP_SIZE + 1 : config.STEP_SIZE, :]
            feat_stacked[f_num, :] = ef.flatten()

        return feat_stacked

    def label_speech(self):
        spec_avg = np.mean(self.audio_spectrograms, axis=1)
        threshold = (np.max(spec_avg) + np.min(spec_avg)) * 0.45
        labels = np.where(spec_avg > threshold, 'Speech', 'Silence')
        
        self.speech_labels = labels

    def extract_eeg_and_audio_features(self):


        
        
