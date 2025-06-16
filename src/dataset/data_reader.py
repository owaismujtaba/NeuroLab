import numpy as np
from pathlib import Path
import numpy as np
import scipy.signal
from scipy import fftpack
from scipy.signal import  hilbert, decimate
import numpy.matlib as matlib
from mne.filter import filter_data
from sklearn.preprocessing import LabelEncoder

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

        self.target_sr = 16000

        self._read_data()
        self._clean_eeg()
        if config.REF == 'ESR':
            self._electrode_shaft_referencing()
        self._audio_processor()
        self.extract_eeg_and_audio_features()
        
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
        print(self.eeg.shape)

    def _clean_eeg(self):
        styled_print('', 'Cleaning sEEG', 'green')
        clean_data = []
        clean_channels = []
        channels = self.channels
        for i in range(channels.shape[0]):
            if '+' in channels[i][0]: #EKG/MRK/etc channels
                continue
            elif channels[i][0][0] == 'E': #Empty channels
                continue
            elif channels[i][0][:2] == 'el': #Empty channels
                continue
            elif channels[i][0][0] in ['F','C','T','O','P']: #Other channels
                continue        
            else:
                clean_channels.append(channels[i])
                clean_data.append(self.eeg[:,i])
        
        self.eeg =  np.transpose(np.array(clean_data,dtype="float64"))
        self.channels = clean_channels
        print(self.eeg.shape)

    def _audio_processor(self):
       
        audio = decimate(self.audio,int(config.AUDIO_SR / self.target_sr))
        self.audio = np.int16(audio/np.max(np.abs(audio)) * 32767)  
    
    def _name_vector(self, elecs):
        model_order = config.MODEL_ORDER
        names = matlib.repmat(
            elecs.astype(np.dtype(('U', 10))), 1, 2 * model_order + 1
        ).T

        for i, offset in enumerate(range(-model_order, model_order + 1)):
            names[i, :] = [e[0] + 'T' + str(offset) for e in elecs]

        return names.flatten()


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
        print(feat.shape)
        return feat
        
    def extract_mel_spectrograms(self,audio):  
        styled_print('', 'Audio Spectrograms', 'green')
          
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
        
        self.audio_spectrograms = spectrogram
        print(self.audio_spectrograms.shape)
        return spectrogram
    
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

        print(feat_stacked.shape)
        return feat_stacked

    def label_speech(self):
        spec_avg = np.mean(self.audio_spectrograms, axis=1)
        threshold = (np.max(spec_avg) + np.min(spec_avg)) * 0.45
        labels = np.where(spec_avg > threshold, 'Speech', 'Silence')
        le = LabelEncoder()
        le = le.fit(labels)
        self.speech_labels = le.transform(labels)

    def extract_eeg_and_audio_features(self):
        eeg_features = self.extract_freq_band_envelope()
        eeg_stacked_features = self.stack_features(eeg_features)

        audio_features = self.extract_mel_spectrograms(self.audio)
        self.label_speech()
        target = self.speech_labels[config.MODEL_ORDER*config.STEP_SIZE: self.speech_labels.shape[0]-config.MODEL_ORDER*config.STEP_SIZE]
        spectrograms =audio_features[config.MODEL_ORDER*config.STEP_SIZE: audio_features.shape[0]-config.MODEL_ORDER*config.STEP_SIZE]

        if spectrograms.shape[0]!=eeg_stacked_features.shape[0]:
            t_len = np.min([spectrograms.shape[0],eeg_stacked_features.shape[0]])
            spectrograms = spectrograms[:t_len,:]
            eeg_stacked_features = eeg_stacked_features[:t_len,:]
            target = target[:t_len]

        self.channels = np.array(self.channels)
        self.target = target
        self.eeg_features = eeg_stacked_features
        self.feature_names = self._name_vector(self.channels)
        self.channels = self.channels.flatten()
        self.audio_features = spectrograms
        
        
        
        return self.eeg_features, self.audio_features, self.feature_names, self.target, self.channels
        

        
        
