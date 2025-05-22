import numpy as np
import os
import librosa
import soundfile as sf
from pathlib import Path
from src.utils.graphics import styled_print
import config as config


class AudioReconstructor:
    """
    Reconstruct audio waveform from mel or log-mel spectrograms using the Griffin-Lim algorithm.
    """

    def __init__(self):
        """
        Initialize the reconstructor.

        Args:
            sample_rate (int): Sampling rate of the original audio.
            n_fft (int): FFT window size used during spectrogram creation.
            frame_shift (float): Frame shift in seconds (e.g., 0.01 for 10ms).
            is_log_mel (bool): Set to True if the spectrogram is log-mel scaled.
        """
        styled_print("📊", "Initializing Audio Reconstruction Class", "yellow", panel=True)

        self.sample_rate = 16000
        self.frame_shift = config.FRAME_SHIFT
        self.n_fft = 2048
        self.hop_length = int(self.frame_shift * self.sample_rate)
        self.directory = Path(config.CUR_DIR, 'Audios')

    def reconstruct(self,subject_id, mel_spec, griffin_lim_iter=30, type='actual'):
        """
        Reconstruct audio from a mel or log-mel spectrogram.

        Args:
            mel_spec (np.ndarray): Spectrogram of shape (frames, n_mels).
            output_path (str): Path to save the reconstructed audio file.
            griffin_lim_iter (int): Number of iterations for Griffin-Lim algorithm.

        Returns:
            np.ndarray: Reconstructed audio waveform.
        """
        styled_print("", "Reconstruction started", "green", panel=False)

        mel_spec = mel_spec.T  

        stft_mag = librosa.feature.inverse.mel_to_stft(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.n_fft
        )

        audio = librosa.griffinlim(
            stft_mag,
            n_iter=griffin_lim_iter,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            init='random',       # Faster convergence than zero init
            momentum=0.99        # Speed up convergence
        )

        os.makedirs(self.directory, exist_ok=True)
        filename = f'{type}_{subject_id}.wav'
        filepath = Path(self.directory, filename)
        sf.write(filepath, audio, self.sample_rate)

        print(
            f"✅ Audio saved to: '{filepath}' | Duration: {len(audio) / self.sample_rate:.2f} seconds"
        )

        
