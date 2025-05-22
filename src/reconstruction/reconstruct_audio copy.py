import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

import config as config


class MelSpectrogramReconstructor:
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
        
        self.sample_rate = 16000
        self.frame_shift = config.FRAME_SHIFT
        self.n_fft = 2048
        self.hop_length = int(self.frame_shift * self.sample_rate)
        self.directory = Path(config.CUR_DIR, 'Audios')

    def reconstruct(self,subject_id,  mel_spec, griffin_lim_iter=60):
        """
        Reconstruct audio from a mel or log-mel spectrogram.

        Args:
            mel_spec (np.ndarray): Spectrogram of shape (frames, n_mels).
            output_path (str): Path to save the reconstructed audio file.
            griffin_lim_iter (int): Number of iterations for Griffin-Lim algorithm.

        Returns:
            np.ndarray: Reconstructed audio waveform.
        """
        mel_spec = mel_spec.T  # shape becomes (n_mels, time)

        if self.is_log_mel:
            mel_spec = np.exp(mel_spec)

        stft_mag = librosa.feature.inverse.mel_to_stft(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.n_fft
        )

        audio = librosa.griffinlim(
            stft_mag,
            n_iter=griffin_lim_iter,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )

        sf.write(output_path, audio, self.sample_rate)

        print(
            f"✅ Audio saved to: '{output_path}' | Duration: {len(audio) / self.sample_rate:.2f} seconds"
        )

        return audio
