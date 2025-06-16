from src.dataset.data_reader import DataReader
from src.dataset.data_loader import DataLoader

from src.model.model import NeuralNetwork, NeuroInceptDecoder, LinearRegressionModel, LSTMModel
from src.model.trainer import ModelTrainer
from src.utils.utils import z_score_normalize
from src.utils.graphics import styled_print
from src.reconstruction.reconstruct_audio import AudioReconstructor
from src.reconstruction.spectrograms_loader import SpectrogramFoldLoader

import config as config
import pdb


def training_pipeline(subject_id='21'):
    reader = DataReader(subject_id=subject_id)
    reader.eeg_features
    loader = DataLoader(data_reader=reader)
    audio_features, eeg_features = loader.get_audio_and_eeg_features()
    input_shape = (eeg_features.shape[1],)
    
    if config.TRAIN_MODEL == 'NeuroIncept':
        styled_print('', 'Using NN Model', 'cyan', panel=True)

        model = NeuroInceptDecoder(input_shape=input_shape, output_shape=23)
        
        trainer = ModelTrainer(
            model_name='NeuroIncept',
            subject_id=subject_id
        )

    if config.TRAIN_MODEL == 'NN':
        styled_print('', 'Using NN Model', 'cyan', panel=True)

        model = NeuralNetwork(input_shape=input_shape)
        
        trainer = ModelTrainer(
            model_name='NeuralNetwork_R_1',
            subject_id=subject_id
        )

    if config.TRAIN_MODEL == 'RNN':
        styled_print('', 'Using LR Model', 'cyan', panel=True)

        model = NeuroInceptDecoder(input_shape=input_shape)
    
        trainer = ModelTrainer(
            model_name='NeuroInceptDecoder',
            subject_id=subject_id
        )
    
    if config.TRAIN_MODEL == 'LR':
        styled_print('', 'Using LR Model', 'cyan', panel=True)
        model = LinearRegressionModel(input_shape=input_shape)

        trainer = ModelTrainer(
            model_name='LR',
            subject_id=subject_id
        )

    if config.TRAIN_MODEL == 'LSTM':
        styled_print('', 'Using LSTM Model', 'cyan', panel=True)
        model = LSTMModel(input_shape=input_shape)

        trainer = ModelTrainer(
            model_name='LSTM',
            subject_id=subject_id
        )

    
    eeg_features = z_score_normalize(eeg_features)
    trainer.train_model(model=model, X=eeg_features, y=audio_features)


if config.TRANING:
    for subject in range(1, 31):
        if subject<10:
            subject = f'0{subject}'
        training_pipeline(subject_id=str(subject))


if config.RECONSTRUCT_AUDIO:
    reconstructor = AudioReconstructor()
    
    for subject in range(1, 31):
        if subject<10:
            subject = f'0{subject}'
        data_loader = SpectrogramFoldLoader(subject_id=subject)
        reconstructor.reconstruct(
            subject_id=subject, 
            mel_spec=data_loader.actual_spectrograms, 
            type='actual'
        )

        reconstructor.reconstruct(
            subject_id=subject, 
            mel_spec=data_loader.predicted_spectrograms, 
            type='predicted'
        )
        break