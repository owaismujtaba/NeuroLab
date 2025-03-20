from src.dataset.data_reader import DataReader
from src.dataset.data_loader import DataLoader

from src.model.model import NeuralNetwork, NeuroInceptDecoder, LinearRegressionModel
from src.model.trainer import ModelTrainer
from src.utils.utils import z_score_normalize

import config as config
import pdb


def training_pipeline(subject_id='21'):
    reader = DataReader(subject_id=subject_id)
    reader.eeg_features
    loader = DataLoader(data_reader=reader)
    audio_features, eeg_features = loader.get_audio_and_eeg_features()
    input_shape = (eeg_features.shape[1],)
    
    

    if config.TRAIN_MODEL == 'NeuralNetwork':
        model = NeuralNetwork(input_shape=input_shape)
        
        trainer = ModelTrainer(
            model_name='NeuralNetworkFinal',
            subject_id=subject_id
        )

    if config.TRAIN_MODEL == 'RNN':
        model = NeuroInceptDecoder(input_shape=input_shape)
    
        trainer = ModelTrainer(
            model_name='NeuroInceptDecoder',
            subject_id=subject_id
        )
    
    if config.TRAIN_MODEL == 'LR':
        model = LinearRegressionModel(input_shape=input_shape)

        trainer = ModelTrainer(
            model_name='LR',
            subject_id=subject_id
        )

    
    eeg_features = z_score_normalize(eeg_features)
    trainer.train_model(model=model, X=eeg_features, y=audio_features)

for subject in range(1, 31):
    if subject<10:
        subject = f'0{subject}'
    training_pipeline(subject_id=str(subject))
    