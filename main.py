from src.dataset.data_reader import DataReader
from src.dataset.data_loader import DataLoader

from src.model.model import NeuralNetwork, NeuroInceptDecoder
from src.model.trainer import ModelTrainer
from src.utils.utils import z_score_normalize
import pdb


def training_pipeline(subject_id='21'):
    reader = DataReader(subject_id=subject_id)
    reader.eeg_features
    loader = DataLoader(data_reader=reader)
    audio_features, eeg_features = loader.get_audio_and_eeg_features()
    input_shape = (eeg_features.shape[1],)
    
    #model = NeuroInceptDecoder(input_shape=input_shape)
    model = NeuralNetwork(input_shape=input_shape)
    '''
    trainer = ModelTrainer(
        model_name='NeuroInceptDecoder',
        subject_id=subject_id
    )
    '''

    trainer = ModelTrainer(
        model_name='Neural',
        subject_id=subject_id
    )
    '''eeg_features  = eeg_features.reshape(eeg_features.shape[0], eeg_features.shape[1], 1)'''
    eeg_features = z_score_normalize(eeg_features)
    trainer.train_model(model=model, X=eeg_features, y=audio_features)

for subject in range(21,31 ):
    training_pipeline(subject_id=str(subject))
    