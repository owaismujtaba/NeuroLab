from src.dataset.data_reader import DataReader
from src.dataset.data_loader import DataLoader

from src.model.model import NeuralNetwork
from src.model.trainer import ModelTrainer
import pdb


def training_pipeline(subject_id='21'):
    pdb.set_trace()
    reader = DataReader(subject_id=subject_id)
    reader.eeg_features
    loader = DataLoader(data_reader=reader)
    audio_features, eeg_features = loader.get_audio_and_eeg_features()
    input_shape = eeg_features.shape
    output_shape = audio_features.shape
    
    model = NeuralNetwork(
        input_shape=(None, input_shape),
        output_shape=(None, output_shape)
    )

    trainer = ModelTrainer(
        model_name='NeuralNetwrok',
        subject_id=subject_id
    )

training_pipeline(subject_id='21')