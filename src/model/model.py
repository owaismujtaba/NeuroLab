import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import pdb

import config as config
from src.utils.utils import calculate_pcc_spectrgorams

early_stopping = EarlyStopping(
    monitor='val_loss',   # Monitor validation loss
    patience=30,    # Number of epochs to wait before stopping
    restore_best_weights=True,  # Restore best weights when stopping
    verbose=1
)

class RNNModel:
    def __init__(self, input_shape=(2247,), output_units=23, rnn_units=256, dropout_rate=0.2):
       
        """
        Build a speech decoder model using RNN layers (GRU).
        
        Parameters:
            - input_shape: Shape of the input data (e.g., [time_steps, features])
            - output_units: The number of units in the output layer (e.g., number of speech features)
            - rnn_units: Number of units in the GRU layer (default is 256)
            - dropout_rate: Dropout rate to prevent overfitting (default is 0.2)
        
        Returns:
            - model: Keras model object
        """
        inputs = layers.Input(shape=input_shape)
        
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.GRU(rnn_units, return_sequences=True)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.GRU(rnn_units, return_sequences=True)(x)
        
        outputs = layers.Dense(output_units, activation='linear')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')  

        self.model = model

    def train(self, X_train, y_train):
        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train = y_train.reshape(y_train.shape[0], -1)
        self.model.fit(X_train, y_train,
            batch_size=32, 
            epochs=config.EPOCHS, 
            validation_split=0.10,
            callbacks=[early_stopping]
        ) 
       



class NeuralNetwork:
    def __init__(self, input_shape=(2247, ), output_shape=(23)):

        self.model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(output_shape) 
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
    def train(self, X_train, y_train):
        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train = y_train.reshape(y_train.shape[0], -1)
        self.model.fit(X_train, y_train,
            batch_size=32, 
            epochs=config.EPOCHS, 
            validation_split=0.10,
            callbacks=[early_stopping]
        )