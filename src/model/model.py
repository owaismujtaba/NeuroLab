import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, Model, Input
from sklearn.linear_model import LinearRegression

import pdb

import config as config
from src.utils.utils import calculate_pcc_spectrgorams

early_stopping = EarlyStopping(
    monitor='val_loss',   # Monitor validation loss
    patience=5,    # Number of epochs to wait before stopping
    restore_best_weights=True,  # Restore best weights when stopping
    verbose=1
)

class LSTMModel:
    def __init__(self, input_shape=(2247, 1), output_shape=(23,)):
        self.input_shape = (input_shape[0], 1)
        self.output_shape = output_shape

    def _create_model(self):
        self.model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(256, return_sequences=True),
            layers.LSTM(512, return_sequences=True),
            layers.LSTM(512),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.output_shape[0]) 
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
    def train(self, X_train, y_train):
        self._create_model()
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Ensure correct shape for RNN
        y_train = y_train.reshape(y_train.shape[0], -1)
        self.model.fit(X_train, y_train,
            batch_size=config.BATCH_SIZE, 
            epochs=config.EPOCHS, 
            validation_split=0.10,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )


class NeuroInceptDecoder:
    def __init__(self, input_shape=(2247, ), output_shape=23, gru_units=64):
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.gru_units = gru_units


    def _create_model(self):

        inputs = Input(shape=(self.input_shape[0], 1))   # Add channel dimension (F, 1)

        # **First Inception Module**
        x = self.inception_module(inputs)

        # **GRU Module**
        x = layers.GRU(self.gru_units, return_sequences=True)(x)
        x = layers.GRU(self.gru_units * 2, return_sequences=True)(x)
        x = layers.GRU(self.gru_units * 4, return_sequences=False)(x)

        # **Second Inception Module**
        x = layers.Reshape((x.shape[-1], 1))(x)  # Reshape to apply Conv1D
        x = self.inception_module(x)

        # **Flatten before Fully Connected Layers**
        x = layers.Flatten()(x)

        # **Dense Layers**
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)

        # **Output Layer**
        outputs = layers.Dense(self.output_shape, activation='linear')(x)

        # **Create Model**
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse')

    def inception_module(self, input_tensor):
        
        """Applies Inception-style convolutions to the input tensor."""
        conv_1x1 = layers.Conv1D(64, 1, padding='same', activation='relu')(input_tensor)
        conv_3x3 = layers.Conv1D(64, 3, padding='same', activation='relu')(input_tensor)
        conv_5x5 = layers.Conv1D(64, 5, padding='same', activation='relu')(input_tensor)

        max_pool = layers.MaxPooling1D(3, strides=1, padding='same')(input_tensor)
        max_pool_conv = layers.Conv1D(64, 1, padding='same', activation='relu')(max_pool)

        return layers.Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5, max_pool_conv])

    def train(self, X_train, y_train):
        self._create_model()
        X_train = X_train.reshape(X_train.shape[0], self.input_shape[0], 1)
        y_train = y_train.reshape(y_train.shape[0], -1)
        self.model.fit(X_train, y_train,
            batch_size=config.BATCH_SIZE, 
            epochs=config.EPOCHS, 
            validation_split=0.10,
            callbacks=[early_stopping]
        )


class NeuralNetwork:
    def __init__(self, input_shape=(2247, ), output_shape=(23)):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def _create_model(self):
        self.model = keras.Sequential([
            layers.Input(shape=self.input_shape),
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
            layers.Dense(self.output_shape) 
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
    def train(self, X_train, y_train):
        self._create_model()
        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train = y_train.reshape(y_train.shape[0], -1)
        self.model.fit(X_train, y_train,
            batch_size=config.BATCH_SIZE, 
            epochs=config.EPOCHS, 
            validation_split=0.10,
            callbacks=[early_stopping]
        )


class LinearRegressionModel:
    def __init__(self, input_shape=(2247,), output_shape=(23,)):
        self.input_shape = input_shape
        self.output_shape = output_shape
        

    def _create_model(self):
        self.model = LinearRegression()


    def train(self, X_train, y_train):
        self._create_model()
        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train = y_train.reshape(y_train.shape[0], -1)
        
        self.model.fit(X_train, y_train)   
        
    def predict(self, X_test):
        X_test = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict(X_test)