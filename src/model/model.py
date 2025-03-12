import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pdb

early_stopping = EarlyStopping(
            monitor='val_loss',   # Monitor validation loss
            patience=20,    # Number of epochs to wait before stopping
            restore_best_weights=True,  # Restore best weights when stopping
            verbose=1
        )



class NeuralNetwork:
    def __init__(self, input_shape, output_shape):
        self.model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_shape)  # Single output for regression
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
    def train(self, X_train, y_train):
        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train = y_train.reshape(y_train.shape[0], -1)
        self.model.fit(X_train, y_train,
            batch_size=32, 
            epochs=1000, 
            validation_split=0.10,
            callbacks=[early_stopping]
        )