import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import load_model
from data_loader import train_data
from model import Two_stream_Model
import math
import pickle
import os

def create_model(learn_rate= 0.0001):

    model = Two_stream_Model().model
    loss_fn = keras.losses.binary_crossentropy

    adam = keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss=loss_fn, optimizer=adam, metrics=['accuracy', 'mae'])
    return model

def train( num_epochs=200):

    X_train, Y_train, X_val, Y_val = train_data()

    model = create_model()
    checkpoint_1 = keras.callbacks.ModelCheckpoint('./model_check'+'_{epoch:02d}.h5', monitor='val_loss', save_best_only=False, period=10, save_weights_only=False)
    checkpoint_2 = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    history = model.fit(X_train, Y_train,
        batch_size=10,
        epochs=num_epochs,
        validation_data=(X_val, Y_val),
        shuffle=True,
        callbacks=[checkpoint_1,checkpoint_2],
        verbose=2)
    
    model.save('./two_stream.h5')

    return model



if __name__ == "__main__":

        model = train()