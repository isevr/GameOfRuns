# Necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
tf.data.experimental.enable_debug_mode()
import os
import matplotlib.pyplot as plt

def model_training(data, labels):
    epochs = 2
    X = pd.DataFrame(data)
    X = X.values.reshape(-1,11,10)
    X = X / 255
    y = pd.DataFrame(labels)
    y = keras.utils.to_categorical(y, 2)

    # Model training
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(50, (3, 3), strides=1, padding="same", activation="relu", input_shape=(11,10,1)))   
    model.add(keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(keras.layers.Conv2D(25,(3,3), strides=1, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=512, activation="relu"))
    model.add(keras.layers.Dense(units=2, activation="softmax"))

    model.compile(loss="binary_crossentropy", metrics=['accuracy','precision','recall','f1_score'], optimizer='rmsprop')

    history = model.fit(X, y, epochs=epochs, verbose=1, validation_split=0.2, callbacks=[early_stop])

    if not os.path.exists('static'):
        os.makedirs('static')

    # Save plots
    plt.figure(figsize=(12,5))

    plt.subplot(3,2,1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xticks([i for i in range(epochs)])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train','validation'],loc='upper right')
    # plt.savefig('static/plot_loss.png')

    plt.subplot(3,2,2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train','validation'],loc='lower right')
    plt.xticks([i for i in range(epochs)])
    # plt.savefig('static/plot_accuracy.png')

    plt.subplot(3,2,3)
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.xlabel('epochs')
    plt.ylabel('precision')
    plt.legend(['train','validation'],loc='lower right')
    plt.xticks([i for i in range(epochs)])
    # plt.savefig('static/plot_precision.png')

    plt.subplot(3,2,4)
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.xlabel('epochs')
    plt.ylabel('recall')
    plt.legend(['train','validation'],loc='lower right')
    plt.xticks([i for i in range(epochs)])
    # plt.savefig('static/plot_recall.png')

    plt.subplot(3,2,5)
    plt.plot(history.history['f1_score'])
    plt.plot(history.history['val_f1_score'])
    plt.xlabel('epochs')
    plt.ylabel('f1_score')
    plt.legend(['train','validation'],loc='lower right')
    plt.xticks([i for i in range(epochs)])
    # plt.savefig('static/plot_f1_score.png')

    plt.tight_layout()  
    plt.savefig('static/model_training_plots.png')
    # plt.close()

    # Saves model
    model.save('pretrained_models/runs_predictor.keras')
