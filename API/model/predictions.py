# Libraries
from sklearn.metrics import classification_report
import tensorflow as tf
tf.data.experimental.enable_debug_mode()
import pandas as pd
import numpy as np

# Load model and data
model = tf.keras.models.load_model('runs_predictor.h5')
X = pd.read_csv('data.csv')
del X['Unnamed: 0']
X = X.values.reshape(-1,20,10)
X = X / 255

# Print predictions
preds = model.predict(X)
y_pred = np.argmax(preds, axis = 1)
y_pred = np.where(y_pred == 1, 'run', 'no_run')

with open('predictions.txt', 'w') as f:
    for prediction in y_pred: 
        f.write(str(prediction) + '\n')