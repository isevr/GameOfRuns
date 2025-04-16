import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os
from joblib import load


class SequenceOptimization:
    def __init__(self, model):
        self.model = load_model(model)
        
    def loss(self, x_partial, x_missing):
        full_input = tf.concat([x_partial, x_missing], axis=0)  
        full_input = tf.expand_dims(full_input, axis=0)  # batch dim

        # model predictions
        predictions = self.model(full_input)

        # probability of the 'run' class
        run_probability = predictions[0, 1]

        # negative log probability 
        return -tf.math.log(run_probability + 1e-8)  # epsilon to avoid log(0)

    def confidence_metric(self, loss_value, length, max_len=10, alpha=1.0):
        normalized_length = length / max_len
        
        confidence = (1 / (1 + alpha * loss_value)) * normalized_length
        return confidence

    def opt_loop(self, partial_sequence, steps=500):
        
        partial_len = partial_sequence.shape[0]

        partial_sequence_reshaped = partial_sequence.reshape(partial_len, 11, 1)

        x_missing = tf.Variable(np.random.rand(10 - partial_len, 11, 1), dtype=tf.float32)

        optimizer = tf.optimizers.Adam(learning_rate=0.01)

        for step in range(steps):
            with tf.GradientTape() as tape:
                loss_value = self.loss(partial_sequence_reshaped, x_missing)

            grads = tape.gradient(loss_value, [x_missing])

            optimizer.apply_gradients(zip(grads, [x_missing]))

        missing = np.clip(np.round(x_missing.numpy()), 0, 3)
                
        self.full = np.concatenate([partial_sequence_reshaped, missing], axis=0)

        exp_full = np.expand_dims(self.full, axis=0)

        self.pred = np.argmax(self.model.predict(exp_full))

        self.confidence = self.confidence_metric(loss_value.numpy(), partial_len)
        
        return self.full, self.pred, self.confidence
    
    def decode(self, columns):

        full_df = pd.DataFrame(self.full.reshape(-1,110)).astype(int)
        full_df.columns = columns

        # TODO: add encoders loading

        rules = []
        for encoder, column in zip(encoders, full_df.columns):
            rules.append(encoder.inverse_transform(full_df[column])[0])

        generated_rules = pd.DataFrame(rules).T
        generated_rules.columns = columns

        return generated_rules
