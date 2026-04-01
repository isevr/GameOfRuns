import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os
from joblib import load


class SequenceOptimization:
    def __init__(self, model, encoder_dir):
        self.model = load_model(model)
        self.encoder_dir = encoder_dir
        
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
        """
        Converts the optimized numerical sequence back to original text labels.
        """
        # Reshape the optimized 'full' array back to (10 events, 11 features)
        # self.full is (10, 11, 1) from opt_loop
        flat_data = self.full.reshape(10, 11)
        decoded_rows = []

        # Iterate through each of the 10 events
        for event_idx in range(10):
            event_row = {}
            # Map each feature (column) to its corresponding encoder
            for col_idx, col_name in enumerate(columns):
                encoder_path = os.path.join(self.encoder_dir, f"{col_name}.joblib")
                
                if os.path.exists(encoder_path):
                    le = load(encoder_path)
                    val = int(flat_data[event_idx, col_idx])
                    # Handle cases where the opt_loop might produce an index 
                    # slightly out of the original encoder range due to clipping
                    try:
                        event_row[col_name] = le.inverse_transform([val])[0]
                    except:
                        event_row[col_name] = "Unknown"
                else:
                    event_row[col_name] = flat_data[event_idx, col_idx]
            
            decoded_rows.append(event_row)

        generated_rules = pd.DataFrame(decoded_rows)
        return generated_rules