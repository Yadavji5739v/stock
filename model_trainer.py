import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
import joblib

# Make tensorflow optional
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class ModelTrainer:
    def __init__(self, ticker, model_path=None):
        self.ticker = ticker
        self.model_path = model_path or f'models/{ticker}_lstm.h5'
        self.fallback_path = f'models/{ticker}_fallback.joblib'
        if not os.path.exists('models'):
            os.makedirs('models')

    def build_model(self, input_shape):
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_fallback(self, X_train, y_train):
        """Trains a RandomForest as a fallback if TensorFlow is missing."""
        print(f"Training fallback model for {self.ticker}...")
        # Flatten X for RandomForest (RandomForest doesn't need sequences like LSTM)
        X_flat = X_train.reshape(X_train.shape[0], -1)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_flat, y_train)
        joblib.dump(model, self.fallback_path)
        return model

    def load_model(self):
        # 1. Try Loading LSTM if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            if os.path.exists(self.model_path):
                try:
                    return tf.keras.models.load_model(self.model_path), "LSTM"
                except:
                    pass
        
        # 2. Try Loading Fallback if it exists
        if os.path.exists(self.fallback_path):
            try:
                return joblib.load(self.fallback_path), "Fallback (RF)"
            except:
                pass
                
        return None, None

    def predict(self, model, model_type, X):
        if model_type == "LSTM":
            return model.predict(X)
        elif model_type == "Fallback (RF)":
            X_flat = X.reshape(X.shape[0], -1)
            return model.predict(X_flat).reshape(-1, 1)
        return None
