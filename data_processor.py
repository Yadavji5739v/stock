import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

class DataProcessor:
    def __init__(self, file_path='dataset/indian_stocks_all_history.csv'):
        self.file_path = file_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self, ticker):
        """Loads data for a specific ticker and processes it."""
        # Using chunking or efficient loading if possible, but for a specific ticker, 
        # it's better to filter during read if we use a library like dask, 
        # but here we'll use pandas and filter.
        
        # Optimization: Read only necessary columns
        cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        
        # Read the whole file might be slow (260MB), let's see. 
        # In a real app we might use a database or parquet.
        df = pd.read_csv(self.file_path, usecols=cols)
        df = df[df['ticker'] == ticker].copy()
        
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True)
        
        return df

    def add_features(self, df):
        """Adds technical indicators."""
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        df['Returns'] = df['close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Drop rows with NaN values created by rolling windows
        df.dropna(inplace=True)
        return df

    def prepare_lstm_data(self, df, feature_col='close', window_size=60):
        """Scales data and creates windows for LSTM."""
        data = df[[feature_col]].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaled_data

if __name__ == "__main__":
    processor = DataProcessor()
    # Test with RELIANCE
    df = processor.load_data('RELIANCE')
    df = processor.add_features(df)
    print(f"Loaded {len(df)} rows for RELIANCE")
    print(df.head())
