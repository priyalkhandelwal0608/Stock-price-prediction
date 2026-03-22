import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    data = df[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []

    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)

    # reshape for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler