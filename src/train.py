from sklearn.linear_model import LinearRegression

def train_model(X, y):
    # Flatten input (LSTM format → ML format)
    X = X.reshape(X.shape[0], -1)

    model = LinearRegression()
    model.fit(X, y)

    return model