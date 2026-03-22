def predict(model, data, scaler):
    data = data.reshape(data.shape[0], -1)
    predictions = model.predict(data)

    # Optional: reshape to match expected output
    return predictions