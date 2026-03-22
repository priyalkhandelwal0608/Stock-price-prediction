# =========================================
# Stock Prediction Web App (FINAL CLEAN)
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# =========================================
# UI
# =========================================
st.title("📈 Stock Price Prediction App")
st.write("Upload stock CSV file (must contain 'Close' column)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "Close" not in df.columns:
        st.error("CSV must contain 'Close' column")
    else:
        data = df[['Close']].values

        # Preprocessing
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], -1)

        # Train
        model = LinearRegression()
        model.fit(X, y)

        # Predictions
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        y_actual = scaler.inverse_transform(y.reshape(-1, 1))

        # Future input
        future_days = st.slider("Select Future Days", 1, 10, 3)

        last_60 = scaled_data[-60:]
        temp_input = last_60.flatten().tolist()

        future_preds = []
        for i in range(future_days):
            x_input = np.array(temp_input[-60:]).reshape(1, -1)
            pred = model.predict(x_input)[0]
            temp_input.append(pred)
            future_preds.append(pred)

        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

        # Trend
        latest_actual = y_actual[-1][0]
        latest_pred = future_preds[0][0]

        trend = "UP 📈" if latest_pred > latest_actual else "DOWN 📉"
        signal = "BUY 🟢" if latest_pred > latest_actual else "SELL 🔴"
        change = ((latest_pred - latest_actual) / latest_actual) * 100

        # Display
        st.subheader("📊 Prediction Summary")
        st.write(f"Trend: {trend}")
        st.write(f"Recommendation: {signal}")
        st.write(f"Expected Change: {change:.2f}%")

        st.subheader("🔮 Future Predictions")
        st.write(future_preds.flatten())

        # Graph
        st.subheader("📈 Stock Price Graph")

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(y_actual)), y_actual, label="Actual")
        plt.plot(range(len(predictions)), predictions, label="Predicted")

        future_x = np.arange(len(y_actual), len(y_actual) + future_days)
        plt.plot(future_x, future_preds, linestyle='dashed', marker='o', label="Future")

        plt.legend()
        plt.title("Stock Price Prediction")

        st.pyplot(plt)

else:
    st.info("👆 Upload a CSV file to start")