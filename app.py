
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Prophet vs LSTM Forecasting", layout="wide")

st.title("📈 Prophet vs LSTM Time Series Forecasting")

st.markdown("Upload your own CSV file with columns `ds` (date) and `y` (value), or use our built-in sample dataset.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Custom data uploaded successfully!")
else:
    st.info("No file uploaded. Using sample dataset.")
    months = pd.date_range(start='2022-01-01', periods=24, freq='M')
    sales = [120, 130, 150, 140, 135, 145, 170, 165, 180, 250, 300, 320,
             125, 135, 155, 145, 140, 150, 175, 170, 185, 260, 310, 330]
    df = pd.DataFrame({'ds': months, 'y': sales})

# Display dataset
st.subheader("📊 Input Data")
st.write(df.tail())

# Forecast button
if st.button("Run Forecast"):
    st.subheader("🔮 Prophet Forecast")
    prophet = Prophet(yearly_seasonality=True)
    prophet.add_seasonality(name='flu_season', period=12, fourier_order=4)
    prophet.fit(df)
    future = prophet.make_future_dataframe(periods=3, freq='M')
    forecast_prophet = prophet.predict(future)

    st.line_chart(forecast_prophet[['ds', 'yhat']].set_index('ds').tail(12))

    st.subheader("🤖 LSTM Forecast")

    data = df['y'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    window = 6
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)

    # Forecast
    input_seq = scaled[-window:].reshape(1, window, 1)
    preds = []
    for _ in range(3):
        pred = model.predict(input_seq, verbose=0)[0]
        preds.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[pred]], axis=1)

    lstm_forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    # Plot
    future_months = pd.date_range(start=df['ds'].max() + pd.DateOffset(months=1), periods=3, freq='M')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['ds'], df['y'], label="Actual Sales")
    ax.plot(forecast_prophet['ds'], forecast_prophet['yhat'], '--', label="Prophet Forecast")
    ax.plot(future_months, lstm_forecast, ':o', label="LSTM Forecast")
    ax.legend()
    ax.set_title("Prophet vs LSTM Forecast")
    ax.grid(True)
    st.pyplot(fig)
