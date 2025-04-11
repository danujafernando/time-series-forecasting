import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

st.title("Time Series Forecasting App")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

decomposition_type = st.radio("Choose decomposition type", ["Additive", "Multiplicative"])

model_choice = st.selectbox("Choose a forecasting model", ["ARIMA",
"ETS", "Prophet"])

forecast_horizon = st.slider("Select forecast horizon (months)", 1, 24, 12)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Dataset:", df.head())

    time_col = st.selectbox("Select the time column", df.columns)
    target_col = st.selectbox("Select the value column", df.columns, 1)

    # Convert time column to datetime
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)

    try:
        result = seasonal_decompose(df[target_col], model=decomposition_type.lower(), period=12)
        st.write("Seasonal Decomposition")
        result.plot()
        st.pyplot(plt.gcf())
    except Exception as e:
        st.warning(f"Decomposition failed: {e}")

    st.subheader("Forecasting Results")
    if model_choice == "ARIMA":
        model = ARIMA(df[target_col], order=(1,1,1)).fit()
        forecast = model.forecast(steps=forecast_horizon)
        st.line_chart(forecast)
        st.write("ARIMA Summary:", model.summary())

    elif model_choice == "ETS":
        model = ExponentialSmoothing(df[target_col], trend="add", seasonal="add", seasonal_periods=12).fit()
        forecast = model.forecast(forecast_horizon)
        st.line_chart(forecast)
        st.write("ETS Summary:", model.summary())

    elif model_choice == "Prophet":
        prophet_df = df[[target_col]].reset_index().rename(columns={time_col: "ds", target_col: "y"})
        m = Prophet()
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=forecast_horizon, freq='M')
        forecast = m.predict(future)
        fig = m.plot(forecast)
        st.pyplot(fig)

    st.subheader("Evaluation Metrics")
    try:
        test_actual = df[target_col][-forecast_horizon:]
        if model_choice == "Prophet":
            forecast_values = forecast[['ds', 'yhat']].tail(forecast_horizon).set_index('ds')['yhat']
        else:
            forecast_values = forecast

        mae = mean_absolute_error(test_actual, forecast_values)
        rmse = np.sqrt(mean_squared_error(test_actual, forecast_values))
        st.metric("MAE", round(mae, 2))
        st.metric("RMSE", round(rmse, 2))
    except:
        st.info("Evaluation metrics not available (not enough data).")