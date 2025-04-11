# 📈 Time Series Forecasting App (Streamlit)

This is a simple Streamlit web app for uploading a time series CSV file (e.g., daily sales data), visualizing it, and forecasting future values using ARIMA or Prophet models.

---

## 🔧 Features

- Upload your own CSV file (must include `date` and `sales` columns)
- View interactive time series charts
- Forecast future values (1 to 30 days)
- Uses ARIMA model (Prophet version optional)

---

## 📁 Example CSV Format

```csv
date,sales
2024-03-01,120
2024-03-02,135
2024-03-03,150
...
```
## 🧠 Notes
- Make sure your CSV uses ```YYYY-MM-DD``` for dates.
- ARIMA works well for many datasets, but Prophet adds better support for trends and seasonality.
- You can easily customize this app for other kinds of time series (temperature, stock prices, etc.).
---

# 🚀 Getting Start

## 1. Clone this Repo
```bash
git clone git@github.com:danujafernando/time-series-forecasting.git 
```

## 2. 📦 Install required packages
```bash 
pip3 install streamlit pandas matplotlib statsmodels scikit-learn
pip3 install prophet
```

## 3. ▶️ Run the app
```bash 
streamlit run app.py
```
or
```bash 
python3 -m streamlit run app.py
```

## 🎉 Have Fun!