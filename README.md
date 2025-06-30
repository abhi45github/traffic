# 🚦 TrafficTelligence: Traffic Volume Prediction using Machine Learning

TrafficTelligence is an intelligent traffic volume forecasting system that uses machine learning to estimate and predict traffic flow based on historical weather and temporal data. Built with Python and Streamlit, it provides an interactive UI for real-time predictions.

---

## 📌 Features

- Predict traffic volume based on:
  - Weather conditions (clear, rain, snow, etc.)
  - Time-based features (hour, weekday, month)
- Interactive web UI using **Streamlit**
- Model trained using **Random Forest Regressor**
- CSV-based historical traffic data
- Real-time predictions with sliders and dropdowns

---

## 🧠 Technical Architecture

1. **Data Preprocessing**
   - Convert date-time into hour, weekday, month
   - One-hot encode categorical variables

2. **Model Training**
   - Use `RandomForestRegressor` from scikit-learn
   - Trained on cleaned traffic dataset

3. **UI and Prediction**
   - Streamlit app for user input and prediction display
   - Backend loads trained model to make predictions

---

## 📂 Project
