import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("traffic_model.pkl")

st.title("🚦 Traffic Volume Predictor")

# Input form
holiday = st.selectbox("Holiday", ['None', 'Labor Day', 'Christmas Day'])
weather_main = st.selectbox("Weather Main", ['Clear', 'Clouds', 'Rain', 'Snow'])
weather_description = st.selectbox("Weather Description", ['sky is clear', 'light rain', 'scattered clouds', 'mist'])

temp = st.slider("Temperature (K)", 260, 320, 290)
rain_1h = st.slider("Rain (last 1h)", 0.0, 10.0, 0.0)
snow_1h = st.slider("Snow (last 1h)", 0.0, 10.0, 0.0)
clouds_all = st.slider("Clouds (%)", 0, 100, 40)

hour = st.slider("Hour of Day", 0, 23, 8)
weekday = st.slider("Weekday (0=Mon, 6=Sun)", 0, 6, 2)
month = st.slider("Month", 1, 12, 6)

# When button is clicked
if st.button("Predict Traffic Volume"):
    input_data = pd.DataFrame([{
        'holiday': holiday,
        'weather_main': weather_main,
        'weather_description': weather_description,
        'temp': temp,
        'rain_1h': rain_1h,
        'snow_1h': snow_1h,
        'clouds_all': clouds_all,
        'hour': hour,
        'weekday': weekday,
        'month': month
    }])

    result = model.predict(input_data)
    st.success(f"🚗 Predicted Traffic Volume: {int(result[0])}")
