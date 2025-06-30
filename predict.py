import joblib
import pandas as pd

# Load the trained model
model = joblib.load('traffic_model.pkl')

# Sample input for prediction (one row of test data)
data = {
    'holiday': 'None',
    'temp': 288.45,
    'rain_1h': 0.0,
    'snow_1h': 0.0,
    'clouds_all': 40,
    'weather_main': 'Clear',
    'weather_description': 'sky is clear',
    'hour': 8,
    'weekday': 2,
    'month': 5
}

# Convert to DataFrame
df = pd.DataFrame([data])

# Predict
prediction = model.predict(df)

print(f"🚗 Predicted Traffic Volume: {int(prediction[0])}")
