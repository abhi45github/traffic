import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# STEP 1: Load and preprocess the dataset
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path).sample(5000, random_state=42)  # use 5000 rows
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['weekday'] = df['date_time'].dt.weekday
    df['month'] = df['date_time'].dt.month
    df.drop(columns=['date_time'], inplace=True)

    X = df.drop(columns='traffic_volume')
    y = df['traffic_volume']

    cat_cols = ['holiday', 'weather_main', 'weather_description']
    
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], remainder='passthrough')

    return train_test_split(X, y, test_size=0.2, random_state=42), preprocessor

# STEP 2: Train model
def train_model(X_train, y_train, preprocessor):
    model = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    return model

# STEP 3: Evaluate model
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print("\n📊 Evaluation Results:")
    print(f"  MAE (Mean Absolute Error): {mae:.2f}")
    print(f"  R² Score: {r2:.4f}")

# STEP 4: Run if script is executed directly
if __name__ == "__main__":
    print("🟡 Training started...")

    FILE = "Metro_Interstate_Traffic_Volume.csv"  # File in same folder
    (X_train, X_test, y_train, y_test), preprocessor = load_and_preprocess(FILE)
    print("✅ Data loaded and preprocessed")

    model = train_model(X_train, y_train, preprocessor)
    print("✅ Model trained")

    evaluate_model(model, X_test, y_test)
import joblib
joblib.dump(model, 'traffic_model.pkl')
print("💾 Model saved as 'traffic_model.pkl'")
