import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score, classification_report
from xgboost import XGBRegressor, XGBClassifier
import joblib

from utils.time_series import create_time_series_features

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv('data/cleaned_weather.csv')

# Apply time-series transformation
df = create_time_series_features(df, lags=7, horizon=7)

# -----------------------------
# 2. Feature Selection 🔥 (UPDATED)
# -----------------------------
features = [col for col in df.columns if 'lag' in col or 'roll' in col] + [
    'temp_range',
    'pressure_change',
    'city_name',
    'month',
    'season',

    # 🔥 NEW IMPORTANT FEATURES
    'month_sin',
    'month_cos',
    'city_month_avg_temp'
]

X = df[features]

# -----------------------------
# 3. Train Models (t+1 → t+7)
# -----------------------------
for i in range(1, 8):
    print(f"\n🚀 Training for Day t+{i}")

    y_temp = df[f'target_temp_t+{i}']
    y_rain = df[f'target_rain_t+{i}']

    # Split
    X_train, X_test, y_temp_train, y_temp_test = train_test_split(
        X, y_temp, test_size=0.2, random_state=42
    )

    _, _, y_rain_train, y_rain_test = train_test_split(
        X, y_rain, test_size=0.2, random_state=42
    )

    #  Temperature Model 

    temp_model = XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    temp_model.fit(X_train, y_temp_train)

    temp_preds = temp_model.predict(X_test)
    print("🌡️ Temp MAE:", mean_absolute_error(y_temp_test, temp_preds))

    # 🌧️ Rain Model
    scale_pos_weight = len(y_rain_train[y_rain_train == 0]) / len(y_rain_train[y_rain_train == 1])

    rain_model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    )

    rain_model.fit(X_train, y_rain_train)

    # Probability predictions
    rain_probs = rain_model.predict_proba(X_test)[:, 1]

    # 🔥 Threshold tuning
    threshold = 0.4
    rain_preds = (rain_probs > threshold).astype(int)

    print("🌧️ Rain F1 Score:", f1_score(y_rain_test, rain_preds))
    print("📊 Classification Report:\n", classification_report(y_rain_test, rain_preds))

    # -----------------------------
    # Save Models
    # -----------------------------
    joblib.dump(temp_model, f'models/temp_t+{i}.pkl')
    joblib.dump(rain_model, f'models/rain_t+{i}.pkl')

print("\n✅ All models trained successfully!")