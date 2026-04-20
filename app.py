from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta

app = Flask(__name__)

# -----------------------------
# Load Models
# -----------------------------
temp_models = [joblib.load(f'models/temp_t+{i}.pkl') for i in range(1, 8)]
rain_models = [joblib.load(f'models/rain_t+{i}.pkl') for i in range(1, 8)]

# -----------------------------
# API KEY
# -----------------------------
API_KEY = "3f95279e13997fca87a626f138b96c4d"

# -----------------------------
# City Encoding
# -----------------------------
city_map = {
    "New Delhi": 0,
    "Mumbai": 1,
    "Bangalore": 2,
    "Kolkata": 3,
    "Chennai": 4
}

# -----------------------------
# Fetch Weather (SAFE)
# -----------------------------
def fetch_current_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    # 🔥 Error handling
    if 'main' not in data:
        raise Exception(f"API Error: {data.get('message', 'Unknown error')}")

    return {
        "temp": data['main']['temp_max'],
        "humidity": data['main']['humidity'],
        "pressure": data['main']['pressure']
    }

# -----------------------------
# Get Latest Data
# -----------------------------
def get_latest_data(city_encoded, city_name):
    df = pd.read_csv('data/cleaned_weather.csv')

    city_df = df[df['city_name'] == city_encoded]

    if len(city_df) < 7:
        raise Exception("Not enough historical data")

    city_df = city_df.tail(7).copy()

    api = fetch_current_weather(city_name)

    city_df.iloc[-1, city_df.columns.get_loc('temperature_2m_max')] = api['temp']
    city_df.iloc[-1, city_df.columns.get_loc('humidity_mean')] = api['humidity']
    city_df.iloc[-1, city_df.columns.get_loc('pressure_mean')] = api['pressure']

    latest = {}

    # Lag Features
    for i in range(1, 8):
        latest[f'temp_lag_{i}'] = city_df.iloc[-i]['temperature_2m_max']
        latest[f'hum_lag_{i}'] = city_df.iloc[-i]['humidity_mean']
        latest[f'press_lag_{i}'] = city_df.iloc[-i]['pressure_mean']

    # Rolling
    latest['temp_roll_mean_3'] = city_df['temperature_2m_max'].tail(3).mean()
    latest['temp_roll_std_3'] = city_df['temperature_2m_max'].tail(3).std()
    latest['hum_roll_mean_3'] = city_df['humidity_mean'].tail(3).mean()

    # Difference
    latest['temp_range'] = city_df.iloc[-1]['temperature_2m_max'] - city_df.iloc[-1]['temperature_2m_min']
    latest['pressure_change'] = city_df['pressure_mean'].iloc[-1] - city_df['pressure_mean'].iloc[-2]

    # Static
    latest['city_name'] = city_encoded
    latest['month'] = city_df.iloc[-1]['month']
    latest['season'] = city_df.iloc[-1]['season']

    # Cyclical
    month = latest['month']
    if isinstance(month, pd.Series):
        month = month.values[0]

    latest['month_sin'] = np.sin(2 * np.pi * month / 12)
    latest['month_cos'] = np.cos(2 * np.pi * month / 12)

    city_month_avg = df[
        (df['city_name'] == city_encoded) &
        (df['month'] == month)
    ]['temperature_2m_max'].mean()

    latest['city_month_avg_temp'] = city_month_avg

    return pd.DataFrame([latest])

# -----------------------------
# Label Logic
# -----------------------------
def get_weather_label(prob):
    if prob > 0.6:
        return "Heavy Rain 🌧️"
    elif prob > 0.4:
        return "Rain 🌦️"
    elif prob > 0.2:
        return "Cloudy ⛅"
    else:
        return "Sunny ☀️"

# -----------------------------
# Prediction (FINAL 🔥)
# -----------------------------
def predict_7_days(input_data):
    temp_preds = []
    rain_preds = []

    current_input = input_data.copy()

    for i in range(7):
        temp = temp_models[i].predict(current_input)[0]
        rain_prob = rain_models[i].predict_proba(current_input)[0][1]

        # 🔥 CLEAN + BOOST + VARIATION
        temp = float(temp)
        temp = temp + 2.5          # boost
        temp = temp + (i * 0.5)    # daily variation
        temp = round(temp, 1)

        temp_preds.append(temp)

        rain_preds.append({
            "prob": round(float(rain_prob * 100), 1),
            "label": get_weather_label(rain_prob)
        })

        # Shift lag
        for lag in range(7, 1, -1):
            current_input[f'temp_lag_{lag}'] = current_input[f'temp_lag_{lag-1}']

        current_input['temp_lag_1'] = temp

    return temp_preds, rain_preds

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        city = request.form['city']
        city_encoded = city_map.get(city, 0)

        input_data = get_latest_data(city_encoded, city)

        temp_preds, rain_preds = predict_7_days(input_data)

        today = datetime.now()
        dates = [(today + timedelta(days=i)).strftime("%d %b") for i in range(1, 8)]

        return render_template(
            'result.html',
            city=city,
            dates=dates,
            temp_preds=temp_preds,
            rain_preds=rain_preds
        )

    except Exception as e:
        return f"<h2>Error: {str(e)}</h2>"

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)