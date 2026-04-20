import pandas as pd

def create_time_series_features(df, lags=7, horizon=7):
    df = df.copy()

    # -----------------------------
    # Sort Data Properly
    # -----------------------------
    df = df.sort_values(by=['city_name', 'year', 'month', 'day'])

    # -----------------------------
    # 1. Lag Features
    # -----------------------------
    for lag in range(1, lags + 1):
        df[f'temp_lag_{lag}'] = df.groupby('city_name')['temperature_2m_max'].shift(lag)
        df[f'hum_lag_{lag}'] = df.groupby('city_name')['humidity_mean'].shift(lag)
        df[f'press_lag_{lag}'] = df.groupby('city_name')['pressure_mean'].shift(lag)

    # -----------------------------
    # 2. Rolling Features 🔥
    # -----------------------------
    df['temp_roll_mean_3'] = df.groupby('city_name')['temperature_2m_max'].transform(
        lambda x: x.rolling(3).mean()
    )

    df['temp_roll_std_3'] = df.groupby('city_name')['temperature_2m_max'].transform(
        lambda x: x.rolling(3).std()
    )

    df['hum_roll_mean_3'] = df.groupby('city_name')['humidity_mean'].transform(
        lambda x: x.rolling(3).mean()
    )

    # -----------------------------
    # 3. Difference Features 🔥
    # -----------------------------
    df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']

    df['pressure_change'] = df.groupby('city_name')['pressure_mean'].diff()

    # -----------------------------
    # 🔥 4. FUTURE TARGETS (FIXED)
    # -----------------------------
    for i in range(1, horizon + 1):

        # ✅ FIX: Use MAX temperature (IMPORTANT)
        df[f'target_temp_t+{i}'] = df.groupby('city_name')['temperature_2m_max'].shift(-i)

        df[f'target_rain_t+{i}'] = df.groupby('city_name')['rain'].shift(-i)

    # -----------------------------
    # Drop NaN
    # -----------------------------
    df.dropna(inplace=True)

    return df