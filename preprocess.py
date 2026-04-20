import pandas as pd
import numpy as np

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv('data/weather.csv')

# -----------------------------
# 2. Basic Cleaning
# -----------------------------
df.drop_duplicates(inplace=True)

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# -----------------------------
# 3. Date Features
# -----------------------------
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# -----------------------------
# 🔥 CYCLICAL MONTH FEATURES (IMPORTANT)
# -----------------------------
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)


# Season Encoding
def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Summer
    elif month in [6, 7, 8, 9]:
        return 2  # Monsoon
    else:
        return 3  # Post-Monsoon

df['season'] = df['month'].apply(get_season)


#  CITY-MONTH AVG TEMP (VERY IMPORTANT)
df['city_month_avg_temp'] = df.groupby(
    ['city_name', 'month']
)['temperature_2m_max'].transform('mean')

# -----------------------------
# 4. Handle Missing Values
# -----------------------------
df = df.ffill()
df = df.bfill()

# -----------------------------
# 5. Rain Classification Target 🌧️
# -----------------------------
df['rain'] = df['rain_sum'].apply(lambda x: 1 if x > 0 else 0)

# -----------------------------
# 6. Encode City Name 🔥
# -----------------------------
df['city_name'] = df['city_name'].astype('category').cat.codes

# -----------------------------
# 7. Drop Unnecessary Columns
# -----------------------------
df.drop(['date'], axis=1, inplace=True)


# 8. REMOVE OUTLIERS
numeric_cols = [
    'temperature_2m_max',
    'temperature_2m_min',
    'pressure_mean',
    'humidity_mean',
    'wind_speed_10m_max'
]

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        (df[col] >= Q1 - 1.5 * IQR) &
        (df[col] <= Q3 + 1.5 * IQR)
    ]

# -----------------------------
# 9. Final Check
# -----------------------------
print("\n📊 Temperature Range After Cleaning:")
print("Min:", df['temperature_2m_max'].min())
print("Max:", df['temperature_2m_max'].max())

# -----------------------------
# 10. Save Clean Dataset
# -----------------------------
df.to_csv('data/cleaned_weather.csv', index=False)

print("\n✅ Preprocessing completed successfully!")
print("Shape:", df.shape)
print(df.head())