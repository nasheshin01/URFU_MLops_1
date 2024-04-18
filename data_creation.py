import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys

def generate_weather_data(num_samples):
    # Создаем пустой DataFrame для хранения данных о погоде
    weather_df = pd.DataFrame()

    # Генерируем данные для месяца
    weather_df['month'] = np.random.randint(1, 13, num_samples)

    # Создаем метку для сезона на основе месяца
    weather_df['season'] = weather_df['month'].apply(lambda x: get_season(x))

    # Генерируем данные для температуры в зависимости от сезона
    weather_df['temperature'] = weather_df.apply(lambda row: generate_temperature(row['season']), axis=1)

    # Генерируем данные для осадков в зависимости от сезона
    weather_df['precipitation'] = weather_df.apply(lambda row: generate_precipitation(row['season']), axis=1)

    # Генерируем данные для давления в зависимости от сезона
    weather_df['pressure'] = weather_df.apply(lambda row: generate_pressure(row['season']), axis=1)

    # Вводим случайные аномалии для температуры, осадков и давления
    weather_df['temperature'] = introduce_anomalies(weather_df['temperature'], anomaly_range=(-10, 10))
    weather_df['precipitation'] = introduce_anomalies(weather_df['precipitation'], anomaly_range=(0, 5))
    weather_df['pressure'] = introduce_anomalies(weather_df['pressure'])

    return weather_df.drop('month', axis=1)

def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

def generate_temperature(season):
    # Генерируем температуру в зависимости от сезона
    if season == 'Spring':
        return np.random.normal(15, 5)
    elif season == 'Summer':
        return np.random.normal(25, 5)
    elif season == 'Autumn':
        return np.random.normal(10, 5)
    else:
        return np.random.normal(0, 5)

def generate_precipitation(season):
    # Генерируем осадки в зависимости от сезона
    if season == 'Spring':
        return np.random.choice([0.0, 1.0, 2.0], p=[0.6, 0.3, 0.1])
    elif season == 'Summer':
        return np.random.choice([0.0, 1.0, 2.0, 3.0], p=[0.7, 0.2, 0.05, 0.05])
    elif season == 'Autumn':
        return np.random.choice([0.0, 1.0, 2.0], p=[0.4, 0.4, 0.2])
    else:
        return np.random.choice([0.0, 1.0], p=[0.9, 0.1])

def generate_pressure(season):
    # Генерируем давление в зависимости от сезона
    if season == 'Spring':
        return np.random.normal(1013, 10)
    elif season == 'Summer':
        return np.random.normal(1010, 10)
    elif season == 'Autumn':
        return np.random.normal(1015, 10)
    else:
        return np.random.normal(1020, 10)

def introduce_anomalies(data, anomaly_probability=0.1, anomaly_range=(-20, 20)):
    # Вводим случайные аномалии в данные
    anomalies = np.random.choice([True, False], size=len(data), p=[anomaly_probability, 1-anomaly_probability])
    anomaly_values = np.random.uniform(anomaly_range[0], anomaly_range[1], size=np.sum(anomalies))
    data_copy = data.copy()  # явно копируем срез
    data_copy.loc[anomalies] += anomaly_values
    return data_copy.round(2)

def main():
    data_index = sys.argv[1]
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    # Генерируем данные о погоде
    weather_data = generate_weather_data(10000)

    # Разделяем данные на обучающий и тестовый наборы
    train_df, test_df = train_test_split(weather_data, test_size=0.2, random_state=42)

    # Сохраняем данные
    train_df.to_csv(f'train/weather_data_{data_index}.csv', index=False)
    test_df.to_csv(f'test/weather_data_{data_index}.csv', index=False)

if __name__ == "__main__":
    main()