import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys

def preprocess_data(data):
    # Заменяем категориальный признак "season" на числовые значения
    season_mapping = {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}
    data['season'] = data['season'].map(season_mapping)
    
    # Применяем стандартизацию к числовым признакам
    scaler = StandardScaler()
    numeric_columns = ['temperature', 'precipitation', 'pressure']
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    return data

def main():
    data_index = sys.argv[1]
    # Загружаем данные для обучения
    train_df = pd.read_csv(f'train/weather_data_{data_index}.csv')
    
    # Применяем предварительную обработку данных для обучения
    preprocessed_train_df = preprocess_data(train_df)
    
    # Сохраняем предобработанные данные для обучения
    preprocessed_train_df.to_csv(f'train/preprocessed_data_{data_index}.csv', index=False)

    # Загружаем данные для тестирования
    test_df = pd.read_csv(f'test/weather_data_{data_index}.csv')
    
    # Применяем предварительную обработку данных для тестирования
    preprocessed_test_df = preprocess_data(test_df)
    
    # Сохраняем предобработанные данные для тестирования
    preprocessed_test_df.to_csv(f'test/preprocessed_data_{data_index}.csv', index=False)

if __name__ == "__main__":
    main()