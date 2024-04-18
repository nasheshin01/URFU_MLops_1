import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

def main():
    data_index = sys.argv[1]
    # Загружаем обученную модель
    with open(f'trained_model_{data_index}.pkl', 'rb') as f:
        model = pickle.load(f)

    # Загружаем данные для тестирования
    test_df = pd.read_csv(f'test/preprocessed_data_{data_index}.csv')

    # Предсказываем значения на тестовых данных
    predictions = model.predict(test_df.drop('season', axis=1))

    # Загружаем истинные значения меток сезонов
    true_labels = pd.to_numeric(test_df['season'])

    # Рассчитываем метрики качества модели
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

if __name__ == "__main__":
    main()