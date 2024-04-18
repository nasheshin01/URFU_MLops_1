import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import sys

def main():
    data_index = sys.argv[1]
    # Загружаем предобработанные данные для обучения
    train_df = pd.read_csv(f'train/preprocessed_data_{data_index}.csv')
    x = train_df.drop('season', axis=1)
    target = train_df['season']

    # Разделяем данные на признаки и целевую переменную
    X_train, X_val, y_train, y_val = train_test_split(x, target, test_size=0.2, random_state=42)

    # Создаем и обучаем модель логистической регрессии
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Оцениваем качество модели
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Сохраняем обученную модель
    with open(f'trained_model_{data_index}.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()