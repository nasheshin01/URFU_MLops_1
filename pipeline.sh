#!/bin/bash

# Создаем папки train и test
mkdir -p train
mkdir -p test

# Проверяем, был ли передан аргумент для количества наборов данных
if [ -z "$1" ]; then
    echo "Write number of datasets"
else
    num_datasets="$1"
fi

# Генерируем данные, обучаем модель и тестируем ее для каждого набора данных
for ((i=1; i<=$num_datasets; i++)); do
    echo "__________________"
    echo "Dataset $i"
    echo "__________________"
    
    # Генерируем данные
    python3 data_creation.py "$i"
    
    # Подготавливаем данные для обучения модели
    python3 model_preprocessing.py "$i"
    
    # Обучаем модель
    echo "__________________"
    echo "Train result"
    python3 model_preparation.py "$i"
    
    # Тестируем модель
    echo "__________________"
    echo "Test result"
    python3 model_testing.py "$i"
done