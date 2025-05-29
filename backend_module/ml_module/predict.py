import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from ml_module.preprocess import preprocess_data

# Пути к модели и scaler
MODEL_PATH = "ml_module/saved_model/model.keras"
SCALER_PATH = "ml_module/saved_model/minmax_scaler.pkl"

# Словарь для сопоставления кода типу личности
personality_types = {
    "ENTJ": "Предприниматель",
    "ESTJ": "Администратор",
    "ENFJ": "Наставник",
    "ESFJ": "Энтузиаст",
    "ENTP": "Искатель",
    "ENFP": "Советчик",
    "ESTP": "Маршал",
    "ESFP": "Политик",
    "INTJ": "Аналитик",
    "ISTJ": "Инспектр",
    "INFJ": "Гуманист",
    "ISFJ": "Хранитель",
    "INTP": "Критик",
    "INFP": "Лирик",
    "ISTP": "Мастер",
    "ISFP": "Посредник"
}


# Загрузка модели и scaler при импорте модуля
print("📦 Загрузка модели и scaler...")
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Функция для преобразования числа в букву
def prediction_to_letter(value, true_letter, false_letter):
    return true_letter if value >= 0.5 else false_letter


def make_predictions(input_data):
    """
    Выполняет предсказания на новых данных.

    :param input_data: DataFrame или массив с входными данными
    :return: Массив предсказаний
    """
    print("🚀 Запущена функция make_predictions")

    # Преобразуем в DataFrame при необходимости
    if isinstance(input_data, np.ndarray):
        input_data = pd.DataFrame(input_data)

    # Предобработка: получаем X (y нам не нужен)
    X, _ = preprocess_data(input_data, scaler)

    # Предсказания
    predictions = (model.predict(X))[0].tolist()

    print("✅ Завершена функция make_predictions")

    # Получаем последние 4 предсказания
    last_four_predictions = predictions[-4:]
    # Преобразуем предсказания в буквы
    letters = [
            prediction_to_letter(last_four_predictions[0], 'E', 'I'),  # Экстраверт/Интроверт
            prediction_to_letter(last_four_predictions[2], 'S', 'N'),  # Интуитивный/Сенсорный
            prediction_to_letter(last_four_predictions[1], 'F', 'T'),  # Логический/Эмоциональный
            prediction_to_letter(last_four_predictions[3], 'P', 'J')   # Плановый/Спонтанный
             ]

    # Формируем строку из 4 букв
    personality_code = ''.join(letters)

    
    # Получаем название типа личности по коду
    personality_name = personality_types.get(personality_code, "Неизвестный тип")

    # Добавляем результат в массив предсказаний
    predictions.append(personality_name + "(" + personality_code + ")")
    



    return predictions
