import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


def preprocess_data(df, scaler=None):
    print("🚀 Запущена функция preprocess_data")
    """
    Предобработка данных: удаление целевых столбцов и нормализация.

    :param df: DataFrame с входными данными
    :param scaler: Обученный MinMaxScaler (если None, создается новый)
    :return: Кортеж (нормализованные данные, scaler)
    """
    # Удаляем целевые столбцы (если они есть)
    target_columns = [f'real_class_{i + 1}' for i in range(14)]
    df_clean = df.drop(columns=[col for col in target_columns if col in df.columns], errors='ignore')

    # Преобразуем к числовому формату, если надо (мало ли)
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Нормализация данных
    X = scaler.transform(df_clean)

    print("✅ Завершена функция preprocess_data")
    return X, scaler
