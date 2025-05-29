import os
import sys
import logging
import tensorflow as tf
import pandas as pd
from django.conf import settings
from main_app.models import PatientAnalysis, Post, FormTargets
import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout, Add,
    LeakyReLU
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# Добавляем пути к модулям
sys.path.append(os.path.join(settings.BASE_DIR, 'parsing_vk_links'))
sys.path.append(os.path.join(settings.BASE_DIR, 'ml_module'))

# Импортируем необходимые функции
from parsing_vk_links.get_vk_data import parse_vk_data
from ml_module.get_ton import get_neg_neu_pos

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_module/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def prepare_training_data():
    """Подготовка данных для обучения"""
    try:
        import pandas as pd

        # Получаем пациентов для переобучения модели
        recent_patients = FormTargets.objects.all()
        
        if not recent_patients.exists():
            logger.warning("Пациентов для переобучения модели нет")
            return None, None

        # Формируем список пациентов с необходимыми полями
        patients_list = []
        vk_links = []
        for patient in recent_patients:
            vk_raw = (patient.vk_link or '').rstrip('/')
            screen_name = vk_raw.split('/')[-1] if vk_raw else ''
            vk_links.append(patient.vk_link or '')
            patients_list.append({
                'cut_vk_link': screen_name,
                # Основные личностные шкалы
                'extroversion': patient.extroversion,
                'agreeableness': patient.agreeableness,
                'conscientiousness': patient.conscientiousness,
                'neuroticism': patient.neuroticism,
                'openness': patient.openness,
                'negative_affectivity': patient.negative_affectivity,
                'detachment': patient.detachment,
                'antagonism': patient.antagonism,
                'disinhibition': patient.disinhibition,
                'psychoticism': patient.psychoticism,
                # Классификационные признаки
                'extraversion_introversion': patient.extraversion_introversion,
                'logic_ethics': patient.logic_ethics,
                'intuition_sensing': patient.intuition_sensing,
                'rationality_irrationality': patient.rationality_irrationality,
            })

        if not vk_links:
            logger.warning("Нет ссылок на VK для обработки")
            return None, None

        # Получаем данные из VK через parse_vk_data
        main_df, wall_df = parse_vk_data(vk_links)
        
        if main_df is None or wall_df is None:
            logger.warning("Не удалось получить данные из VK")
            return None, None

        # Обрабатываем wall_df через get_neg_neu_pos
        processed_wall_df = get_neg_neu_pos(wall_df, main_df['screen_name'][0])
        
        if processed_wall_df is None:
            logger.warning("Не удалось обработать данные о постах")
            return None, None

        # Создаем DataFrame пациентов и объединяем с main_df по screen_name и cut_vk_link
        patients_df = pd.DataFrame(patients_list)
        merged_main_df = pd.merge(
            patients_df,
            main_df,
            left_on='cut_vk_link',
            right_on='screen_name',
            how='left'
        )

        logger.info(
            f"Успешно подготовлены данные: {len(merged_main_df)} записей профилей, "
            f"{len(processed_wall_df)} обработанных записей постов"
        )
        return merged_main_df, processed_wall_df

    except Exception as e:
        logger.error(f"Ошибка при подготовке данных: {str(e)}")
        return None, None



def preprocess_data(users, wall):
    wall = wall.copy()
    wall['attachments'] = wall['attachments'].fillna('repost')
    wall['text_length'] = wall['text'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)

    users['status_len'] = users['status'].apply(lambda x: len(x) if isinstance(x, str) else 0)

    df = users.merge(wall, left_on='screen_name', right_on='user_id', how='left')

    label_encoder = LabelEncoder()
    df['relation'] = df['relation'].fillna('не указано')
    df['relation'] = label_encoder.fit_transform(df['relation'])
    df['sentiment'] = df['sentiment'].fillna('Нейтральная')
    df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

    nonzero_percentages = df.apply(lambda col: (col != 0).mean())
    df = df.loc[:, nonzero_percentages[nonzero_percentages >= 0.5].index]

    df.rename(columns={'user_id_x': 'user_id'}, inplace=True)

    post_count_ton = df.groupby('user_id')['sentiment'].value_counts().unstack(fill_value=0)
    post_count_ton.columns = ['neg_sum', 'neutral_sum', 'pos_sum']

    custom_stats = df.groupby('user_id')['custom_probability'].agg(
        mean_custom_prob='mean',
        median_custom_prob='median',
        std_custom_prob='std'
    ).reset_index()

    df = df.merge(post_count_ton, on='user_id', how='left')
    df = df.merge(custom_stats, on='user_id', how='left')

    df['mean_text_length'] = df.groupby('user_id')['text_length'].transform('mean')
    df = df.drop('text_length', axis=1)

    label_encoder = LabelEncoder()
    df['sex'] = label_encoder.fit_transform(df['sex'])

    target_columns = [   
                         'extroversion',
                         'agreeableness',
                         'conscientiousness',
                         'neuroticism',
                         'openness',
                         'negative_affectivity',
                         'detachment',
                         'antagonism',
                         'disinhibition',
                         'psychoticism',
                         'extraversion_introversion',
                         'logic_ethics',
                         'intuition_sensing',
                         'rationality_irrationality'
                     ]
    rename_dict = {col: f'real_class_{i+1}' for i, col in enumerate(target_columns)}
    df = df.rename(columns=rename_dict)

    df['real_class_11'] = df['real_class_11'].apply(lambda x: 1 if x == 'E' else 0)
    df['real_class_12'] = df['real_class_12'].apply(lambda x: 1 if x == 'F' else 0)
    df['real_class_13'] = df['real_class_13'].apply(lambda x: 1 if x == 'S' else 0)
    df['real_class_14'] = df['real_class_14'].apply(lambda x: 1 if x == 'P' else 0)

    drop_cols = [  
                'photo_max_orig', 
                'first_name', 
                'last_name', 
                'bdate', 
                'country', 
                'city',
                'home_town', 
                'interests', 
                'books', 
                'tv', 
                'quotes', 
                'games', 
                'movies', 
                'activities',
                'music', 
                'site', 
                'screen_name', 
                'occupation_type', 
                'occupation_name', 
                'religion',
                'inspired_by', 
                'date', 
                'type',
                'mobile_phone ', 
                'home_phone  ', 
                'photos',
                'notes', 
                'gifts', 
                'groups', 
                'followers', 
                'university', 
                'political',
                'people_main', 
                'life_main', 
                'smoking', 
                'alcohol', 
                'user_id_x', 
                'attachments', 
                'text', 
                'user_id_y', 
                'sentiment', 
                'custom_probability',
                'status' # Надо обработать статус
                ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    columns_to_convert = [
                         'has_photo', 
                         'has_mobile', 
                         'can_see_all_posts', 
                         'can_see_audio',
                         'can_write_private_message', 
                         'can_send_friend_request', 
                         'albums',
                         'videos', 
                         'friends', 
                         'pages', 
                         'subscriptions',
                         'can_access_closed', 
                         'mean_text_length', 
                         'neg_sum',
                         'neutral_sum',
                         'pos_sum'
                         ]
    df.loc[:, columns_to_convert] = df[columns_to_convert].fillna(0).round().astype(int)

    for col in ['mean_custom_prob', 'median_custom_prob', 'std_custom_prob']:
        df[col] = df[col].fillna(df[col].mean())

    df = df.drop_duplicates().drop('user_id', axis=1)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df.drop([f'real_class_{i+1}' for i in range(14)], axis=1).values)
    joblib.dump(scaler, 'ml_module/saved_model/minmax_scaler.pkl')

    y = df[[f'real_class_{i+1}' for i in range(14)]].values

    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(input_dim):
    inputs = Input(shape=(input_dim,), name="Input")

    x = Dense(512, kernel_regularizer=l2(0.001), name="Dense_1")(inputs)
    x = BatchNormalization(name="BN_1")(x)
    x = LeakyReLU(alpha=0.1, name="LeakyReLU_1")(x)
    x = Dropout(0.3, name="Dropout_1")(x)

    residual = x
    x = Dense(512, kernel_regularizer=l2(0.001), name="Dense_2")(x)
    x = BatchNormalization(name="BN_2")(x)
    x = LeakyReLU(alpha=0.1, name="LeakyReLU_2")(x)
    x = Dropout(0.3, name="Dropout_2")(x)
    x = Dense(512, kernel_regularizer=l2(0.001), name="Dense_3")(x)
    x = BatchNormalization(name="BN_3")(x)
    x = Add(name="Add_Residual_1")([x, residual])
    x = LeakyReLU(alpha=0.1, name="LeakyReLU_Residual_1")(x)

    x = Dense(256, kernel_regularizer=l2(0.001), name="Dense_4")(x)
    x = BatchNormalization(name="BN_4")(x)
    x = LeakyReLU(alpha=0.1, name="LeakyReLU_4")(x)
    x = Dropout(0.3, name="Dropout_4")(x)

    residual = x
    x = Dense(256, kernel_regularizer=l2(0.001), name="Dense_5")(x)
    x = BatchNormalization(name="BN_5")(x)
    x = LeakyReLU(alpha=0.1, name="LeakyReLU_5")(x)
    x = Dropout(0.3, name="Dropout_5")(x)
    x = Dense(256, kernel_regularizer=l2(0.001), name="Dense_6")(x)
    x = BatchNormalization(name="BN_6")(x)
    x = Add(name="Add_Residual_2")([x, residual])
    x = LeakyReLU(alpha=0.1, name="LeakyReLU_Residual_2")(x)

    x = Dense(128, kernel_regularizer=l2(0.001), name="Dense_7")(x)
    x = BatchNormalization(name="BN_7")(x)
    x = LeakyReLU(alpha=0.1, name="LeakyReLU_7")(x)
    x = Dropout(0.2, name="Dropout_7")(x)

    x = Dense(64, kernel_regularizer=l2(0.001), name="Dense_8")(x)
    x = BatchNormalization(name="BN_8")(x)
    x = LeakyReLU(alpha=0.1, name="LeakyReLU_8")(x)
    x = Dropout(0.2, name="Dropout_8")(x)

    x = Dense(32, name="Dense_9")(x)
    x = BatchNormalization(name="BN_9")(x)
    x = LeakyReLU(alpha=0.1, name="LeakyReLU_9")(x)

    outputs = Dense(14, activation='sigmoid', name="Output")(x)

    return Model(inputs=inputs, outputs=outputs, name="ImprovedDeepRegressionModel")

def train_model(users, wall):
    print("Запущена функция train_model")
    X_train, X_test, y_train, y_test = preprocess_data(users, wall)

    model = build_model(X_train.shape[1])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_loss', verbose=1)

    print("🚀 Обучение модели...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )

    print("📊 Оценка модели...")
    loss, mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"\n🔍 Test Loss (MSE): {loss:.4f}\n🎯 Test MAE: {mae:.4f}")

    preds = model.predict(X_test)
    for i in range(y_test.shape[1]):
        mse = mean_squared_error(y_test[:, i], preds[:, i])
        rmse = np.sqrt(mse)
        mae_val = mean_absolute_error(y_test[:, i], preds[:, i])
        print(f"Target {i+1} --> RMSE: {rmse:.4f}, MAE: {mae_val:.4f}")

    model_save_path = "ml_module/saved_model/model.keras"
    model.save(model_save_path)
    print(f"💾 Модель сохранена в {model_save_path}")

    history_path = "ml_module/saved_model/model_training_history.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history.history, f, ensure_ascii=False, indent=4)
    print(f"📑 История обучения сохранена в {history_path}")
    print("Завершена функция train_model")


if __name__ == "__main__":
    users_df, wall_df = prepare_training_data()
    train_model(users_df, wall_df)
