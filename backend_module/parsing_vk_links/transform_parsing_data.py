from typing import List, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def data_preprocessing(users: pd.DataFrame, wall: pd.DataFrame) -> np.ndarray:
    print("🔧 Запущена функция data_preprocessing")

    # -------------------
    # 1. Обработка текстов и пропусков
    # -------------------
    wall = wall.copy()
    wall['attachments'] = wall['attachments'].fillna('repost')
    wall['text_length'] = wall['text'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)

    users['status_len'] = users['status'].apply(lambda x: len(x) if isinstance(x, str) else 0)

    # -------------------
    # 2. Объединение данных
    # -------------------
    print("📎 Объединение users и wall...")
    df = users.merge(wall, left_on='screen_name', right_on='user_id', how='left')

    # -------------------
    # 3. Кодирование категориальных признаков
    # -------------------
    print("🔤 Обработка категориальных признаков...")
    le = LabelEncoder()
    df['relation'] = le.fit_transform(df['relation'].fillna('не указано'))
    df['sentiment'] = le.fit_transform(df['sentiment'].fillna('Нейтральная'))

    # -------------------
    # 4. Создание новых признаков
    # -------------------
    print("🧮 Создание новых признаков...")

    df.rename(columns={'user_id_x': 'user_id'}, inplace=True)

    sentiment_counts = df.groupby('user_id')['sentiment'].value_counts().unstack(fill_value=0)
    sentiment_counts = sentiment_counts.reindex(columns=[-1, 0, 1], fill_value=0)
    sentiment_counts.columns = ['neg_sum', 'neutral_sum', 'pos_sum']

    prob_stats = df.groupby('user_id')['custom_probability'].agg(
        mean_custom_prob='mean',
        median_custom_prob='median',
        std_custom_prob='std'
    ).reset_index()

    df = df.merge(sentiment_counts, on='user_id', how='left')
    df = df.merge(prob_stats, on='user_id', how='left')

    df['mean_text_length'] = df.groupby('user_id')['text_length'].transform('mean')
    df.drop(columns='text_length', inplace=True)

    # -------------------
    # 5. Кодирование пола
    # -------------------
    df['sex'] = le.fit_transform(df['sex'])

    # -------------------
    # 6. Очистка и нормализация
    # -------------------
    print("🧹 Очистка и нормализация...")

    drop_cols = [
        'photo_max_orig', 'first_name', 'last_name', 'bdate', 'country', 'city',
        'home_town', 'interests', 'books', 'tv', 'quotes', 'games', 'movies', 'activities',
        'music', 'site', 'screen_name', 'occupation_type', 'occupation_name', 'religion',
        'inspired_by', 'date', 'type', 'mobile_phone ', 'home_phone  ', 'photos',
        'notes', 'gifts', 'groups', 'followers', 'university', 'political',
        'people_main', 'life_main', 'smoking', 'alcohol', 'user_id_y', 'attachments',
        'text', 'sentiment', 'custom_probability', 'status', 'user_id'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

    binarize_cols = [
        'has_photo', 'has_mobile', 'can_see_all_posts', 'can_see_audio',
        'can_write_private_message', 'can_send_friend_request', 'albums',
        'videos', 'friends', 'pages', 'subscriptions', 'can_access_closed',
        'mean_text_length', 'neg_sum', 'neutral_sum', 'pos_sum'
    ]
    df[binarize_cols] = df[binarize_cols].fillna(0).replace('', np.nan)
    df[binarize_cols] = df[binarize_cols].fillna(0).round().astype(int)

    fill_mean_cols = ['mean_custom_prob', 'median_custom_prob', 'std_custom_prob']
    for col in fill_mean_cols:
        df[col] = df[col].fillna(df[col].mean())

    df.drop(columns=['can_post', 'can_be_invited_group', 'audios', 'is_closed'], inplace=True, errors='ignore')
    df = df.drop_duplicates().fillna(df.mean(numeric_only=True))

    # -------------------
    # 7. Оставляем только нужные признаки и заполняем пропуски
    # -------------------

    required_columns = [
        'sex', 'relation', 'has_photo', 'has_mobile', 'can_see_all_posts',
        'can_see_audio', 'can_write_private_message', 'can_send_friend_request',
        'albums', 'videos', 'friends', 'pages', 'subscriptions',
        'can_access_closed', 'status_len',
        'neg_sum', 'neutral_sum', 'pos_sum', 'mean_custom_prob',
        'median_custom_prob', 'std_custom_prob', 'mean_text_length'
    ]

    # Проверим, все ли нужные признаки есть
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Предупреждение: отсутствуют следующие признаки и будут заполнены нулями: {missing_cols}")
        for col in missing_cols:
            df[col] = 0  # Добавляем недостающие колонки со значением 0

    # Оставим только нужные признаки и убедимся в порядке
    df = df[required_columns]

    # Заполним NaN на всякий случай
    df = df.fillna(0)

#     # -------------------
#     # 8. Масштабирование
#     # -------------------
#     scaler = MinMaxScaler()
#     print("✅ Завершена функция data_preprocessing")
#     return scaler.fit_transform(df)
    return df
