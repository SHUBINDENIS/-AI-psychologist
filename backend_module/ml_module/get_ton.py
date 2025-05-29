import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Загрузка модели и токенизатора один раз (глобально)
MODEL_NAME = "cointegrated/rubert-tiny-sentiment-balanced"
print("📥 Загрузка модели тональности...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def clean_text(text: str) -> str:
    """Очистка текста от ссылок, тегов, символов и цифр"""
    text = str(text).lower().strip()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def analyze_sentiment(text: str) -> tuple[str, float]:
    """Возвращает (тональность, кастомная вероятность)"""
    if not isinstance(text, str) or not text.strip():
        return "Нейтральная", 0.5
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        labels = ["Негативная", "Нейтральная", "Позитивная"]
        pred_label = labels[probs.argmax()]
        custom_prob = round(probs[1] * 0.5 + probs[2], 3)
        return pred_label, custom_prob
    except Exception as e:
        print(f"⚠️ Ошибка при анализе текста: {e}")
        return "Нейтральная", 0.5


def get_neg_neu_pos(df: pd.DataFrame, screen_name) -> pd.DataFrame:
    """Добавляет к DataFrame столбцы sentiment и custom_probability"""
    print("🚀 Запущена функция get_neg_neu_pos")

    df = df.copy()

    # Проверка на наличие столбца 'text'
    if 'text' not in df.columns:
        print("⚠️ Входной DataFrame не содержит столбца 'text'. Возвращаем нейтральный пример.")
        return pd.DataFrame([{
            'user_id': screen_name,
            'date': '',
            'type': 'post',
            'attachments': '',
            'text': '',
            'sentiment': 'Нейтральная',
            'custom_probability': 0.5
        }])

    df['text'] = df['text'].astype(str).apply(clean_text)
    df = df[df['text'].str.strip() != ''].dropna(subset=['text']).drop_duplicates(subset=['text'])

    print(f"📄 Обработка {len(df)} уникальных текстов...")

    if df.empty:
        print("⚠️ Нет текстов для анализа. Возвращаем один нейтральный пример.")
        return pd.DataFrame([{
            'user_id': screen_name,
            'date': '',
            'type': 'post',
            'attachments': '',
            'text': '',
            'sentiment': 'Нейтральная',
            'custom_probability': 0.5
        }])

    # Применение модели
    df[['sentiment', 'custom_probability']] = df['text'].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )

    df = df.dropna(subset=['sentiment', 'custom_probability'])

    print(f"✅ Обработка завершена. Осталось строк: {len(df)}")
    return df





