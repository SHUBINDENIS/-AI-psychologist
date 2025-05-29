import logging
from ml_module.predict import make_predictions
from ml_module.get_ton import get_neg_neu_pos
from parsing_vk_links.transform_parsing_data import data_preprocessing
from parsing_vk_links.get_vk_data import parse_vk_data

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('parsing_plus_prediction.log')]
)

def parsing_plus_prediction(vk_links):
    """
    Функция для парсинга данных из VK и выполнения предсказаний с использованием ML-модели.

    Args:
        vk_links (list or str): Список ссылок на профили VK или одна ссылка в виде строки.

    Returns:
        list: Результаты предсказаний.
    """
    # Проверка входных данных
    if isinstance(vk_links, str):
        vk_links = [vk_links]
    elif not isinstance(vk_links, list) or not all(isinstance(link, str) for link in vk_links):
        logging.error("Invalid input: vk_links must be a string or a list of strings.")
        return []

    try:
        # Парсинг данных из VK
        logging.info("Starting VK data parsing...")
        main_df, wall_df = parse_vk_data(vk_links)
        logging.info("VK data parsing completed.")

        if main_df.empty and wall_df.empty:
            logging.warning("No data was parsed from the provided VK links.")
            return []

        # Обработка тональности постов
        logging.info("Processing sentiment analysis for wall posts...")
        wall_df = get_neg_neu_pos(wall_df, main_df['screen_name'][0])
        logging.info("Sentiment analysis completed.")

        # Предобработка данных
        logging.info("Preprocessing parsed data...")
        processed_data = data_preprocessing(main_df, wall_df)
        logging.info("Data preprocessing completed.")

        # Выполнение предсказаний
        logging.info("Making predictions with the ML model...")
        predictions = make_predictions(processed_data)
        model_result = [ round(100 * elem) for elem in predictions[:14] ]
        model_result.append(predictions[14])
        logging.info("Predictions completed.")

        # Вывод результатов
        logging.info(f"Predictions: {model_result}")
        return model_result

    except Exception as e:
        logging.error(f"An error occurred during parsing or prediction: {e}")
        return []
    
def convert_answers_in_targets(answers):
    # Преобразование текстовых ответов в числовые значения
    response_mapping = {
        "да": 4,
        "скорее да": 3,
        "скорее нет": 2,
        "нет": 1
    }

    # Извлечение данных из входного словаря
    vk_link = answers["received patient-vk link"]["vk-link"]
    responses = answers["received answers on questions"]

    # Преобразование ответов в числовые значения
    numeric_responses = {key: response_mapping[value] for key, value in responses.items()}

    # Вычисление значений для полей П7–П9 (зависящих от ответов на вопросы)
    P7 = (numeric_responses["response_3"] + numeric_responses["response_5"] +
          numeric_responses["response_18"] + numeric_responses["response_22"] +
          numeric_responses["response_34"] - 5) / 15  # Отстраненность

    P8 = (numeric_responses["response_1"] + numeric_responses["response_16"] +
          numeric_responses["response_23"] + numeric_responses["response_38"] +
          numeric_responses["response_39"] - 5) / 15  # Антагонизм

    P9 = (numeric_responses["response_8"] + numeric_responses["response_11"] +
          numeric_responses["response_26"] + numeric_responses["response_36"] +
          numeric_responses["response_42"] - 5) / 15  # Дезингуляция

    # Вычисление значений для полей П1–П3 (зависящих от других полей)
    extroversion = 1 - P7  # Экстраверсия
    agreeableness = 1 - P8  # Доброжелательность
    conscientiousness = 1 - P9  # Добросовестность

    neuroticism = (numeric_responses["response_6"] + numeric_responses["response_9"] +
                   numeric_responses["response_14"] + numeric_responses["response_33"] +
                   numeric_responses["response_41"] - 5) / 15  # Негативная эмоциональность

    openness = (numeric_responses["response_13"] + numeric_responses["response_24"] +
                numeric_responses["response_28"] + numeric_responses["response_31"] +
                numeric_responses["response_45"] - 5) / 15  # Открытость опыту

    negative_affectivity = (numeric_responses["response_6"] + numeric_responses["response_9"] +
                            numeric_responses["response_14"] + numeric_responses["response_33"] +
                            numeric_responses["response_41"] - 5) / 15  # Негативная аффективность

    detachment = P7  # Отстраненность (уже вычислено выше)

    antagonism = P8  # Антагонизм (уже вычислено выше)

    disinhibition = P9  # Дезингуляция (уже вычислено выше)

    psychoticism =  (numeric_responses["response_7"] + numeric_responses["response_15"] +
                     numeric_responses["response_17"] + numeric_responses["response_35"] +
                     numeric_responses["response_44"] - 5) / 15  # Психотизм


    # Вычисление значений для полей П11–П14 (логические условия)
    extraversion_introversion = "E" if extroversion >= 0.5 else "I"  # Экстраверсия – интроверсия
    logic_ethics = "T" if ((5 - numeric_responses["response_2"] + numeric_responses["response_20"] +
                            numeric_responses["response_27"] + 5 - numeric_responses["response_32"] +
                            numeric_responses["response_40"] - 5) / 15) >= 0.5 else "F"  # Логика – этика
    intuition_sensing = "N" if ((5 - numeric_responses["response_12"] + numeric_responses["response_19"] +
                                 numeric_responses["response_21"] + 5 - numeric_responses["response_29"] +
                                 numeric_responses["response_30"] - 5) / 15) >= 0.5 else "S"  # Интуиция – сенсорика
    rationality_irrationality = "J" if ((numeric_responses["response_4"] + numeric_responses["response_10"] +
                                         5 - numeric_responses["response_25"] + 5 - numeric_responses["response_37"] +
                                         numeric_responses["response_43"] - 5) / 15) >= 0.5 else "P"  # Рациональность – иррациональность
    

    # Формирование результата
    targets = {
        "extroversion": extroversion,
        "agreeableness": agreeableness,
        "conscientiousness": conscientiousness,
        "neuroticism": neuroticism,
        "openness": openness,
        "negative_affectivity": negative_affectivity,
        "detachment": detachment,
        "antagonism": antagonism,
        "disinhibition": disinhibition,
        "psychoticism": psychoticism,
        "extraversion_introversion": extraversion_introversion,
        "logic_ethics": logic_ethics,
        "intuition_sensing": intuition_sensing,
        "rationality_irrationality": rationality_irrationality
    }

    return vk_link, targets


