from django.apps import AppConfig
from django.conf import settings
import threading
import schedule
import time
import logging
import os
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_module/scheduler.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_scheduler():
    """Запуск планировщика задач"""
    while True:
        schedule.run_pending()
        time.sleep(60)  # Проверяем каждую минуту

class DjandoConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main_app'

    def ready(self):
        """
        Метод вызывается при запуске приложения.
        Здесь мы настраиваем автоматическое обновление модели.
        """
        # Импортируем здесь, чтобы избежать циклических импортов
        from ml_module.train_model import train_model, prepare_training_data
        
        def scheduled_training():
            """Функция для выполнения переобучения модели"""
            try:
                logger.info("Запуск запланированного переобучения модели")
                users_df, wall_df = prepare_training_data()
                success = train_model(users_df, wall_df)
                
                if success:
                    logger.info("Переобучение модели успешно завершено")
                else:
                    logger.error("Переобучение модели не удалось")
                    
            except Exception as e:
                logger.error(f"Ошибка при выполнении запланированного переобучения: {str(e)}")

        # Настраиваем расписание: запуск каждый день в 3:00
        schedule.every().day.at("03:00").do(scheduled_training)
        
        # Запускаем планировщик в отдельном потоке
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Планировщик задач для переобучения модели успешно настроен")
