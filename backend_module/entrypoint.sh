#!/bin/bash

# включаем детальный вывод команд
set -x

# Функция для обработки ошибок:
error_handler() {
    echo "Произошла ошибка, но контейнер не завершает работу."
    # Можно запустить интерактивную оболочку для отладки
    exec bash
}

# Если любая команда вернёт ошибку, вызовется функция error_handler
trap 'error_handler' ERR

echo "Applying database migrations..."
python manage.py makemigrations main_app|| { echo "Ошибка при создании миграций"; exit 1; }
python manage.py migrate || { echo "Ошибка при применении миграций"; exit 1; }

echo "Starting server..."
python manage.py runserver 0.0.0.0:8000 || { echo "Ошибка при запуске сервера"; exit 1; }

# Если вы хотите, чтобы контейнер оставался запущенным даже после завершения основных команд,
# можно добавить запуск оболочки или команду, которая не завершится:
exec bash
