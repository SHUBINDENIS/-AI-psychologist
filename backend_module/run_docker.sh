#!/bin/bash
set -e

# Создание директории data, если она не существует
if [ ! -d "data" ]; then
    echo "Создаем директорию data..."
    mkdir data
fi

# Копирование базы данных в директорию data, если она там отсутствует
if [ ! -f "data/db.sqlite3" ] && [ -f "db.sqlite3" ]; then
    echo "Копируем базу данных в директорию data..."
    cp db.sqlite3 data/
fi

# Загрузка переменных из .env файла
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Проверка существования контейнера
if docker ps -a --format '{{.Names}}' | grep -q "^ai_psychologist_backend$"; then
    echo "Останавливаем существующий контейнер..."
    docker stop ai_psychologist_backend
    docker rm ai_psychologist_backend
fi

# Пересборка образа
echo "Пересобираем Docker образ..."
docker build -t ai_psychologist_backend .

# Запуск контейнера с переменными окружения
echo "Запускаем контейнер..."
docker run -d --name ai_psychologist_backend \
    -p 8000:8000 \
    -v "$(pwd)/data":/app/data \
    -e VK_TOKEN="$VK_TOKEN" \
    -e SECRET_KEY="$SECRET_KEY" \
    ai_psychologist_backend

# Проверка успешности запуска
echo "Проверяем статус контейнера..."
sleep 5
if docker ps | grep -q "ai_psychologist_backend"; then
    echo "Контейнер успешно запущен. Логи:"
    docker logs -f ai_psychologist_backend
else
    echo "Ошибка при запуске контейнера. Проверьте логи:"
    docker logs ai_psychologist_backend
    exit 1
fi

# Удаление всех файлов в контейнере (исключая примонтированную директорию data)
# echo "Удаляем все файлы в контейнере (за исключением data)..."
# docker exec ai_psychologist_backend sh -c 'find /app -mindepth 1 -maxdepth 1 -not -name "data" -exec rm -rf {} +'

# # Копирование всех файлов из текущей директории в контейнер
# echo "Копируем все файлы из текущей директории в контейнер..."
# # если в текущей директории есть файлы, которые не должны копироваться, добавьте .dockerignore
# docker cp . ai_psychologist_backend:/app
