# Используем официальный образ Python 3.11 (полная версия для скорости и наличия всех инструментов)
FROM python:3.11.1

# Устанавливаем рабочую директорию
WORKDIR /app

# Обновляем пакетный менеджер и устанавливаем необходимые пакеты
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        sqlite3

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости Python без сохранения кэша для экономии места
RUN pip install -r requirements.txt

# Создаем директорию для базы данных
RUN mkdir -p /app/data

# Копируем entrypoint скрипт и делаем его исполняемым
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Копируем остальные файлы проекта
COPY . .

# Открываем порт для приложения
EXPOSE 8000

USER root

# Запускаем контейнер через entrypoint скрипт
ENTRYPOINT ["./entrypoint.sh"]