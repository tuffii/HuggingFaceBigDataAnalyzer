# HuggingFace Big Data Analyzer

## Аналитика моделей с Hugging Face Hub (Python + PostgreSQL + Grafana)

Этот репозиторий содержит инструменты для выгрузки, обработки и анализа больших данных о моделях с Hugging Face Hub.
Система собирает метаданные, рассчитывает метрики, агрегирует статистику и визуализирует динамику через Grafana.

### Сбор данных

Загрузка информации о моделях HuggingFace через API

Сохранение данных в PostgreSQL

Хранение: model_id, downloads, likes, author, task, library_name, createdAt, лицензия и др.

### Метрики и аналитика

Динамика скачиваний (по месяцам)

Топ-авторов по скачиваниям

Популярность лицензий

Топовые задачи и библиотеки

Метрики в CSV/PNG

Возможность запускать отдельные пайплайны

### Визуализация

Проект интегрирован с Grafana, где строятся интерактивные панели:

тренды скачиваний

пики спроса

сравнение авторов/моделей

популярность фреймворков

активность по датам

статистика моделей в разрезе задач / лицензий / организаций

Архитектура проекта

/db
   connection.py         — подключение к PostgreSQL
   init.sql              — структура таблиц

/metrics
   downloads_time_metrics.py       — динамика скачиваний
   downloads_authors_metrics.py    — топ авторов
   license_metrics.py              — анализ лицензий
   ...
/grafana
   dashboards.json        — (опционально) экспорт панели

/scripts
   fetch_hf_data.py       — загрузка данных из HuggingFace

README.md
pyproject.toml / requirements.txt


Требования

Python 3.10+

PostgreSQL 14+

Grafana 9+

API-доступ к HuggingFace (опционально)

Установка и запуск
1. Клонировать репозиторий
git clone https://github.com/username/HuggingFaceBigDataAnalyzer.git
cd HuggingFaceBigDataAnalyzer

2. Установить зависимости
pip install -r requirements.txt

3. Настроить базу PostgreSQL
Выполнить SQL-скрипт:
psql -U postgres -f db/init.sql

4. Загрузить данные с HuggingFace
python scripts/fetch_hf_data.py

5. Запустить расчёт метрик
python metrics/downloads_time_metrics.py \
    --out downloads.png \
    --out-csv downloads.csv

Графики в Grafana

1. Настроить PostgreSQL Datasource

2. Подключить вашу базу

3. Импортировать дашборд (если есть dashboards.json)

4. Использовать SQL-запросы вида:
SELECT date_trunc('month', createdAt) AS month,
       SUM(downloads) AS downloads
FROM hf_models
GROUP BY month
ORDER BY month;
