# HuggingFace Big Data Analyzer

Проект для сбора данных с HuggingFace Hub и анализа больших данных о моделях нейросетей

## Используемые инструменты
- **Python** — для скриптов сбора данных, расчета метрик и построения графиков  
- **Docker** — для поднятия окружения и сервисов  
- **PostgreSQL** — для хранения всех данных  
- **Grafana** — для визуализации метрик  

## Базы данных
В проекте используется несколько таблиц:  

| Таблица           | Описание |
|------------------|----------|
| `hf_models`      | Основная таблица со всеми данными моделей HuggingFace |
| `hf_architecture`| Данные по архитектурам моделей |
| `hf_downloads`   | Динамика скачиваний моделей |
| `hf_leaders`     | Топ-авторы по количеству скачиваний |
| `hf_license`     | Популярность лицензий |
| `hf_category`    | Категории и pipeline_tag моделей |  

Эти таблицы содержат очищенные данные, которые используются для построения графиков в Grafana

## Запуск проекта

1. **Поднять окружение через Docker**  

```bash
docker-compose up -d
````

2. **Выгрузить данные с HuggingFace**

```bash
python utils/download_dataset.py
```

3. **Построить графики и метрики с помощью Python**

```bash
python metrics/run_all_metrics.py
```

4. **Построить таблицы для Grafana**

```bash
python -m metrics.tables.run_all_tables
```

5. **Открыть Grafana**

* Перейти на `http://localhost:3000`
* Импортировать дашборд из `grafana/dashboards/HuggingFaceMetrics.json`
