import json

from db.connection import get_connection
from db.schema import init_db
from metrics.collector import MetricsCollector
from metrics.report import ReportGenerator
from metrics.visualize import Visualizer
from scraper.fetch_models import fetch_all_models

DB_CONFIG = {
    "dbname": "postgres",
    "user": "BIGDATA",
    "password": "PASSWORD",
    "host": "localhost",
    "port": 5432,
}

HF_TOKEN = ""


# def main():
#     conn = get_connection(DB_CONFIG)
#     init_db(conn)
#     fetch_all_models(conn, HF_TOKEN)
#     conn.close()
#
#
# if __name__ == "__main__":
#     main()

def main():
    conn = get_connection(DB_CONFIG)
    init_db(conn)
    # fetch_all_models(conn, HF_TOKEN)  # запускайте отдельно, если нужно

    collector = MetricsCollector(conn)
    df = collector.load_full_models_df(limit=None)  # без лимита — берёт всю таблицу

    # проверка полей
    check = collector.required_fields_check(df)
    print("Проверка полей:", json.dumps(check, ensure_ascii=False, indent=2))

    numeric = collector.numeric_summary(df, top_likes_n=20)
    top_tags = collector.top_tags(df, top_n=50)
    license_trends = collector.license_trends_by_year(df, years=5)
    pipeline_trend = collector.pipeline_trend_last_n_years(df, n_years=5)
    clusters = collector.cluster_by_architecture(df, n_clusters=8)
    top_companies = collector.top_companies(df, top_n=30)
    domain_ts = collector.time_series_models_by_domain(df, years=5)

    visualizer = Visualizer()
    assets = {}
    # генерируем графики (если данные есть)
    try:
        assets['downloads_hist'] = visualizer.histogram_downloads(df)
    except Exception as e:
        print("Не удалось построить гистограмму downloads:", e)
    try:
        assets['pipeline_counts'] = visualizer.pipeline_bar(numeric.pipeline_counts or {})
    except Exception as e:
        print("Не удалось построить pipeline bar:", e)
    try:
        assets['top_tags'] = visualizer.top_tags_bar(top_tags)
    except Exception as e:
        print("Не удалось построить top tags:", e)
    try:
        assets['domain_trend'] = visualizer.time_series_stacked(domain_ts)
    except Exception as e:
        print("Не удалось построить time series:", e)
    try:
        assets['license_heatmap'] = visualizer.license_trend_heatmap(license_trends)
    except Exception as e:
        print("Не удалось построить license heatmap:", e)

    report = ReportGenerator()
    metrics_package = {
        "numeric_summary": numeric.to_dict(),
        "top_tags": top_tags,
        "license_trends": license_trends,
        "pipeline_trend": pipeline_trend,
        "clusters": clusters,
        "top_companies": top_companies,
        "domain_time_series": domain_ts,
        "fields_check": check
    }

    json_path = report.generate_json_report(metrics_package)
    html_path = report.generate_html_report(metrics_package, assets=assets)

    print("JSON report:", json_path)
    print("HTML report:", html_path)

    conn.close()

if __name__ == "__main__":
    main()