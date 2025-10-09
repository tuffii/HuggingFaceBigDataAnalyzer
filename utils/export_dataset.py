import psycopg2
import json
import gzip
import csv
from tqdm import tqdm
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    "dbname": "postgres",
    "user": "BIGDATA",
    "password": "PASSWORD",
    "host": "localhost",
    "port": 5432,
}

CHUNK_SIZE = 10_000
USE_GZIP = False

def default_serializer(obj):
    """Преобразует datetime и прочие нестандартные типы в строку."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)

def get_total_count(conn):
    """Подсчитывает общее количество записей в таблице."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM hf_models;")
        (count,) = cur.fetchone()
        return count

def export_sample_to_json(conn, output_path="hf_models_sample.json", limit=100):
    """Создает обзорный JSON-файл из первых N записей таблицы hf_models."""
    print(f"🔍 Создаём обзорный JSON-файл ({limit} записей)...")

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM hf_models LIMIT %s;", (limit,))
        rows = cur.fetchall()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=default_serializer)

    print(f"✅ Обзорный JSON создан: {output_path} ({len(rows)} записей)")

def export_to_jsonl(conn, output_path="hf_models.jsonl"):
    """Экспорт таблицы hf_models в JSONL с прогрессом и сжатием."""
    total_rows = get_total_count(conn)
    print(f"📦 Начинаем экспорт {total_rows:,} записей из таблицы hf_models...")

    open_func = gzip.open if USE_GZIP else open
    if USE_GZIP and not output_path.endswith(".gz"):
        output_path += ".gz"

    with conn.cursor(name="stream_cursor", cursor_factory=RealDictCursor) as cur:
        cur.itersize = CHUNK_SIZE
        cur.execute("SELECT * FROM hf_models;")
        with open_func(output_path, "wt", encoding="utf-8") as f, tqdm(total=total_rows, unit="rows") as pbar:
            for row in cur:
                json.dump(row, f, ensure_ascii=False, default=default_serializer)
                f.write("\n")
                pbar.update(1)

    print(f"✅ Экспорт завершён: {output_path}")

def export_to_csv(conn, output_path="hf_models.csv"):
    """Экспорт таблицы hf_models в CSV (для Excel/анализа)."""
    total_rows = get_total_count(conn)
    print(f"📦 Начинаем экспорт {total_rows:,} записей в CSV...")

    with conn.cursor(name="stream_cursor", cursor_factory=RealDictCursor) as cur:
        cur.itersize = CHUNK_SIZE
        cur.execute("SELECT * FROM hf_models;")
        with open(output_path, "w", newline='', encoding="utf-8") as f, tqdm(total=total_rows, unit="rows") as pbar:
            writer = None
            for row in cur:
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
                pbar.update(1)

    print(f"✅ CSV готов: {output_path}")

def main():
    print("🔗 Подключаемся к PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        # export_to_jsonl(conn)
        # export_to_csv(conn)
        export_sample_to_json(conn)
    finally:
        conn.close()
        print("🔒 Соединение с базой закрыто.")

if __name__ == "__main__":
    main()
