from __future__ import annotations
import argparse
from typing import Optional
import pandas as pd
import json
from db.connection import get_connection

DB_CONFIG = {
    "dbname": "postgres",
    "user": "BIGDATA",
    "password": "PASSWORD",
    "host": "localhost",
    "port": 5432,
}

SELECT_SQL = """
SELECT model_id, downloads, raw_data
FROM hf_models
WHERE downloads IS NOT NULL
"""


# ---------------------------
# Утилиты
# ---------------------------
def _safe_json_to_dict(x) -> dict | None:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None


def extract_author(raw_data: dict | None) -> str:
    if raw_data and isinstance(raw_data, dict):
        a = raw_data.get("author")
        if isinstance(a, str) and a.strip():
            return a.strip()
    return "UNKNOWN"


# ---------------------------
# 1. Загрузка данных
# ---------------------------
def fetch_data(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(SELECT_SQL)
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame(columns=["model_id", "downloads", "raw_data", "author"])

    df = pd.DataFrame(rows, columns=["model_id", "downloads", "raw_data"])
    df["downloads"] = pd.to_numeric(df["downloads"], errors="coerce").fillna(0)
    df["raw_data"] = df["raw_data"].apply(_safe_json_to_dict)
    df["author"] = df["raw_data"].apply(extract_author)
    df = df.dropna(subset=["author"])
    return df


# ---------------------------
# 2. Построение топ-авторов
# ---------------------------
def build_authors(df: pd.DataFrame, top_n: int = 31) -> pd.DataFrame:
    grouped = (
        df.groupby("author", as_index=False)["downloads"]
        .sum()
        .sort_values("downloads", ascending=False)
        .head(top_n)
    )
    return grouped


# ---------------------------
# 3. Запись в PostgreSQL
# ---------------------------
def write_to_db(conn, df: pd.DataFrame):
    create_sql = """
    DROP TABLE IF EXISTS hf_leaders;
    CREATE TABLE hf_leaders (
        author TEXT PRIMARY KEY,
        downloads BIGINT DEFAULT 0
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_sql)
        conn.commit()

    insert_sql = "INSERT INTO hf_leaders (author, downloads) VALUES (%s, %s)"
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            cur.execute(insert_sql, [row["author"], int(row["downloads"])])
    conn.commit()


# ---------------------------
# 4. Основной сценарий
# ---------------------------
def main(top_n: int):
    conn = get_connection(DB_CONFIG)
    try:
        df = fetch_data(conn)
    finally:
        conn.close()

    if df.empty:
        return

    authors = build_authors(df, top_n=top_n)

    conn = get_connection(DB_CONFIG)
    try:
        write_to_db(conn, authors)
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Создание таблицы топ-авторов по скачиваниям моделей")
    parser.add_argument("--top", type=int, default=31, help="Количество топ авторов")
    args = parser.parse_args()
    main(args.top)
