from __future__ import annotations
import argparse
from typing import Optional
import pandas as pd
from db.connection import get_connection

DB_CONFIG = {
    "dbname": "postgres",
    "user": "BIGDATA",
    "password": "PASSWORD",
    "host": "localhost",
    "port": 5432,
}

SELECT_SQL = """
SELECT createdat, downloads
FROM hf_models
WHERE createdat IS NOT NULL
"""


# ---------------------------
# 1. Загрузка данных
# ---------------------------
def fetch_data(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(SELECT_SQL)
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame(columns=["createdat", "downloads"])

    df = pd.DataFrame(rows, columns=["createdat", "downloads"])
    df["createdat"] = pd.to_datetime(df["createdat"], utc=True, errors="coerce")
    df["downloads"] = pd.to_numeric(df["downloads"], errors="coerce").fillna(0)
    df = df.dropna(subset=["createdat"])
    return df


# ---------------------------
# 2. Агрегация по месяцам
# ---------------------------
def aggregate_downloads(df: pd.DataFrame, start="2022-06-01", end="2025-12-01") -> pd.DataFrame:
    df = df.copy()
    df["period"] = df["createdat"].dt.to_period("M").dt.to_timestamp(how="start")
    agg = df.groupby("period", as_index=False)["downloads"].sum().sort_values("period")

    # полный диапазон месяцев
    full_idx = pd.date_range(start=start, end=end, freq="MS")
    agg = agg.set_index("period").reindex(full_idx, fill_value=0)
    agg.index.name = "period"
    agg = agg.reset_index()
    return agg


# ---------------------------
# 3. Запись в PostgreSQL
# ---------------------------
def write_to_db(conn, df: pd.DataFrame):
    create_sql = """
    DROP TABLE IF EXISTS hf_downloads;
    CREATE TABLE hf_downloads (
        period DATE PRIMARY KEY,
        downloads BIGINT DEFAULT 0
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_sql)
        conn.commit()

    insert_sql = "INSERT INTO hf_downloads (period, downloads) VALUES (%s, %s)"
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            cur.execute(insert_sql, [row["period"], int(row["downloads"])])
    conn.commit()


# ---------------------------
# 4. Основная функция
# ---------------------------
def main(out_csv: Optional[str] = None):
    conn = get_connection(DB_CONFIG)
    try:
        df = fetch_data(conn)
    finally:
        conn.close()

    if df.empty:
        return

    agg_df = aggregate_downloads(df)

    # Сохраняем CSV
    if out_csv:
        agg_df.to_csv(out_csv, index=False)

    # Сохраняем в базу
    conn = get_connection(DB_CONFIG)
    try:
        write_to_db(conn, agg_df)
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Таблица суммарных скачиваний по месяцам")
    parser.add_argument("--out-csv", default=None, help="Путь для сохранения CSV")
    args = parser.parse_args()
    main(args.out_csv)
