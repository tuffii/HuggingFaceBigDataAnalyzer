from __future__ import annotations
import argparse
from typing import Optional
import pandas as pd
import psycopg2
from db.connection import get_connection


DB_CONFIG = {
    "dbname": "postgres",
    "user": "BIGDATA",
    "password": "PASSWORD",
    "host": "localhost",
    "port": 5432,
}

SELECT_SQL = """
SELECT model_id, license, createdat
FROM hf_models
WHERE createdat IS NOT NULL
"""


# -------------------------------------------------
# 1. Загрузка данных
# -------------------------------------------------
def fetch_data(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(SELECT_SQL)
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame(columns=["model_id", "license", "createdat"])

    df = pd.DataFrame(rows, columns=["model_id", "license", "createdat"])
    df["createdat"] = pd.to_datetime(df["createdat"], utc=True, errors="coerce")

    return df


# -------------------------------------------------
# 2. Агрегация
# -------------------------------------------------
def prepare_aggregation(df: pd.DataFrame, freq="MS", top_n=12) -> pd.DataFrame:
    df = df.copy()

    # нормализация лицензий
    df["license"] = (
        df["license"]
        .fillna("null")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # удаляем null / undefined / "" — идут в other
    bad_licenses = {"null", "undefined", "", "none"}
    df.loc[df["license"].isin(bad_licenses), "license"] = "other"

    # перевод даты в период
    freq_map = {"MS": "M", "W-MON": "W", "D": "D", "Y": "Y"}
    period_freq = freq_map.get(freq.upper(), freq)

    df["period"] = (
        df["createdat"]
        .dt.tz_localize(None)
        .dt.to_period(period_freq)
        .dt.to_timestamp(how="start")
    )

    grouped = (
        df.groupby(["period", "license"])
        .agg(count=("model_id", "count"))
        .reset_index()
    )

    pivot = (
        grouped
        .pivot(index="period", columns="license", values="count")
        .fillna(0)
        .sort_index()
    )

    # обеспечиваем отсутствие дырок по датам
    full_idx = pd.date_range(
        start=pivot.index.min(),
        end=pivot.index.max(),
        freq=freq
    )
    pivot = pivot.reindex(full_idx, fill_value=0)
    pivot.index.name = "period"

    # вычисляем топ
    totals = pivot.sum().sort_values(ascending=False)

    if len(totals) > top_n:
        top_cols = list(totals.iloc[:top_n].index)
        other_cols = [c for c in pivot.columns if c not in top_cols]

        pivot_top = pivot[top_cols].copy()
        pivot_top["other"] = pivot[other_cols].sum(axis=1)
        return pivot_top

    else:
        pivot["other"] = 0
        return pivot


# -------------------------------------------------
# 3. Запись в PostgreSQL
# -------------------------------------------------
def write_to_db(conn, pivot_df: pd.DataFrame):
    cols = list(pivot_df.columns)

    create_sql = f"""
    DROP TABLE IF EXISTS hf_license;
    CREATE TABLE hf_license (
        period DATE NOT NULL,
        {",".join(f'"{c}" INTEGER DEFAULT 0' for c in cols)}
    );
    """

    with conn.cursor() as cur:
        cur.execute(create_sql)
        conn.commit()

    insert_sql = f"""
    INSERT INTO hf_license
    (period, {",".join(f'"{c}"' for c in cols)})
    VALUES ({",".join(["%s"] * (len(cols) + 1))})
    """

    with conn.cursor() as cur:
        for period, row in pivot_df.iterrows():
            cur.execute(
                insert_sql,
                [period] + [int(row[c]) for c in cols]
            )
    conn.commit()


# -------------------------------------------------
# 4. Сохранение CSV
# -------------------------------------------------
def save_csv(pivot_df: pd.DataFrame, out_path: str):
    pivot_df.to_csv(out_path, index_label="period")


# -------------------------------------------------
# 5. Основной сценарий
# -------------------------------------------------
def main(freq: str, top_n: int, out_csv: Optional[str]):
    conn = get_connection(DB_CONFIG)

    try:
        df = fetch_data(conn)
    finally:
        conn.close()

    if df.empty:
        return

    pivot = prepare_aggregation(df, freq=freq, top_n=top_n)

    conn = get_connection(DB_CONFIG)
    try:
        write_to_db(conn, pivot)
    finally:
        conn.close()

    if out_csv:
        save_csv(pivot, out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Таблица top-12 лицензий по периодам")
    parser.add_argument("--freq", default="MS", help="MS / W-MON / D / Y")
    parser.add_argument("--top", type=int, default=12, help="Количество лицензий в таблице")
    parser.add_argument("--out-csv", default=None, help="Путь для экспорта CSV")

    args = parser.parse_args()

    main(args.freq, args.top, args.out_csv)
