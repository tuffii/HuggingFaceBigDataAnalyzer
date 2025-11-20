from __future__ import annotations
import json
import psycopg2
import pandas as pd
from datetime import datetime
from collections import defaultdict
from db.connection import get_connection


DB_CONFIG = {
    "dbname": "postgres",
    "user": "BIGDATA",
    "password": "PASSWORD",
    "host": "localhost",
    "port": 5432,
}


SELECT_SQL = """
SELECT model_id, pipeline_tag, tags, createdat
FROM hf_models
WHERE createdat IS NOT NULL
"""


def normalize_tags(x):
    if x is None:
        return None
    if isinstance(x, list):
        return x
    try:
        p = json.loads(x)
        if isinstance(p, list):
            return p
    except Exception:
        pass
    return None


def load_data(conn):
    with conn.cursor() as cur:
        cur.execute(SELECT_SQL)
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["model_id", "pipeline_tag", "tags", "createdat"])

    df["createdat"] = pd.to_datetime(df["createdat"], utc=True, errors="coerce")
    df["tags"] = df["tags"].apply(normalize_tags)
    df["pipeline_tag"] = df["pipeline_tag"].fillna("NULL").replace("", "NULL")
    return df


def prepare_aggregation(df: pd.DataFrame, top_n=12) -> pd.DataFrame:
    df = df.copy()

    df["period"] = (
        df["createdat"]
        .dt.tz_localize(None)
        .dt.to_period("M")
        .dt.to_timestamp(how="start")
    )

    grouped = (
        df.groupby(["period", "pipeline_tag"])
        .agg(count=("model_id", "count"))
        .reset_index()
    )

    pivot = (
        grouped
        .pivot(index="period", columns="pipeline_tag", values="count")
        .fillna(0)
        .sort_index()
    )

    # --------------------------------
    # Убираем столбец "NULL"
    # --------------------------------
    if "NULL" in pivot.columns:
        pivot = pivot.drop(columns=["NULL"])

    # --------------------------------------------
    # Оставляем TOP-12 колонок, остальные → other
    # --------------------------------------------
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

def write_to_db(conn, pivot_df: pd.DataFrame):
    cols = list(pivot_df.columns)

    create_sql = f"""
    DROP TABLE IF EXISTS hf_pipeline_tag;
    CREATE TABLE hf_pipeline_tag (
        period DATE NOT NULL,
        {",".join(f'"{c}" INTEGER DEFAULT 0' for c in cols)}
    );
    """

    with conn.cursor() as cur:
        cur.execute(create_sql)
        conn.commit()

    insert_sql = f"""
    INSERT INTO hf_pipeline_tag
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


def main():
    conn = get_connection(DB_CONFIG)
    try:
        df = load_data(conn)
        if df.empty:
            return

        pivot = prepare_aggregation(df, top_n=12)
        write_to_db(conn, pivot)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
