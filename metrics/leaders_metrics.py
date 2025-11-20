from __future__ import annotations
import argparse
from typing import Optional
import locale
import json

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from db.connection import get_connection

# Константы
PLOT_DPI = 150
PLOT_SIZE = (14, 8)
TOP_N_AUTHORS = 31

SELECT_SQL = """
SELECT model_id, downloads, raw_data
FROM hf_models
WHERE downloads IS NOT NULL
"""

DB_CONFIG = {
    "dbname": "postgres",
    "user": "BIGDATA",
    "password": "PASSWORD",
    "host": "localhost",
    "port": 5432,
}


def setup_russian_labels():
    import matplotlib
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False
    for loc in ("ru_RU.UTF-8", "ru_RU", "Russian_Russia"):
        try:
            locale.setlocale(locale.LC_TIME, loc)
            break
        except locale.Error:
            pass


# Получение данных
def fetch_data(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(SELECT_SQL)
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["model_id", "downloads", "raw_data"])
    df["downloads"] = pd.to_numeric(df["downloads"], errors="coerce").fillna(0)

    # безопасное извлечение автора
    def extract_author(x):
        if not x:
            return None
        if isinstance(x, dict):
            return x.get("author")
        try:
            return json.loads(x).get("author")
        except (json.JSONDecodeError, TypeError):
            return None

    df["author"] = df["raw_data"].apply(extract_author)
    df = df.dropna(subset=["author"])
    return df



# График
def plot_top_authors(df: pd.DataFrame, out_path: str, top_n: int = TOP_N_AUTHORS):
    if df.empty:
        print("Нет данных для построения графика.")
        return

    grouped = (
        df.groupby("author", as_index=False)["downloads"]
        .sum()
        .sort_values("downloads", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
    ax = plt.gca()
    ax.barh(grouped["author"], grouped["downloads"], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Скачивания", fontsize=12)
    ax.set_ylabel("Автор", fontsize=12)
    ax.set_title(f"Топ-{top_n} авторов по скачиваниям", fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=PLOT_DPI)
    plt.close()


def save_csv(df: pd.DataFrame, out_csv: str, top_n: int = TOP_N_AUTHORS):
    grouped = (
        df.groupby("author", as_index=False)["downloads"]
        .sum()
        .sort_values("downloads", ascending=False)
        .head(top_n)
    )
    grouped.to_csv(out_csv, index=False)


def main(out_png: str, out_csv: Optional[str], top_n: int = TOP_N_AUTHORS):
    setup_russian_labels()
    conn = get_connection(DB_CONFIG)
    try:
        df = fetch_data(conn)
    finally:
        conn.close()

    if out_csv:
        save_csv(df, out_csv, top_n)
    plot_top_authors(df, out_png, top_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Топ авторов по скачиваниям моделей")
    parser.add_argument("--out", required=True, help="Путь к PNG")
    parser.add_argument("--out-csv", default=None, help="Путь к CSV (опционально)")
    parser.add_argument("--top", type=int, default=100, help="Топ авторов")
    args = parser.parse_args()
    main(args.out, args.out_csv, args.top)
