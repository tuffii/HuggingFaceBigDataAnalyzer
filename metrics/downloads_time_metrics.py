from __future__ import annotations
import argparse
from typing import Optional, Tuple
import locale

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from db.connection import get_connection

# Константы
DEFAULT_TIME_FREQ = "MS"
PLOT_DPI = 150
PLOT_SIZE = (14, 7)

SELECT_SQL = """
SELECT model_id, downloads, createdat
FROM hf_models
WHERE createdat IS NOT NULL
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
    df = pd.DataFrame(rows, columns=["model_id", "downloads", "createdat"])
    df["createdat"] = pd.to_datetime(df["createdat"], utc=True, errors="coerce")
    df["downloads"] = pd.to_numeric(df["downloads"], errors="coerce").fillna(0)
    return df


# Подготовка временного ряда
def prepare_time_series(df: pd.DataFrame, time_freq: str = DEFAULT_TIME_FREQ) -> Tuple[pd.DataFrame, int]:
    null_count = (df["downloads"] <= 0).sum()

    freq_map = {"MS": "M", "W-MON": "W", "W-SUN": "W", "D": "D", "Y": "Y"}
    period_freq = freq_map.get(time_freq.upper(), time_freq)

    df["period"] = (
        df["createdat"]
        .dt.tz_localize(None)
        .dt.to_period(period_freq)
        .dt.to_timestamp(how="start")
    )

    grouped = df.groupby("period", as_index=False)["downloads"].sum()
    pivot = grouped.set_index("period").sort_index()
    pivot = pivot.loc[pivot.index >= "2022-06"]
    full_idx = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq=time_freq)
    pivot = pivot.reindex(full_idx, fill_value=0)
    pivot.index.name = "period"

    return pivot, null_count


# График
def plot_time_series(pivot_df: pd.DataFrame, out_path: str, null_count: int):
    if pivot_df.empty:
        print("Нет данных для построения графика.")
        return

    plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
    ax = plt.gca()

    ax.plot(pivot_df.index, pivot_df["downloads"], color="steelblue", linewidth=2)
    ax.set_title("Динамика скачиваний моделей", fontsize=14)
    ax.set_xlabel("Период", fontsize=12)
    ax.set_ylabel("Сумма скачиваний", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")

    if null_count > 0:
        plt.gcf().text(
            0.985, 0.05,
            f"Моделей с 0 скачиваний: {null_count:,}",
            fontsize=9, color="gray", ha="right", va="bottom"
        )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=PLOT_DPI)
    plt.close()


# === Основная функция ===
def main(out_png: str, out_csv: Optional[str], freq: str):
    setup_russian_labels()
    conn = get_connection(DB_CONFIG)
    try:
        df = fetch_data(conn)
    finally:
        conn.close()

    if df.empty:
        print("Нет данных.")
        return

    pivot, null_count = prepare_time_series(df, freq)

    if out_csv:
        pivot.to_csv(out_csv, index_label="period")

    plot_time_series(pivot, out_png, null_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Динамика скачиваний моделей")
    parser.add_argument("--out", required=True, help="Путь к PNG")
    parser.add_argument("--out-csv", default=None, help="Путь к CSV (опционально)")
    parser.add_argument("--freq", default="MS", help="Частота ('MS','W-MON','D','Y')")
    args = parser.parse_args()
    main(args.out, args.out_csv, args.freq)
