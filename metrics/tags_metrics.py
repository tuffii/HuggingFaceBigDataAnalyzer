from __future__ import annotations

import argparse
import ast
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

from db.connection import get_connection

DEFAULT_TIME_FREQ = "YS"

SELECT_SQL = """
SELECT model_id, tags, createdat
FROM hf_models
WHERE createdat IS NOT NULL AND tags IS NOT NULL
"""

DB_CONFIG = {
    "dbname": "postgres",
    "user": "BIGDATA",
    "password": "PASSWORD",
    "host": "localhost",
    "port": 5432,
}

def setup_russian_labels():
    import locale, matplotlib
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False
    for loc in ("ru_RU.UTF-8", "ru_RU", "Russian_Russia"):
        try:
            locale.setlocale(locale.LC_TIME, loc)
            break
        except locale.Error:
            pass

def parse_tags_field(raw_value: str) -> list[str]:
    """Парсинг поля tags в список строк."""
    if not raw_value:
        return []
    try:
        if isinstance(raw_value, list):
            return raw_value
        return ast.literal_eval(raw_value)
    except Exception:
        return []


def fetch_data(conn) -> pd.DataFrame:
    # Получение данных из базы
    with conn.cursor() as cur:
        cur.execute(SELECT_SQL)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["model_id", "tags", "createdat"])

    df = pd.DataFrame(rows, columns=["model_id", "tags", "createdat"])
    df["createdat"] = pd.to_datetime(df["createdat"], utc=True, errors="coerce")
    df["tags"] = df["tags"].apply(parse_tags_field)
    return df


def prepare_tags_time_series(df: pd.DataFrame, time_freq: str = DEFAULT_TIME_FREQ) -> Tuple[pd.DataFrame, int]:
    # Подготовка временного ряда по тегам
    df = df.copy()
    null_count = df["tags"].apply(lambda x: not x).sum()
    df = df[df["tags"].apply(lambda x: bool(x))]
    if df.empty:
        return pd.DataFrame(), null_count

    df_expanded = df.explode("tags").dropna(subset=["tags"])
    df_expanded["tags"] = df_expanded["tags"].astype(str)

    freq_map = {
        "YS": "Y", "Y-JAN": "Y",
        "MS": "M", "M": "M",
        "D": "D",
        "W-MON": "W", "W-SUN": "W",
    }
    period_freq = freq_map.get(time_freq.upper(), time_freq)

    df_expanded["period"] = (
        df_expanded["createdat"]
        .dt.tz_localize(None)
        .dt.to_period(period_freq)
        .dt.to_timestamp(how="start")
    )

    grouped = (
        df_expanded.groupby(["period", "tags"])
        .agg(count=("model_id", "count"))
        .reset_index()
    )

    pivot = grouped.pivot(index="period", columns="tags", values="count").fillna(0).sort_index()
    full_idx = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq=time_freq)
    pivot = pivot.reindex(full_idx, fill_value=0)
    pivot.index.name = "period"

    return pivot, null_count


def plot_stacked_tags(
        pivot_df,
        out_path: str,
        null_count: int,
        top_n: int = 15,
        title: str = "Модели по тегам во времени (стековая диаграмма)",
        dpi: int = 150,
        figsize=(14, 8),
        show_other: bool = True
):
    if pivot_df.empty:
        return

    col_sums = pivot_df.sum(axis=0).sort_values(ascending=False)
    top_cols = list(col_sums.iloc[:top_n].index)
    other_cols = [c for c in pivot_df.columns if c not in top_cols]

    df_top = pivot_df[top_cols].copy()
    other_count = pivot_df[other_cols].sum().sum() if other_cols else 0
    if show_other and other_count > 0:
        df_top["Другое"] = pivot_df[other_cols].sum(axis=1)

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    df_top.plot.area(ax=ax, stacked=True, alpha=0.9)

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Период", fontsize=12)
    ax.set_ylabel("Число моделей", fontsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Легенда
    handles, labels = ax.get_legend_handles_labels()
    if not show_other and other_count > 0:
        handles.append(mpatches.Patch(color="gray"))
        labels.append(f"Другое (всего {int(other_count):,})")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Теги (суммарно)", frameon=False)

    # Показать число моделей без тегов
    ax.text(1.02, -0.15, f"Пустые теги (не на графике): {null_count:,}", transform=ax.transAxes,
            fontsize=11, color="gray", ha="left", va="top")

    # Цифровой формат даты на оси X
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_aggregated_csv(pivot_df: pd.DataFrame, out_csv: str):
    # Сохранение агрегированного CSV
    pivot_df.to_csv(out_csv, index_label="period")


def main(out_png: str, out_csv: Optional[str], freq: str, top_n: int, show_other: bool):
    # Настройка русской локали и шрифтов
    setup_russian_labels()

    # Получение данных из базы
    conn = get_connection(DB_CONFIG)
    try:
        df = fetch_data(conn)
    finally:
        conn.close()

    if df.empty:
        return

    # Подготовка временного ряда по тегам
    pivot, null_count = prepare_tags_time_series(df, time_freq=freq)

    if out_csv and not pivot.empty:
        save_aggregated_csv(pivot, out_csv)

    if not pivot.empty:
        plot_stacked_tags(
            pivot_df=pivot,
            out_path=out_png,
            null_count=null_count,
            top_n=top_n,
            title=f"Модели по тегам во времени ({freq})",
            show_other=show_other
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot tag time series from HF models DB")
    parser.add_argument("--out", dest="out_png", required=True, help="Path to output PNG")
    parser.add_argument("--out-csv", dest="out_csv", default=None, help="Optional: save aggregated CSV")
    parser.add_argument("--freq", default="YS", help="Time frequency for aggregation (e.g. 'YS', 'MS', 'D')")
    parser.add_argument("--top", type=int, default=15, help="Top N tags to show")
    parser.add_argument("--no-other", action="store_true", help="Hide 'Other' category from chart but show its total in legend")
    args = parser.parse_args()

    main(args.out_png, args.out_csv, args.freq, args.top, show_other=not args.no_other)
