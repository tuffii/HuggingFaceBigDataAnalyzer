from __future__ import annotations
import argparse
import os
import json
import locale
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from tqdm import tqdm

from db.connection import get_connection
from utils.embedding_infer import PipelineTagInferencer
from dotenv import load_dotenv

# Загрузка токена из .env
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Константы
DEFAULT_TIME_FREQ = "MS"
PLOT_DPI = 150
PLOT_SIZE = (12, 7)
TOP_N_LINES = 12

_inferencer: Optional[PipelineTagInferencer] = None


def setup_russian_labels():
    """Настройка шрифтов для поддержки кириллицы."""
    import matplotlib
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False
    for loc in ("ru_RU.UTF-8", "ru_RU", "Russian_Russia"):
        try:
            locale.setlocale(locale.LC_TIME, loc)
            break
        except locale.Error:
            pass


def infer_pipeline_tag(tags: Optional[List[str]], model_id: str, other_fields: Dict[str, Any]) -> Optional[str]:
    """Восстановление pipeline_tag для записей с отсутствующим значением."""
    global _inferencer
    if _inferencer is None:
        _inferencer = PipelineTagInferencer()
    return _inferencer.infer_pipeline_tag(tags, model_id, other_fields)


# SQL-запрос и конфигурация БД
SELECT_SQL = """
SELECT model_id, pipeline_tag, tags, createdat
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


def fetch_data(conn) -> pd.DataFrame:
    """Получение данных из базы."""
    with conn.cursor() as cur:
        cur.execute(SELECT_SQL)
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame(columns=["model_id", "pipeline_tag", "tags", "createdat"])

    df = pd.DataFrame(rows, columns=["model_id", "pipeline_tag", "tags", "createdat"])
    df["createdat"] = pd.to_datetime(df["createdat"], utc=True, errors="coerce")

    def norm_tags(x):
        if x is None:
            return None
        if isinstance(x, list):
            return x
        try:
            parsed = json.loads(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return None

    df["tags"] = df["tags"].apply(norm_tags)
    return df


def prepare_time_series(df: pd.DataFrame, time_freq: str = DEFAULT_TIME_FREQ, fill_null_pipeline: bool = False) -> pd.DataFrame:
    """Агрегация данных по временным периодам и pipeline_tag."""
    df = df.copy()

    if fill_null_pipeline:
        tqdm.pandas(desc="Восстановление pipeline_tag")
        mask = df["pipeline_tag"].isna() | df["pipeline_tag"].eq("")
        df.loc[mask, "pipeline_tag"] = df.loc[mask].progress_apply(
            lambda row: infer_pipeline_tag(row["tags"], row["model_id"], row.to_dict()), axis=1
        )

    freq_map = {"MS": "M", "W-MON": "W", "W-SUN": "W", "D": "D"}
    period_freq = freq_map.get(time_freq.upper(), time_freq)

    df["period"] = (
        df["createdat"]
        .dt.tz_localize(None)
        .dt.to_period(period_freq)
        .dt.to_timestamp(how="start")
    )

    df["pipeline_tag_for_group"] = df["pipeline_tag"].fillna("NULL")

    grouped = (
        df.groupby(["period", "pipeline_tag_for_group"])
        .agg(count=("model_id", "count"))
        .reset_index()
    )

    pivot = (
        grouped.pivot(index="period", columns="pipeline_tag_for_group", values="count")
        .fillna(0)
        .sort_index()
    )

    full_idx = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq=time_freq)
    pivot = pivot.reindex(full_idx, fill_value=0)
    pivot.index.name = "period"

    return pivot


def plot_time_series(
    pivot_df: pd.DataFrame,
    out_path: str,
    top_n: int = TOP_N_LINES,
    title: str = "Модели по pipeline_tag",
    dpi: int = PLOT_DPI,
    figsize: Tuple[int, int] = PLOT_SIZE,
):
    """Построение и сохранение графика динамики моделей по pipeline_tag."""
    col_sums = pivot_df.sum(axis=0).sort_values(ascending=False)
    if col_sums.empty:
        return

    null_count = int(col_sums.get("NULL", 0))
    if "NULL" in col_sums:
        col_sums = col_sums.drop("NULL")

    top_cols = list(col_sums.iloc[:top_n].index)
    other_cols = [c for c in pivot_df.columns if c not in top_cols and c != "NULL"]

    df_top = pivot_df[top_cols].copy()
    if other_cols:
        df_top["Другое"] = pivot_df[other_cols].sum(axis=1)

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    for col in tqdm(df_top.columns, desc="Рисуем линии"):
        linewidth = 1.0 + (df_top[col].sum() / (df_top.sum().sum() + 1)) * 4.0
        ax.plot(df_top.index, df_top[col].values, label=f"{col} ({int(df_top[col].sum())})", linewidth=linewidth)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Период (начало)", fontsize=12)
    ax.set_ylabel("Число моделей", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune="lower"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # цифровой формат даты
    plt.xticks(rotation=45, ha="right")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9, title="pipeline_tag")
    plt.tight_layout()

    if null_count > 0:
        plt.text(
            1.02, 0.02,
            f"NULL (отсутствует pipeline_tag): {null_count:,}",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
            ha="left",
            va="bottom",
        )

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_aggregated_csv(pivot_df: pd.DataFrame, out_csv: str):
    """Сохранение агрегированных данных в CSV."""
    pivot_df.to_csv(out_csv, index_label="period")


def main(out_png: str, out_csv: Optional[str], freq: str, top_n: int, fill_null: bool):
    """Основной сценарий выполнения."""
    setup_russian_labels()

    # Получение данных из базы
    conn = get_connection(DB_CONFIG)
    try:
        df = fetch_data(conn)
    finally:
        conn.close()

    if df.empty:
        return

    # Агрегация данных
    pivot = prepare_time_series(df, time_freq=freq, fill_null_pipeline=fill_null)

    # Сохранение CSV
    if out_csv:
        save_aggregated_csv(pivot, out_csv)

    # Построение графика
    plot_time_series(pivot, out_png, top_n=top_n, title=f"Модели по pipeline_tag ({freq})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="График по pipeline_tag из БД HF")
    parser.add_argument("--out", dest="out_png", required=True, help="Путь к PNG")
    parser.add_argument("--out-csv", dest="out_csv", default=None, help="Опционально: сохранить агрегированный CSV")
    parser.add_argument("--freq", default="MS", help="Частота ('MS','W-MON','D')")
    parser.add_argument("--top", type=int, default=12, help="Топ N линий pipeline_tag")
    parser.add_argument("--fill-null", action="store_true", help="Восстановить отсутствующие pipeline_tag")
    args = parser.parse_args()

    main(args.out_png, args.out_csv, args.freq, args.top, args.fill_null)
