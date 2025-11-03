from __future__ import annotations
import argparse
from typing import Optional, Tuple
import locale

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from tqdm import tqdm
from db.connection import get_connection

DEFAULT_TIME_FREQ = "MS"
PLOT_DPI = 150
PLOT_SIZE = (12, 7)
TOP_N_LINES = 12

SELECT_SQL = """
SELECT model_id, license, createdat
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

# --- Русская локаль и кириллица на графиках ---
def setup_russian_labels():
    import matplotlib
    matplotlib.rcParams["font.family"] = "DejaVu Sans"   # есть в matplotlib, поддерживает кириллицу
    matplotlib.rcParams["axes.unicode_minus"] = False
    for loc in ("ru_RU.UTF-8", "ru_RU", "Russian_Russia"):
        try:
            locale.setlocale(locale.LC_TIME, loc)
            break
        except locale.Error:
            pass
# ------------------------------------------------

def fetch_data(conn) -> pd.DataFrame:
    print("🔄 Получаем данные из базы...")
    with conn.cursor() as cur:
        cur.execute(SELECT_SQL)
        rows = cur.fetchall()
    print(f"✅ Получено строк: {len(rows)}")
    if not rows:
        return pd.DataFrame(columns=["model_id", "license", "createdat"])

    df = pd.DataFrame(rows, columns=["model_id", "license", "createdat"])
    print("🧩 Парсим временные метки...")
    df["createdat"] = pd.to_datetime(df["createdat"], utc=True, errors="coerce")
    return df


def prepare_time_series(df: pd.DataFrame, time_freq: str = DEFAULT_TIME_FREQ) -> Tuple[pd.DataFrame, int]:
    print("⏳ Готовим агрегирование по времени...")
    df = df.copy()

    df["license"] = df["license"].fillna("null").astype(str)
    null_count = (df["license"].str.lower() == "null").sum()
    df["license"] = df["license"].str.lower()
    df.loc[df["license"].str.startswith("other"), "license"] = "other"
    df = df[df["license"] != "null"]

    if df.empty:
        print("⚠️ Нет валидных данных по лицензиям.")
        return pd.DataFrame(), null_count

    freq_map = {"MS": "M", "W-MON": "W", "W-SUN": "W", "D": "D", "Y": "Y"}
    period_freq = freq_map.get(time_freq.upper(), time_freq)

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

    pivot = grouped.pivot(index="period", columns="license", values="count").fillna(0).sort_index()

    full_idx = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq=time_freq)
    pivot = pivot.reindex(full_idx, fill_value=0)
    pivot.index.name = "period"

    print("✅ Агрегирование завершено.")
    return pivot, null_count


def plot_time_series(
    pivot_df: pd.DataFrame,
    out_path: str,
    null_count: int,
    top_n: int = TOP_N_LINES,
    title: str = "Модели по лицензиям",
    dpi: int = PLOT_DPI,
    figsize: Tuple[int, int] = (14, 7),
):
    print("🎨 Строим график по лицензиям...")
    col_sums = pivot_df.sum(axis=0).sort_values(ascending=False)
    if col_sums.empty:
        print("⚠️ Нет данных для построения графика.")
        return

    top_cols = list(col_sums.iloc[:top_n].index)
    df_top = pivot_df[top_cols].copy()

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    for col in tqdm(df_top.columns, desc="📈 Рисуем линии"):
        linewidth = 1.0 + (df_top[col].sum() / (df_top.sum().sum() + 1)) * 4.0
        ax.plot(df_top.index, df_top[col].values, label=f"{col} ({int(df_top[col].sum())})", linewidth=linewidth)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Период (начало)", fontsize=12)
    ax.set_ylabel("Число моделей", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune="lower"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))  # русские месяцы при рабочей локали
    plt.xticks(rotation=45, ha="right")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=9,
        title="Лицензия"
    )

    if null_count > 0:
        null_text = f"null (не на графике): {null_count:,}"
        plt.gcf().text(
            0.985,
            0.05,
            null_text,
            fontsize=9,
            color="gray",
            ha="right",
            va="bottom",
        )

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ График сохранён в {out_path} (null = {null_count:,})")


def save_aggregated_csv(pivot_df: pd.DataFrame, out_csv: str):
    print("💾 Сохраняем агрегированный CSV...")
    pivot_df.to_csv(out_csv, index_label="period")
    print(f"✅ CSV сохранён: {out_csv}")


def main(out_png: str, out_csv: Optional[str], freq: str, top_n: int):
    setup_russian_labels()  # <<< включаем кириллицу и русские месяцы

    print("🚀 Запуск пайплайна по лицензиям...")
    conn = get_connection(DB_CONFIG)
    try:
        df = fetch_data(conn)
    finally:
        conn.close()

    if df.empty:
        print("⚠️ Пустая выборка. Завершаем.")
        return

    pivot, null_count = prepare_time_series(df, time_freq=freq)

    if out_csv and not pivot.empty:
        save_aggregated_csv(pivot, out_csv)

    if not pivot.empty:
        plot_time_series(
            pivot,
            out_png,
            null_count,
            top_n=top_n,
            title=f"Модели по лицензиям ({freq})"
        )

    print("🏁 Готово.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="График динамики лицензий моделей (HF)")
    parser.add_argument("--out", dest="out_png", required=True, help="Путь к PNG")
    parser.add_argument("--out-csv", dest="out_csv", default=None, help="Опционально: путь к CSV")
    parser.add_argument("--freq", default="MS", help="Частота ('MS','W-MON','D','Y')")
    parser.add_argument("--top", type=int, default=12, help="Топ N лицензий на графике")
    args = parser.parse_args()

    main(args.out_png, args.out_csv, args.freq, args.top)
