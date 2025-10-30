from __future__ import annotations
import argparse
from typing import Optional, Tuple
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


def fetch_data(conn) -> pd.DataFrame:
    print("🔄 Fetching data from database...")
    with conn.cursor() as cur:
        cur.execute(SELECT_SQL)
        rows = cur.fetchall()
    print(f"✅ Retrieved {len(rows)} rows from DB.")
    if not rows:
        return pd.DataFrame(columns=["model_id", "license", "createdat"])

    df = pd.DataFrame(rows)
    print("🧩 Parsing timestamps...")
    df["createdat"] = pd.to_datetime(df["createdat"], utc=True, errors="coerce")
    return df


def prepare_time_series(df: pd.DataFrame, time_freq: str = DEFAULT_TIME_FREQ) -> Tuple[pd.DataFrame, int]:
    print("⏳ Preparing time series aggregation...")
    df = df.copy()

    # нормализуем лицензии
    df["license"] = df["license"].fillna("null").astype(str)

    # 👇 отделяем null
    null_count = (df["license"].str.lower() == "null").sum()

    # 👇 приводим к нижнему регистру для унификации "other"
    df["license"] = df["license"].str.lower()

    # 👇 группируем все варианты "other" в одну категорию
    df.loc[df["license"].str.startswith("other"), "license"] = "other"

    # убираем null (не нужно на графике)
    df = df[df["license"] != "null"]

    if df.empty:
        print("⚠️ No valid license data found.")
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

    print("✅ Aggregation complete.")
    return pivot, null_count



def plot_time_series(
    pivot_df: pd.DataFrame,
    out_path: str,
    null_count: int,
    top_n: int = TOP_N_LINES,
    title: str = "Models by license over time",
    dpi: int = PLOT_DPI,
    figsize: Tuple[int, int] = PLOT_SIZE
):
    print("🎨 Plotting license time series...")
    col_sums = pivot_df.sum(axis=0).sort_values(ascending=False)
    if col_sums.empty:
        print("⚠️ No data to plot.")
        return

    top_cols = list(col_sums.iloc[:top_n].index)
    df_top = pivot_df[top_cols].copy()

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    for col in tqdm(df_top.columns, desc="📈 Drawing lines"):
        linewidth = 1.0 + (df_top[col].sum() / (df_top.sum().sum() + 1)) * 4.0
        ax.plot(df_top.index, df_top[col].values, label=f"{col} ({int(df_top[col].sum())})", linewidth=linewidth)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Period (start)", fontsize=12)
    ax.set_ylabel("Number of models", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")

    # 👇 текст о количестве null лицензий
    if null_count > 0:
        null_text = f"null (not plotted): {null_count}"
        # переносим текст вниз, чтобы не пересекался с легендой
        plt.gca().text(
            1.02, 0.1, null_text,  # 👈 смещён вниз
            transform=plt.gca().transAxes,
            fontsize=10,
            va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9)
        )

    plt.tight_layout(rect=(0, 0, 0.85, 1))
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"✅ Plot saved to {out_path}")



def save_aggregated_csv(pivot_df: pd.DataFrame, out_csv: str):
    print("💾 Saving aggregated CSV...")
    pivot_df.to_csv(out_csv, index_label="period")
    print(f"✅ CSV saved to {out_csv}")


def main(out_png: str, out_csv: Optional[str], freq: str, top_n: int):
    print("🚀 Starting license statistics pipeline...")
    conn = get_connection(DB_CONFIG)
    try:
        df = fetch_data(conn)
    finally:
        conn.close()

    if df.empty:
        print("⚠️ No rows fetched from DB. Exiting.")
        return

    pivot, null_count = prepare_time_series(df, time_freq=freq)

    print(f"📊 Hidden license counts (not plotted):")

    if out_csv and not pivot.empty:
        save_aggregated_csv(pivot, out_csv)

    if not pivot.empty:
        plot_time_series(
            pivot,
            out_png,
            null_count,
            top_n=top_n,
            title=f"Models by license ({freq})"
        )

    print("🏁 License statistics pipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot license time series from HF models DB")
    parser.add_argument("--out", dest="out_png", required=True, help="Path to output PNG")
    parser.add_argument("--out-csv", dest="out_csv", default=None, help="Optional: save aggregated CSV")
    parser.add_argument("--freq", default="MS", help="Time frequency for aggregation (e.g. 'MS', 'W-MON', 'D', 'Y')")
    parser.add_argument("--top", type=int, default=12, help="Top N licenses to show")
    args = parser.parse_args()

    main(args.out_png, args.out_csv, args.freq, args.top)
