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


def parse_tags_field(raw_value: str) -> list[str]:
    """
    Безопасно парсит колонку tags (возможные форматы:
    - '["a","b","c"]'
    - JSON-строка внутри кавычек
    - null или пустая строка)
    """
    if not raw_value:
        return []
    try:
        # Иногда это уже список или JSON внутри строки
        if isinstance(raw_value, list):
            return raw_value
        return ast.literal_eval(raw_value)
    except Exception:
        return []


def fetch_data(conn) -> pd.DataFrame:
    print("🔄 Fetching data from database...")
    with conn.cursor() as cur:
        cur.execute(SELECT_SQL)
        rows = cur.fetchall()
    print(f"✅ Retrieved {len(rows)} rows from DB.")
    if not rows:
        return pd.DataFrame(columns=["model_id", "tags", "createdat"])

    df = pd.DataFrame(rows, columns=["model_id", "tags", "createdat"])
    print("🧩 Parsing timestamps and tags...")
    df["createdat"] = pd.to_datetime(df["createdat"], utc=True, errors="coerce")
    df["tags"] = df["tags"].apply(parse_tags_field)
    return df


def prepare_tags_time_series(df: pd.DataFrame, time_freq: str = DEFAULT_TIME_FREQ) -> Tuple[pd.DataFrame, int]:
    print("⏳ Preparing tag time series aggregation...")
    df = df.copy()

    null_count = df["tags"].apply(lambda x: not x).sum()
    df = df[df["tags"].apply(lambda x: bool(x))]

    if df.empty:
        print("⚠️ No valid tag data found.")
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

    print("✅ Tag aggregation complete.")
    return pivot, null_count


def plot_stacked_tags(
        pivot_df,
        out_path: str,
        null_count: int,
        top_n: int = 15,
        title: str = "Models by tags over time (stacked)",
        dpi: int = 150,
        figsize=(14, 8),
        show_other: bool = True
):
    if pivot_df.empty:
        print("⚠️ No data to plot.")
        return

    col_sums = pivot_df.sum(axis=0).sort_values(ascending=False)
    top_cols = list(col_sums.iloc[:top_n].index)
    other_cols = [c for c in pivot_df.columns if c not in top_cols]

    df_top = pivot_df[top_cols].copy()
    other_count = pivot_df[other_cols].sum().sum() if other_cols else 0
    if show_other and other_count > 0:
        df_top["Other"] = pivot_df[other_cols].sum(axis=1)

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    df_top.plot.area(ax=ax, stacked=True, alpha=0.9)

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Period", fontsize=12)
    ax.set_ylabel("Number of models", fontsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    if not show_other and other_count > 0:
        handles.append(mpatches.Patch(color="gray"))
        labels.append(f"Other (total {int(other_count):,})")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Tags (total count)")

    ax.text(1.02, -0.15, f"NULL tags (not plotted): {null_count:,}", transform=ax.transAxes,
            fontsize=11, color="gray", ha="left", va="top")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Stacked tags plot saved to {out_path}")


def save_aggregated_csv(pivot_df: pd.DataFrame, out_csv: str):
    print("💾 Saving aggregated tag CSV...")
    pivot_df.to_csv(out_csv, index_label="period")
    print(f"✅ CSV saved to {out_csv}")


def main(out_png: str, out_csv: Optional[str], freq: str, top_n: int):
    print("🚀 Starting tag metrics pipeline...")
    conn = get_connection(DB_CONFIG)
    try:
        df = fetch_data(conn)
    finally:
        conn.close()

    if df.empty:
        print("⚠️ No rows fetched from DB. Exiting.")
        return

    pivot, null_count = prepare_tags_time_series(df, time_freq=freq)

    print(f"📊 Total models without tags (not plotted): {null_count:,}")

    if out_csv and not pivot.empty:
        save_aggregated_csv(pivot, out_csv)

    if not pivot.empty:
        plot_stacked_tags(
            pivot_df=pivot,
            out_path=out_png,
            null_count=null_count,
            top_n=top_n,
            title=f"Models by tags over time ({freq})",
            show_other=not args.no_other
        )

    print("🏁 Tag metrics pipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot tag time series from HF models DB")
    parser.add_argument("--out", dest="out_png", required=True, help="Path to output PNG")
    parser.add_argument("--out-csv", dest="out_csv", default=None, help="Optional: save aggregated CSV")
    parser.add_argument("--freq", default="YS", help="Time frequency for aggregation (e.g. 'YS', 'MS', 'D')")
    parser.add_argument("--top", type=int, default=15, help="Top N tags to show")
    parser.add_argument("--no-other", action="store_true", help="Hide 'Other' category from chart but show its total in legend")
    args = parser.parse_args()

    main(args.out_png, args.out_csv, args.freq, args.top)
