from __future__ import annotations
import argparse
import os
import json
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from tqdm import tqdm
from db.connection import get_connection
from embedding_infer import PipelineTagInferencer


from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


DEFAULT_TIME_FREQ = "MS"
PLOT_DPI = 150
PLOT_SIZE = (12, 7)
TOP_N_LINES = 12


_inferencer: Optional[PipelineTagInferencer] = None


def infer_pipeline_tag(tags: Optional[List[str]], model_id: str, other_fields: Dict[str, Any]) -> Optional[str]:
    global _inferencer
    if _inferencer is None:
        _inferencer = PipelineTagInferencer()
    return _inferencer.infer_pipeline_tag(tags, model_id, other_fields)


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
    print("🔄 Fetching data from database...")
    with conn.cursor() as cur:
        cur.execute(SELECT_SQL)
        rows = cur.fetchall()
    print(f"✅ Retrieved {len(rows)} rows from DB.")
    if not rows:
        return pd.DataFrame(columns=["model_id","pipeline_tag","tags","createdat"])
    df = pd.DataFrame(rows)
    print("🧩 Parsing timestamps and JSON tags...")
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
    print("⏳ Preparing time series aggregation...")
    df = df.copy()

    if fill_null_pipeline:
        tqdm.pandas(desc="🔍 Inferring missing pipeline_tag (embeddings)")
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

    pivot = grouped.pivot(index="period", columns="pipeline_tag_for_group", values="count").fillna(0).sort_index()
    full_idx = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq=time_freq)
    pivot = pivot.reindex(full_idx, fill_value=0)
    pivot.index.name = "period"

    print("✅ Aggregation complete.")
    return pivot

def plot_time_series(
    pivot_df: pd.DataFrame,
    out_path: str,
    top_n: int = TOP_N_LINES,
    title: str = "Models by pipeline_tag over time",
    dpi: int = PLOT_DPI,
    figsize: Tuple[int, int] = PLOT_SIZE,
):
    print("🎨 Plotting time series graph...")
    col_sums = pivot_df.sum(axis=0).sort_values(ascending=False)
    if col_sums.empty:
        print("⚠️ No data to plot.")
        return

    null_count = int(col_sums.get("NULL", 0))
    if "NULL" in col_sums:
        col_sums = col_sums.drop("NULL")

    top_cols = list(col_sums.iloc[:top_n].index)
    other_cols = [c for c in pivot_df.columns if c not in top_cols and c != "NULL"]

    df_top = pivot_df[top_cols].copy()
    if other_cols:
        df_top["Other"] = pivot_df[other_cols].sum(axis=1)

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    for col in tqdm(df_top.columns, desc="📈 Drawing lines"):
        linewidth = 1.0 + (df_top[col].sum() / (df_top.sum().sum() + 1)) * 4.0
        ax.plot(df_top.index, df_top[col].values, label=f"{col} ({int(df_top[col].sum())})", linewidth=linewidth)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Period (start)", fontsize=12)
    ax.set_ylabel("Number of models", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune="lower"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()

    if null_count > 0:
        plt.text(
            1.02,
            0.02,
            f"NULL (missing pipeline_tag): {null_count:,}",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
            ha="left",
            va="bottom",
        )

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Plot saved to {out_path} (NULL count = {null_count:,})")


def save_aggregated_csv(pivot_df: pd.DataFrame, out_csv: str):
    print("💾 Saving aggregated CSV...")
    pivot_df.to_csv(out_csv, index_label="period")
    print(f"✅ CSV saved to {out_csv}")

def main(out_png: str, out_csv: Optional[str], freq: str, top_n: int, fill_null: bool):
    print("🚀 Starting pipeline...")
    conn = get_connection(DB_CONFIG)
    try:
        df = fetch_data(conn)
    finally:
        conn.close()

    if df.empty:
        print("⚠️ No rows fetched from DB. Exiting.")
        return

    pivot = prepare_time_series(df, time_freq=freq, fill_null_pipeline=fill_null)

    if out_csv:
        save_aggregated_csv(pivot, out_csv)

    plot_time_series(pivot, out_png, top_n=top_n, title=f"Models by pipeline_tag ({freq})")
    print("🏁 Pipeline finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pipeline_tag time series from HF models DB")
    parser.add_argument("--out", dest="out_png", required=True, help="Path to output PNG")
    parser.add_argument("--out-csv", dest="out_csv", default=None, help="Optional: save aggregated CSV")
    parser.add_argument("--freq", default="MS", help="Time frequency for aggregation (e.g. 'MS', 'W-MON', 'D')")
    parser.add_argument("--top", type=int, default=12, help="Top N pipeline_tag lines to show")
    parser.add_argument("--fill-null", action="store_true", help="Attempt to infer missing pipeline_tag (not implemented yet)")
    args = parser.parse_args()

    main(args.out_png, args.out_csv, args.freq, args.top, args.fill_null)
