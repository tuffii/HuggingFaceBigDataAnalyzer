from __future__ import annotations
import argparse
import json
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from tqdm import tqdm

from db.connection import get_connection

DEFAULT_TIME_FREQ = "MS"      # месяцы
PLOT_DPI = 150
PLOT_SIZE = (14, 7)
TOP_N_LINES = 12
RECENT_DAYS = 180             # «активность» за последние N дней

SELECT_SQL = """
SELECT model_id, likes, downloads, createdat, lastmodified, raw_data, tags
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

# ---------- Настройка кириллицы и русской локали ----------
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
# ----------------------------------------------------------

def _safe_json_to_dict(x: Any) -> Optional[Dict[str, Any]]:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None

def _parse_author(model_id: Optional[str], raw_data: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Приоритет: raw_data['author'] -> namespace из model_id до '/'.
    """
    if raw_data and isinstance(raw_data, dict):
        a = raw_data.get("author")
        if isinstance(a, str) and a.strip():
            return a.strip()
    if isinstance(model_id, str) and "/" in model_id:
        return model_id.split("/", 1)[0].strip()
    return None

def fetch_data(conn) -> pd.DataFrame:
    print("🔄 Получаем данные из БД...")
    with conn.cursor() as cur:
        cur.execute(SELECT_SQL)
        rows = cur.fetchall()
    print(f"✅ Загружено строк из БД: {len(rows)}")
    if not rows:
        return pd.DataFrame(columns=["model_id", "likes", "downloads", "createdat", "lastmodified", "raw_data", "tags"])

    df = pd.DataFrame(rows, columns=["model_id", "likes", "downloads", "createdat", "lastmodified", "raw_data", "tags"])

    print("🧩 Парсим временные метки и JSON...")
    df["createdat"] = pd.to_datetime(df["createdat"], utc=True, errors="coerce")
    df["lastmodified"] = pd.to_datetime(df["lastmodified"], utc=True, errors="coerce")

    for col in ["likes", "downloads"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["raw_data"] = df["raw_data"].apply(_safe_json_to_dict)

    print("👤 Определяем автора/организацию…")
    df["author"] = df.apply(lambda r: _parse_author(r["model_id"], r["raw_data"]), axis=1)
    df["author"] = df["author"].fillna("UNKNOWN")

    df = df.dropna(subset=["createdat"])
    return df

def _parse_date(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    return pd.to_datetime(s, utc=True, errors="coerce")

def filter_by_period(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    ts_start = _parse_date(start)
    ts_end = _parse_date(end)
    dfx = df.copy()
    if ts_start is not None:
        dfx = dfx[dfx["createdat"] >= ts_start]
    if ts_end is not None:
        dfx = dfx[dfx["createdat"] <= ts_end]
    return dfx

def build_leaderboard(df: pd.DataFrame, recent_days: int = RECENT_DAYS) -> pd.DataFrame:
    dfx = df.copy()
    # корректное tz-aware UTC "сейчас"
    now = pd.Timestamp.now(tz="UTC")
    recent_cut = now - pd.Timedelta(days=recent_days)
    dfx["is_recent"] = dfx["createdat"] >= recent_cut

    grouped = dfx.groupby("author").agg(
        models_total=("model_id", "count"),
        models_recent=("is_recent", "sum"),
        likes_total=("likes", "sum"),
        downloads_total=("downloads", "sum"),
        first_model_date=("createdat", "min"),
        last_model_date=("createdat", "max"),
    ).reset_index()

    grouped = grouped.sort_values(
        by=["models_total", "models_recent", "downloads_total", "likes_total"],
        ascending=[False, False, False, False]
    )
    grouped = grouped.rename(columns={"models_recent": f"models_recent_{recent_days}d"})
    return grouped

def prepare_time_series_by_author(df: pd.DataFrame, time_freq: str = DEFAULT_TIME_FREQ) -> pd.DataFrame:
    print("⏳ Готовим агрегирование по авторам во времени...")
    dfx = df.copy()
    freq_map = {"MS": "M", "W-MON": "W", "W-SUN": "W", "D": "D", "YS": "Y"}
    period_freq = freq_map.get(time_freq.upper(), time_freq)

    dfx["period"] = (
        dfx["createdat"]
        .dt.tz_localize(None)
        .dt.to_period(period_freq)
        .dt.to_timestamp(how="start")
    )

    grouped = (
        dfx.groupby(["period", "author"])
        .agg(count=("model_id", "count"))
        .reset_index()
    )

    pivot = grouped.pivot(index="period", columns="author", values="count").fillna(0).sort_index()

    if not pivot.empty:
        full_idx = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq=time_freq)
        pivot = pivot.reindex(full_idx, fill_value=0)
        pivot.index.name = "period"

    print("✅ Агрегирование завершено.")
    return pivot

def plot_time_series_authors(
    pivot_df: pd.DataFrame,
    out_path: str,
    top_n: int = TOP_N_LINES,
    title: str = "Новые модели по авторам/организациям во времени",
    dpi: int = PLOT_DPI,
    figsize: Tuple[int, int] = PLOT_SIZE,
):
    print("🎨 Строим график по авторам...")
    if pivot_df.empty:
        print("⚠️ Нет данных для построения графика.")
        return

    col_sums = pivot_df.sum(axis=0).sort_values(ascending=False)
    top_cols = list(col_sums.iloc[:top_n].index)
    df_top = pivot_df[top_cols].copy()

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    for col in tqdm(df_top.columns, desc="📈 Рисуем линии"):
        lw = 1.0 + (df_top[col].sum() / (df_top.sum().sum() + 1)) * 4.0
        ax.plot(df_top.index, df_top[col].values, label=f"{col} ({int(df_top[col].sum())})", linewidth=lw)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Период (начало)", fontsize=12)
    ax.set_ylabel("Количество моделей за период", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune="lower"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))  # «янв 2025»
    plt.xticks(rotation=45, ha="right")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ График сохранён в {out_path}")

def save_csv(df: pd.DataFrame, out_csv: str):
    print(f"💾 Сохраняем CSV в {out_csv} ...")
    df.to_csv(out_csv, index=False)
    print("✅ CSV сохранён.")

def main(
    out_png: str,
    out_csv_leaders: Optional[str],
    out_csv_ts: Optional[str],
    freq: str,
    top_n: int,
    recent_days: int,
    start: Optional[str],
    end: Optional[str],
):
    setup_russian_labels()

    print("🚀 Запускаем пайплайн метрики лидеров...")
    conn = get_connection(DB_CONFIG)
    try:
        df = fetch_data(conn)
    finally:
        conn.close()

    if df.empty:
        print("⚠️ Нет строк из БД. Завершаем.")
        return

    # Фильтр по периоду
    df = filter_by_period(df, start, end)
    if df.empty:
        print("⚠️ После фильтрации по периоду данных нет. Завершаем.")
        return

    # Лидерборд
    leaderboard = build_leaderboard(df, recent_days=recent_days)
    if out_csv_leaders:
        save_csv(leaderboard, out_csv_leaders)

    # Тайм-серия
    pivot = prepare_time_series_by_author(df, time_freq=freq)
    if out_csv_ts and not pivot.empty:
        pivot.reset_index().rename(columns={"index": "period"}).to_csv(out_csv_ts, index=False)

    if not pivot.empty:
        title_suffix = ""
        if start or end:
            title_suffix = f" — период: {start or '…'}…{end or '…'}"
        plot_time_series_authors(
            pivot_df=pivot,
            out_path=out_png,
            top_n=top_n,
            title=f"Новые модели по авторам/организациям во времени ({freq}){title_suffix}"
        )

    print("🏁 Пайплайн метрики лидеров успешно завершён.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Определение лидирующих авторов/организаций на Hugging Face")
    parser.add_argument("--out", dest="out_png", required=True, help="Путь к PNG-графику (тайм-серия)")
    parser.add_argument("--leaders-csv", dest="out_csv_leaders", default="leaders.csv", help="CSV с рейтингом авторов/организаций")
    parser.add_argument("--ts-csv", dest="out_csv_ts", default=None, help="Опционально: сохранить тайм-серию (pivot) в CSV")
    parser.add_argument("--freq", default="MS", help="Частота агрегации (например, 'MS', 'W-MON', 'D', 'YS')")
    parser.add_argument("--top", type=int, default=12, help="Сколько авторов показать на графике")
    parser.add_argument("--recent-days", type=int, default=180, help="Окно недавней активности для метрики models_recent_Nd")
    parser.add_argument("--start", default=None, help="Начало периода (ISO: 2023-01-01)")
    parser.add_argument("--end",   default=None, help="Конец периода (ISO: 2025-12-31)")
    args = parser.parse_args()

    main(
        args.out_png,
        args.out_csv_leaders,
        args.out_csv_ts,
        args.freq,
        args.top,
        args.recent_days,
        args.start,
        args.end
    )
