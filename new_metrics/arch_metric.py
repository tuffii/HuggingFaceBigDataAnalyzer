#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from typing import Optional, List, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import matplotlib.dates as mdates

from db.connection import get_connection

# ---------- Константы ----------
DEFAULT_TIME_FREQ = "MS"      # месяцы
PLOT_DPI = 150
PLOT_SIZE = (14, 8)
TOP_N = 8
YEAR_MIN_DEFAULT = 2016       # минимальный год на графике

SELECT_SQL = """
SELECT
  model_id,
  createdat,
  tags,
  pipeline_tag,
  (raw_data::jsonb->>'library_name') AS library_name
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

# ---------- Русская локаль / кириллица ----------
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
# -----------------------------------------------

def _parse_date_arg(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    return pd.to_datetime(s, utc=True, errors="coerce")

_num_re = re.compile(r"^[+-]?\d+(\.\d+)?$")

def parse_created_at(x: Any) -> Optional[pd.Timestamp]:
    """Парсим createdat (ms/s/ISO). Возвращаем UTC-aware Timestamp или NaT."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("UTC") if x.tzinfo else x.tz_localize("UTC")
    if isinstance(x, (int, float)):
        v = float(x)
        if v >= 1e12:   # epoch ms
            return pd.to_datetime(v, unit="ms", utc=True, errors="coerce")
        if v >= 1e9:    # epoch s
            return pd.to_datetime(v, unit="s", utc=True, errors="coerce")
        return pd.NaT
    s = str(x).strip()
    if not s:
        return pd.NaT
    if _num_re.match(s):
        try:
            v = float(s)
            if v >= 1e12:
                return pd.to_datetime(v, unit="ms", utc=True, errors="coerce")
            if v >= 1e9:
                return pd.to_datetime(v, unit="s", utc=True, errors="coerce")
            return pd.NaT
        except Exception:
            return pd.NaT
    return pd.to_datetime(s, utc=True, errors="coerce")

def _norm_tags(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t).strip().lower() for t in x if t is not None]
    try:
        parsed = json.loads(x)
        if isinstance(parsed, list):
            return [str(t).strip().lower() for t in parsed if t is not None]
    except Exception:
        pass
    return []

# ---------- Эвристики архитектур ----------
ARCH_RULES = {
    "transformer": ["transformer","gpt","bert","t5","llama","mistral","falcon",
                    "roberta","xlm","vit","clip","blip","whisper","wav2vec",
                    "deberta","bart","distilbert","albert","electra"],
    "diffusion": ["diffusion","stable-diffusion","sdxl","unet","latent-diffusion",
                  "controlnet","flux","kandinsky","sd3"],
    "cnn": ["cnn","resnet","unet-cnn","efficientnet","mobilenet","densenet","vgg","inception"],
    "rnn": ["rnn","lstm","gru","seq2seq"],
    "gan": ["gan","stylegan","biggan","cyclegan","pix2pix"],
    "vae": ["vae","autoencoder","auto-encoder"],
    "gnn": ["gnn","graphsage","gcn","gat"],
    "mlp": ["mlp","perceptron","feedforward"],
}
LIB_HINTS = {"diffusers": "diffusion"}
PIPE_HINTS = {
    "text-generation": "transformer",
    "image-classification": "cnn",
    "object-detection": "cnn",
    "image-segmentation": "cnn",
    "text-to-image": "diffusion",
    "image-to-image": "diffusion",
    "audio-to-audio": "transformer",
    "automatic-speech-recognition": "transformer",
    "token-classification": "transformer",
    "translation": "transformer",
}

def infer_architecture(model_id: str, tags: List[str],
                       pipeline_tag: Optional[str], library_name: Optional[str]) -> str:
    s_all = " ".join([model_id or ""] + tags).lower()
    for arch, keys in ARCH_RULES.items():
        if any(k in s_all for k in keys):
            return arch
    if library_name and library_name.strip().lower() in LIB_HINTS:
        return LIB_HINTS[library_name.strip().lower()]
    if pipeline_tag and pipeline_tag.strip().lower() in PIPE_HINTS:
        return PIPE_HINTS[pipeline_tag.strip().lower()]
    return "other"
# ---------------------------------------------

def fetch_data(conn, start: Optional[str], end: Optional[str], min_year: int) -> pd.DataFrame:
    print("🔄 Получаем данные из БД...")
    with conn.cursor() as cur:
        cur.execute(SELECT_SQL)
        rows = cur.fetchall()
    print(f"✅ Загружено строк: {len(rows)}")

    df = pd.DataFrame(rows, columns=["model_id","createdat","tags","pipeline_tag","library_name"])
    df["createdat"] = df["createdat"].apply(parse_created_at)
    df = df.dropna(subset=["createdat"])

    ts_start = _parse_date_arg(start) or pd.Timestamp(min_year, 1, 1, tz="UTC")
    # нельзя локализовать tz-aware → берем как есть
    ts_end = _parse_date_arg(end) or (pd.Timestamp.utcnow().tz_convert("UTC") + pd.Timedelta(days=30))
    df = df[(df["createdat"] >= ts_start) & (df["createdat"] <= ts_end)]
    df = df[df["createdat"].dt.year >= min_year]

    df["tags"] = df["tags"].apply(_norm_tags)
    df["arch"] = df.apply(
        lambda r: infer_architecture(r["model_id"], r["tags"],
                                     r.get("pipeline_tag") or None,
                                     r.get("library_name") or None),
        axis=1
    )
    return df

def _apply_min_arch_count(df: pd.DataFrame, min_arch_count: int) -> pd.DataFrame:
    """Архитектуры с суммой < порога сворачиваем в 'other'."""
    if min_arch_count <= 0:
        return df
    totals = df.groupby("arch")["model_id"].count()
    low = set(totals[totals < min_arch_count].index)
    if low:
        df = df.copy()
        df.loc[df["arch"].isin(low), "arch"] = "other"
    return df

def prepare_time_series(
    df: pd.DataFrame,
    time_freq: str,
    share: bool,
    min_year: int,
    drop_other: bool,
    min_arch_count: int,
) -> pd.DataFrame:
    print("⏳ Агрегируем по времени...")
    dfx = _apply_min_arch_count(df, min_arch_count)

    freq_map = {"MS":"M","W-MON":"W","W-SUN":"W","D":"D","YS":"Y"}
    period_freq = freq_map.get(time_freq.upper(), time_freq)

    dfx["period"] = (
        dfx["createdat"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.to_period(period_freq)
        .dt.to_timestamp(how="start")
    )

    year_floor = pd.Timestamp(min_year, 1, 1)
    year_cap = pd.Timestamp(pd.Timestamp.utcnow().year + 1, 12, 31)
    dfx = dfx[(dfx["period"] >= year_floor) & (dfx["period"] <= year_cap)]

    grouped = dfx.groupby(["period","arch"]).agg(count=("model_id","count")).reset_index()
    if grouped.empty:
        return pd.DataFrame()

    pivot = grouped.pivot(index="period", columns="arch", values="count").fillna(0).sort_index()

    start_idx = max(pivot.index.min(), year_floor)
    end_idx = min(pivot.index.max(), year_cap)
    full_idx = pd.date_range(start=start_idx, end=end_idx, freq=time_freq)
    pivot = pivot.reindex(full_idx, fill_value=0)
    pivot.index.name = "period"

    if drop_other and "other" in pivot.columns:
        pivot = pivot.drop(columns=["other"])

    if share:
        row_sums = pivot.sum(axis=1).replace(0, pd.NA)
        pivot = pivot.div(row_sums, axis=0).fillna(0.0)

    print("✅ Готово.")
    return pivot

def _parse_figsize(s: Optional[str]) -> Tuple[int, int]:
    if not s:
        return PLOT_SIZE
    try:
        w, h = s.lower().replace(" ", "").split("x")
        return (int(w), int(h))
    except Exception:
        return PLOT_SIZE

def plot_stacked_architectures(
    pivot_df: pd.DataFrame,
    out_path: str,
    top_n: int,
    share: bool,
    drop_other: bool,
    figsize_str: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    dpi: int = PLOT_DPI,
):
    print("🎨 Рисуем стек-график архитектур...")
    if pivot_df.empty:
        print("⚠️ Нет данных для графика.")
        return

    # --- жестко выкидываем 'other' в отрисовке, если требуется
    df_plot = pivot_df.copy()
    df_plot.columns = [str(c) for c in df_plot.columns]
    if drop_other:
        df_plot = df_plot.loc[:, [c for c in df_plot.columns if c.lower() != "other"]]

    # Топ-N + «другие» (без 'other')
    col_sums = df_plot.sum(axis=0).sort_values(ascending=False)
    top_cols = list(col_sums.iloc[:top_n].index)
    other_cols = [c for c in df_plot.columns if c not in top_cols and c.lower() != "other"]

    df_top = df_plot[top_cols].copy()
    legend_labels = top_cols[:]
    if other_cols:
        df_top["другие"] = df_plot[other_cols].sum(axis=1)
        legend_labels.append("другие")

    # stackplot хочет числа дат
    x_dt = pd.to_datetime(df_top.index)
    x_num = mdates.date2num(x_dt.to_pydatetime())
    y_series = [df_top[c].values for c in df_top.columns]

    figsize = _parse_figsize(figsize_str)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.stackplot(x_num, y_series, labels=legend_labels, alpha=0.9)

    title = "Архитектуры моделей во времени (доли)" if share else "Архитектуры моделей во времени (кол-во)"
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Период", fontsize=12)
    ax.set_ylabel("Доля" if share else "Число моделей", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")
    ax.margins(x=0)

    # Масштаб Y + формат процентов для долей
    if share:
        ax.yaxis.set_major_locator(MaxNLocator(integer=False))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        # если не задано явно — автоматически подрезаем низ, чтобы убрать «пустоту»
        if ymin is None:
            # минимальная сумма «не-трансформерных» долей → ориентир для подрезки
            non_transformer = 1.0 - df_top.get("transformer", pd.Series(0, index=df_top.index)).fillna(0.0)
            # хотим видеть верхние 20–30% шкалы; но не меньше 60% для страховки
            ymin = max(0.60, df_top["transformer"].min() - 0.05) if "transformer" in df_top else 0.60
        if ymax is None:
            ymax = 1.0
        ax.set_ylim(ymin, ymax)
    else:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if (ymin is not None) or (ymax is not None):
            ax.set_ylim(ymin if ymin is not None else ax.get_ylim()[0],
                        ymax if ymax is not None else ax.get_ylim()[1])

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Архитектура", frameon=False, fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(right=0.80)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ График сохранён в {out_path}")

def save_csv(df: pd.DataFrame, out_csv: str, index_label: str = "period"):
    print(f"💾 Сохраняем CSV в {out_csv} ...")
    df.to_csv(out_csv, index_label=index_label)
    print("✅ CSV сохранён.")

def make_arch_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("arch")
              .agg(models_total=("model_id","count"),
                   first_date=("createdat","min"),
                   last_date=("createdat","max"))
              .reset_index()
              .sort_values(["models_total"], ascending=False))

def main(
    out_png: str,
    out_csv: Optional[str],
    out_csv_leader: Optional[str],
    freq: str,
    top: int,
    share: bool,
    start: Optional[str],
    end: Optional[str],
    min_year: int,
    drop_other: bool,
    min_arch_count: int,
    figsize: Optional[str],
    ymin: Optional[float],
    ymax: Optional[float],
):
    setup_russian_labels()
    print("🚀 Пайплайн: кластеризация архитектур...")
    conn = get_connection(DB_CONFIG)
    try:
        df = fetch_data(conn, start, end, min_year)
    finally:
        conn.close()

    if df.empty:
        print("⚠️ Пустой выбор — завершаю.")
        return

    pivot = prepare_time_series(
        df,
        time_freq=freq,
        share=share,
        min_year=min_year,
        drop_other=drop_other,
        min_arch_count=min_arch_count,
    )

    if out_csv and not pivot.empty:
        save_csv(pivot, out_csv, index_label="period")
    if out_csv_leader:
        save_csv(make_arch_leaderboard(df), out_csv_leader, index_label="arch")
    if not pivot.empty:
        plot_stacked_architectures(
            pivot_df=pivot,
            out_path=out_png,
            top_n=top,
            share=share,
            drop_other=drop_other,
            figsize_str=figsize,
            ymin=ymin,
            ymax=ymax,
        )
    print("🏁 Готово.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Кластеризация архитектур нейросетей (эвристики)")
    parser.add_argument("--out", dest="out_png", required=True, help="Путь к PNG-графику")
    parser.add_argument("--out-csv", dest="out_csv", default=None, help="Опционально: CSV тайм-серии")
    parser.add_argument("--out-csv-leader", dest="out_csv_leader", default=None, help="Опционально: CSV сводки")
    parser.add_argument("--freq", default=DEFAULT_TIME_FREQ, help="Частота ('MS','W-MON','D','YS')")
    parser.add_argument("--top", type=int, default=TOP_N, help="Сколько архитектур явно")
    parser.add_argument("--share", action="store_true", help="Показывать доли (0..1)")
    parser.add_argument("--start", default=None, help="Начало периода (ISO)")
    parser.add_argument("--end",   default=None, help="Конец периода (ISO)")
    parser.add_argument("--min-year", type=int, default=YEAR_MIN_DEFAULT, help="Минимальный год на графике")
    parser.add_argument("--drop-other", action="store_true", help="Исключить группу 'other' с графика")
    parser.add_argument("--min-arch-count", type=int, default=0,
                        help="Порог суммарного количества моделей для архитектуры; ниже порога → 'other'")
    parser.add_argument("--figsize", default=None, help="Размер фигуры, формат 'W x H', напр. '16x9'")
    parser.add_argument("--ymin", type=float, default=None, help="Нижняя граница оси Y (например 0.8 для долей)")
    parser.add_argument("--ymax", type=float, default=None, help="Верхняя граница оси Y (например 1.0 для долей)")
    args = parser.parse_args()

    main(
        out_png=args.out_png,
        out_csv=args.out_csv,
        out_csv_leader=args.out_csv_leader,
        freq=args.freq,
        top=args.top,
        share=args.share,
        start=args.start,
        end=args.end,
        min_year=args.min_year,
        drop_other=args.drop_other,
        min_arch_count=args.min_arch_count,
        figsize=args.figsize,
        ymin=args.ymin,
        ymax=args.ymax,
    )
