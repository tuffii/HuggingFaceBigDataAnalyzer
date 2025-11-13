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

# Константы
DEFAULT_TIME_FREQ = "MS"  # Частота агрегации по месяцам
PLOT_DPI = 150
PLOT_SIZE = (14, 8)
TOP_N = 8
YEAR_MIN_DEFAULT = 2016

SELECT_SQL = """
             SELECT model_id, \
                    createdat, \
                    tags, \
                    pipeline_tag, \
                    (raw_data::jsonb->>'library_name') AS library_name
             FROM hf_models
             WHERE createdat IS NOT NULL \
             """

DB_CONFIG = {
    "dbname": "postgres",
    "user": "BIGDATA",
    "password": "PASSWORD",
    "host": "localhost",
    "port": 5432,
}



# Настройка русской локали и шрифтов
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



# Вспомогательные функции парсинга
_num_re = re.compile(r"^[+-]?\d+(\.\d+)?$")


def _parse_date_arg(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    return pd.to_datetime(s, utc=True, errors="coerce")


def parse_created_at(x: Any) -> Optional[pd.Timestamp]:
    """Парсит createdat из различных форматов (ms, s, ISO)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("UTC") if x.tzinfo else x.tz_localize("UTC")
    if isinstance(x, (int, float)):
        v = float(x)
        if v >= 1e12:
            return pd.to_datetime(v, unit="ms", utc=True, errors="coerce")
        if v >= 1e9:
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
        except Exception:
            return pd.NaT
    return pd.to_datetime(s, utc=True, errors="coerce")


def _norm_tags(x: Any) -> List[str]:
    """Приводит список тегов к нижнему регистру и списку строк."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t).strip().lower() for t in x if t]
    try:
        parsed = json.loads(x)
        if isinstance(parsed, list):
            return [str(t).strip().lower() for t in parsed if t]
    except Exception:
        pass
    return []



# Эвристики определения архитектур моделей
ARCH_RULES = {
    "transformer": ["transformer", "gpt", "bert", "t5", "llama", "mistral", "falcon",
                    "roberta", "xlm", "vit", "clip", "blip", "whisper", "wav2vec",
                    "deberta", "bart", "distilbert", "albert", "electra"],
    "diffusion": ["diffusion", "stable-diffusion", "sdxl", "unet", "latent-diffusion",
                  "controlnet", "flux", "kandinsky", "sd3"],
    "cnn": ["cnn", "resnet", "efficientnet", "mobilenet", "densenet", "vgg", "inception"],
    "rnn": ["rnn", "lstm", "gru", "seq2seq"],
    "gan": ["gan", "stylegan", "biggan", "cyclegan", "pix2pix"],
    "vae": ["vae", "autoencoder", "auto-encoder"],
    "gnn": ["gnn", "graphsage", "gcn", "gat"],
    "mlp": ["mlp", "perceptron", "feedforward"],
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
    """Определяет архитектуру модели по эвристикам."""
    s_all = " ".join([model_id or ""] + tags).lower()
    for arch, keys in ARCH_RULES.items():
        if any(k in s_all for k in keys):
            return arch
    if library_name and library_name.strip().lower() in LIB_HINTS:
        return LIB_HINTS[library_name.strip().lower()]
    if pipeline_tag and pipeline_tag.strip().lower() in PIPE_HINTS:
        return PIPE_HINTS[pipeline_tag.strip().lower()]
    return "other"



# Основная логика
def fetch_data(conn, start: Optional[str], end: Optional[str], min_year: int) -> pd.DataFrame:
    """Извлекает и фильтрует данные из базы."""
    with conn.cursor() as cur:
        cur.execute(SELECT_SQL)
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["model_id", "createdat", "tags", "pipeline_tag", "library_name"])
    df["createdat"] = df["createdat"].apply(parse_created_at)
    df = df.dropna(subset=["createdat"])

    ts_start = _parse_date_arg(start) or pd.Timestamp(min_year, 1, 1, tz="UTC")
    ts_end = _parse_date_arg(end) or (pd.Timestamp.utcnow().tz_convert("UTC") + pd.Timedelta(days=30))
    df = df[(df["createdat"] >= ts_start) & (df["createdat"] <= ts_end)]
    df = df[df["createdat"].dt.year >= min_year]

    df["tags"] = df["tags"].apply(_norm_tags)
    df["arch"] = df.apply(lambda r: infer_architecture(
        r["model_id"], r["tags"], r.get("pipeline_tag"), r.get("library_name")), axis=1)

    return df


def _apply_min_arch_count(df: pd.DataFrame, min_arch_count: int) -> pd.DataFrame:
    """Объединяет редкие архитектуры в категорию 'other'."""
    if min_arch_count <= 0:
        return df
    totals = df.groupby("arch")["model_id"].count()
    low = set(totals[totals < min_arch_count].index)
    if low:
        df = df.copy()
        df.loc[df["arch"].isin(low), "arch"] = "other"
    return df


def prepare_time_series(df: pd.DataFrame, time_freq: str, share: bool,
                        min_year: int, drop_other: bool, min_arch_count: int) -> pd.DataFrame:
    """Создает временной ряд количества моделей по архитектурам."""
    dfx = _apply_min_arch_count(df, min_arch_count)
    freq_map = {"MS": "M", "W-MON": "W", "W-SUN": "W", "D": "D", "YS": "Y"}
    period_freq = freq_map.get(time_freq.upper(), time_freq)

    dfx["period"] = (
        dfx["createdat"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.to_period(period_freq)
        .dt.to_timestamp(how="start")
    )

    grouped = dfx.groupby(["period", "arch"]).agg(count=("model_id", "count")).reset_index()
    if grouped.empty:
        return pd.DataFrame()

    pivot = grouped.pivot(index="period", columns="arch", values="count").fillna(0).sort_index()
    if drop_other and "other" in pivot.columns:
        pivot = pivot.drop(columns=["other"])
    if share:
        row_sums = pivot.sum(axis=1).replace(0, pd.NA)
        pivot = pivot.div(row_sums, axis=0).fillna(0.0)
    return pivot



# Визуализация
def _parse_figsize(s: Optional[str]) -> Tuple[int, int]:
    if not s:
        return PLOT_SIZE
    try:
        w, h = s.lower().replace(" ", "").split("x")
        return (int(w), int(h))
    except Exception:
        return PLOT_SIZE


def plot_stacked_architectures(pivot_df: pd.DataFrame, out_path: str,
                               top_n: int, share: bool, drop_other: bool,
                               figsize_str: Optional[str] = None,
                               ymin: Optional[float] = None,
                               ymax: Optional[float] = None,
                               dpi: int = PLOT_DPI,
                               null_count: Optional[int] = None):
    """Строит стек-график распределения архитектур во времени."""
    if pivot_df.empty:
        return

    df_plot = pivot_df.copy()
    df_plot.columns = [str(c) for c in df_plot.columns]
    if drop_other:
        df_plot = df_plot.loc[:, [c for c in df_plot.columns if c.lower() != "other"]]

    col_sums = df_plot.sum(axis=0).sort_values(ascending=False)
    top_cols = list(col_sums.iloc[:top_n].index)
    other_cols = [c for c in df_plot.columns if c not in top_cols and c.lower() != "other"]

    df_top = df_plot[top_cols].copy()
    legend_labels = top_cols[:]
    if other_cols:
        df_top["другие"] = df_plot[other_cols].sum(axis=1)
        legend_labels.append("другие")

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
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")
    ax.margins(x=0)

    if share:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(ymin or 0.6, ymax or 1.0)
    else:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if ymin or ymax:
            ax.set_ylim(ymin or ax.get_ylim()[0], ymax or ax.get_ylim()[1])

    # Легенда
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              title="Архитектура", frameon=False, fontsize=9)

    if null_count is not None:
        null_text = f"Не классифицировано (null): {null_count:,}"
        plt.gcf().text(
            0.985, 0.03,
            null_text,
            fontsize=9,
            color="gray",
            ha="right",
            va="bottom",
        )

    plt.tight_layout()
    plt.subplots_adjust(right=0.80)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# CSV и лидерборд
def save_csv(df: pd.DataFrame, out_csv: str, index_label: str = "period"):
    df.to_csv(out_csv, index_label=index_label)


def make_arch_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Создает таблицу с общим числом моделей по архитектурам."""
    return (df.groupby("arch")
            .agg(models_total=("model_id", "count"),
                 first_date=("createdat", "min"),
                 last_date=("createdat", "max"))
            .reset_index()
            .sort_values(["models_total"], ascending=False))



# Точка входа
def main(out_png: str, out_csv: Optional[str], out_csv_leader: Optional[str],
         freq: str, top: int, share: bool, start: Optional[str], end: Optional[str],
         min_year: int, drop_other: bool, min_arch_count: int,
         figsize: Optional[str], ymin: Optional[float], ymax: Optional[float]):
    setup_russian_labels()
    conn = get_connection(DB_CONFIG)
    try:
        df = fetch_data(conn, start, end, min_year)
    finally:
        conn.close()

    total_all = len(df)
    df_with_arch = df[~df["arch"].isna()]
    total_with_arch = len(df_with_arch)
    total_null = total_all - total_with_arch

    if df.empty:
        return

    pivot = prepare_time_series(df, freq, share, min_year, drop_other, min_arch_count)
    if out_csv and not pivot.empty:
        save_csv(pivot, out_csv)
    if out_csv_leader:
        save_csv(make_arch_leaderboard(df), out_csv_leader, index_label="arch")
    if not pivot.empty:
        plot_stacked_architectures(
            pivot, out_png, top, share, drop_other,
            figsize, ymin, ymax,
            null_count=total_null
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Кластеризация архитектур нейросетей (эвристики)")
    parser.add_argument("--out", dest="out_png", required=True)
    parser.add_argument("--out-csv", dest="out_csv", default=None)
    parser.add_argument("--out-csv-leader", dest="out_csv_leader", default=None)
    parser.add_argument("--freq", default=DEFAULT_TIME_FREQ)
    parser.add_argument("--top", type=int, default=TOP_N)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--min-year", type=int, default=YEAR_MIN_DEFAULT)
    parser.add_argument("--drop-other", action="store_true")
    parser.add_argument("--min-arch-count", type=int, default=0)
    parser.add_argument("--figsize", default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    args = parser.parse_args()

    main(args.out_png, args.out_csv, args.out_csv_leader, args.freq, args.top,
         args.share, args.start, args.end, args.min_year, args.drop_other,
         args.min_arch_count, args.figsize, args.ymin, args.ymax)
