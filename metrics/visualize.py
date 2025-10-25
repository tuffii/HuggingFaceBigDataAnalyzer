from .config import PLOTS_DIR
from loguru import logger
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import json
from pathlib import Path

def save_plt(fig, filename: str):
    path = Path(PLOTS_DIR) / filename
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Сохранён график: {path}")
    return str(path)

class Visualizer:
    def __init__(self, logger_obj=logger):
        self.logger = logger_obj

    def histogram_downloads(self, df: pd.DataFrame, bins=50, out_name="downloads_hist.png"):
        fig, ax = plt.subplots()
        ax.hist(df['downloads_num'].replace([float('inf'), -float('inf')], None).dropna(), bins=bins)
        ax.set_title("Распределение загрузок (downloads)")
        ax.set_xlabel("downloads")
        ax.set_ylabel("frequency")
        return save_plt(fig, out_name)

    def pipeline_bar(self, pipeline_counts: dict, out_name="pipeline_counts.png", top_n=30):
        items = sorted(pipeline_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        labels = [i[0] for i in items]
        vals = [i[1] for i in items]
        fig, ax = plt.subplots(figsize=(10, max(4, len(labels)*0.3)))
        ax.barh(range(len(labels))[::-1], vals)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[::-1])
        ax.set_title("Количество моделей по pipeline_tag")
        return save_plt(fig, out_name)

    def top_tags_bar(self, tags_counts: list, out_name="top_tags.png"):
        labels = [t for t,c in tags_counts]
        vals = [c for t,c in tags_counts]
        fig, ax = plt.subplots(figsize=(10, max(4, len(labels)*0.3)))
        ax.barh(range(len(labels))[::-1], vals)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[::-1])
        ax.set_title("Топ тегов")
        return save_plt(fig, out_name)

    def time_series_stacked(self, ts_dict: dict, out_name="time_series_stacked.html", png_name="time_series_stacked.png"):
        """
        ts_dict: {year: {category: count}}
        Сохраняет интерактивный html (Plotly) и статичный PNG.
        """
        # преобразуем в DataFrame
        rows = []
        for year, per in ts_dict.items():
            for k,v in per.items():
                rows.append({"year": int(year), "category": k, "count": v})
        if not rows:
            self.logger.warning("time_series_stacked: пустые данные")
            return None
        df = pd.DataFrame(rows)
        pivot = df.pivot(index='year', columns='category', values='count').fillna(0)
        # Plotly stacked area
        fig = px.area(pivot.reset_index(), x='year', y=pivot.columns, title="Динамика по категориям")
        html_path = Path(PLOTS_DIR) / out_name
        fig.write_html(str(html_path))
        self.logger.info(f"Interactive chart saved: {html_path}")
        # also static
        try:
            png_path = Path(PLOTS_DIR) / png_name
            fig.write_image(str(png_path), width=1200, height=600)
            self.logger.info(f"Static PNG saved: {png_path}")
        except Exception as e:
            self.logger.warning("Не получилось сохранить PNG из plotly (нужны дополнительные зависимости), сохраняю PNG через matplotlib")
            # fallback matplotlib
            ax = pivot.plot(kind='area', stacked=True, figsize=(12,6))
            fig2 = ax.get_figure()
            fig2.savefig(Path(PLOTS_DIR) / png_name, bbox_inches='tight')
            fig2.clf()
        return {"html": str(html_path), "png": str(Path(PLOTS_DIR) / png_name)}

    def license_trend_heatmap(self, license_trends: dict, out_name="license_trends.png"):
        # license_trends: {year: {license: count}}
        import numpy as np
        years = sorted(license_trends.keys())
        licenses = sorted({lic for y in years for lic in license_trends[y].keys()})
        mat = []
        for y in years:
            row = [license_trends[y].get(lic, 0) for lic in licenses]
            mat.append(row)
        df = pd.DataFrame(mat, index=years, columns=licenses)
        fig, ax = plt.subplots(figsize=(max(8, len(licenses)*0.5), max(4, len(years)*0.6)))
        c = ax.imshow(df.values, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(licenses)))
        ax.set_xticklabels(licenses, rotation=90)
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        fig.colorbar(c, ax=ax)
        ax.set_title("Тенденции использования лицензий (heatmap)")
        return save_plt(fig, out_name)
