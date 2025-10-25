import json
from .config import REPORTS_DIR
from pathlib import Path
from loguru import logger
from typing import Dict, Any
from jinja2 import Template
import datetime

class ReportGenerator:
    def __init__(self, logger_obj=logger):
        self.logger = logger_obj

    def generate_json_report(self, metrics_dict: Dict[str, Any], filename: str = None) -> str:
        if filename is None:
            filename = f"report_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path = Path(REPORTS_DIR) / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=2, default=str)
        self.logger.info(f"JSON отчёт сохранён: {path}")
        return str(path)

    def generate_html_report(self, metrics_data, assets=None):
        """Генерирует HTML-отчет с графиками и таблицами."""

        # 🧩 если assets — словарь, собираем все пути в один список
        all_assets = []
        if isinstance(assets, dict):
            for v in assets.values():
                if isinstance(v, (list, tuple)):
                    all_assets.extend(v)
                elif isinstance(v, str):
                    all_assets.append(v)
        elif isinstance(assets, (list, tuple)):
            all_assets = list(assets)
        elif isinstance(assets, str):
            all_assets = [assets]

        html_parts = []
        for p in all_assets:
            if not isinstance(p, str):
                continue
            if p.endswith(".html"):
                with open(p, "r", encoding="utf-8") as f:
                    html_parts.append(f.read())
            elif p.endswith(".png"):
                html_parts.append(f'<img src="{p}" style="max-width:100%;">')

        template = Template("""
        <html>
        <head>
            <meta charset="utf-8">
            <title>HuggingFace Big Data Metrics</title>
        </head>
        <body>
            <h1>📊 Отчёт по метрикам Hugging Face</h1>
            <p>Дата генерации: {{ date }}</p>
            <h2>📈 Числовые метрики</h2>
            <pre>{{ metrics_json }}</pre>
            <h2>🖼 Графические метрики</h2>
            {{ html_content | safe }}
        </body>
        </html>
        """)

        html_content = template.render(
            date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics_json=json.dumps(metrics_data, indent=2, ensure_ascii=False),
            html_content="\n".join(html_parts),
        )

        output_dir = Path("metrics_output1/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML отчёт сохранён: {output_path}")
        return str(output_path)

