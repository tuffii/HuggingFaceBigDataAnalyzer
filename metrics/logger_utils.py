import json
import os
import datetime
from pathlib import Path
from loguru import logger

LOG_DIR = Path("metrics_output1/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
UNKNOWN_LOG_PATH = LOG_DIR / f"unknown_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


def log_unknown(category: str, model_id: str, data: dict):
    """
    Логирует нераспознанную категорию (other / unknown) с метаданными.

    :param category: тип категории (license, architecture, pipeline, tag и т.д.)
    :param model_id: идентификатор модели
    :param data: словарь с контекстом (например, сырые теги, license, config и т.д.)
    """
    try:
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "category": category,
            "model_id": model_id,
            "data": data,
        }

        with open(UNKNOWN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.warning(f"[UNCATEGORIZED] {category} → {model_id} ({list(data.keys())})")

    except Exception as e:
        logger.error(f"Ошибка при логировании неизвестного элемента ({category}): {e}")
