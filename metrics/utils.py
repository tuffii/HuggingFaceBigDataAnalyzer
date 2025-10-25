import json
import re
from typing import Optional
from .logger_utils import log_unknown

from loguru import logger


def safe_parse_json(maybe_json):
    if maybe_json is None:
        return None
    if isinstance(maybe_json, (dict, list)):
        return maybe_json
    try:
        return json.loads(maybe_json)
    except Exception:
        return None


def extract_company_from_model_id(model_id: str) -> Optional[str]:
    """Частая стратегия: author/model_name => берем часть до '/' как компанию/автора."""
    if not model_id:
        return None
    parts = model_id.split("/")
    if len(parts) >= 2:
        return parts[0].lower()
    return None


def extract_architecture_from_tags_or_raw(tags, raw_data) -> Optional[str]:
    """
    Попытки извлечь архитектуру:
    - искать теги вида 'architecture:bert' или 'arch:bert' или 'model_family:...'
    - искать в raw_data поля вроде 'library_name'/'transformersInfo'/'config'/'model_index'
    Возвращает строку или None.
    """
    try:
        # 1) по tags (ожидаем список строк)
        if tags:
            for t in tags:
                if isinstance(t, str):
                    m = re.match(r'(?i)(architecture|arch|model_family|family|base_model):\s*(.+)', t)
                    if m:
                        arch = m.group(2).strip().lower()
                        logger.debug(f"архитектура из тега: {arch}")
                        return arch
                    # общие ключи
                    m2 = re.match(r'(?i)(bert|gpt|vit|t5|llama|resnet|conv|unet|whisper|wav2vec|clip|dpt|swin|deit)', t)
                    if m2:
                        return m2.group(1).lower()
        # 2) raw_data и config (json)
        raw = None
        if isinstance(raw_data, str):
            try:
                raw = json.loads(raw_data)
            except Exception:
                raw = None
        elif isinstance(raw_data, dict):
            raw = raw_data
        if raw:
            # check common keys
            for key in ("library_name", "transformersInfo", "transformers_info", "config"):
                v = raw.get(key)
                if not v:
                    continue
                if isinstance(v, str):
                    # try find arch like 'bert'
                    m = re.search(r'(?i)(bert|gpt|t5|llama|vit|resnet|unet|whisper|wav2vec|clip|swin|deit)', v)
                    if m:
                        return m.group(1).lower()
                elif isinstance(v, dict):
                    # look for 'model_type' or 'architectures'
                    if 'model_type' in v:
                        return str(v['model_type']).lower()
                    if 'architectures' in v and isinstance(v['architectures'], list) and v['architectures']:
                        return str(v['architectures'][0]).lower()
        return None
    except Exception as e:
        logger.exception("Ошибка в extract_architecture_from_tags_or_raw: {}", e)
        return None


def ensure_list_from_json_field(field):
    if field is None:
        return []
    if isinstance(field, list):
        return field
    try:
        return json.loads(field)
    except Exception:
        return []
