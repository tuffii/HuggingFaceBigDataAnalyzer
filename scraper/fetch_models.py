import json
import time
from huggingface_hub import HfApi
from db.utils import extract_license, insert_model

def safe_serialize(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_serialize(v) for v in obj]
    if hasattr(obj, "dict"):
        return safe_serialize(obj.dict)
    return str(obj)

def fetch_all_models(conn, hf_token: str):
    """
    Полная выгрузка всех публичных моделей Hugging Face (~2M+).
    HuggingFaceHub сам подгружает данные постранично.
    """
    api = HfApi(token=hf_token)
    total = 0
    print("🚀 Начинаю полную выгрузку моделей с Hugging Face...")

    # Без ограничения limit — Hugging Face возвращает всё, подгружая постранично
    for m in api.list_models(full=True):
        try:
            tags = safe_serialize(getattr(m, "tags", None))
            model_data = {
                "modelId": m.id,
                "pipeline_tag": getattr(m, "pipeline_tag", None),
                "license": extract_license(tags),
                "downloads": getattr(m, "downloads", None),
                "private": str(getattr(m, "private", False)),
                "tags": json.dumps(tags, ensure_ascii=False),
                "likes": getattr(m, "likes", None),
                "createdAt": getattr(m, "created_at", None),
                "lastModified": getattr(m, "last_modified", None),
                "cardData": json.dumps(safe_serialize(getattr(m, "card_data", None)), ensure_ascii=False),
                "config": json.dumps(safe_serialize(getattr(m, "config", None)), ensure_ascii=False),
                "gated": str(getattr(m, "gated", False)),
                "siblings": json.dumps(safe_serialize(getattr(m, "siblings", None)), ensure_ascii=False),
                "raw_data": json.dumps(safe_serialize(m.__dict__), ensure_ascii=False),
            }

            insert_model(conn, model_data)
            total += 1

            if total % 100 == 0:
                print(f"✅ Загружено {total} моделей...")

            if total % 500 == 0:
                time.sleep(1)

        except Exception as inner_e:
            print(f"⚠️ Ошибка при вставке модели {getattr(m, 'id', '?')}: {inner_e}")
            conn.rollback()

    print(f"🎉 Полная выгрузка завершена. Всего обработано: {total} моделей.")
