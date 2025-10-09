def extract_license(tags):
    """Извлекает лицензию из списка тегов."""
    if not tags or not isinstance(tags, list):
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split("license:")[1]
    return None


def insert_model(conn, model_data):
    """Вставляет запись о модели в БД."""
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO hf_models (
        model_id, pipeline_tag, license, downloads, private, tags,
        likes, createdAt, lastModified, cardData, config,
        gated, siblings, raw_data
        ) VALUES (
        %(modelId)s, %(pipeline_tag)s, %(license)s, %(downloads)s, %(private)s, %(tags)s,
        %(likes)s, %(createdAt)s, %(lastModified)s, %(cardData)s, %(config)s,
        %(gated)s, %(siblings)s, %(raw_data)s
        )
        ON CONFLICT (model_id) DO NOTHING;
        """, model_data)
    conn.commit()
