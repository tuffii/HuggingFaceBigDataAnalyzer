def init_db(conn):
    """Создает таблицу для хранения информации о моделях."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS hf_models (
            id SERIAL PRIMARY KEY,
            model_id TEXT UNIQUE,
            pipeline_tag TEXT,
            license TEXT,
            downloads INTEGER,
            private TEXT,
            tags JSONB,
            likes INTEGER,
            createdAt TIMESTAMP,
            lastModified TIMESTAMP,
            cardData JSONB,
            config JSONB,
            gated TEXT,
            siblings JSONB,
            raw_data JSONB
            );
        """)
    conn.commit()
print("✅ Таблица hf_models готова.")
