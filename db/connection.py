import psycopg2
from psycopg2.extras import RealDictCursor


def get_connection(db_config):
    """Создает и возвращает подключение к базе данных."""
    return psycopg2.connect(
        dbname=db_config["dbname"],
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"],
        cursor_factory=RealDictCursor
    )
