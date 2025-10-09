from db.connection import get_connection
from db.schema import init_db
from scraper.fetch_models import fetch_all_models

DB_CONFIG = {
    "dbname": "postgres",
    "user": "BIGDATA",
    "password": "PASSWORD",
    "host": "localhost",
    "port": 5432,
}

HF_TOKEN = ""


def main():
    conn = get_connection(DB_CONFIG)
    init_db(conn)
    fetch_all_models(conn, HF_TOKEN)
    conn.close()


if __name__ == "__main__":
    main()
