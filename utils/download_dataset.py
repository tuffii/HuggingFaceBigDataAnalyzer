from db.connection import get_connection
from db import schema
from scraper.fetch_models import fetch_all_models
from dotenv import load_dotenv
import os

DB_CONFIG = {
    "dbname": "postgres",
    "user": "BIGDATA",
    "password": "PASSWORD",
    "host": "localhost",
    "port": 5432,
}

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")


def main():
    conn = get_connection(DB_CONFIG)
    schema.init_db(conn)
    fetch_all_models(conn, os.getenv("HF_TOKEN"))
    conn.close()


if __name__ == "__main__":
    main()
