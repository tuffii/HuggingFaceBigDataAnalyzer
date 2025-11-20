import subprocess
import pathlib
import sys

TABLES_DIR = pathlib.Path(__file__).parent
TABLE_MODULES = [f.stem for f in TABLES_DIR.glob("*.py") if f.name != "run_all_tables.py"]

def run_all_tables():
    for mod in TABLE_MODULES:
        print(f"\nЗапуск {mod}...")
        result = subprocess.run(
            [sys.executable, "-m", f"metrics.tables.{mod}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        if result.stderr:
            print(result.stderr)
        if result.returncode == 0:
            print(f"{mod}.py выполнен успешно.")
        else:
            print(f"Ошибка при выполнении {mod}.py (код {result.returncode})")

if __name__ == "__main__":
    run_all_tables()
