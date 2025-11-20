from __future__ import annotations
import sys
import os
import threading
import subprocess
from dotenv import load_dotenv

load_dotenv()
PYTHON_EXEC = sys.executable
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRAPH_DIR = os.path.join(RESULTS_DIR, "graph")
CSV_DIR = os.path.join(RESULTS_DIR, "csv")

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


METRICS = [
    {
        "name": "license_metrics",
        "script": "license_metrics.py",
        "args": [
            "--out", os.path.join(GRAPH_DIR, "license.png"),
            "--out-csv", os.path.join(CSV_DIR, "license.csv"),
            "--freq", "MS",
            "--top", "12",
        ],
    },
    {
        "name": "category_metrics",
        "script": "category_metrics.py",
        "args": [
            "--out", os.path.join(GRAPH_DIR, "category.png"),
            "--out-csv", os.path.join(CSV_DIR, "category.csv"),
            "--freq", "MS",
            "--top", "12",
        ],
    },
    {
        "name": "architecture_metrics",
        "script": "architecture_metrics.py",
        "args": [
            "--out", os.path.join(GRAPH_DIR, "architecture.png"),
            "--out-csv", os.path.join(CSV_DIR, "architecture.csv"),
            "--freq", "MS",
            "--top", "12",
        ],
    },
    {
        "name": "downloads_metrics",
        "script": "downloads_metrics.py",
        "args": [
            "--out", os.path.join(GRAPH_DIR, "downloads.png"),
            "--out-csv", os.path.join(CSV_DIR, "downloads.csv"),
            "--freq", "MS",
        ],
    },
    {
        "name": "leaders_metrics",
        "script": "leaders_metrics.py",
        "args": [
            "--out", os.path.join(GRAPH_DIR, "leaders.png"),
            "--out-csv", os.path.join(CSV_DIR, "leaders.csv"),
            "--top", "31",
        ],
    },
    {
        "name": "prognosis",
        "script": "prognosis.py",
        "args": [
            "--out", os.path.join(GRAPH_DIR, "prognosis.png"),
            "--out-csv", os.path.join(CSV_DIR, "prognosis.csv"),
        ],
    },
]


def run_metric(metric: dict):
    """Запускает один пайплайн в отдельном процессе"""
    name = metric["name"]
    script = metric["script"]
    args = metric["args"]

    print(f"\n[START] {name}")

    script_path = os.path.join(BASE_DIR, "metrics", script)

    try:
        result = subprocess.run(
            [PYTHON_EXEC, script_path] + args,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=os.environ.copy(),
        )
        print(f"[DONE] {name}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {name}")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)

def main():
    print("Loading .env and initializing HF_TOKEN")
    token = os.environ.get("HF_TOKEN", "")
    print(f"HF_TOKEN = {token[:10]}... (hidden)")

    print("Starting all metrics in parallel threads...\n")

    threads = []
    for metric in METRICS:
        t = threading.Thread(target=run_metric, args=(metric,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("\nAll metrics pipelines finished!")

if __name__ == "__main__":
    main()
