from __future__ import annotations
import sys
import os
import threading
import subprocess
from dotenv import load_dotenv

load_dotenv()
PYTHON_EXEC = sys.executable
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# корень проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# папка для результатов
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

METRICS = [
    {
        "name": "license_metrics",
        "script": "license_metrics.py",
        "args": [
            "--out", os.path.join(RESULTS_DIR, "license.png"),
            "--out-csv", os.path.join(RESULTS_DIR, "license.csv"),
            "--freq", "MS",
            "--top", "12",
        ],
    },
    {
        "name": "pipeline_tag_metrics",
        "script": "pipeline_tag_metrics.py",
        "args": [
            "--out", os.path.join(RESULTS_DIR, "pipeline_tag.png"),
            "--out-csv", os.path.join(RESULTS_DIR, "pipeline_tag.csv"),
            "--freq", "MS",
            "--top", "12",
        ],
    },
    {
        "name": "tags_metrics",
        "script": "tags_metrics.py",
        "args": [
            "--out", os.path.join(RESULTS_DIR, "tags.png"),
            "--out-csv", os.path.join(RESULTS_DIR, "tags.csv"),
            "--freq", "YS",
            "--top", "15",
            "--no-other",
        ],
    },
    {
        "name": "leaders_metrics",
        "script": "leaders_metrics.py",
        "args": [
            "--out", os.path.join(RESULTS_DIR, "leaders.png"),
            "--out-csv", os.path.join(RESULTS_DIR, "leaders.csv"),
            "--freq", "MS",
            "--top", "12",
        ],
    },
    {
        "name": "architecture_metric",
        "script": "architecture_metrics.py",
        "args": [
            "--out", os.path.join(RESULTS_DIR, "architecture.png"),
            "--out-csv", os.path.join(RESULTS_DIR, "architecture.csv"),
            "--freq", "MS",
            "--top", "12",
        ],
    },
    {
        "name": "downloads_time_metrics",
        "script": "downloads_time_metrics.py",
        "args": [
            "--out", os.path.join(RESULTS_DIR, "downloads_time.png"),
            "--out-csv", os.path.join(RESULTS_DIR, "downloads_time.csv"),
            "--freq", "MS",
        ],
    },
    {
        "name": "downloads_authors_metrics",
        "script": "downloads_authors_metrics.py",
        "args": [
            "--out", os.path.join(RESULTS_DIR, "downloads_authors.png"),
            "--out-csv", os.path.join(RESULTS_DIR, "downloads_authors.csv"),
            "--top", "31",
        ],
    },
    {
        "name": "prognosis",
        "script": "prognosis.py",
        "args": [
            "--out", os.path.join(RESULTS_DIR, "prognosis.png"),
            "--out-csv", os.path.join(RESULTS_DIR, "prognosis.csv"),
        ],
    },
]

def run_metric(metric: dict):
    """Запускает один пайплайн в отдельном процессе"""
    name = metric["name"]
    script = metric["script"]
    args = metric["args"]

    print(f"\n[START] {name}")
    try:
        result = subprocess.run(
            [PYTHON_EXEC, os.path.join(BASE_DIR, script)] + args,
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
