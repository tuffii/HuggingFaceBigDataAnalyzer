from __future__ import annotations
import sys
import os
import threading
import subprocess
from dotenv import load_dotenv

load_dotenv()
PYTHON_EXEC = sys.executable
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# гарантируем наличие каталога для результатов
os.makedirs("results", exist_ok=True)

METRICS = [
    {
        "name": "license_time_series",
        "script": "new_metrics/license_time_series.py",
        "args": [
            "--out", "results/license.png",
            "--out-csv", "results/license.csv",
            "--freq", "MS",
            "--top", "12",
        ],
    },
    {
        "name": "pipeline_tag_time_series",
        "script": "new_metrics/pipeline_tag_time_series.py",
        "args": [
            "--out", "results/pipeline_plot.png",
            "--out-csv", "results/pipeline_data.csv",
            "--freq", "MS",
            "--top", "12",
        ],
    },
    {
        "name": "tags_metrics",
        "script": "new_metrics/tags_metrics.py",
        "args": [
            "--out", "results/tags.png",
            "--out-csv", "results/tags.csv",
            "--freq", "YS",
            "--top", "15",
            "--no-other",  # <-- раньше здесь была пустая строка ""
        ],
    },
]


def run_metric(metric: dict):
    """Запускает один пайплайн в отдельном процессе"""
    name = metric["name"]
    script = metric["script"]
    args = metric["args"]

    print(f"\n🚀 [START] {name}")
    try:
        result = subprocess.run(
            [PYTHON_EXEC, script] + args,
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        print(f"✅ [DONE] {name}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ [FAIL] {name}")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)


def main():
    print("🌍 Loading .env and initializing HF_TOKEN")
    token = os.environ.get("HF_TOKEN", "")
    print(f"HF_TOKEN = {token[:10]}... (hidden)")

    print("🧩 Starting all metrics in parallel threads...\n")

    threads = []
    for metric in METRICS:
        t = threading.Thread(target=run_metric, args=(metric,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("\n🏁 All metrics pipelines finished!")


if __name__ == "__main__":
    main()
