from pathlib import Path
from loguru import logger

BASE_DIR = Path.cwd() / "metrics_output1"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"

for d in (BASE_DIR, PLOTS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Настройка логирования (логирует в stdout и в файл)
logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")  # stdout
logger.add(str(BASE_DIR / "metrics.log"), rotation="10 MB", level="DEBUG")
