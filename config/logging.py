"""Настройка логирования OpenManus на основе dict-конфига (load_config)."""
import logging
import logging.handlers
import os
from typing import Any, Dict

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(config: Dict[str, Any]) -> None:
    """Настроить корневой логгер: уровень из config['log_level'], файл — из env OPENMANUS_LOG_FILE."""
    level = getattr(logging, str(config.get("log_level", "INFO")).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_FORMAT)

    log_file = os.environ.get("OPENMANUS_LOG_FILE")
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
