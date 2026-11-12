"""Настройки процесса поверх .env из корня проекта."""

import os
import math
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET")


@dataclass(frozen=True)
class Settings:
    database_path: Path
    log_level: str
    ollama_enabled: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_timeout_seconds: float = 120.0


def load_settings() -> Settings:
    values = {**dotenv_values(PROJECT_ROOT / ".env", interpolate=False), **os.environ}
    log_level = values.get("LOG_LEVEL", "INFO")
    if log_level is None:
        raise ValueError("LOG_LEVEL: требуется значение")
    log_level = log_level.upper()
    if log_level not in LOG_LEVELS:
        raise ValueError(f"LOG_LEVEL: допустимы {', '.join(LOG_LEVELS)}")

    raw_path = values.get("DATABASE_PATH", "data/theses.sqlite3")
    if raw_path is None or not raw_path.strip() or "\x00" in raw_path:
        raise ValueError("DATABASE_PATH: требуется непустой путь к файлу SQLite без NUL")
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        path = path.resolve()
        if path.exists() and not path.is_file():
            raise ValueError("DATABASE_PATH: требуется путь к файлу SQLite, а не каталогу")
        if any(parent.exists() and not parent.is_dir() for parent in path.parents):
            raise ValueError("DATABASE_PATH: родительский путь должен быть каталогом")
    except (OSError, RuntimeError) as error:
        raise ValueError("DATABASE_PATH: не удалось разрешить путь к файлу SQLite") from error
    enabled = values.get("OLLAMA_ENABLED", "false")
    if enabled is None or enabled.lower() not in ("true", "false"):
        raise ValueError("OLLAMA_ENABLED: допустимы true или false")
    base_url = values.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    if not base_url:
        raise ValueError("OLLAMA_BASE_URL: требуется локальный HTTP-адрес Ollama")
    try:
        timeout = float(values.get("OLLAMA_TIMEOUT_SECONDS", "120"))
    except (ValueError, TypeError) as error:
        raise ValueError("OLLAMA_TIMEOUT_SECONDS: требуется число") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("OLLAMA_TIMEOUT_SECONDS: требуется конечное положительное число")
    return Settings(
        database_path=path, log_level=log_level, ollama_enabled=enabled.lower() == "true",
        ollama_base_url=base_url, ollama_timeout_seconds=timeout,
    )
