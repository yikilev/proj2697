"""Настройки из окружения процесса; .env автоматически не читается."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET")


@dataclass(frozen=True)
class Settings:
    search_mode: Literal["bm25", "hybrid"]
    database_path: Path
    log_level: str


def load_settings() -> Settings:
    search_mode = os.environ.get("SEARCH_MODE", "bm25")
    if search_mode not in ("bm25", "hybrid"):
        raise ValueError("SEARCH_MODE: допустимы только bm25 или hybrid")

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    if log_level not in LOG_LEVELS:
        raise ValueError(f"LOG_LEVEL: допустимы {', '.join(LOG_LEVELS)}")

    raw_path = os.environ.get("DATABASE_PATH", "data/theses.sqlite3")
    if not raw_path.strip() or "\x00" in raw_path:
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
    return Settings(search_mode=search_mode, database_path=path, log_level=log_level)
