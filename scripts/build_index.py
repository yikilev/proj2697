"""Полная индексация через Ollama BGE-M3 с транзакционной записью SQLite."""

import argparse
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from config.settings import load_settings
from external.embedder import Encoder, EncoderError, OllamaEncoder
from infrastructure.storage import IndexStorage, SQLiteStorage
from services.indexing import build_index


def main(
    argv: Sequence[str] | None = None,
    *,
    encoder: Encoder | None = None,
    storage: IndexStorage | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Полная индексация UTF-8 JSON-корпуса ВКР")
    parser.add_argument(
        "--input", type=Path, required=True,
        help="UTF-8 JSON-массив карточек; относительный путь от корня проекта, не от текущего каталога",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Число карточек в батче (по умолчанию 32)")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size должен быть положительным")

    input_path = args.input
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parents[1] / input_path
    try:
        if encoder is None or storage is None:
            settings = load_settings()
            if encoder is None:
                encoder = OllamaEncoder(settings.ollama_base_url, settings.ollama_timeout_seconds)
            if storage is None:
                storage = SQLiteStorage(settings.database_path)
        result = build_index(input_path, encoder=encoder, storage=storage, batch_size=args.batch_size)
    except (OSError, ValueError, EncoderError, sqlite3.Error) as error:
        parser.error(f"Индексация не выполнена: {error}")
    print(f"Проиндексировано карточек: {result.cards_count}; обрезано текстов: {result.truncated_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
