"""CLI индексации. Реальные адаптеры ещё не подключены."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from external.embedder import Encoder, EncoderError
from infrastructure.storage import IndexStorage
from services.indexing import build_index


def main(
    argv: Sequence[str] | None = None,
    *,
    encoder: Encoder | None = None,
    storage: IndexStorage | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Полная индексация UTF-8 JSON-корпуса ВКР")
    parser.add_argument("--input", type=Path, required=True, help="Путь к JSON-массиву карточек")
    args = parser.parse_args(argv)
    if encoder is None or storage is None:
        parser.error("Адаптеры энкодера и хранилища не подключены; передайте encoder и storage в main().")

    input_path = args.input
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parents[1] / input_path
    try:
        result = build_index(input_path, encoder=encoder, storage=storage)
    except (OSError, ValueError, EncoderError) as error:
        parser.error(f"Индексация не выполнена: {error}")
    print(f"Проиндексировано карточек: {result.cards_count}; обрезано текстов: {result.truncated_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
