"""Преобразование выгрузки ВКР МИЭМ в JSON-тело PUT /theses."""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from api.schemas import ReplaceThesesRequest
from domain.models import Thesis


KEYWORD_MARKER = re.compile(r"\b(?:ключевые\s+слова|key\s*words)\s*[:–—-]", re.I)
KEYWORD_END = re.compile(
    r"\.(?=[ \t]*(?:\n|$))"
    r"|\n(?=[ \t]*(?:Актуальность|Объект|Цель|Методы|Результаты|Об[ъь]?[её]м|"
    r"Данная работа|Выпускная квалификационная работа)\b)",
    re.I,
)


def split_annotation(annotation: str) -> tuple[str, list[str]]:
    text = annotation.replace("\r\n", "\n").replace("\r", "\n")
    # Мягкий перенос явно обозначает разрыв слова; обычный дефис неоднозначен.
    text = re.sub(r"\u00ad\s*\n\s*", "", text).replace("\u00ad", "")
    match = KEYWORD_MARKER.search(text)
    keywords = []
    if match:
        tail = text[match.end():]
        boundary = KEYWORD_END.search(tail)
        end = boundary.start() if boundary else len(tail)
        for item in re.split(r"[,;]", tail[:end]):
            keyword = " ".join(item.split()).strip(" .")
            if keyword and keyword.casefold() not in {word.casefold() for word in keywords}:
                keywords.append(keyword)
        remaining = tail[boundary.end():] if boundary else ""
        text = text[:match.start()] + "\n" + remaining
    return " ".join(text.split()), keywords


def prepare_import(source: Path) -> ReplaceThesesRequest:
    cards = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            source_url = record["source_url"]
            url = urlsplit(source_url)
            page = re.fullmatch(r"/edu/vkr/(\d+)/?", url.path)
            if url.hostname != "www.hse.ru" or page is None:
                raise ValueError("Ожидается ссылка https://www.hse.ru/edu/vkr/<номер>")
            abstract, keywords = split_annotation(record["annotation"])
            cards.append(Thesis(
                id=f"hse-{page.group(1)}",
                title=record["title"],
                abstract=abstract,
                keywords=keywords,
                author=record["author_raw"],
                year=record["defense_year"],
                url=source_url,
            ))
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"{source}:{number}: {error}") from error
    return ReplaceThesesRequest(cards=cards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Исходный UTF-8 JSONL; относительный путь от текущего рабочего каталога",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help='UTF-8 JSON-объект {"cards": [...]} для PUT /theses; путь от текущего каталога; файл перезаписывается',
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Входной и выходной файлы должны различаться")
    try:
        payload = prepare_import(args.input)
        args.output.write_text(
            json.dumps(payload.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Сохранено {len(payload.cards)} карточек в {args.output}; "
        f"с ключевыми словами: {sum(bool(card.keywords) for card in payload.cards)}"
    )


if __name__ == "__main__":
    main()
