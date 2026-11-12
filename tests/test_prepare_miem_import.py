import json

import pytest

from scripts.prepare_miem_import import prepare_import, split_annotation


@pytest.mark.parametrize(("text", "abstract", "keywords"), [
    (
        "Описание.\r\nКлючевые слова: CVSS, EPSS; KEV, CVSS.",
        "Описание.", ["CVSS", "EPSS", "KEV"],
    ),
    (
        "Ключевые слова: ГОСТ 34.12, FIPS 204, ML-DSA.\nОбъект исследования — подпись.",
        "Объект исследования — подпись.", ["ГОСТ 34.12", "FIPS 204", "ML-DSA"],
    ),
    (
        "Описание.\nКлючевые слова: ПЛИС, архи\u00ad\r\nтектура системы.",
        "Описание.", ["ПЛИС", "архитектура системы"],
    ),
    (
        "Ключевые слова: PIN-блок, ГОСТ,\nТехнический Комитет №26\nДанная работа посвящена защите.",
        "Данная работа посвящена защите.", ["PIN-блок", "ГОСТ", "Технический Комитет №26"],
    ),
    (
        "Описание.\nКлючевые слова – MATLAB\nОбъем 51 страниц, 10 рисунков.",
        "Описание. Объем 51 страниц, 10 рисунков.", ["MATLAB"],
    ),
    ("Распознавание ключевых слов на микроконтроллере.",
     "Распознавание ключевых слов на микроконтроллере.", []),
])
def test_keywords_preserve_annotation_and_technical_terms(text, abstract, keywords):
    assert split_annotation(text) == (abstract, keywords)


def test_mapping_and_duplicate_ids(tmp_path):
    source = tmp_path / "works.jsonl"
    record = {
        "title": "Название", "annotation": "Описание.",
        "source_url": "https://www.hse.ru/edu/vkr/123",
        "author_raw": "Автор", "defense_year": 2026,
        "supervisor_id": "person", "parser_version": "0.0.3",
    }
    line = json.dumps(record, ensure_ascii=False)
    source.write_text(line + "\n", encoding="utf-8")
    payload = prepare_import(source)
    assert payload.model_dump(mode="json") == {"cards": [{
        "id": "hse-123", "title": "Название", "abstract": "Описание.",
        "url": record["source_url"], "keywords": [], "author": "Автор", "year": 2026,
    }]}
    source.write_text(line + "\n" + line, encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        prepare_import(source)


def test_invalid_record_reports_source_line(tmp_path):
    source = tmp_path / "works.jsonl"
    source.write_text('\n{"title": "Без ссылки"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"works.jsonl:2:"):
        prepare_import(source)
