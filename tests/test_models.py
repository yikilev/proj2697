import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from domain.models import Thesis


@pytest.fixture
def card():
    return {
        "id": "thesis-001",
        "title": "Гибридный поиск",
        "abstract": "Поиск по описаниям ВКР.",
        "url": "https://example.org/theses/001",
    }


def test_normalizes_all_strings_and_serializes_url(card):
    padded = {key: f" \t{value}\n " for key, value in card.items()}
    thesis = Thesis(**padded, keywords=[" BM25 ", "\tэмбеддинги\n"], author=" Иван Иванов ")
    assert thesis.model_dump(mode="json") == {
        **card,
        "keywords": ["BM25", "эмбеддинги"],
        "author": "Иван Иванов",
        "year": None,
    }


def test_optional_fields_and_independent_keyword_defaults(card):
    first = Thesis(**card)
    second = Thesis(**card)
    first.keywords.append("BM25")
    assert second.keywords == []
    assert second.author is None
    assert second.year is None


@pytest.mark.parametrize("field", ["id", "title", "abstract", "url"])
def test_required_fields_cannot_be_omitted(card, field):
    del card[field]
    with pytest.raises(ValidationError):
        Thesis(**card)


@pytest.mark.parametrize("field", ["id", "title", "abstract", "url"])
@pytest.mark.parametrize("value", ["", " \n\t ", None, 123, True, [], b"text"])
def test_required_fields_reject_empty_strings_and_wrong_types(card, field, value):
    card[field] = value
    with pytest.raises(ValidationError):
        Thesis(**card)


@pytest.mark.parametrize("value", ["/", "thesis/001", " /thesis-001 "])
def test_id_rejects_slashes(card, value):
    with pytest.raises(ValidationError):
        Thesis(**{**card, "id": value})


@pytest.mark.parametrize("url", ["ftp://example.org/a", "file:///tmp/a", "/theses/1", "example.org", "https://"])
def test_url_requires_http_or_https_with_host(card, url):
    with pytest.raises(ValidationError):
        Thesis(**{**card, "url": url})


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_accepts_http_and_https(card, scheme):
    url = f"{scheme}://example.org/theses/001"
    assert str(Thesis(**{**card, "url": url}).url) == url


@pytest.mark.parametrize("overrides", [
    {"keywords": None}, {"keywords": "BM25"}, {"keywords": [1]},
    {"keywords": ("BM25",)}, {"author": 123}, {"year": "2025"},
    {"year": True}, {"year": 2025.0}, {"unexpected": "value"},
])
def test_rejects_wrong_optional_types_and_extra_fields(card, overrides):
    with pytest.raises(ValidationError):
        Thesis(**{**card, **overrides})


def test_sample_corpus_validates_and_round_trips():
    path = Path(__file__).resolve().parents[1] / "data" / "sample_theses.json"
    for record in json.loads(path.read_text(encoding="utf-8")):
        thesis = Thesis.model_validate(record)
        assert thesis.model_dump(mode="json") == record
        assert Thesis.model_validate_json(thesis.model_dump_json()) == thesis
