import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from external.embedder import EncodedDocuments, ModelDescription, validate_vectors
from infrastructure.storage import IndexMetadata
from scripts.build_index import main
from services.indexing import (
    TEXT_PREPARATION_VERSION,
    IndexingResult,
    build_index,
    build_search_text,
    load_index,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ModelDescription(identifier="test-encoder", version="revision-1", dimension=2, normalization="l2")


class FakeEncoder:
    model = MODEL

    def __init__(self, outputs=()):
        self.outputs = outputs
        self.batches = []

    def encode_query(self, text):
        return np.array([1, 0], dtype=np.float32)

    def encode_documents(self, texts):
        self.batches.append(list(texts))
        if self.outputs:
            output = self.outputs[len(self.batches) - 1]
            if isinstance(output, Exception):
                raise output
            return output
        return EncodedDocuments(np.tile([1.0, 0.0], (len(texts), 1)), 0)


class MemoryStorage:
    def __init__(self, snapshot=None):
        self.snapshot = snapshot
        self.replacements = []

    def load_snapshot(self):
        return self.snapshot

    def replace_snapshot(self, snapshot):
        self.replacements.append(snapshot)
        self.snapshot = snapshot


@pytest.fixture
def records():
    return [
        {
            "id": card_id, "title": " Поиск ВКР ", "abstract": "Описание работы",
            "url": f"https://example.org/{card_id}", "keywords": ["BM25", "эмбеддинги"],
        }
        for card_id in ("z", "a", "b")
    ]


@pytest.fixture
def corpus(tmp_path, records):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def previous_snapshot(corpus):
    storage = MemoryStorage()
    build_index(corpus, encoder=FakeEncoder(), storage=storage)
    return storage.snapshot


def test_import_batches_id_mapping_metadata_and_truncation(corpus):
    encoder = FakeEncoder([
        EncodedDocuments(np.array([[1, 0], [0, 1]]), 1),
        EncodedDocuments(np.array([[-1, 0]]), 1),
    ])
    storage = MemoryStorage()
    result = build_index(corpus, encoder=encoder, storage=storage, batch_size=2)
    assert result == IndexingResult(cards_count=3, truncated_count=2)
    assert len(storage.replacements) == 1
    snapshot = storage.snapshot
    assert [card.id for card in snapshot.cards] == list(snapshot.vector_ids) == ["z", "a", "b"]
    np.testing.assert_array_equal(snapshot.vectors, [[1, 0], [0, 1], [-1, 0]])
    assert snapshot.metadata == IndexMetadata(model=MODEL, text_preparation_version=TEXT_PREPARATION_VERSION)
    assert encoder.batches == [
        [build_search_text(card) for card in snapshot.cards[:2]],
        [build_search_text(snapshot.cards[2])],
    ]
    assert encoder.batches[0][0] == "Поиск ВКР\nBM25 эмбеддинги\nОписание работы"
    assert not snapshot.vectors.flags.writeable
    # Адаптер может переиспользовать свой буфер: опубликованные данные независимы.
    encoder.outputs[0].vectors[:] = 0
    np.testing.assert_array_equal(snapshot.vectors[:2], [[1, 0], [0, 1]])


def test_sample_corpus_imports_without_model_or_network():
    storage = MemoryStorage()
    result = build_index(ROOT / "data/sample_theses.json", encoder=FakeEncoder(), storage=storage)
    assert result == IndexingResult(6, 0)
    assert len(load_index(storage=storage, model=MODEL).cards) == 6


@pytest.mark.parametrize("payload", [[], {}, None, "text", [1]])
def test_rejects_empty_or_wrong_corpus(corpus, payload, previous_snapshot):
    corpus.write_text(json.dumps(payload), encoding="utf-8")
    encoder, storage = FakeEncoder(), MemoryStorage(previous_snapshot)
    with pytest.raises(ValueError):
        build_index(corpus, encoder=encoder, storage=storage)
    assert encoder.batches == []
    assert storage.replacements == []
    assert storage.snapshot is previous_snapshot


@pytest.mark.parametrize("changes", [
    {"id": " z "}, {"id": "bad/id"}, {"title": " "}, {"abstract": None},
    {"url": "ftp://example.org"}, {"keywords": [1]}, {"year": True}, {"extra": "field"},
])
def test_rejects_bad_cards_and_duplicate_normalized_ids(corpus, records, changes, previous_snapshot):
    records[-1].update(changes)
    corpus.write_text(json.dumps(records), encoding="utf-8")
    encoder, storage = FakeEncoder(), MemoryStorage(previous_snapshot)
    with pytest.raises(ValueError):
        build_index(corpus, encoder=encoder, storage=storage)
    assert encoder.batches == []
    assert storage.replacements == []
    assert storage.snapshot is previous_snapshot


@pytest.mark.parametrize("content", [b"not json", b"[", b"\xff"])
def test_invalid_json_or_utf8_leaves_previous_index(corpus, content, previous_snapshot):
    corpus.write_bytes(content)
    storage = MemoryStorage(previous_snapshot)
    with pytest.raises(ValueError):
        build_index(corpus, encoder=FakeEncoder(), storage=storage)
    assert storage.replacements == []
    assert storage.snapshot is previous_snapshot


def test_missing_input_leaves_previous_index(tmp_path, previous_snapshot):
    storage = MemoryStorage(previous_snapshot)
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path / "missing.json", encoder=FakeEncoder(), storage=storage)
    assert storage.replacements == []
    assert storage.snapshot is previous_snapshot


@pytest.mark.parametrize("vectors", [
    [1, 0], [[[1, 0]]], [], [[1, 0], [0, 1]], [[1]], [[]], [[1, 0, 0]],
    [[0, 0]], [[2, 0]], [[0.99, 0]], [[float("nan"), 0]],
    [[float("inf"), 0]], [[-float("inf"), 0]], [[1e300, 1e300]],
    [["1", "0"]], [[True, False]], [[1 + 1j, 0]],
])
def test_bad_encoder_vectors_in_later_batch_preserve_previous_index(corpus, vectors, previous_snapshot):
    encoder = FakeEncoder([
        EncodedDocuments(np.array([[1, 0], [0, 1]]), 0),
        EncodedDocuments(np.array(vectors), 0),
    ])
    storage = MemoryStorage(previous_snapshot)
    with pytest.raises(ValueError):
        build_index(corpus, encoder=encoder, storage=storage, batch_size=2)
    assert len(encoder.batches) == 2
    assert storage.replacements == []
    assert storage.snapshot is previous_snapshot


@pytest.mark.parametrize("truncated_count", [-1, 2, 0.5, True, "1", None])
def test_invalid_truncation_count_preserves_previous_index(corpus, truncated_count, previous_snapshot):
    encoder = FakeEncoder([
        EncodedDocuments(np.array([[1, 0], [0, 1]]), 0),
        EncodedDocuments(np.array([[1, 0]]), truncated_count),
    ])
    storage = MemoryStorage(previous_snapshot)
    with pytest.raises(ValueError, match="truncated_count"):
        build_index(corpus, encoder=encoder, storage=storage, batch_size=2)
    assert storage.replacements == []
    assert storage.snapshot is previous_snapshot


def test_encoder_failure_after_first_batch_preserves_previous_index(corpus, previous_snapshot):
    encoder = FakeEncoder([
        EncodedDocuments(np.array([[1, 0], [0, 1]]), 0), RuntimeError("encoder failed"),
    ])
    storage = MemoryStorage(previous_snapshot)
    with pytest.raises(RuntimeError, match="encoder failed"):
        build_index(corpus, encoder=encoder, storage=storage, batch_size=2)
    assert storage.replacements == []
    assert storage.snapshot is previous_snapshot


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_rejects_invalid_batch_size(corpus, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        build_index(corpus, encoder=FakeEncoder(), storage=MemoryStorage(), batch_size=batch_size)


def test_does_not_estimate_tokens_or_truncate_text(corpus, records):
    records[0]["abstract"] = "слово " * 10000
    corpus.write_text(json.dumps(records), encoding="utf-8")
    encoder = FakeEncoder()
    result = build_index(corpus, encoder=encoder, storage=MemoryStorage())
    assert encoder.batches[0][0].endswith(records[0]["abstract"].strip())
    assert result.truncated_count == 0


def test_load_accepts_independent_card_and_vector_order(previous_snapshot):
    snapshot = replace(previous_snapshot, vector_ids=("b", "z", "a"), vectors=np.array([[0, 1], [-1, 0], [1, 0]]))
    storage = MemoryStorage(snapshot)
    loaded = load_index(storage=storage, model=MODEL)
    assert loaded.vector_ids == ("b", "z", "a")
    assert [card.id for card in loaded.cards] == ["z", "a", "b"]
    np.testing.assert_array_equal(loaded.vectors, snapshot.vectors)
    snapshot.vectors[:] = 0
    snapshot.cards[0].title = "changed"
    snapshot.cards[0].keywords.append("changed")
    assert loaded.cards[0].title == "Поиск ВКР"
    assert "changed" not in loaded.cards[0].keywords
    assert loaded.vectors[0, 1] == 1
    assert not loaded.vectors.flags.writeable
    assert storage.replacements == []


@pytest.mark.parametrize("changes", [
    {"identifier": "another-model"}, {"version": "revision-2"},
    {"dimension": 3}, {"normalization": "none"},
])
def test_load_rejects_each_incompatible_model_field(previous_snapshot, changes):
    stored_model = ModelDescription(**{**MODEL.model_dump(), **changes})
    metadata = IndexMetadata(model=stored_model, text_preparation_version=TEXT_PREPARATION_VERSION)
    storage = MemoryStorage(replace(previous_snapshot, metadata=metadata))
    with pytest.raises(ValueError, match="incompatible"):
        load_index(storage=storage, model=MODEL)


def test_load_rejects_incompatible_text_preparation(previous_snapshot):
    metadata = IndexMetadata(model=MODEL, text_preparation_version="old")
    with pytest.raises(ValueError, match="incompatible"):
        load_index(storage=MemoryStorage(replace(previous_snapshot, metadata=metadata)), model=MODEL)


@pytest.mark.parametrize("changes", [
    {"identifier": ""}, {"version": " "}, {"dimension": 0}, {"dimension": -1},
    {"dimension": True}, {"dimension": "2"}, {"normalization": "unknown"},
])
def test_invalid_model_description(changes):
    with pytest.raises(ValueError):
        ModelDescription(**{**MODEL.model_dump(), **changes})


def test_rejects_unnormalized_encoder(corpus):
    encoder, storage = FakeEncoder(), MemoryStorage()
    encoder.model = ModelDescription(**{**MODEL.model_dump(), "normalization": "none"})
    with pytest.raises(ValueError, match="L2"):
        build_index(corpus, encoder=encoder, storage=storage)
    assert encoder.batches == storage.replacements == []


def test_load_requires_an_existing_index():
    with pytest.raises(ValueError, match="missing"):
        load_index(storage=MemoryStorage(), model=MODEL)


@pytest.mark.parametrize("vector_ids", [("z", "a"), ("z", "a", "unknown"), ("z", "a", "a"), ("z", "a", "b", "extra")])
def test_load_rejects_missing_extra_and_duplicate_vector_ids(previous_snapshot, vector_ids):
    with pytest.raises(ValueError, match="IDs"):
        load_index(storage=MemoryStorage(replace(previous_snapshot, vector_ids=vector_ids)), model=MODEL)


@pytest.mark.parametrize("vectors", [
    [1, 0], [[1, 0]], [[1, 0, 0]] * 3, [[0, 0]] * 3,
    [[float("nan"), 0]] * 3, [[float("inf"), 0]] * 3, [[2, 0]] * 3,
])
def test_load_rejects_corrupt_vectors(previous_snapshot, vectors):
    with pytest.raises(ValueError):
        load_index(storage=MemoryStorage(replace(previous_snapshot, vectors=np.array(vectors))), model=MODEL)


def test_load_rejects_empty_duplicate_or_invalid_cards(previous_snapshot):
    for cards in [(), (previous_snapshot.cards[0],) * 3]:
        with pytest.raises(ValueError):
            load_index(storage=MemoryStorage(replace(previous_snapshot, cards=cards)), model=MODEL)
    previous_snapshot.cards[0].title = " "
    with pytest.raises(ValueError):
        load_index(storage=MemoryStorage(previous_snapshot), model=MODEL)


def test_float32_rounding_is_accepted_for_documents_and_query():
    vector = np.array([1, 1], dtype=np.float32) / np.sqrt(np.float32(2))
    np.testing.assert_array_equal(validate_vectors(vector, shape=(2,)), vector)
    np.testing.assert_array_equal(validate_vectors([vector], shape=(1, 2)), [vector])


def test_storage_failure_is_not_reported_as_success(corpus):
    class FailingStorage(MemoryStorage):
        def replace_snapshot(self, snapshot):
            raise OSError("write failed")

    with pytest.raises(OSError, match="write failed"):
        build_index(corpus, encoder=FakeEncoder(), storage=FailingStorage())


def test_cli_with_explicit_dependencies(corpus, capsys):
    storage = MemoryStorage()
    assert main(["--input", str(corpus)], encoder=FakeEncoder(), storage=storage) == 0
    assert "карточек: 3; обрезано текстов: 0" in capsys.readouterr().out
    assert len(storage.replacements) == 1


def test_cli_relative_path_is_resolved_from_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage = MemoryStorage()
    assert main(["--input", "data/sample_theses.json"], encoder=FakeEncoder(), storage=storage) == 0
    assert len(storage.snapshot.cards) == 6


def test_cli_data_error_has_nonzero_exit(corpus, capsys):
    corpus.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["--input", str(corpus)], encoder=FakeEncoder(), storage=MemoryStorage())
    assert error.value.code == 2
    assert "Индексация не выполнена" in capsys.readouterr().err


def test_cli_without_adapters_fails_clearly():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.build_index", "--input", "data/sample_theses.json"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "Адаптеры энкодера и хранилища не подключены" in result.stderr
    assert "Traceback" not in result.stderr



def test_cli_expected_encoder_failure_has_clean_error_and_preserves_index(corpus, previous_snapshot, capsys):
    from external.embedder import EncoderError

    storage = MemoryStorage(previous_snapshot)
    with pytest.raises(SystemExit) as caught:
        main(["--input", str(corpus)], encoder=FakeEncoder([EncoderError("unavailable")]), storage=storage)
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Индексация не выполнена: unavailable" in captured.err
    assert storage.snapshot is previous_snapshot
    assert storage.replacements == []


def test_cli_programming_error_is_not_disguised_as_encoder_outage(corpus):
    with pytest.raises(RuntimeError, match="bug"):
        main(["--input", str(corpus)], encoder=FakeEncoder([RuntimeError("bug")]), storage=MemoryStorage())


def test_float32_roundoff_cannot_publish_an_unloadable_index(corpus, previous_snapshot):
    # В float64 норма ещё в допуске; сохранение float32 выводит её за границу.
    encoder = FakeEncoder([EncodedDocuments(np.tile([0.999989001, 0], (3, 1)), 0)])
    storage = MemoryStorage(previous_snapshot)
    with pytest.raises(ValueError, match="L2-normalized"):
        build_index(corpus, encoder=encoder, storage=storage)
    assert storage.snapshot is previous_snapshot
    assert storage.replacements == []
