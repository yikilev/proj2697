import json
import sqlite3
import struct
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from domain.models import Thesis
from external.embedder import ModelDescription
from infrastructure.storage import IndexMetadata, IndexSnapshot, SQLiteStorage
from main import AppDependencies, create_app
from services.indexing import build_index, load_index
from tests.test_indexing import FakeEncoder


@pytest.fixture
def snapshot():
    cards = (
        Thesis(id="z", title="Поиск ВКР", abstract="Описание\nработы", keywords=["BM25", "ё"],
               author="Иван Иванов", year=2025, url="https://example.org/z?q=1#part"),
        Thesis(id="a", title="Другая работа", abstract="Описание", url="http://example.org/a"),
    )
    metadata = IndexMetadata(
        model=ModelDescription(identifier="test", version="revision-2", dimension=2, normalization="l2"),
        text_preparation_version="text-v3",
    )
    # Порядок векторов намеренно отличается от порядка карточек.
    return IndexSnapshot(cards, ("a", "z"), np.array([[0.6, 0.8], [-1, 0]], dtype=np.float32), metadata)


@pytest.fixture
def storage(tmp_path):
    return SQLiteStorage(tmp_path / "nested" / "index ?#.sqlite")


def execute(storage, sql, parameters=()):
    with closing(sqlite3.connect(storage.path)) as connection, connection:
        return connection.execute(sql, parameters).fetchall()


def assert_snapshot(actual, expected):
    assert {card.id: card for card in actual.cards} == {card.id: card for card in expected.cards}
    assert actual.metadata == expected.metadata
    actual_vectors = dict(zip(actual.vector_ids, actual.vectors))
    for card_id, vector in zip(expected.vector_ids, expected.vectors):
        np.testing.assert_array_equal(actual_vectors[card_id], vector)
    assert len(actual_vectors) == len(expected.vector_ids)
    assert actual.vectors.dtype == np.float32
    assert not actual.vectors.flags.writeable


def test_roundtrip_new_adapter_all_card_fields_metadata_and_id_mapping(storage, snapshot):
    storage.replace_snapshot(snapshot)
    loaded = SQLiteStorage(storage.path).load_snapshot()
    assert_snapshot(loaded, snapshot)
    assert [card.id for card in loaded.cards] == ["a", "z"]
    assert execute(storage, "SELECT vector FROM vectors WHERE card_id = 'a'")[0][0] == struct.pack("<ff", 0.6, 0.8)
    assert SQLiteStorage(storage.path).load_cards() == loaded.cards


@pytest.mark.parametrize("previous_kind", ["cards", "snapshot"])
def test_snapshot_completely_replaces_previous_corpus(storage, snapshot, previous_kind):
    if previous_kind == "cards":
        storage.replace_cards(snapshot.cards)
    else:
        storage.replace_snapshot(snapshot)
    card = snapshot.cards[0].model_copy(update={"id": "new"})
    metadata = snapshot.metadata.model_copy(update={"text_preparation_version": "next"})
    updated = IndexSnapshot((card,), ("new",), np.array([[0, 1]], dtype=np.float32), metadata)
    storage.replace_snapshot(updated)
    assert_snapshot(SQLiteStorage(storage.path).load_snapshot(), updated)
    assert execute(storage, "SELECT id FROM cards") == [("new",)]
    assert execute(storage, "SELECT card_id FROM vectors") == [("new",)]


def test_cards_only_creation_replacement_and_invalidation(storage, snapshot):
    storage.replace_cards(snapshot.cards)
    assert SQLiteStorage(storage.path).load_cards() == tuple(reversed(snapshot.cards))
    assert storage.load_snapshot() is None
    storage.replace_snapshot(snapshot)
    card = snapshot.cards[0].model_copy(update={"id": "new"})
    storage.replace_cards((card,))
    reloaded = SQLiteStorage(storage.path)
    assert reloaded.load_cards() == (card,)
    assert reloaded.load_snapshot() is None
    assert execute(storage, "SELECT * FROM vectors") == []
    assert execute(storage, "SELECT kind, metadata FROM index_state") == [("cards", None)]
    with pytest.raises(ValueError, match="missing"):
        load_index(storage=reloaded, model=snapshot.metadata.model)


@pytest.mark.parametrize("previous_kind", ["cards", "snapshot"])
@pytest.mark.parametrize("next_kind", ["cards", "snapshot"])
def test_write_failure_after_deletes_and_inserts_rolls_back(storage, snapshot, previous_kind, next_kind):
    if previous_kind == "snapshot":
        storage.replace_snapshot(snapshot)
    else:
        storage.replace_cards(snapshot.cards)
    # Последний INSERT падает, когда прежние данные уже удалены, а новые записаны.
    execute(storage, """
        CREATE TRIGGER fail_publication BEFORE INSERT ON index_state
        BEGIN SELECT RAISE(ABORT, 'publication failed'); END
    """)
    card = snapshot.cards[0].model_copy(update={"id": "new"})
    updated = IndexSnapshot((card,), ("new",), np.array([[0, 1]], dtype=np.float32), snapshot.metadata)
    with pytest.raises(sqlite3.IntegrityError, match="publication failed"):
        if next_kind == "snapshot":
            storage.replace_snapshot(updated)
        else:
            storage.replace_cards((card,))
    reloaded = SQLiteStorage(storage.path)
    assert reloaded.load_cards() == tuple(reversed(snapshot.cards))
    if previous_kind == "snapshot":
        assert_snapshot(reloaded.load_snapshot(), snapshot)
    else:
        assert reloaded.load_snapshot() is None


@pytest.mark.parametrize("kind", ["cards", "snapshot"])
@pytest.mark.parametrize("bad_cards", ["empty", "duplicates", "invalid"])
def test_rejects_bad_corpus_without_changing_database(storage, snapshot, kind, bad_cards):
    storage.replace_snapshot(snapshot)
    cards = {
        "empty": (),
        "duplicates": (snapshot.cards[0], snapshot.cards[0]),
        "invalid": (snapshot.cards[0].model_copy(update={"title": " "}),),
    }[bad_cards]
    with pytest.raises(ValueError):
        if kind == "cards":
            storage.replace_cards(cards)
        else:
            storage.replace_snapshot(replace(snapshot, cards=cards))
    assert_snapshot(storage.load_snapshot(), snapshot)


@pytest.mark.parametrize("changes", [
    {"vector_ids": ("a", "a")}, {"vector_ids": ("a", "unknown")},
    {"vectors": np.array([[1, 0]])}, {"vectors": np.array([[1, 0], [0, 0]])},
    {"vectors": np.array([[float("nan"), 0], [0, 1]])},
])
def test_rejects_invalid_snapshot_before_creating_database(storage, snapshot, changes):
    with pytest.raises(ValueError):
        storage.replace_snapshot(replace(snapshot, **changes))
    assert not storage.path.parent.exists()


def test_missing_storage_reads_do_not_create_file_or_directory(storage):
    assert storage.load_snapshot() is None
    assert storage.load_cards() is None
    assert not storage.path.parent.exists()


@pytest.mark.parametrize("contents", [b"", b"not a SQLite database"])
def test_existing_empty_or_invalid_file_is_not_missing(storage, snapshot, contents):
    storage.path.parent.mkdir()
    storage.path.write_bytes(contents)
    for read in (storage.load_snapshot, storage.load_cards):
        with pytest.raises((ValueError, sqlite3.DatabaseError)):
            read()
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        storage.replace_cards(snapshot.cards)
    assert storage.path.read_bytes() == contents


def test_replace_does_not_repair_incomplete_schema(storage, snapshot):
    storage.replace_cards(snapshot.cards)
    execute(storage, "DROP TABLE vectors")
    before = storage.path.read_bytes()
    with pytest.raises(ValueError, match="Invalid SQLite schema"):
        storage.replace_cards((snapshot.cards[0],))
    assert storage.path.read_bytes() == before


@pytest.mark.parametrize("sql,parameters", [
    ("DELETE FROM index_state", ()),
    ("DELETE FROM cards", ()),
    ("DELETE FROM vectors", ()),
    ("DELETE FROM vectors WHERE card_id = 'a'", ()),
    ("UPDATE vectors SET card_id = 'unknown' WHERE card_id = 'a'", ()),
    ("UPDATE cards SET keywords = 'not JSON'", ()),
    ("UPDATE cards SET keywords = '[1]'", ()),
    ("UPDATE cards SET title = ''", ()),
    ("UPDATE cards SET url = 'ftp://example.org'", ()),
    ("UPDATE cards SET year = 'unknown'", ()),
    ("UPDATE index_state SET metadata = '{}'", ()),
    ("UPDATE index_state SET metadata = 'not JSON'", ()),
    ("UPDATE index_state SET metadata = json_set(metadata, '$.model.dimension', 3)", ()),
    ("UPDATE vectors SET vector = ? WHERE card_id = 'a'", (b"\x00",)),
    ("UPDATE vectors SET vector = ? WHERE card_id = 'a'", (struct.pack("<ff", float("nan"), 0),)),
    ("UPDATE vectors SET vector = ? WHERE card_id = 'a'", (struct.pack("<ff", float("inf"), 0),)),
    ("UPDATE vectors SET vector = ? WHERE card_id = 'a'", (struct.pack("<ff", 0, 0),)),
    ("UPDATE vectors SET vector = ? WHERE card_id = 'a'", (struct.pack(">ff", 0.6, 0.8),)),
    ("UPDATE index_state SET kind = 'cards', metadata = NULL", ()),
    ("DROP TABLE vectors", ()),
])
def test_corrupt_saved_representation_is_not_masked(storage, snapshot, sql, parameters):
    storage.replace_snapshot(snapshot)
    execute(storage, sql, parameters)
    for read in (storage.load_snapshot, storage.load_cards):
        with pytest.raises((ValueError, sqlite3.DatabaseError)):
            read()


def test_empty_cards_only_state_is_corrupt(storage, snapshot):
    storage.replace_cards(snapshot.cards)
    execute(storage, "DELETE FROM cards")
    with pytest.raises(ValueError, match="nonempty"):
        storage.load_snapshot()
    with pytest.raises(ValueError, match="nonempty"):
        storage.load_cards()


def test_read_transaction_keeps_all_parts_consistent_during_replacement(storage, snapshot, monkeypatch):
    storage.replace_snapshot(snapshot)
    execute(storage, "PRAGMA journal_mode = WAL")
    transaction = storage._transaction
    replacement_completed = []

    @contextmanager
    def concurrent_transaction(*, write):
        with transaction(write=write) as connection:
            def replace_after_reading_state(sql):
                if sql.startswith("SELECT id, title"):
                    # WAL позволяет другому соединению завершить запись между SELECT.
                    SQLiteStorage(storage.path).replace_cards((snapshot.cards[0],))
                    replacement_completed.append(True)
            connection.set_trace_callback(replace_after_reading_state)
            yield connection

    monkeypatch.setattr(storage, "_transaction", concurrent_transaction)
    assert_snapshot(storage.load_snapshot(), snapshot)
    assert replacement_completed == [True]
    assert SQLiteStorage(storage.path).load_snapshot() is None
    assert SQLiteStorage(storage.path).load_cards() == (snapshot.cards[0],)


def test_shared_adapter_can_be_called_from_different_threads(storage, snapshot):
    storage.replace_snapshot(snapshot)

    def roundtrip(_):
        storage.replace_snapshot(snapshot)
        return storage.load_snapshot()

    with ThreadPoolExecutor(max_workers=4) as executor:
        for loaded in executor.map(roundtrip, range(8)):
            assert_snapshot(loaded, snapshot)


def test_connections_close_after_success_and_failure(storage, snapshot, monkeypatch):
    connect = sqlite3.connect
    connections = []

    def track_connection(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", track_connection)
    storage.replace_snapshot(snapshot)
    storage.load_snapshot()
    execute(storage, "CREATE TRIGGER fail BEFORE INSERT ON cards BEGIN SELECT RAISE(ABORT, 'fail'); END")
    with pytest.raises(sqlite3.IntegrityError):
        storage.replace_cards(snapshot.cards)
    execute(storage, "DELETE FROM vectors")
    with pytest.raises(ValueError):
        storage.load_cards()
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_indexing_service_with_real_sqlite(tmp_path, storage, snapshot):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([card.model_dump(mode="json") for card in snapshot.cards]), encoding="utf-8")
    encoder = FakeEncoder()
    result = build_index(corpus, encoder=encoder, storage=storage, batch_size=1)
    loaded = load_index(storage=SQLiteStorage(storage.path), model=encoder.model)
    assert result.cards_count == len(loaded.cards) == 2
    assert len(encoder.batches) == 2
    assert {card.id for card in loaded.cards} == {"a", "z"}
    app = create_app(dependency_factory=lambda: AppDependencies(encoder, SQLiteStorage(storage.path)))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["documents_count"] == 2
        response = client.post("/theses/search", json={"query": "Поиск"})
        assert response.status_code == 200
        assert response.json()["count"] == 2
        assert response.json()["results"][0]["id"] == "z"
        assert client.get("/theses/z").json() == snapshot.cards[0].model_dump(mode="json")
    assert len(encoder.batches) == 2
