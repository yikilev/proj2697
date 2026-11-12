import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Event, get_ident

import pytest
from fastapi.testclient import TestClient

from infrastructure.storage import SQLiteStorage
from main import AppDependencies, create_app
from services.search import BM25SearchService
from tests.test_search_service import FakeEncoder, snapshot as snapshot


def card(card_id="first", title="Поиск"):
    return {
        "id": card_id, "title": title, "abstract": "Описание работы",
        "keywords": ["BM25"], "author": "Иван Иванов", "year": 2025,
        "url": f"https://example.org/{card_id.strip()}",
    }


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "corpus" / "theses.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    monkeypatch.setenv("SEARCH_MODE", "bm25")
    return SQLiteStorage(path)


@pytest.fixture
def running(database):
    with TestClient(create_app()) as client:
        yield client, database


def assert_corpus(client, database, expected):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok", "index_loaded": True, "documents_count": len(expected),
        "search_mode": "bm25", "embedding_model": None,
    }
    assert [item.model_dump(mode="json") for item in database.load_cards()] == expected
    for item in expected:
        assert client.get(f"/theses/{item['id']}").json() == item
    response = client.post("/theses/search", json={"query": "Описание", "debug": True})
    assert response.status_code == 200
    hits = response.json()["results"]
    assert {hit["id"] for hit in hits} == {item["id"] for item in expected}
    assert all(hit["semantic_rank"] is None and hit["semantic_score"] is None for hit in hits)


def test_first_import_replacement_repeat_and_restart(database):
    with TestClient(create_app()) as client:
        assert client.get("/docs").status_code == 200
        response = client.put("/theses", json={"cards": [card(" first ")]})
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "documents_count": 1, "search_mode": "bm25"}
        assert_corpus(client, database, [card()])
        response = client.post("/theses/search", json={"query": "Поиск"})
        assert response.json()["results"][0]["id"] == "first"
        replacement = [card("new", "Роботы"), card("next", "Машины")]
        for _ in range(2):
            response = client.put("/theses", json={"cards": replacement})
            assert response.status_code == 200
            assert response.json()["documents_count"] == 2
            assert_corpus(client, database, replacement)
            assert client.get("/theses/first").status_code == 404
            assert client.post("/theses/search", json={"query": "Поиск"}).json()["count"] == 0
            assert client.post("/theses/search", json={"query": "Роботы"}).json()["results"][0]["id"] == "new"
    with TestClient(create_app()) as restarted:
        assert_corpus(restarted, SQLiteStorage(database.path), replacement)
        assert restarted.get("/theses/first").status_code == 404


@pytest.mark.parametrize("payload", [
    {}, {"cards": []}, {"cards": None}, {"cards": {}}, {"cards": [None]},
    {"cards": [card()], "path": "corpus.json"},
    {"cards": [card(), card(" first ")]},
    *({"cards": [{**card(), field: value}]} for field, value in [
        ("title", " "), ("abstract", None), ("id", "bad/id"), ("id", 123),
        ("url", "file:///tmp/corpus"), ("year", "2025"), ("keywords", "BM25"),
        ("keywords", [123]), ("unknown", True),
    ]),
])
@pytest.mark.parametrize("loaded", [False, True])
def test_invalid_import_is_422_and_preserves_state(running, payload, loaded):
    client, database = running
    if loaded:
        assert client.put("/theses", json={"cards": [card()]}).status_code == 200
    before = database.path.read_bytes() if loaded else None
    service = client.app.state.search_service
    response = client.put("/theses", json=payload)
    assert response.status_code == 422
    assert client.app.state.search_service is service
    if loaded:
        assert database.path.read_bytes() == before
        assert_corpus(client, database, [card()])
    else:
        assert not database.path.exists()
        assert client.get("/health").status_code == 503


def test_openapi_describes_both_health_states_and_json_import(running):
    client, _ = running
    schema = client.get("/openapi.json").json()
    health = schema["paths"]["/health"]["get"]["responses"]
    assert health["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/HealthResponse")
    assert health["503"]["content"]["application/json"]["schema"]["$ref"].endswith("/NotReadyHealthResponse")
    put = schema["paths"]["/theses"]["put"]
    assert set(put["requestBody"]["content"]) == {"application/json"}
    assert {"200", "409", "422", "503"} <= put["responses"].keys()


def test_build_failure_keeps_database_and_search(running, monkeypatch):
    client, database = running
    client.put("/theses", json={"cards": [card()]})
    before = database.path.read_bytes()

    def fail(*args):
        raise RuntimeError("BM25 build failed")

    monkeypatch.setattr("services.search.BM25Index", fail)
    with pytest.raises(RuntimeError, match="BM25 build failed"):
        client.put("/theses", json={"cards": [card("new")]})
    assert database.path.read_bytes() == before
    assert_corpus(client, database, [card()])


def test_write_failure_rolls_back_and_keeps_search(running):
    client, database = running
    client.put("/theses", json={"cards": [card()]})
    with closing(sqlite3.connect(database.path)) as connection, connection:
        connection.execute("""
            CREATE TRIGGER fail BEFORE INSERT ON index_state
            BEGIN SELECT RAISE(ABORT, 'internal SQL /private/path'); END
        """)
    response = client.put("/theses", json={"cards": [card("new")]})
    assert response.status_code == 503
    assert response.json() == {"detail": "Хранилище временно недоступно; корпус не заменён"}
    assert_corpus(client, database, [card()])
    assert client.get("/theses/new").status_code == 404
    with closing(sqlite3.connect(database.path)) as connection, connection:
        connection.execute("DROP TRIGGER fail")
    assert client.put("/theses", json={"cards": [card("new")]}).status_code == 200
    assert_corpus(client, database, [card("new")])


@pytest.mark.parametrize("error", [PermissionError("/private/path"), sqlite3.OperationalError("locked SQL")])
def test_storage_unavailable_is_503(running, monkeypatch, error):
    client, database = running

    def fail(cards):
        raise error

    monkeypatch.setattr(client.app.state.storage, "replace_cards", fail)
    response = client.put("/theses", json={"cards": [card()]})
    assert response.status_code == 503
    assert str(error) not in response.text
    assert not database.path.exists()
    assert client.get("/health").status_code == 503


@pytest.mark.parametrize("error", [TypeError("bug"), sqlite3.ProgrammingError("bug")])
def test_write_programming_errors_propagate(running, monkeypatch, error):
    client, _ = running

    def fail(cards):
        raise error

    monkeypatch.setattr(client.app.state.storage, "replace_cards", fail)
    with pytest.raises(type(error), match="bug"):
        client.put("/theses", json={"cards": [card()]})


def test_concurrent_imports_serialize_write_and_publication(running, monkeypatch):
    client, database = running
    client.put("/theses", json={"cards": [card()]})
    state = client.app.state
    first_saved, second_waiting, second_saved = Event(), Event(), Event()
    release_first, release_second = Event(), Event()
    real_lock = state.import_lock

    class ObservedLock:
        def __enter__(self):
            if first_saved.is_set():
                second_waiting.set()
            real_lock.acquire()

        def __exit__(self, *args):
            real_lock.release()

    monkeypatch.setattr(state, "import_lock", ObservedLock())
    replace = state.storage.replace_cards

    def controlled_replace(cards):
        if cards[0].id == "second":
            # Вторая запись допускается только после публикации первой.
            assert state.search_service.get_thesis("first-update") is not None
        replace(cards)
        if cards[0].id == "first-update":
            first_saved.set()
            assert release_first.wait(5)
        else:
            second_saved.set()
            assert release_second.wait(5)

    monkeypatch.setattr(state.storage, "replace_cards", controlled_replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.put, "/theses", json={"cards": [card("first-update")]})
        try:
            assert first_saved.wait(5)
            second = pool.submit(client.put, "/theses", json={"cards": [card("second")]})
            assert second_waiting.wait(5)
            assert client.get("/theses/first").status_code == 200
            assert not first.done() and not second.done()
            release_first.set()
            assert first.result(timeout=5).status_code == 200
            assert second_saved.wait(5)
            assert client.get("/theses/first-update").status_code == 200
            assert not second.done()
        finally:
            release_first.set()
            release_second.set()
        assert second.result(timeout=5).status_code == 200
    assert_corpus(client, database, [card("second")])
    assert client.get("/theses/first-update").status_code == 404


def test_in_flight_search_finishes_on_one_old_snapshot(running, monkeypatch):
    client, database = running
    old = [card("same", "Старый поиск"), card("removed", "Старый поиск")]
    client.put("/theses", json={"cards": old})
    service = client.app.state.search_service
    started, release = Event(), Event()
    search = service.search

    def controlled_search(request):
        started.set()
        assert release.wait(5)
        return search(request)

    monkeypatch.setattr(service, "search", controlled_search)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, "/theses/search", json={"query": "Поиск"})
        try:
            assert started.wait(5)
            updated = [card("same", "Новый поиск")]
            assert client.put("/theses", json={"cards": updated}).status_code == 200
            assert_corpus(client, database, updated)
        finally:
            release.set()
        response = pending.result(timeout=5)
    assert response.status_code == 200
    hits = response.json()["results"]
    assert {hit["id"] for hit in hits} == {"same", "removed"}
    assert all(hit["title"] == "Старый поиск" for hit in hits)


def test_build_and_write_run_outside_event_loop(running, monkeypatch):
    client, database = running
    threads = []
    build = BM25SearchService.from_cards
    replace = SQLiteStorage.replace_cards

    def record_build(cards):
        threads.append(get_ident())
        assert not database.path.exists()
        return build(cards)

    def record_replace(self, cards):
        threads.append(get_ident())
        return replace(self, cards)

    monkeypatch.setattr(BM25SearchService, "from_cards", record_build)
    monkeypatch.setattr(SQLiteStorage, "replace_cards", record_replace)
    assert client.put("/theses", json={"cards": [card()]}).status_code == 200
    assert len(threads) == 2
    assert all(thread != client.portal.call(get_ident) for thread in threads)


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_actual_hybrid_rejects_import_without_changing_sqlite(database, snapshot, monkeypatch, mode):
    database.replace_snapshot(snapshot)
    before = database.path.read_bytes()
    encoder = FakeEncoder()
    monkeypatch.setenv("SEARCH_MODE", mode)
    app = create_app(dependency_factory=lambda: AppDependencies(encoder, database))
    with TestClient(app) as client:
        original = client.post("/theses/search", json={"query": "Поиск"}).json()
        response = client.put("/theses", json={"cards": [card()]})
        assert response.status_code == 409
        assert database.path.read_bytes() == before
        assert client.get("/health").json()["search_mode"] == "hybrid"
        assert client.post("/theses/search", json={"query": "Поиск"}).json() == original
