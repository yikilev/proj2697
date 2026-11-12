import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router
from api.schemas import SearchRequest
from domain.models import Thesis
from infrastructure.storage import SQLiteStorage
from services.indexing import read_corpus
from services.search import BM25SearchService
from tests.test_search_service import DIAGNOSTICS, snapshot as snapshot


@pytest.fixture
def cards():
    return tuple(
        Thesis(
            id=card_id, title=title, abstract="Описание работы", keywords=["ВКР"],
            author="Автор", year=2025, url=f"https://example.org/{card_id}",
        )
        for card_id, title in [("2", "Поиск"), ("z", "Ёлка"), ("10", "Поиск")]
    )


@pytest.fixture
def storage(tmp_path, cards):
    storage = SQLiteStorage(tmp_path / "index.sqlite")
    storage.replace_cards(cards)
    return storage


@pytest.fixture
def service(storage):
    return BM25SearchService(storage=storage)


@pytest.fixture
def client(service):
    # Изолированная проверка маршрутов; штатный lifespan проверяется в test_startup.
    app = FastAPI()
    app.state.search_service = service
    app.include_router(router)
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("top_k,ids", [(1, ["10"]), (2, ["10", "2"]), (100, ["10", "2"])])
def test_http_matches_top_k_stable_order_and_debug(client, cards, top_k, ids):
    payload = {"query": " \nПОИСК!\t ", "top_k": top_k}
    plain = client.post("/theses/search", json=payload)
    assert plain.status_code == 200
    body = plain.json()
    assert body["query"] == "ПОИСК!"
    assert body["count"] == len(ids)
    assert [hit["id"] for hit in body["results"]] == ids
    for _ in range(3):
        debug = client.post("/theses/search", json={**payload, "debug": True})
        assert debug.status_code == 200
        debug_body = debug.json()
        for rank, hit in enumerate(debug_body["results"], start=1):
            assert hit["score"] == 1 / (60 + rank)
            assert hit["bm25_rank"] == rank
            assert hit["bm25_score"] is not None
            assert hit["semantic_score"] is None
            assert hit["semantic_rank"] is None
        debug_body["results"] = [
            {key: value for key, value in hit.items() if key not in DIAGNOSTICS}
            for hit in debug_body["results"]
        ]
        assert debug_body == body
    original_cards = {card.id: card.model_dump(mode="json") for card in cards}
    for hit in body["results"]:
        assert set(hit) == set(Thesis.model_fields) | {"score"}
        assert {key: value for key, value in hit.items() if key != "score"} == original_cards[hit["id"]]


@pytest.mark.parametrize("debug", [False, True])
def test_http_no_overlap_is_empty(client, debug):
    response = client.post("/theses/search", json={"query": "космонавтика", "debug": debug})
    assert response.status_code == 200
    assert response.json() == {"query": "космонавтика", "count": 0, "results": []}


@pytest.mark.parametrize("query,ids", [
    ("ЕЛКА", ["z"]), ("описание", ["10", "2", "z"]),
    ("вкр", ["10", "2", "z"]), ("Автор", []), ("2025", []), ("example", []),
])
def test_search_text_fields_and_tokenization(service, query, ids):
    assert [hit.id for hit in service.search(SearchRequest(query=query)).results] == ids


def test_card_lookup_health_and_openapi(client, cards):
    card = client.get("/theses/2")
    assert card.status_code == 200
    assert card.json() == cards[0].model_dump(mode="json")
    missing = client.get("/theses/missing")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Работа не найдена"}
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok", "index_loaded": True, "documents_count": 3,
        "search_mode": "bm25", "embedding_model": None,
    }
    schema = client.get("/openapi.json").json()["components"]["schemas"]["HealthResponse"]
    assert {"search_mode", "embedding_model"} <= set(schema["required"])
    assert schema["properties"]["search_mode"]["enum"] == ["bm25", "hybrid"]
    assert {item["type"] for item in schema["properties"]["embedding_model"]["anyOf"]} == {"string", "null"}


@pytest.mark.parametrize("payload", [
    {}, {"query": " "}, {"query": "!?🙂"}, {"query": "я" * 4001},
    {"query": 123}, {"query": "поиск", "mode": "bm25"},
    *({"query": "поиск", "top_k": value} for value in [0, 101, "10", True]),
    {"query": "поиск", "debug": "true"},
])
def test_invalid_search_is_422(client, payload):
    response = client.post("/theses/search", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.parametrize("kind", ["cards", "snapshot"])
def test_reads_sqlite_once_without_vectors(storage, snapshot, monkeypatch, kind):
    if kind == "snapshot":
        storage.replace_snapshot(snapshot)
    connect = sqlite3.connect
    connections = []

    def authorize(action, table, column, database, trigger):
        if action == sqlite3.SQLITE_READ and table == "vectors":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def guarded_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connection.set_authorizer(authorize)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", guarded_connect)
    service = BM25SearchService(storage=storage)
    for _ in range(3):
        assert service.search(SearchRequest(query="Поиск")).count == 2
        assert service.get_thesis("2").title == "Поиск"
        assert service.health()["embedding_model"] is None
    assert len(connections) == 1


def test_reindex_visible_only_to_new_service_and_returned_cards_are_isolated(storage, service, cards):
    request = SearchRequest(query="Поиск", debug=True)
    original = service.search(request).model_dump()
    card = service.get_thesis("2")
    card.title = "Изменено"
    card.keywords.clear()
    response = service.search(request)
    response.results[0].keywords.clear()
    storage.replace_cards((cards[2].model_copy(update={"id": "new", "title": "Роботы"}),))
    assert service.search(request).model_dump() == original
    assert service.get_thesis("new") is None
    assert service.health()["documents_count"] == 3
    restarted = BM25SearchService(storage=SQLiteStorage(storage.path))
    assert restarted.search(request).count == 0
    assert restarted.search(SearchRequest(query="Роботы")).results[0].id == "new"
    assert restarted.get_thesis("2") is None
    assert restarted.health()["documents_count"] == 1


@pytest.mark.parametrize("size", [1, 3, 105])
def test_candidate_limit_and_nonpositive_bm25_scores(storage, cards, size):
    storage.replace_cards(tuple(
        cards[0].model_copy(update={"id": f"{i:03}"}) for i in reversed(range(size))
    ))
    service = BM25SearchService(storage=storage)
    response = service.search(SearchRequest(query="Поиск", top_k=100, debug=True))
    assert response.count == min(size, 100)
    assert [hit.id for hit in response.results] == [f"{i:03}" for i in range(min(size, 100))]
    assert all(hit.bm25_score < 0 for hit in response.results)
    assert [hit.score for hit in response.results] == [1 / (60 + rank) for rank in range(1, response.count + 1)]


def test_missing_corpus_fails_without_creating_database(tmp_path):
    path = tmp_path / "absent" / "index.sqlite"
    with pytest.raises(ValueError, match="BM25 corpus is missing or empty"):
        BM25SearchService(storage=SQLiteStorage(path))
    assert not path.parent.exists()


def test_bm25_score_order_precedes_id_order(storage, cards):
    storage.replace_cards((
        cards[0].model_copy(update={"id": "z", "title": "Поиск поиск"}),
        cards[0].model_copy(update={"id": "a", "title": "Поиск"}),
        *(cards[1].model_copy(update={"id": f"other-{i}"}) for i in range(3)),
    ))
    service = BM25SearchService(storage=storage)
    response = service.search(SearchRequest(query="Поиск", debug=True))
    assert [hit.id for hit in response.results] == ["z", "a"]
    assert response.results[0].bm25_score > response.results[1].bm25_score > 0
    assert [hit.score for hit in response.results] == [1 / 61, 1 / 62]


def test_zero_bm25_score_does_not_remove_matching_card(storage, cards):
    storage.replace_cards(cards[:2])
    response = BM25SearchService(storage=storage).search(SearchRequest(query="Поиск", debug=True))
    assert response.count == 1
    assert response.results[0].id == "2"
    assert response.results[0].bm25_score == 0
    assert response.results[0].score == 1 / 61


def test_empty_saved_corpus_fails(storage):
    with closing(sqlite3.connect(storage.path)) as connection, connection:
        connection.execute("DELETE FROM cards")
    with pytest.raises(ValueError, match="Stored corpus must be nonempty"):
        BM25SearchService(storage=storage)


def test_real_sample_cards_in_temporary_sqlite(tmp_path):
    cards = read_corpus(Path(__file__).resolve().parents[1] / "data" / "sample_theses.json")
    storage = SQLiteStorage(tmp_path / "sample.sqlite")
    storage.replace_cards(cards)
    service = BM25SearchService(storage=SQLiteStorage(storage.path))
    response = service.search(SearchRequest(query="Rasdaman", debug=True))
    assert response.count > 0
    expected = {card.id: card for card in cards}
    for hit in response.results:
        assert hit.model_dump(include=set(Thesis.model_fields)) == expected[hit.id].model_dump()
        assert hit.semantic_score is hit.semantic_rank is None
    assert service.health()["documents_count"] == len(cards)
