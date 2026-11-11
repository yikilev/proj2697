import runpy
from dataclasses import replace
from threading import get_ident
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from external.embedder import EncoderError
from main import AppDependencies, create_app
from tests.test_search_service import (
    DIAGNOSTICS, MODEL, FakeEncoder, MemoryStorage, snapshot as snapshot,
)


class FakeAssembly:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.calls = 0

    def __call__(self):
        self.calls += 1
        self.encoder = FakeEncoder()
        self.storage = MemoryStorage(self.snapshot)
        return AppDependencies(encoder=self.encoder, storage=self.storage)


@pytest.fixture
def running(snapshot):
    assembly = FakeAssembly(snapshot)
    app = create_app(dependency_factory=assembly)
    with TestClient(app) as client:
        yield client, assembly


def test_lifespan_initializes_once_and_health_never_encodes(snapshot, monkeypatch):
    import services.search as search_module

    bm25 = Mock(wraps=search_module.BM25Index)
    semantic = Mock(wraps=search_module.SemanticIndex)
    monkeypatch.setattr(search_module, "BM25Index", bm25)
    monkeypatch.setattr(search_module, "SemanticIndex", semantic)
    assembly = FakeAssembly(snapshot)
    app = create_app(dependency_factory=assembly)
    assert assembly.calls == 0
    with TestClient(app) as client:
        assert assembly.calls == assembly.storage.reads == 1
        assert assembly.encoder.queries == assembly.encoder.batches == []
        for _ in range(3):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {
                "status": "ok", "index_loaded": True,
                "documents_count": 3, "embedding_model": MODEL.identifier,
            }
            assert client.get("/theses/2").status_code == 200
        assert assembly.encoder.queries == []
        for _ in range(3):
            assert client.post("/theses/search", json={"query": "Поиск"}).status_code == 200
        assert assembly.calls == assembly.storage.reads == bm25.call_count == semantic.call_count == 1
        assert assembly.encoder.queries == ["Поиск"] * 3
        assert assembly.encoder.batches == []
    assert app.state.search_service is None


@pytest.mark.parametrize("top_k,ids", [(1, ["10"]), (2, ["10", "2"]), (100, ["10", "2", "z"])])
def test_search_http_ranking_and_debug(running, top_k, ids):
    client, assembly = running
    payload = {"query": " \nПоиск\t ", "top_k": top_k, "debug": False}
    plain = client.post("/theses/search", json=payload)
    debug = client.post("/theses/search", json={**payload, "debug": True})
    assert plain.status_code == debug.status_code == 200
    body = plain.json()
    assert body["query"] == "Поиск"
    assert body["count"] == len(ids)
    assert [hit["id"] for hit in body["results"]] == ids
    assert body["results"][0]["score"] == 1 / 61 + 1 / 62
    assert all(DIAGNOSTICS.isdisjoint(hit) for hit in body["results"])
    debug_body = debug.json()
    assert all(DIAGNOSTICS <= hit.keys() for hit in debug_body["results"])
    debug_body["results"] = [
        {key: value for key, value in hit.items() if key not in DIAGNOSTICS}
        for hit in debug_body["results"]
    ]
    assert debug_body == body
    assert assembly.encoder.queries == ["Поиск", "Поиск"]


def test_debug_keeps_null_for_missing_bm25_channel(running):
    client, _ = running
    response = client.post("/theses/search", json={"query": "космонавтика", "debug": True})
    assert response.status_code == 200
    for hit in response.json()["results"]:
        assert hit["bm25_score"] is None
        assert hit["bm25_rank"] is None
        assert hit["semantic_rank"] is not None


def test_debug_keeps_null_for_missing_semantic_channel(snapshot):
    cards = tuple(snapshot.cards[0].model_copy(update={
        "id": f"{i:03}", "title": "Поиск" if i == 100 else "Другое",
    }) for i in range(101))
    snapshot = replace(snapshot, cards=cards, vector_ids=tuple(card.id for card in cards),
                       vectors=np.tile([1.0, 0.0], (101, 1)))
    with TestClient(create_app(dependency_factory=FakeAssembly(snapshot))) as client:
        response = client.post("/theses/search", json={"query": "Поиск", "top_k": 100, "debug": True})
        assert response.status_code == 200
        hit = next(hit for hit in response.json()["results"] if hit["id"] == "100")
        assert hit["bm25_rank"] == 1
        assert hit["semantic_score"] is None
        assert hit["semantic_rank"] is None


@pytest.mark.parametrize("payload", [
    {}, {"query": ""}, {"query": " \n\t "}, {"query": "!?🙂"},
    {"query": "я" * 4001}, {"query": None}, {"query": 123},
    {"query": "Поиск", "extra": True},
    *({"query": "Поиск", "top_k": value} for value in [0, 101, "10", 1.5, True, None]),
    *({"query": "Поиск", "debug": value} for value in [0, "true", None]),
])
def test_invalid_request_is_422_without_inference(running, payload):
    client, assembly = running
    response = client.post("/theses/search", json=payload)
    assert response.status_code == 422
    assert set(response.json()) == {"detail"}
    errors = response.json()["detail"]
    assert isinstance(errors, list) and errors
    assert all({"loc", "msg", "type"} <= error.keys() for error in errors)
    assert all(error["loc"][0] == "body" for error in errors)
    assert assembly.encoder.queries == []


def test_malformed_json_is_422(running):
    client, assembly = running
    response = client.post("/theses/search", content="{", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert set(response.json()) == {"detail"}
    assert response.json()["detail"][0]["type"] == "json_invalid"
    assert assembly.encoder.queries == []


def test_card_and_unknown_id(running, snapshot):
    client, assembly = running
    response = client.get("/theses/2")
    assert response.status_code == 200
    assert response.json() == snapshot.cards[0].model_dump(mode="json")
    missing = client.get("/theses/missing")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Работа не найдена"}
    assert assembly.encoder.queries == []


def test_expected_encoder_error_is_503_and_health_does_not_probe_model(running):
    client, assembly = running
    assembly.encoder.error = EncoderError("unavailable")
    response = client.post("/theses/search", json={"query": "Поиск"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Энкодер временно недоступен"}
    assert client.get("/health").status_code == 200
    assert assembly.encoder.queries == ["Поиск"]
    assembly.encoder.error = None
    assert client.post("/theses/search", json={"query": "Поиск"}).status_code == 200


@pytest.mark.parametrize("error", [RuntimeError("bug"), TypeError("bug"), ValueError("bug")])
def test_programming_errors_propagate(running, error):
    client, assembly = running
    assembly.encoder.error = error
    with pytest.raises(type(error)) as caught:
        client.post("/theses/search", json={"query": "Поиск"})
    assert caught.value is error


def test_invalid_encoder_vector_is_not_reported_as_expected_outage(running):
    client, assembly = running
    assembly.encoder.query_vector = np.array([0.0, 0.0])
    with pytest.raises(ValueError, match="L2-normalized"):
        client.post("/theses/search", json={"query": "Поиск"})


@pytest.mark.parametrize("problem,match", [
    ("missing", "Index is missing"), ("model", "metadata is incompatible"),
    ("ids", "IDs must match"), ("vectors", "Vectors must have shape"),
])
def test_invalid_snapshot_aborts_startup(snapshot, problem, match):
    if problem == "missing":
        snapshot = None
    elif problem == "model":
        snapshot = replace(snapshot, metadata=snapshot.metadata.model_copy(update={
            "model": MODEL.model_copy(update={"version": "other"}),
        }))
    elif problem == "ids":
        snapshot = replace(snapshot, vector_ids=("2", "10", "unknown"))
    else:
        snapshot = replace(snapshot, vectors=np.ones((3, 3)))
    assembly = FakeAssembly(snapshot)
    app = create_app(dependency_factory=assembly)
    with pytest.raises(ValueError, match=match):
        with TestClient(app):
            pytest.fail("Startup must fail")
    assert app.state.search_service is None
    assert assembly.calls == assembly.storage.reads == 1
    assert assembly.encoder.queries == assembly.encoder.batches == []


def test_missing_integrations_abort_startup():
    with pytest.raises(RuntimeError, match="Адаптеры энкодера и хранилища не подключены"):
        with TestClient(create_app()):
            pytest.fail("Startup must fail")


def test_factory_failure_aborts_startup():
    factory = Mock(side_effect=EncoderError("cannot load model"))
    with pytest.raises(EncoderError, match="cannot load model"):
        with TestClient(create_app(dependency_factory=factory)):
            pytest.fail("Startup must fail")
    factory.assert_called_once_with()


def test_health_without_lifespan_cannot_report_ready():
    client = TestClient(create_app())
    responses = [
        client.get("/health"), client.get("/theses/2"),
        client.post("/theses/search", json={"query": "Поиск"}),
    ]
    for response in responses:
        assert response.status_code == 503
        assert response.json() == {"detail": "Поисковый сервис не инициализирован"}


def test_import_main_does_not_prepare_service_or_open_database(monkeypatch):
    service = Mock(side_effect=AssertionError("Service created during import"))
    connect = Mock(side_effect=AssertionError("Database opened during import"))
    monkeypatch.setattr("services.search.SearchService", service)
    monkeypatch.setattr("sqlite3.connect", connect)
    module = runpy.run_module("main")
    assert module["app"].state.search_service is None
    service.assert_not_called()
    connect.assert_not_called()


def test_synchronous_service_calls_run_outside_event_loop(running, monkeypatch):
    client, _ = running
    service = client.app.state.search_service
    threads = []

    def record_thread(method):
        def call(*args):
            threads.append(get_ident())
            return method(*args)
        return call

    for name in ("search", "get_thesis", "health"):
        monkeypatch.setattr(service, name, record_thread(getattr(service, name)))
    event_loop_thread = client.portal.call(get_ident)
    assert client.post("/theses/search", json={"query": "Поиск"}).status_code == 200
    assert client.get("/theses/2").status_code == 200
    assert client.get("/health").status_code == 200
    assert len(threads) == 3
    assert all(thread != event_loop_thread for thread in threads)
