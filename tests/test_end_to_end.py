"""Прикладной путь через настоящие сервисы и HTTP, с тестовыми адаптерами."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from external.embedder import EncodedDocuments, EncoderError, ModelDescription
from main import AppDependencies, create_app
from services.indexing import build_index


SAMPLE = Path(__file__).resolve().parents[1] / "data/sample_theses.json"
DIAGNOSTICS = {"bm25_score", "semantic_score", "bm25_rank", "semantic_rank"}


class DeterministicEncoder:
    model = ModelDescription(identifier="test-only", version="1", dimension=2, normalization="l2")

    def __init__(self):
        self.batches = []
        self.queries = []
        self.fail_batch = None

    def encode_documents(self, texts):
        self.batches.append(list(texts))
        if len(self.batches) == self.fail_batch:
            raise EncoderError("test encoding failure")
        # Фиксированные направления по уникальным словам примера корпуса.
        directions = {
            "UTAUT2": [0, 1], "ISM": [-1, 0], "налогообложение": [0, -1],
            "RoBERTa": [0.6, 0.8], "Rasdaman": [1, 0], "РКИ": [0.6, -0.8],
        }
        vectors = []
        for text in texts:
            matches = [vector for word, vector in directions.items() if word in text]
            assert len(matches) == 1, "Тестовый текст должен иметь одно фиксированное направление"
            vectors.append(matches[0])
        return EncodedDocuments(np.array(vectors, dtype=np.float32), 0)

    def encode_query(self, text):
        self.queries.append(text)
        return np.array([1, 0], dtype=np.float32)


class MemoryStorage:
    def __init__(self):
        self.snapshot = None
        self.reads = 0
        self.writes = 0

    def load_snapshot(self):
        self.reads += 1
        return self.snapshot

    def replace_snapshot(self, snapshot):
        self.snapshot = snapshot
        self.writes += 1


@pytest.fixture
def indexed_sample():
    encoder, storage = DeterministicEncoder(), MemoryStorage()
    result = build_index(SAMPLE, encoder=encoder, storage=storage, batch_size=2)
    assert (result.cards_count, result.truncated_count) == (6, 0)
    assert len(encoder.batches) == 3
    assert storage.writes == 1
    return storage


def test_sample_index_lifespan_http_search_card_and_health(indexed_sample):
    storage = indexed_sample
    encoder = DeterministicEncoder()
    records = {record["id"]: record for record in json.loads(SAMPLE.read_text(encoding="utf-8"))}
    # Семантический порядок задан направлениями fake; Rasdaman совпадает
    # только с одной карточкой BM25. Ожидания не вычисляются поисковым кодом.
    ids = [
        "hse-1162011207", "hse-1161668678", "hse-1163761122",
        "hse-1154627811", "hse-1159982720", "hse-1159021673",
    ]
    scores = [2 / 61, 1 / 62, 1 / 63, 1 / 64, 1 / 65, 1 / 66]
    app = create_app(dependency_factory=lambda: AppDependencies(encoder, storage))
    assert storage.reads == 0
    with TestClient(app) as client:
        assert storage.reads == 1
        for top_k in (1, 3, 10, 100):
            payload = {"query": " \nRasdaman\t", "top_k": top_k}
            plain = client.post("/theses/search", json=payload)
            debug = client.post("/theses/search", json={**payload, "debug": True})
            assert plain.status_code == debug.status_code == 200
            body = plain.json()
            assert body["query"] == "Rasdaman"
            assert body["count"] == min(top_k, 6)
            assert [hit["id"] for hit in body["results"]] == ids[:top_k]
            assert [hit["score"] for hit in body["results"]] == pytest.approx(scores[:top_k])
            for hit in body["results"]:
                assert DIAGNOSTICS.isdisjoint(hit)
                assert {key: value for key, value in hit.items() if key != "score"} == records[hit["id"]]
                card = client.get(f"/theses/{hit['id']}")
                assert card.status_code == 200
                assert card.json() == records[hit["id"]]
            debug_body = debug.json()
            assert [hit["semantic_rank"] for hit in debug_body["results"]] == list(range(1, min(top_k, 6) + 1))
            assert [hit["semantic_score"] for hit in debug_body["results"]] == pytest.approx([1, 0.6, 0.6, 0, 0, -1][:top_k])
            assert debug_body["results"][0]["bm25_rank"] == 1
            for hit in debug_body["results"][1:]:
                assert hit["bm25_rank"] is hit["bm25_score"] is None
            debug_body["results"] = [
                {key: value for key, value in hit.items() if key not in DIAGNOSTICS}
                for hit in debug_body["results"]
            ]
            assert debug_body == body
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok", "index_loaded": True, "documents_count": 6,
            "embedding_model": encoder.model.identifier,
        }
        assert encoder.queries == ["Rasdaman"] * 8
        assert encoder.batches == []
        assert storage.reads == 1


def test_reindex_is_visible_only_after_lifespan_restart(indexed_sample, tmp_path):
    storage = indexed_sample
    encoders = []

    def assemble():
        encoder = DeterministicEncoder()
        encoders.append(encoder)
        return AppDependencies(encoder, storage)

    app = create_app(dependency_factory=assemble)
    query = {"query": "Rasdaman", "debug": True}
    with TestClient(app) as client:
        original = client.post("/theses/search", json=query).json()
        old_card = client.get("/theses/hse-1162011207").json()
        replacement = [{**old_card, "title": "Новая редакция", "url": "https://example.org/revised"}]
        path = tmp_path / "replacement.json"
        path.write_text(json.dumps(replacement), encoding="utf-8")
        build_index(path, encoder=DeterministicEncoder(), storage=storage)
        assert storage.writes == 2
        assert client.post("/theses/search", json=query).json() == original
        assert client.get("/theses/hse-1162011207").json() == old_card
        assert client.get("/health").json()["documents_count"] == 6
        assert storage.reads == 1
    with TestClient(app) as client:
        response = client.post("/theses/search", json=query)
        assert response.status_code == 200
        assert response.json()["count"] == 1
        assert response.json()["results"][0]["title"] == "Новая редакция"
        assert client.get("/theses/hse-1162011207").json() == replacement[0]
        assert client.get("/theses/hse-1154627811").status_code == 404
        assert client.get("/health").json()["documents_count"] == 1
        assert storage.reads == len(encoders) == 2
        assert all(encoder.batches == [] for encoder in encoders)


@pytest.mark.parametrize("failure", ["duplicate", "encoding"])
def test_failed_reimport_preserves_published_sample(indexed_sample, tmp_path, failure):
    storage = indexed_sample
    previous = storage.snapshot
    path = SAMPLE
    encoder = DeterministicEncoder()
    if failure == "duplicate":
        records = json.loads(SAMPLE.read_text(encoding="utf-8"))
        records.append(records[0])
        path = tmp_path / "duplicate.json"
        path.write_text(json.dumps(records), encoding="utf-8")
    else:
        encoder.fail_batch = 2
    with pytest.raises((ValueError, EncoderError)):
        build_index(path, encoder=encoder, storage=storage, batch_size=2)
    assert storage.snapshot is previous
    assert storage.writes == 1
    app = create_app(dependency_factory=lambda: AppDependencies(DeterministicEncoder(), storage))
    with TestClient(app) as client:
        response = client.post("/theses/search", json={"query": "Rasdaman"})
        assert response.status_code == 200
        assert response.json()["count"] == 6
        assert response.json()["results"][0]["id"] == "hse-1162011207"
        assert client.get("/health").json()["documents_count"] == 6
