import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Lock
from time import sleep
from unittest.mock import Mock

import numpy as np
import pytest

from api.schemas import SearchRequest
from domain.models import Thesis
from external.embedder import EncodedDocuments, EncoderError, ModelDescription
from infrastructure.storage import IndexMetadata, IndexSnapshot
from services.indexing import TEXT_PREPARATION_VERSION, build_index
from services.search import SearchService


MODEL = ModelDescription(identifier="fake", version="1", dimension=2, normalization="l2")
DIAGNOSTICS = {"bm25_score", "semantic_score", "bm25_rank", "semantic_rank"}


class FakeEncoder:
    model = MODEL

    def __init__(self):
        self.queries = []
        self.batches = []
        self.query_vector = np.array([1, 0], dtype=np.float32)
        self.error = None

    def encode_query(self, text):
        self.queries.append(text)
        if self.error is not None:
            raise self.error
        return self.query_vector

    def encode_documents(self, texts):
        self.batches.append(list(texts))
        return EncodedDocuments(np.tile([1.0, 0.0], (len(texts), 1)), 0)


class MemoryStorage:
    def __init__(self, snapshot=None):
        self.snapshot = snapshot
        self.reads = 0

    def load_snapshot(self):
        self.reads += 1
        return self.snapshot

    def replace_snapshot(self, snapshot):
        self.snapshot = snapshot


@pytest.fixture
def snapshot():
    cards = tuple(
        Thesis(
            id=card_id, title=title, abstract="Описание работы", keywords=["ВКР"],
            author="Автор", year=2025, url=f"https://example.org/{card_id}",
        )
        for card_id, title in [("2", "Поиск"), ("z", "Другое"), ("10", "Поиск")]
    )
    # Порядок строк матрицы намеренно отличается от порядка карточек.
    return IndexSnapshot(
        cards, ("z", "2", "10"), np.array([[-1, 0], [1, 0], [0, 1]], dtype=np.float32),
        IndexMetadata(model=MODEL, text_preparation_version=TEXT_PREPARATION_VERSION),
    )


@pytest.fixture
def prepared(snapshot):
    encoder, storage = FakeEncoder(), MemoryStorage(snapshot)
    return SearchService(storage=storage, encoder=encoder), encoder, storage


@pytest.mark.parametrize("top_k,expected_ids", [(1, ["10"]), (2, ["10", "2"]), (10, ["10", "2", "z"]), (100, ["10", "2", "z"])])
def test_search_fuses_channels_limits_and_resolves_ties(prepared, top_k, expected_ids):
    service, encoder, _ = prepared
    response = service.search(SearchRequest(search_mode="hybrid", query=" \nПоиск\t ", top_k=top_k, debug=True))
    assert response.query == "Поиск"
    assert encoder.queries == ["Поиск"]
    assert response.count == len(response.results) == len(set(expected_ids))
    assert [hit.id for hit in response.results] == expected_ids
    assert response.results[0].score == 1 / 61 + 1 / 62
    assert response.results[0].bm25_rank == 1
    assert response.results[0].semantic_rank == 2
    if top_k > 1:
        assert response.results[1].score == response.results[0].score
        assert response.results[1].bm25_rank == 2
        assert response.results[1].semantic_rank == 1
    for hit in response.results:
        assert hit.model_dump(include=set(Thesis.model_fields)) == service.get_thesis(hit.id).model_dump()


def test_no_word_overlap_still_uses_semantic_ranking(prepared):
    service, _, _ = prepared
    response = service.search(SearchRequest(search_mode="hybrid", query="космонавтика", debug=True))
    assert [hit.id for hit in response.results] == ["2", "10", "z"]
    assert [hit.semantic_score for hit in response.results] == [1, 0, -1]
    assert [hit.score for hit in response.results] == [1 / 61, 1 / 62, 1 / 63]
    assert all(hit.bm25_rank is None and hit.bm25_score is None for hit in response.results)


def test_debug_only_changes_diagnostics_and_repeated_search_is_stable(prepared):
    service, _, _ = prepared
    plain = service.search(SearchRequest(search_mode="hybrid", query="Поиск")).model_dump(mode="json")
    for _ in range(3):
        debug = service.search(SearchRequest(search_mode="hybrid", query="Поиск", debug=True)).model_dump(mode="json")
        assert all(DIAGNOSTICS <= hit.keys() for hit in debug["results"])
        debug["results"] = [
            {key: value for key, value in hit.items() if key not in DIAGNOSTICS}
            for hit in debug["results"]
        ]
        assert debug == plain
        assert service.search(SearchRequest(search_mode="hybrid", query="Поиск")).model_dump(mode="json") == plain


def test_card_lookup_and_health_do_not_encode_or_read_storage(prepared, snapshot):
    service, encoder, storage = prepared
    encoder.error = EncoderError("unavailable")
    for _ in range(2):
        assert service.get_thesis("2") == snapshot.cards[0]
        assert "score" not in service.get_thesis("2").model_dump()
        assert service.get_thesis("missing") is None
        assert service.health() == {
            "status": "ok", "index_loaded": True, "documents_count": 3,
            "embedding_model": "fake",
            "search_mode": "hybrid",
        }
    assert encoder.queries == encoder.batches == []
    assert storage.reads == 1


def test_snapshot_is_isolated_from_storage_and_returned_card_mutations(prepared, snapshot):
    service, _, storage = prepared
    request = SearchRequest(search_mode="hybrid", query="Поиск", debug=True)
    original = service.search(request).model_dump()
    snapshot.cards[0].title = "Изменено"
    snapshot.cards[0].keywords.append("изменено")
    snapshot.vectors[:] = [0, 1]
    storage.snapshot = None
    card = service.get_thesis("2")
    card.title = "Изменено снаружи"
    card.keywords.clear()
    response = service.search(request)
    response.results[0].title = "Изменена выдача"
    response.results[0].keywords.clear()
    assert service.search(request).model_dump() == original
    assert service.get_thesis("2").title == "Поиск"
    assert service.get_thesis("2").keywords == ["ВКР"]
    assert storage.reads == 1


@pytest.mark.parametrize("error", [EncoderError("model unavailable"), RuntimeError("bug"), TypeError("bug")])
def test_encoder_errors_propagate_without_fallback_and_lock_is_released(prepared, error):
    service, encoder, storage = prepared
    encoder.error = error
    with pytest.raises(type(error)) as caught:
        service.search(SearchRequest(search_mode="hybrid", query="Поиск"))
    assert caught.value is error
    encoder.error = None
    assert service.search(SearchRequest(search_mode="hybrid", query="Поиск")).count == 3
    assert storage.reads == 1


@pytest.mark.parametrize("vector", [[1], [[1, 0]], [0, 0], [2, 0], [float("nan"), 0], [float("inf"), 0], ["1", "0"], [1 + 0j, 0]])
def test_invalid_encoder_output_remains_an_error(prepared, vector):
    service, encoder, _ = prepared
    encoder.query_vector = np.array(vector)
    with pytest.raises(ValueError):
        service.search(SearchRequest(search_mode="hybrid", query="Поиск"))


@pytest.mark.parametrize("size", [1, 3, 105])
def test_indexes_are_built_once_and_channel_limits_are_capped(snapshot, monkeypatch, size):
    import services.search as search_module

    cards = tuple(snapshot.cards[0].model_copy(update={"id": f"{i:03}"}) for i in reversed(range(size)))
    snapshot = replace(
        snapshot, cards=cards, vector_ids=tuple(card.id for card in cards),
        vectors=np.tile([1.0, 0.0], (size, 1)),
    )
    bm25_factory = Mock(wraps=search_module.BM25Index)
    semantic_factory = Mock(wraps=search_module.SemanticIndex)
    monkeypatch.setattr(search_module, "BM25Index", bm25_factory)
    monkeypatch.setattr(search_module, "SemanticIndex", semantic_factory)
    encoder, storage = FakeEncoder(), MemoryStorage(snapshot)
    service = SearchService(storage=storage, encoder=encoder)
    bm25_search = Mock(wraps=service._bm25.search)
    semantic_search = Mock(wraps=service._semantic.search)
    monkeypatch.setattr(service._bm25, "search", bm25_search)
    monkeypatch.setattr(service._semantic, "search", semantic_search)
    for top_k in [1, 10, 100]:
        result = service.search(SearchRequest(search_mode="hybrid", query="Поиск", top_k=top_k))
        assert [hit.id for hit in result.results] == [f"{i:03}" for i in range(min(size, top_k))]
        assert len({hit.id for hit in result.results}) == result.count
    assert bm25_factory.call_count == semantic_factory.call_count == storage.reads == 1
    for search in (bm25_search, semantic_search):
        assert search.call_count == 3
        assert all(call.kwargs["limit"] == min(size, 100) for call in search.call_args_list)
    assert encoder.batches == []


def test_candidate_cutoff_is_applied_before_fusion(snapshot):
    cards = tuple(snapshot.cards[0].model_copy(update={"id": f"{i:03}"}) for i in range(101))
    # BM25: 000..099; semantic: 100,000..098. Документ 100 имеет только
    # семантический вклад 1/61, а документ 099 — только BM25-вклад 1/160.
    vectors = np.tile([0.0, 1.0], (101, 1))
    vectors[-1] = [1, 0]
    snapshot = replace(snapshot, cards=cards, vector_ids=tuple(card.id for card in cards), vectors=vectors)
    service = SearchService(storage=MemoryStorage(snapshot), encoder=FakeEncoder())
    results = service.search(SearchRequest(search_mode="hybrid", query="Поиск", top_k=100, debug=True)).results
    assert [hit.id for hit in results] == (
        [f"{i:03}" for i in range(61)] + ["100"] + [f"{i:03}" for i in range(61, 99)]
    )
    assert results[61].score == 1 / 61
    assert results[61].bm25_rank is None
    assert results[61].semantic_rank == 1


def test_build_load_and_search_never_reencode_corpus(tmp_path, snapshot):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([card.model_dump(mode="json") for card in snapshot.cards]), encoding="utf-8")
    encoder, storage = FakeEncoder(), MemoryStorage()
    build_index(path, encoder=encoder, storage=storage, batch_size=2)
    batches = list(encoder.batches)
    assert len(batches) == 2
    service = SearchService(storage=storage, encoder=encoder)
    assert encoder.queries == []
    for query in ["Поиск", "космонавтика", "ВКР"]:
        assert service.search(SearchRequest(search_mode="hybrid", query=query)).count == 3
    service.health()
    service.get_thesis("2")
    assert encoder.batches == batches
    assert encoder.queries == ["Поиск", "космонавтика", "ВКР"]
    assert storage.reads == 1


@pytest.mark.parametrize("problem", ["missing", "model", "ids"])
def test_preparation_rejects_unusable_snapshot_without_encoding(snapshot, problem):
    if problem == "missing":
        snapshot = None
    elif problem == "model":
        snapshot = replace(snapshot, metadata=IndexMetadata(
            model=MODEL.model_copy(update={"version": "other"}),
            text_preparation_version=TEXT_PREPARATION_VERSION,
        ))
    else:
        snapshot = replace(snapshot, vector_ids=("z", "2", "unknown"))
    encoder = FakeEncoder()
    with pytest.raises(ValueError):
        SearchService(storage=MemoryStorage(snapshot), encoder=encoder)
    assert encoder.batches == encoder.queries == []


def test_concurrent_queries_serialize_model_encoding(snapshot):
    class ConcurrentEncoder(FakeEncoder):
        def __init__(self):
            super().__init__()
            self.counter_lock = Lock()
            self.active = 0
            self.max_active = 0

        def encode_query(self, text):
            with self.counter_lock:
                self.active += 1
                self.max_active = max(self.active, self.max_active)
            try:
                sleep(0.01)  # Отдаём исполнение другим поисковым потокам.
                return super().encode_query(text)
            finally:
                with self.counter_lock:
                    self.active -= 1

    encoder = ConcurrentEncoder()
    storage = MemoryStorage(snapshot)
    service = SearchService(storage=storage, encoder=encoder)
    barrier = Barrier(4)

    def search(_):
        barrier.wait(timeout=5)
        return service.search(SearchRequest(search_mode="hybrid", query="Поиск")).model_dump()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(search, range(4)))
    assert all(result == results[0] for result in results)
    assert encoder.max_active == 1
    assert len(encoder.queries) == 4
    assert storage.reads == 1


def test_vectors_with_accepted_roundoff_work_from_import_through_search(tmp_path, snapshot):
    class RoundedEncoder(FakeEncoder):
        def encode_documents(self, texts):
            return EncodedDocuments(np.tile([1.0000105, 0.0], (len(texts), 1)), 0)

    encoder = RoundedEncoder()
    encoder.query_vector = np.array([1.0000105, 0.0])
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([card.model_dump(mode="json") for card in snapshot.cards]), encoding="utf-8")
    storage = MemoryStorage()
    build_index(path, encoder=encoder, storage=storage)
    service = SearchService(storage=storage, encoder=encoder)
    assert service.search(SearchRequest(search_mode="hybrid", query="Поиск")).count == 3
