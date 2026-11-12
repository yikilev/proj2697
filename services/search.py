"""Поиск по одному снимку в памяти и объединение рейтингов через RRF."""

from collections.abc import Sequence
from _thread import LockType
from threading import Lock
from typing import Self

from api.schemas import DebugSearchResult, SearchRequest, SearchResponse
from domain.models import FusedHit, SearchHit, Thesis
from external.embedder import Encoder, EncoderError, validate_vectors
from infrastructure.search import BM25Index, SemanticIndex
from infrastructure.storage import IndexSnapshot, IndexStorage, SQLiteStorage
from services.indexing import build_search_text, load_index, validate_index_snapshot


CANDIDATE_LIMIT = 100


class SearchService:
    """Поисковый снимок; экземпляры приложения используют общую блокировку энкодера."""

    def __init__(
        self, *, storage: IndexStorage, encoder: Encoder, encode_lock: LockType | None = None,
    ) -> None:
        snapshot = load_index(storage=storage, model=encoder.model)
        self._initialize(snapshot, encoder, encode_lock)

    @classmethod
    def from_snapshot(
        cls, snapshot: IndexSnapshot, *, encoder: Encoder, encode_lock: LockType,
    ) -> Self:
        """Построить поиск до сохранения и публикации нового снимка."""
        service = cls.__new__(cls)
        service._initialize(validate_index_snapshot(snapshot, model=encoder.model), encoder, encode_lock)
        return service

    def _initialize(
        self, snapshot: IndexSnapshot, encoder: Encoder, encode_lock: LockType | None,
    ) -> None:
        self._model = snapshot.metadata.model
        self._encoder = encoder
        self._encode_lock = encode_lock if encode_lock is not None else Lock()
        self._cards = {card.id: card for card in snapshot.cards}
        self._candidate_limit = min(CANDIDATE_LIMIT, len(self._cards))
        self._bm25 = BM25Index(
            [card.id for card in snapshot.cards],
            [build_search_text(card) for card in snapshot.cards],
        )
        self._semantic = SemanticIndex(snapshot.vector_ids, snapshot.vectors)

    def search(self, request: SearchRequest) -> SearchResponse:
        semantic_hits = []
        if request.search_mode == "hybrid":
            with self._encode_lock:
                # Копия защищает от повторного использования буфера адаптером.
                query_vector = validate_vectors(
                    self._encoder.encode_query(request.query), shape=(self._model.dimension,),
                )
            semantic_hits = self._semantic.search(query_vector, limit=self._candidate_limit)
        bm25_hits = self._bm25.search(request.query, limit=self._candidate_limit)
        hits = reciprocal_rank_fusion(bm25_hits, semantic_hits, top_k=request.top_k)
        results = [
            DebugSearchResult(
                **self._cards[hit.id].model_dump(),
                score=hit.score,
                bm25_score=hit.bm25_score,
                semantic_score=hit.semantic_score,
                bm25_rank=hit.bm25_rank,
                semantic_rank=hit.semantic_rank,
            )
            for hit in hits
        ]
        return SearchResponse.from_results(request, results)

    def get_thesis(self, thesis_id: str) -> Thesis | None:
        card = self._cards.get(thesis_id)
        return card.model_copy(deep=True) if card is not None else None

    def health(self) -> dict[str, str | bool | int]:
        return {
            "status": "ok",
            "index_loaded": True,
            "documents_count": len(self._cards),
            "search_mode": "hybrid",
            "embedding_model": self._model.identifier,
        }


class BM25SearchService:
    """Лексический поиск по карточкам SQLite, загруженным один раз при создании."""

    def __init__(self, *, storage: SQLiteStorage) -> None:
        cards = storage.load_cards(validate_snapshot=False)
        if not cards:
            raise ValueError("BM25 corpus is missing or empty")
        self._initialize(cards)

    @classmethod
    def from_cards(cls, cards: tuple[Thesis, ...]) -> Self:
        """Построить независимый снимок из валидированного непустого корпуса."""
        service = cls.__new__(cls)
        service._initialize(cards)
        return service

    def _initialize(self, cards: tuple[Thesis, ...]) -> None:
        self._cards = {card.id: card.model_copy(deep=True) for card in cards}
        self._candidate_limit = min(CANDIDATE_LIMIT, len(cards))
        self._bm25 = BM25Index(
            [card.id for card in cards],
            [build_search_text(card) for card in cards],
        )

    def search(self, request: SearchRequest) -> SearchResponse:
        if request.search_mode == "hybrid":
            raise EncoderError("Гибридный поиск недоступен: эмбеддер не подключён")
        bm25_hits = self._bm25.search(request.query, limit=self._candidate_limit)
        hits = reciprocal_rank_fusion(bm25_hits, [], top_k=request.top_k)
        results = [
            DebugSearchResult(
                **self._cards[hit.id].model_dump(),
                score=hit.score,
                bm25_score=hit.bm25_score,
                semantic_score=hit.semantic_score,
                bm25_rank=hit.bm25_rank,
                semantic_rank=hit.semantic_rank,
            )
            for hit in hits
        ]
        return SearchResponse.from_results(request, results)

    def get_thesis(self, thesis_id: str) -> Thesis | None:
        card = self._cards.get(thesis_id)
        return card.model_copy(deep=True) if card is not None else None

    def health(self) -> dict[str, str | bool | int | None]:
        return {
            "status": "ok",
            "index_loaded": True,
            "documents_count": len(self._cards),
            "search_mode": "bm25",
            "embedding_model": None,
        }


def reciprocal_rank_fusion(
    bm25_hits: Sequence[SearchHit],
    semantic_hits: Sequence[SearchHit],
    top_k: int = 10,
) -> list[FusedHit]:
    """Объединить уже отсортированные каналы с уникальными ID в каждом."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    bm25 = {hit.id: (rank, hit.score) for rank, hit in enumerate(bm25_hits, start=1)}
    semantic = {hit.id: (rank, hit.score) for rank, hit in enumerate(semantic_hits, start=1)}
    if len(bm25) != len(bm25_hits) or len(semantic) != len(semantic_hits):
        raise ValueError("Document IDs must be unique within each channel")

    results = []
    for document_id in bm25.keys() | semantic.keys():
        bm25_rank, bm25_score = bm25.get(document_id, (None, None))
        semantic_rank, semantic_score = semantic.get(document_id, (None, None))
        score = (1 / (60 + bm25_rank) if bm25_rank is not None else 0.0) + (
            1 / (60 + semantic_rank) if semantic_rank is not None else 0.0
        )
        results.append(FusedHit(
            id=document_id,
            score=score,
            bm25_score=bm25_score,
            semantic_score=semantic_score,
            bm25_rank=bm25_rank,
            semantic_rank=semantic_rank,
        ))
    return sorted(results, key=lambda hit: (-hit.score, hit.id))[:top_k]
