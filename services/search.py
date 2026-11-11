"""Поиск по одному снимку в памяти и объединение рейтингов через RRF."""

from collections.abc import Sequence
from threading import Lock

from api.schemas import DebugSearchResult, SearchRequest, SearchResponse
from domain.models import FusedHit, SearchHit, Thesis
from external.embedder import Encoder, validate_vectors
from infrastructure.search import BM25Index, SemanticIndex
from infrastructure.storage import IndexStorage
from services.indexing import build_search_text, load_index


CANDIDATE_LIMIT = 100


class SearchService:
    """Один экземпляр владеет энкодером для всех поисковых потоков приложения."""

    def __init__(self, *, storage: IndexStorage, encoder: Encoder) -> None:
        snapshot = load_index(storage=storage, model=encoder.model)
        self._model = snapshot.metadata.model
        self._encoder = encoder
        self._encode_lock = Lock()
        self._cards = {card.id: card for card in snapshot.cards}
        self._candidate_limit = min(CANDIDATE_LIMIT, len(self._cards))
        self._bm25 = BM25Index(
            [card.id for card in snapshot.cards],
            [build_search_text(card) for card in snapshot.cards],
        )
        self._semantic = SemanticIndex(snapshot.vector_ids, snapshot.vectors)

    def search(self, request: SearchRequest) -> SearchResponse:
        with self._encode_lock:
            # Копия под блокировкой защищает от повторного использования буфера адаптером.
            query_vector = validate_vectors(
                self._encoder.encode_query(request.query), shape=(self._model.dimension,),
            )
        bm25_hits = self._bm25.search(request.query, limit=self._candidate_limit)
        semantic_hits = self._semantic.search(query_vector, limit=self._candidate_limit)
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
            "embedding_model": self._model.identifier,
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
