"""Локальные поисковые индексы без БД и эмбеддера."""

import re
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from rank_bm25 import BM25Okapi

from domain.models import SearchHit
from external.embedder import L2_ATOL, L2_RTOL


def tokenize(text: str) -> list[str]:
    """Одинаковая Unicode-токенизация документов и запросов; без подчёркиваний."""
    return re.findall(r"[^\W_]+", text.lower().replace("ё", "е"))


class BM25Index:
    def __init__(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(ids) != len(texts):
            raise ValueError("IDs and texts must have the same length")
        if len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique")
        self._ids = tuple(ids)
        tokens = [tokenize(text) for text in texts]
        self._token_sets = [set(document) for document in tokens]
        # rank_bm25 cannot build an index with an empty vocabulary.
        self._bm25 = BM25Okapi(tokens) if any(tokens) else None

    def search(self, query: str, limit: int = 100) -> list[SearchHit]:
        if limit < 0:
            raise ValueError("Candidate limit must be non-negative")
        query_tokens = tokenize(query)
        if not query_tokens or self._bm25 is None or limit == 0:
            return []
        scores = self._bm25.get_scores(query_tokens)
        hits = [
            SearchHit(document_id, float(score))
            for document_id, score, tokens in zip(self._ids, scores, self._token_sets)
            if not tokens.isdisjoint(query_tokens)
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.id))[:limit]


class SemanticIndex:
    """Точное скалярное произведение готовых L2-нормализованных векторов."""

    def __init__(self, ids: Sequence[str], vectors: ArrayLike) -> None:
        matrix = np.array(vectors, dtype=float, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] == 0:
            raise ValueError("Vectors must have shape (number of IDs, dimension > 0)")
        if len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique")
        if not np.isfinite(matrix).all() or not np.allclose(
            np.linalg.norm(matrix, axis=1), 1.0, rtol=L2_RTOL, atol=L2_ATOL
        ):
            raise ValueError("Document vectors must be finite and L2-normalized")
        self._ids = tuple(ids)
        matrix.flags.writeable = False
        self._vectors = matrix

    def search(self, query_vector: ArrayLike, limit: int = 100) -> list[SearchHit]:
        if limit < 0:
            raise ValueError("Candidate limit must be non-negative")
        query = np.asarray(query_vector, dtype=float)
        if query.shape != (self._vectors.shape[1],):
            raise ValueError("Query vector dimension must match document vectors")
        if not np.isfinite(query).all() or not np.isclose(
            np.linalg.norm(query), 1.0, rtol=L2_RTOL, atol=L2_ATOL
        ):
            raise ValueError("Query vector must be finite and L2-normalized")
        scores = self._vectors @ query
        hits = [
            SearchHit(document_id, float(score))
            for document_id, score in zip(self._ids, scores)
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.id))[:limit]
