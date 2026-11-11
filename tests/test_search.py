import math

import numpy as np
import pytest

from domain.models import FusedHit, SearchHit, Thesis
from infrastructure.search import BM25Index, SemanticIndex, tokenize
from services.indexing import build_search_text
from services.search import reciprocal_rank_fusion


def test_search_text_contains_only_title_keywords_and_abstract():
    thesis = Thesis(
        id="secret-id", title="Поиск", keywords=["BM25", "нейросети"],
        abstract="Описание работы", author="Секретный Автор", year=2025,
        url="https://secret.example.org/",
    )
    text = build_search_text(thesis)
    assert text == "Поиск\nBM25 нейросети\nОписание работы"
    index = BM25Index([thesis.id], [text])
    assert index.search("Секретный Автор 2025 https secret example org id") == []
    for query in ["поиск", "bm25", "нейросети", "описание", "работы"]:
        assert [hit.id for hit in index.search(query)] == [thesis.id]
    thesis.keywords = []
    assert build_search_text(thesis) == "Поиск\n\nОписание работы"


@pytest.mark.parametrize("text,expected", [
    ("ЁЖ, ёлка! E5 BM25 foo_bar C++ 2025", ["еж", "елка", "e5", "bm25", "foo", "bar", "c", "2025"]),
    ("中文 café १२", ["中文", "café", "१२"]),
    ("поиск\nработы\tтекст", ["поиск", "работы", "текст"]),
    (" —_!?🙂 ", []),
    ("", []),
])
def test_tokenize(text, expected):
    assert tokenize(text) == expected


def test_bm25_uses_same_tokenization_for_document_and_query():
    index = BM25Index(["z", "a"], ["ЁЖ_BM25", "другое"])
    assert index.search("еж") == index.search(" ЁЖ! ") == [SearchHit("z", 0.0)]
    assert index.search("bm25") == [SearchHit("z", 0.0)]


@pytest.mark.parametrize("query", ["несуществующее", "поисковый", "__!?", ""])
def test_bm25_no_overlap_returns_no_candidates(query):
    assert BM25Index(["a", "b"], ["поиск", "описание"]).search(query) == []


def test_bm25_keeps_zero_scores_but_excludes_nonmatches():
    index = BM25Index(["z", "10", "2", "a"], ["поиск", "поиск", "другое", "другое"])
    assert index.search("поиск") == [SearchHit("10", 0.0), SearchHit("z", 0.0)]
    assert index.search("поиск", limit=1) == [SearchHit("10", 0.0)]


def test_bm25_keeps_negative_scores_and_lexical_tie_order():
    hits = BM25Index(
        ["2", "10", "a"], ["поиск текст", "поиск текст", "другое"]
    ).search("поиск")
    assert [hit.id for hit in hits] == ["10", "2"]
    assert all(hit.score < 0 for hit in hits)
    assert hits[0].score == hits[1].score


def test_bm25_ranks_by_score_before_id_and_applies_limit():
    index = BM25Index(["a", "z", "b", "c", "d"], ["поиск", "поиск поиск", "мир", "мир", "мир"])
    hits = index.search("поиск", limit=1)
    assert [hit.id for hit in hits] == ["z"]
    # N=5, df=2, document length=2, average length=6/5; Okapi defaults.
    expected = math.log(3.5 / 2.5) * 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 2 / 1.2))
    assert hits[0].score == pytest.approx(expected)


def test_bm25_does_not_round_before_sorting():
    index = BM25Index(
        ["a", "z", "b", "c", "d"],
        ["поиск " * 10000, "поиск " * 10001, "мир", "мир", "мир"],
    )
    hits = index.search("поиск")
    assert [hit.id for hit in hits] == ["z", "a"]
    assert 0 < hits[0].score - hits[1].score < 1e-6
    assert index.search("поиск", limit=1) == hits[:1]


@pytest.mark.parametrize("texts", [[], ["!?", "_"]])
def test_bm25_empty_vocabulary(texts):
    index = BM25Index([str(i) for i in range(len(texts))], texts)
    assert index.search("поиск") == []


@pytest.fixture
def semantic_index():
    return SemanticIndex(
        ["z", "2", "10", "a", "negative"],
        [[1, 0], [0.6, 0.8], [0.6, 0.8], [0, 1], [-1, 0]],
    )


def test_semantic_exact_scores_order_and_negative_similarity(semantic_index):
    assert semantic_index.search([1, 0]) == [
        SearchHit("z", 1.0), SearchHit("10", 0.6), SearchHit("2", 0.6),
        SearchHit("a", 0.0), SearchHit("negative", -1.0),
    ]
    assert semantic_index.search([0, 1], limit=1) == [SearchHit("a", 1.0)]


@pytest.mark.parametrize("limit,expected", [(0, []), (1, ["z"]), (2, ["z", "10"]), (100, ["z", "10", "2", "a", "negative"])])
def test_semantic_candidate_limit(semantic_index, limit, expected):
    assert [hit.id for hit in semantic_index.search([1, 0], limit)] == expected


def test_semantic_does_not_round_before_sorting():
    similarity = 1 - 1e-10
    index = SemanticIndex(["a", "z"], [[similarity, math.sqrt(1 - similarity**2)], [1, 0]])
    assert index.search([1, 0], limit=1) == [SearchHit("z", 1.0)]
    assert index.search([1, 0])[1].score == similarity


def test_semantic_snapshot_is_independent_of_input_mutation():
    ids = ["a"]
    matrix = np.array([[1.0, 0.0]], dtype=np.float32)
    index = SemanticIndex(ids, matrix)
    ids[0] = "b"
    matrix[:] = [[0.0, 1.0]]
    assert index.search([1, 0]) == [SearchHit("a", 1.0)]


def test_semantic_empty_matrix():
    assert SemanticIndex([], np.empty((0, 2))).search([1, 0]) == []


@pytest.mark.parametrize("ids,texts", [(["a"], []), (["a", "a"], ["x", "y"])])
def test_bm25_rejects_misaligned_or_duplicate_ids(ids, texts):
    with pytest.raises(ValueError):
        BM25Index(ids, texts)


@pytest.mark.parametrize("ids,vectors", [
    (["a"], [[1, 0], [0, 1]]), (["a", "a"], [[1, 0], [0, 1]]),
    (["a"], [1, 0]), (["a"], [[]]), (["a"], [[0, 0]]),
    (["a"], [[2, 0]]), (["a"], [[float("nan"), 0]]), (["a"], [[float("inf"), 0]]),
])
def test_semantic_rejects_invalid_snapshot(ids, vectors):
    with pytest.raises(ValueError):
        SemanticIndex(ids, vectors)


@pytest.mark.parametrize("query", [[1], [[1, 0]], [0, 0], [2, 0], [float("nan"), 0], [float("inf"), 0]])
def test_semantic_rejects_invalid_query(semantic_index, query):
    with pytest.raises(ValueError):
        semantic_index.search(query)


def test_rrf_merges_ids_and_preserves_channel_scores_and_one_based_ranks():
    bm25 = [SearchHit("b", 0.0), SearchHit("a", -0.5)]
    semantic = [SearchHit("a", 0.9), SearchHit("c", 0.8)]
    assert reciprocal_rank_fusion(bm25, semantic) == [
        FusedHit("a", 1 / 62 + 1 / 61, -0.5, 0.9, 2, 1),
        FusedHit("b", 1 / 61, 0.0, None, 1, None),
        FusedHit("c", 1 / 62, None, 0.8, None, 2),
    ]
    assert bm25 == [SearchHit("b", 0.0), SearchHit("a", -0.5)]


@pytest.mark.parametrize("missing_channel", ["bm25", "semantic"])
def test_rrf_missing_channel_contributes_zero(missing_channel):
    hits = [SearchHit("z", 0.0), SearchHit("a", -1.0)]
    results = reciprocal_rank_fusion([], hits) if missing_channel == "bm25" else reciprocal_rank_fusion(hits, [])
    assert [hit.id for hit in results] == ["z", "a"]
    assert [hit.score for hit in results] == [1 / 61, 1 / 62]
    assert all(getattr(hit, f"{missing_channel}_rank") is None for hit in results)
    assert all(getattr(hit, f"{missing_channel}_score") is None for hit in results)


def test_rrf_ties_use_string_ids_before_limiting():
    bm25 = [SearchHit("2", 1.0), SearchHit("10", 0.0)]
    semantic = [SearchHit("10", 1.0), SearchHit("2", 0.0)]
    results = reciprocal_rank_fusion(bm25, semantic)
    assert [hit.id for hit in results] == ["10", "2"]
    assert results[0].score == results[1].score == 1 / 61 + 1 / 62
    assert reciprocal_rank_fusion(bm25, semantic, top_k=1) == results[:1]


def test_rrf_does_not_round_before_sorting():
    bm25 = [SearchHit(f"b{i}", 100.0 - i) for i in range(97)] + [SearchHit("z", 2.0), SearchHit("a", 1.0)]
    semantic = [SearchHit(f"s{i}", 100.0 - i) for i in range(98)] + [SearchHit("a", 2.0), SearchHit("z", 1.0)]
    results = reciprocal_rank_fusion(bm25, semantic, top_k=200)
    ids = [hit.id for hit in results]
    z_score = results[ids.index("z")].score
    a_score = results[ids.index("a")].score
    assert ids.index("z") < ids.index("a")
    assert z_score == 1 / 158 + 1 / 160
    assert a_score == 2 / 159
    assert 0 < z_score - a_score < 1e-6


@pytest.mark.parametrize("top_k,expected_count", [(0, 0), (1, 1), (2, 2), (100, 3)])
def test_rrf_limit_and_unique_results(top_k, expected_count):
    hits = [SearchHit("a", 1.0), SearchHit("b", 0.5), SearchHit("c", 0.0)]
    results = reciprocal_rank_fusion(hits, hits, top_k=top_k)
    assert len(results) == len({hit.id for hit in results}) == expected_count


def test_rrf_empty_channels():
    assert reciprocal_rank_fusion([], []) == []


@pytest.mark.parametrize("channel", ["bm25", "semantic"])
def test_rrf_rejects_duplicates_within_channel(channel):
    hits = [SearchHit("a", 1.0), SearchHit("a", 0.5)]
    with pytest.raises(ValueError, match="unique"):
        reciprocal_rank_fusion(hits if channel == "bm25" else [], hits if channel == "semantic" else [])


def test_negative_limits_are_rejected(semantic_index):
    with pytest.raises(ValueError):
        BM25Index(["a"], ["поиск"]).search("поиск", limit=-1)
    with pytest.raises(ValueError):
        semantic_index.search([1, 0], limit=-1)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], top_k=-1)


def test_default_candidate_limits():
    ids = [f"{i:03}" for i in reversed(range(105))]
    bm25 = BM25Index(ids, ["поиск"] * 105)
    semantic = SemanticIndex(ids, [[1, 0]] * 105)
    expected = sorted(ids)[:100]
    assert [hit.id for hit in bm25.search("поиск")] == expected
    assert [hit.id for hit in semantic.search([1, 0])] == expected
    assert bm25.search("поиск", limit=0) == []
