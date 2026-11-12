import json

import pytest
from pydantic import ValidationError

from api.schemas import (
    DebugSearchResult, HealthResponse, NotReadyHealthResponse,
    SearchRequest, SearchResponse, SearchResult,
)


DIAGNOSTICS = {"bm25_score", "semantic_score", "bm25_rank", "semantic_rank"}


@pytest.mark.parametrize("field,value", [
    ("status", "ok"), ("index_loaded", True), ("documents_count", 1),
    ("embedding_model", 123), ("search_mode", "invalid"),
])
def test_not_ready_health_rejects_contradictory_values(field, value):
    with pytest.raises(ValidationError):
        NotReadyHealthResponse.model_validate({field: value})


@pytest.mark.parametrize("mode,model", [("bm25", None), ("hybrid", "bge-m3:latest")])
def test_health_embedding_model_is_required_but_nullable(mode, model):
    payload = {
        "status": "ok", "index_loaded": True, "documents_count": 1,
        "search_mode": mode, "embedding_model": model,
    }
    assert HealthResponse.model_validate(payload).model_dump() == payload
    del payload["embedding_model"]
    with pytest.raises(ValidationError):
        HealthResponse.model_validate(payload)


@pytest.mark.parametrize("field,value", [
    ("status", "error"), ("index_loaded", False), ("documents_count", 0),
    ("search_mode", "auto"), ("embedding_model", 123),
])
def test_health_rejects_invalid_values(field, value):
    payload = {
        "status": "ok", "index_loaded": True, "documents_count": 1,
        "search_mode": "bm25", "embedding_model": None,
    }
    with pytest.raises(ValidationError):
        HealthResponse.model_validate({**payload, field: value})


@pytest.mark.parametrize("query", [" Поиск ВКР \n", "42", "ёж", "中文", "C++", "а" * 4000])
def test_request_normalizes_query_and_sets_defaults(query):
    request = SearchRequest(query=query)
    assert request.model_dump() == {"query": query.strip(), "top_k": 10, "debug": False, "search_mode": "bm25"}


def test_query_length_is_checked_after_strip():
    assert len(SearchRequest(query=" \n" + "я" * 4000 + "\t ").query) == 4000


@pytest.mark.parametrize("payload", [
    {}, {"query": ""}, {"query": " \n\t "}, {"query": "!? —_🙂"},
    {"query": "я" * 4001}, {"query": None}, {"query": 123},
    {"query": True}, {"query": []}, {"query": {}},
    {"query": "поиск", "unknown": 1},
])
def test_rejects_invalid_requests_in_python_and_json(payload):
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(payload)
    with pytest.raises(ValidationError):
        SearchRequest.model_validate_json(json.dumps(payload))


def test_query_rejects_bytes():
    with pytest.raises(ValidationError):
        SearchRequest(query=b"search")


@pytest.mark.parametrize("top_k", [1, 10, 100])
def test_top_k_inclusive_bounds(top_k):
    assert SearchRequest(query="ВКР", top_k=top_k).top_k == top_k


@pytest.mark.parametrize("top_k", [0, -1, 101, "10", 10.0, 1.5, True, False, None, []])
def test_top_k_requires_strict_integer_in_range(top_k):
    payload = {"query": "ВКР", "top_k": top_k}
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(payload)
    with pytest.raises(ValidationError):
        SearchRequest.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("debug", [True, False])
def test_debug_accepts_booleans(debug):
    assert SearchRequest(query="ВКР", debug=debug).debug is debug


@pytest.mark.parametrize("debug", [0, 1, "true", "false", 0.0, None, [], {}])
def test_debug_rejects_non_booleans(debug):
    payload = {"query": "ВКР", "debug": debug}
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(payload)
    with pytest.raises(ValidationError):
        SearchRequest.model_validate_json(json.dumps(payload))


@pytest.fixture
def ranked_results():
    card = {
        "title": "Гибридный поиск", "abstract": "Поиск ВКР по теме.",
        "url": "https://example.org/theses/1", "keywords": ["BM25"],
        "author": "Иван Иванов", "year": 2025,
    }
    return [
        DebugSearchResult(
            **card, id="thesis-2", score=0.032,
            bm25_score=-0.5, semantic_score=0.8, bm25_rank=1, semantic_rank=2,
        ),
        DebugSearchResult(
            **card, id="thesis-1", score=0.016,
            bm25_score=None, semantic_score=0.7, bm25_rank=None, semantic_rank=3,
        ),
        DebugSearchResult(
            **card, id="thesis-3", score=0.015,
            bm25_score=0.0, semantic_score=None, bm25_rank=4, semantic_rank=None,
        ),
    ]


@pytest.mark.parametrize("debug", [False, True])
def test_response_serialization_obeys_debug_and_preserves_ranking(ranked_results, debug):
    request = SearchRequest(query=" \nПоиск ВКР\t ", debug=debug)
    response = SearchResponse.from_results(request, ranked_results)
    expected_results = [result.model_dump(mode="json") for result in ranked_results]
    if not debug:
        expected_results = [
            {key: value for key, value in result.items() if key not in DIAGNOSTICS}
            for result in expected_results
        ]
    expected = {"query": "Поиск ВКР", "count": 3, "results": expected_results}
    assert response.model_dump(mode="json") == expected
    assert json.loads(response.model_dump_json()) == expected
    assert SearchResponse.model_validate_json(response.model_dump_json()) == response
    # A public response must not remove diagnostics from the original results.
    assert ranked_results[0].bm25_score == -0.5
    assert DIAGNOSTICS <= ranked_results[1].model_dump().keys()


@pytest.mark.parametrize("debug", [False, True])
def test_empty_response(debug):
    response = SearchResponse.from_results(SearchRequest(query="ВКР", debug=debug), [])
    assert json.loads(response.model_dump_json()) == {"query": "ВКР", "count": 0, "results": []}


@pytest.mark.parametrize("field", sorted(DIAGNOSTICS))
def test_debug_result_requires_all_diagnostic_fields(ranked_results, field):
    payload = ranked_results[0].model_dump()
    del payload[field]
    with pytest.raises(ValidationError):
        DebugSearchResult.model_validate(payload)
    with pytest.raises(ValidationError):
        SearchResponse(query="ВКР", count=1, results=[payload])


def test_plain_result_rejects_diagnostics(ranked_results):
    with pytest.raises(ValidationError):
        SearchResult.model_validate(ranked_results[0].model_dump())


def test_response_rejects_inconsistent_count(ranked_results):
    with pytest.raises(ValidationError):
        SearchResponse(query="ВКР", count=0, results=ranked_results)


@pytest.mark.parametrize("field,value", [
    ("score", "0.5"), ("score", float("nan")), ("score", float("inf")),
    ("bm25_rank", 0), ("semantic_rank", True), ("semantic_rank", 1.5),
])
def test_result_rejects_invalid_scores_and_ranks(ranked_results, field, value):
    with pytest.raises(ValidationError):
        DebugSearchResult.model_validate({**ranked_results[0].model_dump(), field: value})
