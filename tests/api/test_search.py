"""HTTP contract tests for Board-scoped retrieval."""

from __future__ import annotations

from uuid import UUID

import pytest
from fastapi.testclient import TestClient

import src.api.search as search_api
from src.main import create_app
from src.retrieval.schemas import SearchChunk, SearchResult

_API_KEY = "test-service-key"
_BOARD_ID = UUID("11111111-1111-4111-8111-111111111111")
_OTHER_BOARD_ID = UUID("33333333-3333-4333-8333-333333333333")
_QUESTION = "What is the approval process?"


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("AI_SERVICE_API_KEY", _API_KEY)
    return TestClient(create_app())


def _search(client: TestClient, **overrides):
    payload = {
        "question": _QUESTION,
        "board_id": str(_BOARD_ID),
        "top_k": 3,
        "candidate_k": 20,
    }
    payload.update(overrides)
    return client.post(
        "/v1/search",
        headers={"Authorization": f"Bearer {_API_KEY}"},
        json=payload,
    )


def _search_result(
    board_id: UUID = _BOARD_ID,
    chunks: tuple[SearchChunk, ...] | None = None,
) -> SearchResult:
    return SearchResult(
        question=_QUESTION,
        board_id=str(board_id),
        chunks=chunks
        if chunks is not None
        else (
            SearchChunk(
                chunk_id="44444444-4444-4444-8444-444444444444",
                document_id="55555555-5555-4555-8555-555555555555",
                chunk_index=2,
                file_name="board-policy.pdf",
                content="Approvals require the project lead and board owner.",
                page_number=3,
                start_offset=120,
                end_offset=180,
                rerank_score=0.94,
                rrf_score=0.04,
            ),
        ),
    )


def test_search_forwards_board_and_limits_and_returns_public_citations(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def search_documents(question, board_id, *, top_k, candidate_k):
        observed.update(
            question=question,
            board_id=board_id,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        return _search_result()

    monkeypatch.setattr(search_api, "search_documents", search_documents)

    response = _search(api_client, top_k=2, candidate_k=12)

    assert response.status_code == 200
    body = response.json()
    assert body["board_id"] == str(_BOARD_ID)
    assert body["question"] == _QUESTION
    assert body["chunks"][0] == {
        "chunk_id": "44444444-4444-4444-8444-444444444444",
        "document_id": "55555555-5555-4555-8555-555555555555",
        "chunk_index": 2,
        "file_name": "board-policy.pdf",
        "content": "Approvals require the project lead and board owner.",
        "page_number": 3,
        "start_offset": 120,
        "end_offset": 180,
        "rerank_score": 0.94,
        "rrf_score": 0.04,
    }
    assert "storage_key" not in response.text
    assert observed == {
        "question": _QUESTION,
        "board_id": _BOARD_ID,
        "top_k": 2,
        "candidate_k": 12,
    }


def test_search_returns_empty_results_for_board_without_matches(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        search_api,
        "search_documents",
        lambda question, board_id, **kwargs: _search_result(chunks=()),
    )

    response = _search(api_client)

    assert response.status_code == 200
    assert response.json()["chunks"] == []


def test_search_rejects_result_from_another_board(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        search_api,
        "search_documents",
        lambda question, board_id, **kwargs: _search_result(board_id=_OTHER_BOARD_ID),
    )

    response = _search(api_client)

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Search returned data outside the requested Board"
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"question": " \t "},
        {"board_id": "not-a-uuid"},
        {"top_k": 0},
        {"top_k": 6},
        {"candidate_k": 0},
        {"candidate_k": 21},
        {"top_k": 4, "candidate_k": 3},
    ],
)
def test_search_validates_question_scope_and_candidate_limits(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
) -> None:
    def must_not_search(*args, **kwargs):
        raise AssertionError("invalid search request reached retrieval")

    monkeypatch.setattr(search_api, "search_documents", must_not_search)

    response = _search(api_client, **overrides)

    assert response.status_code == 422


def test_search_requires_service_key(api_client: TestClient) -> None:
    response = api_client.post(
        "/v1/search",
        json={"question": _QUESTION, "board_id": str(_BOARD_ID)},
    )

    assert response.status_code == 401


def test_search_failure_does_not_return_exception_details(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_search(*args, **kwargs):
        raise RuntimeError("private storage_key=local://secret and token=abc123")

    monkeypatch.setattr(search_api, "search_documents", fail_search)

    response = _search(api_client)

    assert response.status_code == 500
    assert response.json() == {"detail": "Search failed"}
    assert "secret" not in response.text
    assert "abc123" not in response.text


def test_swagger_documents_search_json_schema_and_bearer_auth(
    api_client: TestClient,
) -> None:
    docs = api_client.get("/docs")
    schema = api_client.get("/openapi.json").json()
    operation = schema["paths"]["/v1/search"]["post"]

    assert docs.status_code == 200
    assert operation["requestBody"]["content"]["application/json"]
    assert operation["security"] == [{"AIServiceBearer": []}]
    assert schema["components"]["schemas"]["SearchRequest"]
    response_schema = schema["components"]["schemas"]["SearchChunkResponse"]
    assert "storage_key" not in response_schema["properties"]
