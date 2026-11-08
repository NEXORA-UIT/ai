"""HTTP contract for grounded answers in the internal Swagger demo."""

from __future__ import annotations

from importlib import import_module
from uuid import UUID

from fastapi.testclient import TestClient

from src.main import create_app
from src.retrieval.schemas import RerankedChunk, SourceChunk

BOARD_ID = "11111111-1111-4111-8111-111111111111"
OTHER_BOARD_ID = "33333333-3333-4333-8333-333333333333"
API_KEY = "test-service-key"
CHUNK_ID = "44444444-4444-4444-8444-444444444444"


def _chunk(board_id: str = BOARD_ID) -> RerankedChunk:
    return RerankedChunk(
        source=SourceChunk(
            board_id=board_id,
            chunk_id=CHUNK_ID,
            document_id="55555555-5555-4555-8555-555555555555",
            chunk_index=0,
            file_name="guide.txt",
            storage_key="private://never-leak",
            content="The knowledge base stores project documents.",
            page_number=None,
            start_offset=0,
            end_offset=44,
        ),
        rerank_score=0.9,
        rrf_score=0.03,
    )


def _post(client: TestClient, **overrides):
    payload = {
        "question": "What does the knowledge base store?",
        "board_id": BOARD_ID,
        "top_k": 5,
        "candidate_k": 20,
    }
    payload.update(overrides)
    return client.post(
        "/v1/answer",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
    )


def test_missing_prompt_returns_sanitized_error(monkeypatch) -> None:
    from src.answering import service

    def fail_generation(*_args, **_kwargs):
        raise FileNotFoundError("/app/src/answering/system_prompt.md")

    monkeypatch.setenv("AI_SERVICE_API_KEY", API_KEY)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(service, "_retrieve_evidence", lambda *_args, **_kw: [_chunk()])
    monkeypatch.setattr(service, "answer_with_sources", fail_generation)

    with TestClient(create_app()) as client:
        response = _post(client)

    assert response.status_code == 500
    assert response.json() == {"detail": "Answer generation failed"}


def test_answer_returns_grounded_text_and_verified_citation(monkeypatch) -> None:
    retrieval_module = import_module("src.retrieval.search_documents")
    provider_module = import_module("src.providers.deepseek.client")

    monkeypatch.setenv("AI_SERVICE_API_KEY", API_KEY)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    observed: dict[str, object] = {}

    def retrieve(question, board_id, *, top_k, candidate_k):
        observed.update(
            question=question,
            board_id=board_id,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        return [_chunk()]

    class FakeDeepSeekClient:
        def __init__(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def complete(self, _prompt):
            return {
                "answer": "It stores project documents.",
                "cited_chunk_ids": [CHUNK_ID],
            }

    monkeypatch.setattr(
        retrieval_module, "retrieve_reranked_chunks", retrieve, raising=False
    )
    monkeypatch.setattr(provider_module, "DeepSeekAnswerClient", FakeDeepSeekClient)

    with TestClient(create_app()) as client:
        response = _post(client)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "question": "What does the knowledge base store?",
        "board_id": BOARD_ID,
        "answer": "It stores project documents.",
        "refused": False,
        "citations": [
            {
                "chunk_id": CHUNK_ID,
                "document_id": "55555555-5555-4555-8555-555555555555",
                "file_name": "guide.txt",
                "page_number": None,
                "start_offset": 0,
                "end_offset": 44,
            }
        ],
    }
    assert observed == {
        "question": "What does the knowledge base store?",
        "board_id": UUID(BOARD_ID),
        "top_k": 5,
        "candidate_k": 20,
    }
    assert "private://" not in response.text


def test_answer_refuses_empty_board_without_provider_key(monkeypatch) -> None:
    retrieval_module = import_module("src.retrieval.search_documents")

    monkeypatch.setenv("AI_SERVICE_API_KEY", API_KEY)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        retrieval_module,
        "retrieve_reranked_chunks",
        lambda *_args, **_kwargs: [],
        raising=False,
    )
    with TestClient(create_app()) as client:
        response = _post(client)
    assert response.status_code == 200, response.text
    assert response.json()["refused"] is True
    assert response.json()["citations"] == []


def test_answer_rejects_foreign_board_evidence(monkeypatch) -> None:
    retrieval_module = import_module("src.retrieval.search_documents")

    monkeypatch.setenv("AI_SERVICE_API_KEY", API_KEY)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setattr(
        retrieval_module,
        "retrieve_reranked_chunks",
        lambda *_args, **_kwargs: [_chunk(OTHER_BOARD_ID)],
        raising=False,
    )
    with TestClient(create_app()) as client:
        response = _post(client)
    assert response.status_code == 500
    assert "private://" not in response.text


def test_answer_reports_missing_provider_key_only_when_evidence_exists(
    monkeypatch,
) -> None:
    retrieval_module = import_module("src.retrieval.search_documents")
    monkeypatch.setenv("AI_SERVICE_API_KEY", API_KEY)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        retrieval_module,
        "retrieve_reranked_chunks",
        lambda *_args, **_kwargs: [_chunk()],
        raising=False,
    )
    with TestClient(create_app()) as client:
        response = _post(client)
    assert response.status_code == 503
    assert response.json() == {"detail": "DeepSeek API key is not configured"}


def test_answer_requires_service_key_and_valid_request(monkeypatch) -> None:
    monkeypatch.setenv("AI_SERVICE_API_KEY", API_KEY)
    with TestClient(create_app()) as client:
        unauthorized = client.post(
            "/v1/answer", json={"question": "hello", "board_id": BOARD_ID}
        )
        invalid = _post(client, top_k=5, candidate_k=3)
    assert unauthorized.status_code == 401
    assert invalid.status_code == 422
