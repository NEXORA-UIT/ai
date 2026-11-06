"""HTTP-level upload-to-search journey for the Swagger demo."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from fastapi.testclient import TestClient

import src.api.documents as documents_api
import src.api.search as search_api
from src.main import create_app
from src.retrieval.schemas import SearchChunk, SearchResult

_API_KEY = "test-service-key"
_BOARD_ID = UUID("11111111-1111-4111-8111-111111111111")
_OTHER_BOARD_ID = UUID("33333333-3333-4333-8333-333333333333")
_UPLOADER_ID = UUID("22222222-2222-4222-8222-222222222222")
_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "synthetic_smoke_doc.txt"


def test_demo_flow_upload_then_search_is_board_scoped_and_public(
    monkeypatch, tmp_path
) -> None:
    ingested_documents: list[dict[str, object]] = []
    search_requests: list[dict[str, object]] = []
    monkeypatch.setenv("AI_SERVICE_API_KEY", _API_KEY)
    monkeypatch.setenv("AI_UPLOAD_DIRECTORY", str(tmp_path / "uploads"))

    def fake_ingest_file_to_storage(**kwargs):
        file_path = Path(kwargs["file_path"])
        document = {
            "document_id": str(kwargs["document_id"]),
            "board_id": str(kwargs["board_id"]),
            "file_name": file_path.name,
            "content": file_path.read_text(encoding="utf-8"),
        }
        ingested_documents.append(document)
        return document["document_id"], SimpleNamespace(
            total_chunks=1, total_tokens=277
        )

    def fake_search_documents(question, board_id, *, top_k, candidate_k):
        search_requests.append(
            {
                "question": question,
                "board_id": str(board_id),
                "top_k": top_k,
                "candidate_k": candidate_k,
            }
        )
        visible_documents = [
            document
            for document in ingested_documents
            if document["board_id"] == str(board_id)
        ]
        chunks = tuple(
            SearchChunk(
                chunk_id="44444444-4444-4444-8444-444444444444",
                document_id=document["document_id"],
                chunk_index=0,
                file_name=document["file_name"],
                content=document["content"],
                page_number=None,
                start_offset=0,
                end_offset=len(document["content"]),
                rerank_score=0.9,
                rrf_score=0.03,
            )
            for document in visible_documents[:top_k]
        )
        return SearchResult(
            question=question,
            board_id=str(board_id),
            chunks=chunks,
        )

    monkeypatch.setattr(
        documents_api, "ingest_file_to_storage", fake_ingest_file_to_storage
    )
    monkeypatch.setattr(search_api, "search_documents", fake_search_documents)

    with TestClient(create_app()) as client:
        upload_response = client.post(
            "/v1/documents",
            headers={"Authorization": f"Bearer {_API_KEY}"},
            data={"board_id": str(_BOARD_ID), "uploaded_by": str(_UPLOADER_ID)},
            files={
                "file": (
                    _FIXTURE_PATH.name,
                    _FIXTURE_PATH.read_bytes(),
                    "text/plain",
                )
            },
        )
        assert upload_response.status_code == 201, upload_response.text
        uploaded_document = upload_response.json()
        assert uploaded_document["status"] == "READY"
        assert uploaded_document["chunk_count"] == 1
        assert uploaded_document["total_tokens"] == 277
        assert str(tmp_path) not in upload_response.text

        search_response = client.post(
            "/v1/search",
            headers={"Authorization": f"Bearer {_API_KEY}"},
            json={
                "question": "What does the Nexora Knowledge Base do?",
                "board_id": str(_BOARD_ID),
                "top_k": 5,
                "candidate_k": 20,
            },
        )
        assert search_response.status_code == 200, search_response.text
        search_result = search_response.json()
        assert search_result["board_id"] == str(_BOARD_ID)
        assert len(search_result["chunks"]) == 1
        assert search_result["chunks"][0]["document_id"] == uploaded_document[
            "document_id"
        ]
        assert "Module Tri thức và RAG" in search_result["chunks"][0]["content"]
        assert "storage_key" not in search_response.text
        assert str(tmp_path) not in search_response.text
        assert search_requests[0] == {
            "question": "What does the Nexora Knowledge Base do?",
            "board_id": str(_BOARD_ID),
            "top_k": 5,
            "candidate_k": 20,
        }

        other_board_response = client.post(
            "/v1/search",
            headers={"Authorization": f"Bearer {_API_KEY}"},
            json={
                "question": "What does the Nexora Knowledge Base do?",
                "board_id": str(_OTHER_BOARD_ID),
                "top_k": 5,
                "candidate_k": 20,
            },
        )
        assert other_board_response.status_code == 200, other_board_response.text
        assert other_board_response.json()["chunks"] == []
        assert search_requests[1] == {
            "question": "What does the Nexora Knowledge Base do?",
            "board_id": str(_OTHER_BOARD_ID),
            "top_k": 5,
            "candidate_k": 20,
        }
