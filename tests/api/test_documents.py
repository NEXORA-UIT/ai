"""HTTP contract tests for synchronous document ingestion."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

import src.api.documents as documents_api
from src.main import create_app

_API_KEY = "test-service-key"
_BOARD_ID = UUID("11111111-1111-4111-8111-111111111111")
_UPLOADER_ID = UUID("22222222-2222-4222-8222-222222222222")


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    monkeypatch.setenv("AI_SERVICE_API_KEY", _API_KEY)
    monkeypatch.setenv("AI_UPLOAD_DIRECTORY", str(tmp_path / "uploads"))
    return TestClient(create_app())


def _upload(client: TestClient, filename: str, content: bytes):
    return client.post(
        "/v1/documents",
        headers={"Authorization": f"Bearer {_API_KEY}"},
        data={"board_id": str(_BOARD_ID), "uploaded_by": str(_UPLOADER_ID)},
        files={"file": (filename, content, "text/plain")},
    )


def test_upload_persists_file_and_returns_public_ingestion_summary(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    observed: dict[str, object] = {}

    def ingest_file_to_storage(**kwargs):
        observed.update(kwargs)
        return str(kwargs["document_id"]), SimpleNamespace(
            total_chunks=3, total_tokens=128
        )

    monkeypatch.setattr(documents_api, "ingest_file_to_storage", ingest_file_to_storage)

    response = _upload(api_client, "../../board-policy.txt", b"BOARD_SCOPED_CONTENT")

    assert response.status_code == 201, response.text
    assert response.json() == {
        "document_id": str(observed["document_id"]),
        "status": "READY",
        "chunk_count": 3,
        "total_tokens": 128,
    }
    stored_path = observed["file_path"]
    assert stored_path == (
        tmp_path
        / "uploads"
        / str(observed["document_id"])
        / "board-policy.txt"
    )
    assert stored_path.read_bytes() == b"BOARD_SCOPED_CONTENT"
    assert observed["board_id"] == _BOARD_ID
    assert observed["uploaded_by"] == _UPLOADER_ID
    assert observed["document_id"] == UUID(str(observed["document_id"]))
    assert str(stored_path) not in response.text
    assert "storage_key" not in response.text


def test_upload_requires_service_key(api_client: TestClient) -> None:
    response = api_client.post(
        "/v1/documents",
        data={"board_id": str(_BOARD_ID), "uploaded_by": str(_UPLOADER_ID)},
        files={"file": ("policy.txt", b"content", "text/plain")},
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    ("filename", "content", "status_code"),
    [
        ("empty.txt", b"", 422),
        ("notes.exe", b"binary", 415),
        ("misnamed.pdf", b"not a pdf", 415),
    ],
)
def test_upload_rejects_invalid_files_before_ingestion(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    content: bytes,
    status_code: int,
) -> None:
    def must_not_ingest(**kwargs):
        raise AssertionError("invalid file reached ingestion pipeline")

    monkeypatch.setattr(documents_api, "ingest_file_to_storage", must_not_ingest)

    response = _upload(api_client, filename, content)

    assert response.status_code == status_code


def test_upload_rejects_file_above_configured_limit(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(documents_api, "MAX_UPLOAD_SIZE_BYTES", 4)

    response = _upload(api_client, "large.txt", b"12345")

    assert response.status_code == 413
    assert response.json() == {"detail": "Uploaded file exceeds the size limit"}


def test_upload_rejects_invalid_scope_uuids(api_client: TestClient) -> None:
    response = api_client.post(
        "/v1/documents",
        headers={"Authorization": f"Bearer {_API_KEY}"},
        data={"board_id": "not-a-uuid", "uploaded_by": str(_UPLOADER_ID)},
        files={"file": ("policy.txt", b"content", "text/plain")},
    )

    assert response.status_code == 422


def test_upload_failure_does_not_return_exception_or_filesystem_details(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_path = r"D:\private\secret.pdf"

    def fail_ingestion(**kwargs):
        raise RuntimeError(f"failed at {secret_path} with secret token=abc123")

    monkeypatch.setattr(documents_api, "ingest_file_to_storage", fail_ingestion)

    response = _upload(api_client, "policy.txt", b"content")

    assert response.status_code == 500
    assert response.json() == {"detail": "Document ingestion failed"}
    assert secret_path not in response.text
    assert "abc123" not in response.text


def test_swagger_documents_multipart_upload_and_bearer_auth(
    api_client: TestClient,
) -> None:
    docs = api_client.get("/docs")
    schema = api_client.get("/openapi.json").json()
    operation = schema["paths"]["/v1/documents"]["post"]

    assert docs.status_code == 200
    assert operation["requestBody"]["content"].get("multipart/form-data")
    assert operation["security"] == [{"AIServiceBearer": []}]
    assert schema["components"]["schemas"]["DocumentUploadResponse"]
