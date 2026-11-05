"""AI-only end-to-end acceptance for TXT ingestion and Board-scoped retrieval."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from src.chunking.tokenizer import WhitespaceTokenCounter
from src.embedding.config import DEFAULT_NORMALIZE_EMBEDDINGS
from src.ingestion.pipeline import ingest_file_to_storage
from src.ingestion.schemas import EmbeddedChunk, ExtractionResult
from src.retrieval.dense import search_by_text

_QUERY_VECTOR = [1.0, 0.0] + [0.0] * 1022
_TARGET_VECTOR = [0.8, 0.6] + [0.0] * 1022
_DISTRACTOR_VECTOR = [0.0, 1.0] + [0.0] * 1022
_TARGET_PASSAGE = "The release date is 2026-12-01."


class _DeterministicEmbedder:
    """Provide normalized vectors so this database test needs no model download."""

    model_name = "test/e2e-embedder"
    revision = "test-revision"
    dimension = 1024

    def __init__(self) -> None:
        self.queries: list[list[str]] = []

    def embed_chunks(self, chunks: list, batch_size: int = 8) -> list[EmbeddedChunk]:
        del batch_size
        return [
            EmbeddedChunk(
                chunk=chunk,
                dense_embedding=(
                    _TARGET_VECTOR
                    if "release date" in chunk.content.casefold()
                    else _DISTRACTOR_VECTOR
                ),
            )
            for chunk in chunks
        ]

    def embed_texts(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        del batch_size
        self.queries.append(texts)
        return [_QUERY_VECTOR.copy() for _ in texts]


class _TextFileExtractor:
    """Read the TXT fixture while keeping Tika outside this deterministic test."""

    def __init__(self, **_: object) -> None:
        pass

    def extract_file(self, file_path: str | Path) -> ExtractionResult:
        path = Path(file_path)
        return ExtractionResult(
            raw_text=path.read_text(encoding="utf-8"),
            file_name=path.name,
            mime_type="text/plain",
            metadata={},
            page_count=None,
            has_form_feeds=False,
        )


@pytest.mark.integration
@pytest.mark.database
def test_txt_ingestion_to_scoped_retrieval_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retrieval_database,
) -> None:
    """Persist a TXT through ingestion, then retrieve its source without leaks."""
    from src.ingestion import pipeline

    storage, seed_document = retrieval_database
    board_a = str(uuid.uuid4())
    board_b = str(uuid.uuid4())
    document_id = uuid.uuid4()
    uploaded_by = uuid.uuid4()
    embedder = _DeterministicEmbedder()
    file_path = tmp_path / "nexora_release.txt"
    file_path.write_text(
        f"{_TARGET_PASSAGE}\n\n"
        "The customer support address is support@example.invalid.",
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "TikaClient", _TextFileExtractor)

    try:
        persisted_document_id, embedded_document = ingest_file_to_storage(
            file_path=file_path,
            board_id=board_a,
            uploaded_by=uploaded_by,
            storage=storage,
            document_id=document_id,
            embedder=embedder,
            token_counter=WhitespaceTokenCounter(),
            chunk_size=6,
            chunk_overlap=0,
        )

        assert persisted_document_id == str(document_id)
        assert embedded_document.total_chunks == 2

        stored = storage.get_document_with_chunks(persisted_document_id)
        assert stored is not None
        assert stored.document.status == "READY"
        assert stored.document.board_id == board_a
        assert [chunk.content for chunk in stored.chunks] == [
            _TARGET_PASSAGE,
            "The customer support address is support@example.invalid.",
        ]
        assert len(stored.chunks[0].embedding) == 1024

        compatible_metadata = {
            "model_id": embedder.model_name,
            "model_revision": embedder.revision,
            "dimension": embedder.dimension,
            "normalized": DEFAULT_NORMALIZE_EMBEDDINGS,
        }
        seed_document(
            board_b,
            "READY",
            "board-b-closest",
            _QUERY_VECTOR,
            metadata=compatible_metadata,
        )
        seed_document(
            board_a,
            "PROCESSING",
            "processing-closest",
            _QUERY_VECTOR,
            metadata=compatible_metadata,
        )
        seed_document(
            board_a,
            "FAILED",
            "failed-closest",
            _QUERY_VECTOR,
            metadata=compatible_metadata,
        )

        results = search_by_text(
            "When is the release date?",
            board_a,
            top_k=1,
            embedder=embedder,
            storage=storage,
        )

        assert embedder.queries == [["When is the release date?"]]
        assert len(results) == 1
        result = results[0]
        assert result.document_id == persisted_document_id
        assert result.board_id == board_a
        assert result.chunk_index == 0
        assert result.file_name == "nexora_release.txt"
        assert result.storage_key.startswith("local://")
        assert result.content == _TARGET_PASSAGE
        assert result.start_offset == 0
        assert result.end_offset == len(_TARGET_PASSAGE)
        assert result.cosine_distance == pytest.approx(0.2, abs=1e-5)
    finally:
        connection = storage.get_connection()
        try:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM documents WHERE id = %s;", (document_id,)
                    )
        finally:
            connection.close()
