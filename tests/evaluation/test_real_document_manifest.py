"""Dataset manifest checks, including synthetic metadata-only test fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.manifest import load_evaluation_dataset
from evaluation.schemas import DatasetKind

_BOARD_ID = "11111111-1111-4111-8111-111111111111"
_DOCUMENT_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"
_CHUNK_ID = "a1000000-0000-4000-8000-000000000001"


def _write_dataset(
    root: Path,
    *,
    kind: str = "real_document_quality",
    source_documents: list[dict[str, str]] | None = None,
    human_labels_reviewed: bool = True,
    expected_chunk_id: str = _CHUNK_ID,
    document_status: str = "READY",
    embedding_revision: str = "test-revision",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "chunks.jsonl"
    snapshot.write_text(
        json.dumps(
            {
                "document_id": _DOCUMENT_ID,
                "board_id": _BOARD_ID,
                "file_name": "guide.pdf",
                "storage_key": "synthetic://manifest-test/guide.pdf",
                "status": document_status,
                "metadata": {
                    "model_id": "BAAI/bge-m3",
                    "model_revision": embedding_revision,
                    "dimension": 1024,
                    "normalized": True,
                },
                "chunks": [
                    {
                        "chunk_id": _CHUNK_ID,
                        "chunk_index": 0,
                        "content": "Approved procedure source text.",
                        "page_number": 1,
                        "start_offset": 0,
                        "end_offset": 32,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    queries = root / "queries.jsonl"
    queries.write_text(
        json.dumps(
            {
                "question": "What is the approved procedure?",
                "board_id": _BOARD_ID,
                "expected_chunk_ids": [expected_chunk_id],
                "answerable": True,
                "expected_facts": ["approved procedure"],
                "expected_citation_ids": [expected_chunk_id],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_documents = source_documents or [
        {
            "document_id": _DOCUMENT_ID,
            "path": "sources/guide.pdf",
            "source_url": "https://example.test/guide.pdf",
            "license": "CC-BY-4.0",
            "provenance": "Public publisher download, captured 2026-10-01",
        }
    ]
    for source in source_documents:
        source_path = root / source["path"]
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(b"%PDF-1.7\nmanifest-test-fixture\n")
        source["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = {
        "kind": kind,
        "version": "test-v1",
        "chunk_snapshot": snapshot.name,
        "chunk_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "queries": queries.name,
        "queries_sha256": hashlib.sha256(queries.read_bytes()).hexdigest(),
        "embedding": {"model_id": "BAAI/bge-m3", "revision": "test-revision"},
        "chunking": {"version": "test-chunking-v1"},
        "human_labels_reviewed": human_labels_reviewed,
        "source_documents": source_documents,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_real_manifest_requires_provenance_license_and_matching_hash(
    tmp_path: Path,
) -> None:
    dataset = load_evaluation_dataset(_write_dataset(tmp_path / "real"))

    assert dataset.kind is DatasetKind.REAL_DOCUMENT_QUALITY
    assert dataset.version == "test-v1"
    assert dataset.cases[0].expected_chunk_ids == (_CHUNK_ID,)
    assert dataset.source_document_count == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_url", ""),
        ("license", ""),
        ("provenance", ""),
        ("path", "sources/guide.txt"),
    ],
)
def test_real_manifest_rejects_missing_provenance_or_non_pdf_docx(
    tmp_path: Path, field: str, value: str
) -> None:
    source = {
        "path": "sources/guide.pdf",
        "source_url": "https://example.test/guide.pdf",
        "license": "CC-BY-4.0",
        "provenance": "Public download",
    }
    source[field] = value
    root = _write_dataset(tmp_path / "real", source_documents=[source])

    with pytest.raises(ValueError):
        load_evaluation_dataset(root)


def test_real_manifest_rejects_changed_source_checksum(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "real")
    (root / "sources" / "guide.pdf").write_bytes(b"changed")

    with pytest.raises(ValueError, match="checksum"):
        load_evaluation_dataset(root)


def test_real_manifest_requires_human_reviewed_labels(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "real", human_labels_reviewed=False)

    with pytest.raises(ValueError, match="human"):
        load_evaluation_dataset(root)


@pytest.mark.parametrize(
    ("status", "revision"),
    [("PROCESSING", "test-revision"), ("READY", "different-revision")],
)
def test_real_manifest_requires_ready_chunks_from_declared_embedding(
    tmp_path: Path, status: str, revision: str
) -> None:
    root = _write_dataset(
        tmp_path / "real", document_status=status, embedding_revision=revision
    )

    with pytest.raises(ValueError, match="READY|embedding"):
        load_evaluation_dataset(root)


def test_real_manifest_links_source_document_to_frozen_snapshot(
    tmp_path: Path,
) -> None:
    source = {
        "document_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb2",
        "path": "sources/guide.pdf",
        "source_url": "https://example.test/guide.pdf",
        "license": "CC-BY-4.0",
        "provenance": "Public publisher download, captured 2026-10-01",
    }
    root = _write_dataset(tmp_path / "real", source_documents=[source])

    with pytest.raises(ValueError, match="document IDs"):
        load_evaluation_dataset(root)


def test_real_manifest_rejects_relevant_ids_missing_from_frozen_snapshot(
    tmp_path: Path,
) -> None:
    root = _write_dataset(tmp_path / "real", expected_chunk_id="missing-chunk")

    with pytest.raises(ValueError, match="snapshot"):
        load_evaluation_dataset(root)


def test_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "real")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["queries"] = "../outside.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="inside"):
        load_evaluation_dataset(root)
