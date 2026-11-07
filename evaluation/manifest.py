"""Load and verify versioned evaluation datasets and frozen chunk snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

from evaluation.schemas import DatasetKind, GoldenCase, load_golden_cases


@dataclass(frozen=True)
class EvaluationDataset:
    """Validated dataset identity shared by every retrieval variant."""

    root: Path
    kind: DatasetKind
    version: str
    chunk_snapshot_path: Path
    chunk_snapshot_sha256: str
    queries_path: Path
    query_snapshot_sha256: str
    embedding_model_id: str
    embedding_revision: str
    chunking_version: str
    cases: tuple[GoldenCase, ...]
    snapshot_chunk_boards: dict[str, str | None]
    source_document_count: int

    @property
    def embedding_fingerprint(self) -> str:
        return f"{self.embedding_model_id}@{self.embedding_revision}"


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest {field} must be a nonempty string")
    return value.strip()


def _contained_file(root: Path, value: object, field: str) -> Path:
    relative = _nonempty_string(value, field)
    candidate_path = Path(relative)
    if candidate_path.is_absolute() or ".." in candidate_path.parts:
        raise ValueError(f"manifest {field} must resolve inside the dataset")
    resolved = (root / candidate_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest {field} must resolve inside the dataset") from exc
    if not resolved.is_file():
        raise ValueError(f"manifest {field} does not exist: {resolved}")
    return resolved


def _verified_sha256(path: Path, expected: object, field: str) -> str:
    expected_hash = _nonempty_string(expected, field).lower()
    if len(expected_hash) != 64 or any(
        char not in "0123456789abcdef" for char in expected_hash
    ):
        raise ValueError(f"manifest {field} must be a SHA-256 hex digest")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"{field} checksum does not match {path.name}")
    return actual_hash


def _read_chunk_snapshot(
    path: Path,
) -> tuple[dict[str, str | None], list[dict[str, object]]]:
    chunk_boards: dict[str, str | None] = {}
    documents: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("snapshot rows must be JSON objects")
            inherited_board = row.get("board_id")
            chunks = row.get("chunks")
            if chunks is not None:
                if not isinstance(chunks, list):
                    raise ValueError("chunks must be a list")
                documents.append(row)
                for chunk in chunks:
                    if not isinstance(chunk, dict):
                        raise ValueError("snapshot chunks must be JSON objects")
                    _add_chunk(chunk_boards, chunk.get("chunk_id"), inherited_board)
            else:
                _add_chunk(chunk_boards, row.get("chunk_id"), inherited_board)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid chunk snapshot at {path}:{line_number}") from exc
    if not chunk_boards:
        raise ValueError(f"chunk snapshot is empty: {path}")
    return chunk_boards, documents


def _add_chunk(
    chunk_boards: dict[str, str | None], chunk_id: object, board_id: object
) -> None:
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("snapshot chunk_id must be a nonempty string")
    if board_id is not None and not isinstance(board_id, str):
        raise ValueError("snapshot board_id must be a string when present")
    if board_id is not None:
        try:
            board_id = str(UUID(board_id))
        except ValueError as exc:
            raise ValueError("snapshot board_id must be a valid UUID") from exc
    if chunk_id in chunk_boards:
        raise ValueError(f"duplicate chunk ID in snapshot: {chunk_id}")
    chunk_boards[chunk_id] = board_id


def _verify_source_documents(root: Path, source_documents: object) -> set[str]:
    if not isinstance(source_documents, list) or not source_documents:
        raise ValueError("real-document datasets require source documents")
    document_ids: set[str] = set()
    for index, source in enumerate(source_documents):
        if not isinstance(source, dict):
            raise ValueError(f"source_documents[{index}] must be an object")
        try:
            document_id = str(UUID(source.get("document_id")))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(
                f"source_documents[{index}].document_id must be a UUID"
            ) from exc
        if document_id in document_ids:
            raise ValueError("source document IDs must be unique")
        document_ids.add(document_id)
        path = _contained_file(
            root, source.get("path"), f"source_documents[{index}].path"
        )
        if path.suffix.lower() not in {".pdf", ".docx"}:
            raise ValueError("real sources must be PDF or DOCX files")
        source_url = _nonempty_string(
            source.get("source_url"), f"source_documents[{index}].source_url"
        )
        parsed_url = urlsplit(source_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("source_url must be an HTTP(S) provenance link")
        _nonempty_string(source.get("license"), f"source_documents[{index}].license")
        _nonempty_string(
            source.get("provenance"), f"source_documents[{index}].provenance"
        )
        _verified_sha256(
            path, source.get("sha256"), f"source_documents[{index}].sha256"
        )
    return document_ids


def _validate_real_snapshot_documents(
    documents: list[dict[str, object]],
    source_document_ids: set[str],
    model_id: str,
    revision: str,
) -> None:
    if not documents:
        raise ValueError(
            "real-document chunk snapshots must use the guarded seeder JSONL format"
        )
    snapshot_document_ids: set[str] = set()
    for index, document in enumerate(documents):
        try:
            document_id = str(UUID(document.get("document_id")))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"snapshot document {index} must have a UUID ID") from exc
        if document_id in snapshot_document_ids:
            raise ValueError("snapshot document IDs must be unique")
        snapshot_document_ids.add(document_id)
        if document.get("status") != "READY":
            raise ValueError("real-document snapshots may contain only READY documents")
        metadata = document.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("snapshot document metadata must be an object")
        if (
            metadata.get("model_id") != model_id
            or metadata.get("model_revision") != revision
            or metadata.get("dimension") != 1024
            or metadata.get("normalized") is not True
        ):
            raise ValueError(
                "snapshot embedding metadata does not match the declared embedding"
            )
    if snapshot_document_ids != source_document_ids:
        raise ValueError(
            "source and snapshot document IDs must match for reproducibility"
        )


def load_evaluation_dataset(dataset_path: Path) -> EvaluationDataset:
    """Validate a manifest, its source files, frozen snapshot and query labels."""
    root = Path(dataset_path).resolve()
    if not root.is_dir():
        raise ValueError(f"evaluation dataset directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"evaluation manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evaluation manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("evaluation manifest must be a JSON object")
    try:
        kind = DatasetKind(manifest.get("kind"))
    except ValueError as exc:
        raise ValueError(
            "manifest kind must identify a supported dataset kind"
        ) from exc

    version = _nonempty_string(manifest.get("version"), "version")
    snapshot_path = _contained_file(
        root, manifest.get("chunk_snapshot"), "chunk_snapshot"
    )
    snapshot_hash = _verified_sha256(
        snapshot_path, manifest.get("chunk_snapshot_sha256"), "chunk_snapshot_sha256"
    )
    queries_path = _contained_file(root, manifest.get("queries"), "queries")
    query_hash = _verified_sha256(
        queries_path, manifest.get("queries_sha256"), "queries_sha256"
    )
    embedding = manifest.get("embedding")
    chunking = manifest.get("chunking")
    if not isinstance(embedding, dict) or not isinstance(chunking, dict):
        raise ValueError("manifest embedding and chunking must be objects")
    embedding_model_id = _nonempty_string(
        embedding.get("model_id"), "embedding.model_id"
    )
    embedding_revision = _nonempty_string(
        embedding.get("revision"), "embedding.revision"
    )
    chunking_version = _nonempty_string(chunking.get("version"), "chunking.version")

    cases = load_golden_cases(queries_path)
    snapshot_chunk_boards, snapshot_documents = _read_chunk_snapshot(snapshot_path)
    for case in cases:
        for chunk_id in case.expected_chunk_ids:
            if chunk_id not in snapshot_chunk_boards:
                raise ValueError(
                    f"labeled relevant chunk {chunk_id} is missing from "
                    "the frozen snapshot"
                )
            chunk_board = snapshot_chunk_boards[chunk_id]
            if chunk_board is not None and chunk_board != case.board_id:
                raise ValueError(
                    f"labeled relevant chunk {chunk_id} belongs to another "
                    "Board in the snapshot"
                )

    source_count = 0
    if kind is DatasetKind.REAL_DOCUMENT_QUALITY:
        if manifest.get("human_labels_reviewed") is not True:
            raise ValueError("real-document labels must be marked human reviewed")
        source_document_ids = _verify_source_documents(
            root, manifest.get("source_documents")
        )
        _validate_real_snapshot_documents(
            snapshot_documents,
            source_document_ids,
            embedding_model_id,
            embedding_revision,
        )
        source_count = len(source_document_ids)
    elif "source_documents" in manifest:
        documents = manifest["source_documents"]
        if not isinstance(documents, list):
            raise ValueError("source_documents must be a list")
        source_count = len(documents)

    return EvaluationDataset(
        root=root,
        kind=kind,
        version=version,
        chunk_snapshot_path=snapshot_path,
        chunk_snapshot_sha256=snapshot_hash,
        queries_path=queries_path,
        query_snapshot_sha256=query_hash,
        embedding_model_id=embedding_model_id,
        embedding_revision=embedding_revision,
        chunking_version=chunking_version,
        cases=cases,
        snapshot_chunk_boards=snapshot_chunk_boards,
        source_document_count=source_count,
    )
