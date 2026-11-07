"""Explicit command line interface for document ingestion and database persistence."""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

# Ensure stdout and stderr support UTF-8 on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from scripts.db.bootstrap import parse_safe_db_info
from scripts.ingestion.console_config import (
    CONSOLE_SEPARATOR_WIDTH,
    STORED_CHUNK_PREVIEW_CHARS,
    UNKNOWN_METADATA_LABEL,
)
from scripts.local_env import load_local_env
from src.ingestion.pipeline import ingest_file_to_storage
from src.persistence.document_storage import DocumentStorage


def run_inspect(storage: DocumentStorage, document_id: str) -> int:
    """Inspect and display stored document and chunks without dumping raw vectors."""
    stored = storage.get_document_with_chunks(document_id)
    if not stored:
        print(f"Document with ID {document_id} not found in database.", file=sys.stderr)
        return 1

    doc = stored.document
    print("=" * CONSOLE_SEPARATOR_WIDTH)
    print("NEXORA STORED DOCUMENT INSPECTION")
    print("=" * CONSOLE_SEPARATOR_WIDTH)
    print(f"Document ID:      {doc.id}")
    print(f"Board ID:         {doc.board_id}")
    print(f"Uploaded By:      {doc.uploaded_by}")
    print(f"File Name:        {doc.file_name}")
    print(f"Storage Key:      {doc.storage_key}")
    print(f"MIME Type:        {doc.mime_type}")
    print(f"File Size:        {doc.file_size:,} bytes")
    print(f"SHA-256:          {doc.checksum}")
    print(f"Status:           {doc.status}")
    print(f"Processing Token: {doc.processing_token or '(None - completed)'}")
    print(f"Created At:       {doc.created_at}")
    print(f"Updated At:       {doc.updated_at}")
    print("-" * CONSOLE_SEPARATOR_WIDTH)
    print(f"Total Chunks:     {len(stored.chunks)}")
    print(
        f"Model ID:         "
        f"{doc.metadata.get('model_id', UNKNOWN_METADATA_LABEL)}"
    )
    print(
        f"Model Revision:   "
        f"{doc.metadata.get('model_revision', UNKNOWN_METADATA_LABEL)}"
    )
    tokens_desc = (
        f"{doc.metadata.get('total_tokens', UNKNOWN_METADATA_LABEL)} "
        f"(sum of chunk tokens, incl. overlap)"
    )
    print(f"Total Tokens:     {tokens_desc}")
    print(
        f"Chunking Version: "
        f"{doc.metadata.get('chunking_version', UNKNOWN_METADATA_LABEL)}"
    )
    print(
        f"Cleaning Version: "
        f"{doc.metadata.get('cleaning_version', UNKNOWN_METADATA_LABEL)}"
    )

    if doc.metadata.get("error_stage"):
        print("-" * CONSOLE_SEPARATOR_WIDTH)
        print(f"ERROR STAGE:      {doc.metadata.get('error_stage')}")
        print(f"ERROR TYPE:       {doc.metadata.get('error_type')}")
        print(f"ERROR MESSAGE:    {doc.metadata.get('error_message')}")
        print(f"FAILED AT:        {doc.metadata.get('failed_at')}")

    if stored.chunks:
        print("-" * CONSOLE_SEPARATOR_WIDTH)
        print(f"--- SAMPLE CHUNKS (1 of {len(stored.chunks)}) ---")
        first_chunk = stored.chunks[0]
        print(f"Chunk #{first_chunk.chunk_index} [ID: {first_chunk.id}]")
        p_num = first_chunk.page_number
        s_off = first_chunk.start_offset
        e_off = first_chunk.end_offset
        print(f"Page: {p_num} | Offsets: [{s_off}, {e_off})")
        print(f"Tokens: {first_chunk.metadata.get('token_count', 'N/A')}")
        vec_dim = len(first_chunk.embedding)
        val0 = first_chunk.embedding[0]
        print(f"Vector Dimensions: {vec_dim} floats (preview: [{val0:.4f}, ...])")
        preview_text = first_chunk.content[:STORED_CHUNK_PREVIEW_CHARS].replace(
            "\n", " "
        )
        print(f"Content Preview: {preview_text}...")

    print("=" * CONSOLE_SEPARATOR_WIDTH)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.ingestion.cli",
        description="Nexora AI Ingestion CLI: Ingest files into pgvector.",
    )
    parser.add_argument(
        "-f", "--file", type=str, help="Path to local file (.txt, .pdf, .docx)."
    )
    parser.add_argument("--board-id", type=str, help="Target Board UUID.")
    parser.add_argument("--uploaded-by", type=str, help="Uploader User UUID.")
    parser.add_argument(
        "--document-id",
        type=str,
        help="Existing document UUID (for re-processing).",
    )
    parser.add_argument(
        "--reclaim",
        action="store_true",
        help="Reclaim ownership if document is stuck in PROCESSING.",
    )
    parser.add_argument("--database-url", type=str, help="Override DATABASE_URL.")
    parser.add_argument(
        "--inspect",
        type=str,
        metavar="DOC_UUID",
        help="Inspect a stored document by UUID.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI main entry point."""
    load_local_env(Path(__file__).parent.parent.parent / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)

    db_url = (
        args.database_url
        or os.environ.get("DATABASE_URL")
        or os.environ.get("TEST_DATABASE_URL")
    )

    storage = DocumentStorage(database_url=db_url)

    # Inspect a stored document
    if args.inspect:
        return run_inspect(storage, args.inspect)

    # Ingest File
    if not args.file:
        print("ERROR: -f / --file is required for ingestion.", file=sys.stderr)
        return 1

    if not args.board_id:
        print("ERROR: --board-id is required for ingestion.", file=sys.stderr)
        return 1

    try:
        board_uuid = uuid.UUID(args.board_id)
    except ValueError:
        print(f"ERROR: Invalid UUID for --board-id: '{args.board_id}'", file=sys.stderr)
        return 1

    uploaded_by_uuid: uuid.UUID
    if args.uploaded_by:
        try:
            uploaded_by_uuid = uuid.UUID(args.uploaded_by)
        except ValueError:
            print(
                f"ERROR: Invalid UUID for --uploaded-by: '{args.uploaded_by}'",
                file=sys.stderr,
            )
            return 1
    elif args.document_id:
        # Reprocess can retain existing uploader; placeholder for initial claim call
        uploaded_by_uuid = uuid.UUID("00000000-0000-0000-0000-000000000000")
    else:
        print(
            "ERROR: --uploaded-by is required when creating a new document.",
            file=sys.stderr,
        )
        return 1

    doc_uuid: uuid.UUID | None = None
    if args.document_id:
        try:
            doc_uuid = uuid.UUID(args.document_id)
        except ValueError:
            print(
                f"ERROR: Invalid UUID for --document-id: '{args.document_id}'",
                file=sys.stderr,
            )
            return 1

    db_info = parse_safe_db_info(db_url)
    h = db_info.get("host")
    p = db_info.get("port")
    d = db_info.get("database")
    print("=" * CONSOLE_SEPARATOR_WIDTH)
    print("NEXORA DOCUMENT INGESTION TO DATABASE")
    print("=" * CONSOLE_SEPARATOR_WIDTH)
    print(f"File:             {args.file}")
    print(f"Board ID:         {board_uuid}")
    print(f"Target Doc ID:    {doc_uuid or '(New document)'}")
    print(f"Database Target:  host={h}:{p}, db={d}")
    print(f"Reclaim Mode:     {args.reclaim}")
    print("-" * CONSOLE_SEPARATOR_WIDTH)

    start_time = time.perf_counter()
    try:
        persisted_id, embedded_doc = ingest_file_to_storage(
            file_path=args.file,
            board_id=board_uuid,
            uploaded_by=uploaded_by_uuid,
            storage=storage,
            document_id=doc_uuid,
            reclaim=args.reclaim,
        )
        total_time = time.perf_counter() - start_time
        print("STATUS:           SUCCESS (READY)")
        print(f"Persisted Doc ID: {persisted_id}")
        print(f"Total Chunks:     {embedded_doc.total_chunks}")
        print(f"Total Tokens:     {embedded_doc.total_tokens}")
        print(f"Model ID:         {embedded_doc.model_name}")
        print(f"Model Revision:   {embedded_doc.model_revision}")
        print(f"Vector Dimension: {embedded_doc.dimension}")
        print(f"Elapsed Time:     {total_time:.2f}s")
        print("=" * CONSOLE_SEPARATOR_WIDTH)
        return 0
    except Exception as exc:
        total_time = time.perf_counter() - start_time
        print(f"STATUS:           FAILED ({type(exc).__name__})", file=sys.stderr)
        print(f"Error Details:    {exc}", file=sys.stderr)
        print(f"Elapsed Time:     {total_time:.2f}s", file=sys.stderr)
        print("=" * CONSOLE_SEPARATOR_WIDTH, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
