"""Console and JSON rendering for the ingestion preview command."""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass

from src.ingestion.console_config import (
    CHUNK_PREVIEW_CHARS,
    CLEANED_TEXT_PREVIEW_CHARS,
    CONSOLE_SEPARATOR_WIDTH,
    EMBEDDED_CHUNK_PREVIEW_CHARS,
    EMBEDDED_CHUNK_SAMPLE_COUNT,
    EMBEDDING_PREVIEW_DIMENSIONS,
    UNKNOWN_METADATA_LABEL,
)
from src.ingestion.schemas import (
    ChunkedDocument,
    CleanedDocument,
    EmbeddedDocument,
    ExtractionResult,
)


@dataclass(frozen=True)
class PreviewResult:
    extraction: ExtractionResult
    cleaned: CleanedDocument
    chunked: ChunkedDocument | None = None
    embedded: EmbeddedDocument | None = None
    model_load_seconds: float = 0.0
    encode_seconds: float = 0.0


def _render_json(args: Namespace, result: PreviewResult) -> None:
    extraction, cleaned, chunked, embedded_doc = (
        result.extraction,
        result.cleaned,
        result.chunked,
        result.embedded,
    )
    model_load_time, encode_time = result.model_load_seconds, result.encode_seconds
    data = {
        "file_name": cleaned.file_name,
        "mime_type": cleaned.mime_type,
        "cleaning_version": cleaned.cleaning_version,
        "char_count": cleaned.char_count,
        "line_count": cleaned.line_count,
        "raw_char_count": len(extraction.raw_text),
        "page_count": extraction.page_count,
        "has_form_feeds": extraction.has_form_feeds,
        "page_spans": [
            {
                "page_number": span.page_number,
                "start_offset": span.start_offset,
                "end_offset": span.end_offset,
                "length": span.end_offset - span.start_offset,
            }
            for span in (cleaned.page_spans or [])
        ]
        if cleaned.page_spans
        else None,
        "metadata": cleaned.metadata,
        "cleaned_text": cleaned.cleaned_text,
    }
    if chunked:
        data["chunking"] = {
            "total_chunks": chunked.total_chunks,
            "total_tokens": chunked.total_tokens,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "tokenizer": chunked.config.get("tokenizer_name"),
            "tokenizer_revision": chunked.config.get("tokenizer_revision"),
            "chunking_version": chunked.chunking_version,
            "chunks": [
                {
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "char_count": c.char_count,
                    "start_offset": c.start_offset,
                    "end_offset": c.end_offset,
                    "page_number": c.page_number,
                    "content": c.content,
                }
                for c in chunked.chunks
            ],
        }
    if embedded_doc:
        data["embedding"] = {
            "model_name": embedded_doc.model_name,
            "model_revision": embedded_doc.model_revision,
            "dimension": embedded_doc.dimension,
            "normalized": embedded_doc.normalized,
            "device": args.device,
            "batch_size": args.batch_size,
            "model_load_time_seconds": round(model_load_time, 3),
            "encode_time_seconds": round(encode_time, 3),
            "total_embedded_chunks": len(embedded_doc.chunks),
            "chunks": [
                {
                    "chunk_index": ec.chunk.chunk_index,
                    "token_count": ec.chunk.token_count,
                    "page_number": ec.chunk.page_number,
                    "l2_norm": round(sum(x * x for x in ec.dense_embedding) ** 0.5, 6),
                    "embedding_preview": [
                        round(x, 4)
                        for x in ec.dense_embedding[:EMBEDDING_PREVIEW_DIMENSIONS]
                    ],
                    **(
                        {"dense_embedding": ec.dense_embedding}
                        if args.export_vectors
                        else {}
                    ),
                }
                for ec in embedded_doc.chunks
            ],
        }
    if args.raw:
        data["raw_text"] = extraction.raw_text
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _render_console(args: Namespace, result: PreviewResult) -> None:
    extraction, cleaned, chunked, embedded_doc = (
        result.extraction,
        result.cleaned,
        result.chunked,
        result.embedded,
    )
    model_load_time, encode_time = result.model_load_seconds, result.encode_seconds
    print("=" * CONSOLE_SEPARATOR_WIDTH)
    print("NEXORA DOCUMENT EXTRACTION & CLEANING PREVIEW")
    print("=" * CONSOLE_SEPARATOR_WIDTH)
    print(f"File Name:        {cleaned.file_name}")
    print(f"MIME Type:        {cleaned.mime_type}")
    print(f"Declared Pages:   {extraction.page_count or UNKNOWN_METADATA_LABEL}")
    print(f"Form Feed Breaks: {'Yes' if extraction.has_form_feeds else 'No'}")
    print(f"Raw Text Chars:   {len(extraction.raw_text)}")
    print(f"Cleaned Chars:    {cleaned.char_count}")
    print(f"Cleaned Lines:    {cleaned.line_count}")
    print(f"Cleaning Version: {cleaned.cleaning_version}")

    if cleaned.page_spans:
        print(f"\nPage Spans ({len(cleaned.page_spans)} detected):")
        for span in cleaned.page_spans:
            diff = span.end_offset - span.start_offset
            print(
                f"  - Page {span.page_number:2d}: "
                f"[{span.start_offset:5d} -> {span.end_offset:5d}] "
                f"({diff:4d} chars)"
            )
    else:
        print("\nPage Spans:       None (Document has no reliable page break markers)")

    if chunked:
        rev = chunked.config.get("tokenizer_revision")
        tok_label = (
            "Whitespace (Test Mode)"
            if args.whitespace_tokenizer
            else f"BGE-M3 @ {rev[:7] if rev else 'latest'}"
        )
        print("\n" + "=" * CONSOLE_SEPARATOR_WIDTH)
        print(f"NEXORA RAG RECURSIVE CHUNKING PREVIEW [{tok_label}]")
        print("=" * CONSOLE_SEPARATOR_WIDTH)
        print(f"Total Chunks:     {chunked.total_chunks}")
        print(
            f"Total Tokens:     {chunked.total_tokens} "
            "(sum across chunks, includes overlap)"
        )
        print(
            f"Chunk Size:       {args.chunk_size} content tokens (excl. special tokens)"
        )
        print(
            f"Target Overlap:   {args.chunk_overlap} tokens "
            "(snapped to sentence/paragraph boundaries)"
        )
        print(f"Tokenizer:        {chunked.config.get('tokenizer_name')}")
        if rev:
            print(f"Revision:         {rev}")
        print(f"Chunking Version: {chunked.chunking_version}")

        print("\n--- CHUNK BREAKDOWN ---")
        for c in chunked.chunks:
            p_str = f"Page {c.page_number}" if c.page_number else "None"
            print(
                f"\nChunk #{c.chunk_index} "
                f"[Tokens: {c.token_count} | Chars: {c.char_count} | "
                f"Offsets: {c.start_offset}->{c.end_offset} | Page: {p_str}]"
            )
            print("-" * CONSOLE_SEPARATOR_WIDTH)
            print(c.content[:CHUNK_PREVIEW_CHARS])
            if len(c.content) > CHUNK_PREVIEW_CHARS:
                print("...")

    if embedded_doc:
        print("\n" + "=" * CONSOLE_SEPARATOR_WIDTH)
        print("NEXORA RAG DENSE EMBEDDING PREVIEW [BAAI/bge-m3]")
        print("=" * CONSOLE_SEPARATOR_WIDTH)
        print(f"Total Vectors:     {len(embedded_doc.chunks)}")
        print(f"Vector Dimension:  {embedded_doc.dimension}")
        print(f"L2 Normalized:     {embedded_doc.normalized}")
        print(f"Model ID:          {embedded_doc.model_name}")
        print(f"Model Revision:    {embedded_doc.model_revision}")
        print(f"Device:            {args.device}")
        print(f"Batch Size:        {args.batch_size}")
        print(f"Model Load Time:   {model_load_time:.2f}s")
        print(
            f"Encode Time:       {encode_time:.2f}s "
            f"({(encode_time / max(len(embedded_doc.chunks), 1)) * 1000:.1f} ms/chunk)"
        )

        print("\n--- EMBEDDED CHUNK SAMPLES ---")
        for ec in embedded_doc.chunks[:EMBEDDED_CHUNK_SAMPLE_COUNT]:
            c = ec.chunk
            vec = ec.dense_embedding
            p_str = f"Page {c.page_number}" if c.page_number else "None"
            vec_preview = [round(x, 4) for x in vec[:EMBEDDING_PREVIEW_DIMENSIONS]]
            norm = sum(x * x for x in vec) ** 0.5
            print(
                f"\nChunk #{c.chunk_index} "
                f"[Tokens: {c.token_count} | Page: {p_str} | L2 Norm: {norm:.4f}]"
            )
            print(f"First {EMBEDDING_PREVIEW_DIMENSIONS} Dimensions: {vec_preview} ...")
            print("-" * CONSOLE_SEPARATOR_WIDTH)
            print(c.content[:EMBEDDED_CHUNK_PREVIEW_CHARS])
            if len(c.content) > EMBEDDED_CHUNK_PREVIEW_CHARS:
                print("...")
        if len(embedded_doc.chunks) > EMBEDDED_CHUNK_SAMPLE_COUNT:
            print(
                f"\n... and "
                f"{len(embedded_doc.chunks) - EMBEDDED_CHUNK_SAMPLE_COUNT} "
                "more embedded chunks."
            )
    elif not chunked:
        print(
            f"\n--- CLEANED TEXT PREVIEW (First {CLEANED_TEXT_PREVIEW_CHARS} chars) ---"
        )
        print(cleaned.cleaned_text[:CLEANED_TEXT_PREVIEW_CHARS])
        if len(cleaned.cleaned_text) > CLEANED_TEXT_PREVIEW_CHARS:
            print("...\n--- END OF PREVIEW ---")

    if args.raw:
        print("\n--- RAW TEXT ---")
        print(extraction.raw_text)


def render_preview(args: Namespace, result: PreviewResult) -> None:
    if args.json:
        _render_json(args, result)
    else:
        _render_console(args, result)
