"""Command-line preview tool for document extraction, cleaning, and chunking.

Usage:
    python -m src.ingestion.preview --file path/to/document.pdf
    python -m src.ingestion.preview --file path/to/document.pdf --json
    python -m src.ingestion.preview --file path/to/document.pdf --chunk
"""

import argparse
import json
import sys
import time
from pathlib import Path

from src.ingestion.chunker import chunk_document
from src.ingestion.exceptions import IngestionError
from src.ingestion.pipeline import extract_and_clean
from src.ingestion.schemas import EmbeddedDocument
from src.ingestion.tokenizer import BgeM3Tokenizer, WhitespaceTokenCounter


def main() -> int:
    # Ensure stdout/stderr handles UTF-8 on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Preview document extraction, text cleaning, and chunking."
    )
    parser.add_argument(
        "--file",
        "-f",
        required=True,
        type=Path,
        help="Path to local document (.txt, .pdf, .docx)",
    )
    parser.add_argument(
        "--endpoint",
        "-e",
        default=None,
        help="Tika server endpoint (default: http://127.0.0.1:9998)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full result as JSON",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Include full raw text in console output",
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Enable recursive chunking preview",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size limit in tokens (default: 512 content tokens)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Target chunk overlap in tokens (default: 64 tokens)",
    )
    parser.add_argument(
        "--whitespace-tokenizer",
        action="store_true",
        help="Use fast whitespace tokenizer for testing/offline mode instead of BGE-M3",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help=(
            "Enable dense embedding preview with BAAI/bge-m3 "
            "(automatically enables chunking)"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for embedding inference (default: cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size for embedding (default: 8)",
    )
    parser.add_argument(
        "--export-vectors",
        action="store_true",
        help="Include complete 1024-d vectors in JSON output (default: excluded)",
    )

    args = parser.parse_args()

    # Early validation of embedding and chunking parameters
    # before invoking external services
    if args.embed:
        args.chunk = True
        if args.batch_size < 1:
            print(
                f"[ERROR] batch-size must be >= 1, got {args.batch_size}",
                file=sys.stderr,
            )
            return 1
        if args.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                print(
                    "[ERROR] CUDA was explicitly requested (--device cuda), but "
                    "torch.cuda.is_available() is False. "
                    "Nexora does not fall back silently.",
                    file=sys.stderr,
                )
                return 1

    if args.chunk:
        if args.chunk_size < 1:
            print(
                f"[ERROR] chunk-size must be >= 1, got {args.chunk_size}",
                file=sys.stderr,
            )
            return 1
        if args.chunk_overlap < 0:
            print(
                f"[ERROR] chunk-overlap must be >= 0, got {args.chunk_overlap}",
                file=sys.stderr,
            )
            return 1
        if args.chunk_overlap >= args.chunk_size:
            print(
                f"[ERROR] chunk-overlap ({args.chunk_overlap}) must be < "
                f"chunk-size ({args.chunk_size})",
                file=sys.stderr,
            )
            return 1

    try:
        extraction, cleaned = extract_and_clean(
            file_path=args.file,
            tika_endpoint=args.endpoint,
            timeout=args.timeout,
        )
    except IngestionError as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[FATAL] Unexpected error: {exc}", file=sys.stderr)
        return 1

    chunked = None
    if args.chunk:
        try:
            tokenizer = (
                WhitespaceTokenCounter()
                if args.whitespace_tokenizer
                else BgeM3Tokenizer()
            )
            chunked = chunk_document(
                cleaned_doc=cleaned,
                token_counter=tokenizer,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        except Exception as exc:
            print(f"[ERROR] Chunking failed: {exc}", file=sys.stderr)
            return 1

    embedded_doc = None
    model_load_time = 0.0
    encode_time = 0.0
    if args.embed and chunked:
        try:
            from src.embedding.embedder import BgeM3Embedder

            t0 = time.perf_counter()
            embedder = BgeM3Embedder(device=args.device, batch_size=args.batch_size)
            embedder._get_model()
            t1 = time.perf_counter()
            model_load_time = t1 - t0

            t2 = time.perf_counter()
            embedded_chunks = embedder.embed_chunks(
                chunked.chunks, batch_size=args.batch_size
            )
            t3 = time.perf_counter()
            encode_time = t3 - t2

            embedded_doc = EmbeddedDocument(
                document_name=chunked.document_name,
                total_chunks=len(embedded_chunks),
                total_tokens=chunked.total_tokens,
                chunks=embedded_chunks,
                model_name=embedder.model_name,
                model_revision=embedder.revision or "",
                dimension=embedder.dimension,
                normalized=True,
                chunking_version=chunked.chunking_version,
                cleaning_version=cleaned.cleaning_version,
            )
        except Exception as exc:
            print(f"[ERROR] Embedding failed: {exc}", file=sys.stderr)
            return 1

    if args.json:
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
                        "l2_norm": round(
                            sum(x * x for x in ec.dense_embedding) ** 0.5, 6
                        ),
                        "embedding_preview": [
                            round(x, 4) for x in ec.dense_embedding[:5]
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
        return 0

    print("=" * 60)
    print("NEXORA DOCUMENT EXTRACTION & CLEANING PREVIEW")
    print("=" * 60)
    print(f"File Name:        {cleaned.file_name}")
    print(f"MIME Type:        {cleaned.mime_type}")
    print(f"Declared Pages:   {extraction.page_count or 'Unknown'}")
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
        print("\n" + "=" * 60)
        print(f"NEXORA RAG RECURSIVE CHUNKING PREVIEW [{tok_label}]")
        print("=" * 60)
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
            print("-" * 60)
            print(c.content[:300])
            if len(c.content) > 300:
                print("...")

    if embedded_doc:
        print("\n" + "=" * 60)
        print("NEXORA RAG DENSE EMBEDDING PREVIEW [BAAI/bge-m3]")
        print("=" * 60)
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
        for ec in embedded_doc.chunks[:3]:
            c = ec.chunk
            vec = ec.dense_embedding
            p_str = f"Page {c.page_number}" if c.page_number else "None"
            vec_preview = [round(x, 4) for x in vec[:5]]
            norm = sum(x * x for x in vec) ** 0.5
            print(
                f"\nChunk #{c.chunk_index} "
                f"[Tokens: {c.token_count} | Page: {p_str} | L2 Norm: {norm:.4f}]"
            )
            print(f"First 5 Dimensions: {vec_preview} ...")
            print("-" * 60)
            print(c.content[:200])
            if len(c.content) > 200:
                print("...")
        if len(embedded_doc.chunks) > 3:
            print(f"\n... and {len(embedded_doc.chunks) - 3} more embedded chunks.")
    elif not chunked:
        print("\n--- CLEANED TEXT PREVIEW (First 500 chars) ---")
        print(cleaned.cleaned_text[:500])
        if len(cleaned.cleaned_text) > 500:
            print("...\n--- END OF PREVIEW ---")

    if args.raw:
        print("\n--- RAW TEXT ---")
        print(extraction.raw_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
