"""Command-line preview tool for document extraction, cleaning, and chunking.

Usage:
    python -m src.ingestion.preview --file path/to/document.pdf
    python -m src.ingestion.preview --file path/to/document.pdf --json
    python -m src.ingestion.preview --file path/to/document.pdf --chunk
"""

import argparse
import sys
import time
from pathlib import Path

from src.chunking.chunker import chunk_document
from src.chunking.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from src.chunking.tokenizer import BgeM3Tokenizer, WhitespaceTokenCounter
from src.embedding.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DENSE_DIMENSION,
    DEFAULT_DEVICE,
)
from src.ingestion.exceptions import IngestionError
from src.ingestion.extraction.tika_client import (
    DEFAULT_TIKA_ENDPOINT,
    DEFAULT_TIMEOUT_SECONDS,
)
from src.ingestion.pipeline import embed_chunked_document, extract_and_clean
from src.ingestion.preview_output import PreviewResult, render_preview


def _build_parser() -> argparse.ArgumentParser:
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
        help=f"Tika server endpoint (default: {DEFAULT_TIKA_ENDPOINT})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
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
        default=DEFAULT_CHUNK_SIZE,
        help=(
            f"Chunk size limit in tokens (default: {DEFAULT_CHUNK_SIZE} content tokens)"
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Target chunk overlap in tokens (default: {DEFAULT_CHUNK_OVERLAP})",
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
        choices=[DEFAULT_DEVICE, "cuda"],
        default=DEFAULT_DEVICE,
        help="Device for embedding inference (default: cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch size for embedding (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--export-vectors",
        action="store_true",
        help=(
            "Include complete "
            f"{DEFAULT_DENSE_DIMENSION}-d vectors in JSON output "
            "(default: excluded)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Ensure stdout/stderr handles UTF-8 on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = _build_parser().parse_args(argv)

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
            embedder.load_model()
            t1 = time.perf_counter()
            model_load_time = t1 - t0

            t2 = time.perf_counter()
            embedded_doc = embed_chunked_document(
                chunked, cleaned, embedder=embedder, batch_size=args.batch_size
            )
            encode_time = time.perf_counter() - t2
        except Exception as exc:
            print(f"[ERROR] Embedding failed: {exc}", file=sys.stderr)
            return 1

    render_preview(
        args,
        PreviewResult(
            extraction, cleaned, chunked, embedded_doc, model_load_time, encode_time
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
