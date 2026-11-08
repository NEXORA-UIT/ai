"""Opt-in CLI for running the locked A/B/C benchmark on TEST_DATABASE_URL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import UUID

from evaluation.adapters import build_retrieval_variants
from evaluation.manifest import EvaluationDataset, load_evaluation_dataset
from evaluation.run_retrieval import RerankerProfile, run_retrieval
from evaluation.schemas import DatasetKind
from src.embedding.embedder import BgeM3Embedder
from src.persistence.document_storage import DocumentStorage
from src.reranking.config import get_rerank_candidate_limit
from src.reranking.cross_encoder import get_default_reranker


def validate_test_database_target(
    database_url: str, expected_database_name: str
) -> str:
    """Require an explicit TEST_DATABASE_URL naming the expected _test DB."""
    parsed = urlsplit(database_url)
    database_name = unquote(parsed.path.lstrip("/"))
    if parsed.scheme not in {"postgres", "postgresql"} or not database_name:
        raise ValueError("TEST_DATABASE_URL must point to one PostgreSQL database")
    if not expected_database_name.endswith("_test"):
        raise ValueError("expected database name must end in _test")
    if database_name != expected_database_name:
        raise ValueError("TEST_DATABASE_URL does not match expected database name")
    return database_name


def _verify_database_and_seed(
    storage: DocumentStorage,
    expected_database_name: str,
    dataset: EvaluationDataset,
) -> None:
    connection = storage.get_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database();")
                row = cursor.fetchone()
                if row is None or row[0] != expected_database_name:
                    raise ValueError(
                        "connected database does not match expected _test DB"
                    )
                chunk_ids = list(dataset.snapshot_chunk_boards)
                try:
                    typed_chunk_ids = [UUID(chunk_id) for chunk_id in chunk_ids]
                except (TypeError, ValueError, AttributeError) as exc:
                    raise ValueError(
                        "database-backed benchmark snapshots must use UUID chunk IDs"
                    ) from exc
                cursor.execute(
                    """
                    SELECT dc.id::text, d.board_id::text
                    FROM document_chunks AS dc
                    JOIN documents AS d ON d.id = dc.document_id
                    WHERE dc.id = ANY(%(chunk_ids)s::uuid[]);
                    """,
                    {"chunk_ids": typed_chunk_ids},
                )
                rows = cursor.fetchall()
                actual_boards = {
                    str(chunk_id): str(board_id) for chunk_id, board_id in rows
                }
                if set(actual_boards) != set(chunk_ids):
                    raise ValueError(
                        "TEST_DATABASE_URL is missing chunks from the frozen snapshot; "
                        "seed the dataset before benchmarking"
                    )
                for chunk_id, expected_board in dataset.snapshot_chunk_boards.items():
                    if (
                        expected_board is not None
                        and actual_boards[chunk_id] != expected_board
                    ):
                        raise ValueError(
                            "seeded chunk Board IDs do not match the frozen snapshot"
                        )
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/golden/real_documents"),
    )
    parser.add_argument("--expected-database-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--allow-synthetic-correctness", action="store_true")
    args = parser.parse_args(argv)

    try:
        dataset = load_evaluation_dataset(args.dataset)
        if (
            dataset.kind is DatasetKind.SYNTHETIC_CORRECTNESS
            and not args.allow_synthetic_correctness
        ):
            raise ValueError(
                "synthetic data requires --allow-synthetic-correctness and "
                "cannot be presented as a real quality benchmark"
            )
        database_url = os.environ.get("TEST_DATABASE_URL", "")
        database_name = validate_test_database_target(
            database_url, args.expected_database_name
        )
        storage = DocumentStorage(database_url=database_url)
        _verify_database_and_seed(storage, database_name, dataset)
        reranker = get_default_reranker()
        variants = build_retrieval_variants(
            embedder=BgeM3Embedder(),
            storage=storage,
            reranker=reranker,
        )
        report = run_retrieval(
            args.dataset,
            variants,
            k=args.k,
            candidate_k=args.candidate_k,
            output_dir=args.output_dir,
            reranker_profile=RerankerProfile(
                model_id=reranker.model_id,
                model_revision=reranker.model_revision,
                candidate_limit=min(
                    args.candidate_k, get_rerank_candidate_limit(top_k=args.k)
                ),
                max_length=reranker.max_length,
                explicit_warmup=False,
            ),
        )
    except Exception as exc:
        parser.error(str(exc))
    print(
        f"Wrote {report.dataset_kind.value} A/B/C report for "
        f"{report.dataset_version} to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
