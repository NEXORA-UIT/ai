from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ingestion_modules_are_grouped_by_domain():
    expected_files = (
        "src/ingestion/extraction/__init__.py",
        "src/ingestion/extraction/file_validator.py",
        "src/ingestion/extraction/tika_client.py",
        "src/ingestion/processing/__init__.py",
        "src/ingestion/processing/chunker.py",
        "src/ingestion/processing/cleaner.py",
        "src/ingestion/processing/tokenizer.py",
        "src/ingestion/persistence/__init__.py",
        "src/ingestion/persistence/storage.py",
    )

    missing = [path for path in expected_files if not (REPO_ROOT / path).is_file()]

    assert not missing, f"ingestion modules are missing from target packages: {missing}"


def test_helper_scripts_are_grouped_by_purpose():
    expected_files = (
        "scripts/db/__init__.py",
        "scripts/db/bootstrap_dev_db.sql",
        "scripts/db/check_db.py",
        "scripts/ingestion/__init__.py",
        "scripts/ingestion/run_tika.py",
    )

    missing = [path for path in expected_files if not (REPO_ROOT / path).is_file()]

    assert not missing, f"helper scripts are missing from target packages: {missing}"


def test_ingestion_tests_are_grouped_by_domain():
    expected_files = (
        "tests/ingestion/fixtures.py",
        "tests/ingestion/extraction/test_file_validator.py",
        "tests/ingestion/extraction/test_tika_client.py",
        "tests/ingestion/extraction/test_integration_tika.py",
        "tests/ingestion/processing/test_chunk_schemas.py",
        "tests/ingestion/processing/test_chunker.py",
        "tests/ingestion/processing/test_chunker_integration.py",
        "tests/ingestion/processing/test_chunker_regression.py",
        "tests/ingestion/processing/test_cleaner.py",
        "tests/ingestion/processing/test_embedded_schemas.py",
        "tests/ingestion/processing/test_tokenizer.py",
        "tests/ingestion/persistence/test_storage_unit.py",
        "tests/ingestion/persistence/test_storage_integration.py",
        "tests/ingestion/workflow/test_pipeline.py",
        "tests/ingestion/workflow/test_pipeline_smoke_db.py",
        "tests/ingestion/workflow/test_pipeline_storage_unit.py",
    )

    missing = [path for path in expected_files if not (REPO_ROOT / path).is_file()]

    assert not missing, f"ingestion tests are missing from domain folders: {missing}"


def test_documentation_is_grouped_by_topic():
    expected_files = (
        "docs/architecture/rag-architecture.md",
        "docs/guides/ingestion/chunking.md",
        "docs/guides/ingestion/database.md",
        "docs/guides/ingestion/embedding.md",
        "docs/guides/ingestion/extraction.md",
        "docs/plans/implementation/2026-11-01-m8-search-documents.md",
        "docs/plans/specs/2026-11-01-rag-retrieval-m2-m8-design.md",
        "docs/refactor/2026-11-05-ai-structure-refactor.md",
    )

    missing = [path for path in expected_files if not (REPO_ROOT / path).is_file()]

    assert not missing, f"documentation is missing from target folders: {missing}"
