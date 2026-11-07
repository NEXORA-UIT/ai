import subprocess
import sys


def test_importing_ingestion_package_does_not_load_database_driver() -> None:
    script = (
        "import sys; import src.ingestion; "
        "from src.ingestion import DocumentStatus; "
        "assert DocumentStatus.READY == 'READY'; "
        "assert 'psycopg' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_ingestion_storage_export_resolves_to_shared_persistence() -> None:
    from src.ingestion import DocumentStorage
    from src.persistence.document_storage import (
        DocumentStorage as SharedDocumentStorage,
    )

    assert DocumentStorage is SharedDocumentStorage
