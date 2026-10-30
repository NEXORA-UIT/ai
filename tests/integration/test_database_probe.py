import os
from unittest.mock import Mock

import pytest

from nexora_ai.infrastructure.database import (
    PgvectorExtensionMissing,
    probe_pgvector,
)


def test_probe_reports_missing_pgvector_without_creating_extension(monkeypatch):
    connection = Mock()
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=False)
    connection.execute.return_value.fetchone.return_value = None
    connect = Mock(return_value=connection)
    monkeypatch.setattr("nexora_ai.infrastructure.database.psycopg.connect", connect)

    with pytest.raises(PgvectorExtensionMissing, match="CREATE EXTENSION"):
        probe_pgvector("postgresql://user:secret@localhost/db")

    connect.assert_called_once_with(
        "postgresql://user:secret@localhost/db", connect_timeout=5
    )
    assert connection.execute.call_count == 1
    assert "CREATE EXTENSION" not in connection.execute.call_args.args[0]


@pytest.mark.integration
def test_backend_database_has_working_pgvector():
    database_url = os.getenv("DATABASE_URL")
    if database_url is None:
        pytest.skip("DATABASE_URL is not configured for the backend integration test")

    version = probe_pgvector(database_url)
    assert version
