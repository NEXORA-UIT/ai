from unittest.mock import MagicMock

from scripts.db import bootstrap


def test_bootstrap_refuses_a_different_connected_database(monkeypatch):
    connection = MagicMock()
    database = connection.__enter__.return_value
    cursor = database.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = ("other_db",)
    monkeypatch.setattr(bootstrap.psycopg, "connect", lambda *a, **kw: connection)

    assert bootstrap.run_bootstrap("postgresql://localhost/dev_db", "dev_db") == 1
    cursor.execute.assert_called_once_with("SELECT current_database();")
