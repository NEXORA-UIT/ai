import math

import pytest

from scripts import check_db
from src import database


class FakeConnection:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query):
        self.queries.append(query)
        assert query.lstrip().upper().startswith("SELECT ")
        return self

    def fetchone(self):
        return next(self.rows)


def test_probe_returns_version_after_vector_operation(monkeypatch):
    conn = FakeConnection([("0.8.0",), (math.sqrt(2),)])
    monkeypatch.setattr(database.psycopg, "connect", lambda *a, **kw: conn)
    assert database.probe_pgvector("postgresql://localhost/db") == "0.8.0"
    assert len(conn.queries) == 2


def test_probe_rejects_missing_extension(monkeypatch):
    conn = FakeConnection([None])
    monkeypatch.setattr(database.psycopg, "connect", lambda *a, **kw: conn)
    with pytest.raises(RuntimeError):
        database.probe_pgvector("postgresql://localhost/db")


def test_probe_rejects_broken_vector_distance(monkeypatch):
    conn = FakeConnection([("0.8.0",), (0.0,)])
    monkeypatch.setattr(database.psycopg, "connect", lambda *a, **kw: conn)
    with pytest.raises(RuntimeError):
        database.probe_pgvector("postgresql://localhost/db")


def test_script_requires_url(monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert check_db.main() != 0
    assert "DATABASE_URL" in capsys.readouterr().out


def test_script_hides_credentials_on_connection_failure(monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@localhost/db")

    def fail(url):
        raise RuntimeError(url)

    monkeypatch.setattr(check_db, "probe_pgvector", fail)
    assert check_db.main() != 0
    assert "secret" not in capsys.readouterr().out


def test_script_reports_success(monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@localhost/db")
    monkeypatch.setattr(check_db, "probe_pgvector", lambda url: "0.8.0")
    assert check_db.main() == 0
    output = capsys.readouterr().out
    assert "0.8.0" in output
    assert "secret" not in output


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (database.psycopg.OperationalError("secret"), "connection"),
        (database.MissingVectorExtension("secret"), "extension"),
        (database.VectorOperationError("secret"), "operation"),
    ],
)
def test_script_reports_safe_failure_category(monkeypatch, capsys, error, category):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@localhost/db")

    def fail(url):
        raise error

    monkeypatch.setattr(check_db, "probe_pgvector", fail)
    assert check_db.main() != 0
    output = capsys.readouterr().out.lower()
    assert category in output
    assert "secret" not in output


def test_probe_classifies_vector_query_error(monkeypatch):
    class BrokenVectorConnection(FakeConnection):
        def execute(self, query):
            if "<->" in query:
                raise database.psycopg.ProgrammingError("secret")
            return super().execute(query)

    conn = BrokenVectorConnection([("0.8.0",)])
    monkeypatch.setattr(database.psycopg, "connect", lambda *a, **kw: conn)
    with pytest.raises(database.VectorOperationError):
        database.probe_pgvector("postgresql://localhost/db")
