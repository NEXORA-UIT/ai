"""Read-only PostgreSQL and pgvector capability probe."""

from math import isclose

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

DATABASE_CONNECT_TIMEOUT_SECONDS = 5
VECTOR_DISTANCE_TOLERANCE = 1e-6


class PgvectorExtensionMissing(RuntimeError):
    """Raised when the connected database has no installed vector extension."""


def probe_pgvector(database_url: str) -> str:
    """Return the installed extension version after testing a vector operation.

    This function is intentionally read-only. Extension installation and schema
    changes belong to Prisma migrations owned by the Express backend.
    """
    with psycopg.connect(
        database_url, connect_timeout=DATABASE_CONNECT_TIMEOUT_SECONDS
    ) as connection:
        extension = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = %s",
            ("vector",),
        ).fetchone()
        if extension is None:
            raise PgvectorExtensionMissing(
                "PostgreSQL extension 'vector' is missing; enable it with a Prisma "
                "migration containing CREATE EXTENSION vector"
            )

        register_vector(connection)
        distance = connection.execute(
            "SELECT %s::vector <-> %s::vector",
            (Vector([1.0, 0.0, 0.0]), Vector([0.0, 1.0, 0.0])),
        ).fetchone()[0]
        if not isclose(distance, 2**0.5, rel_tol=VECTOR_DISTANCE_TOLERANCE):
            raise RuntimeError(
                "pgvector distance operation returned an unexpected value"
            )

    return str(extension[0])
