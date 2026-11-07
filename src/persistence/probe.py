"""Read-only PostgreSQL and pgvector connectivity probe."""

import math

import psycopg

from src.persistence.config import DATABASE_PROBE_CONNECT_TIMEOUT_SECONDS


class MissingVectorExtension(RuntimeError):
    """The vector extension is absent."""


class VectorOperationError(RuntimeError):
    """The vector distance operation failed."""


def probe_pgvector(database_url: str) -> str:
    with psycopg.connect(
        database_url,
        connect_timeout=DATABASE_PROBE_CONNECT_TIMEOUT_SECONDS,
    ) as connection:
        row = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        if row is None:
            raise MissingVectorExtension
        try:
            distance = connection.execute(
                "SELECT '[1,0]'::vector <-> '[0,1]'::vector"
            ).fetchone()
        except psycopg.Error as exc:
            raise VectorOperationError from exc
        if distance is None or not math.isclose(distance[0], math.sqrt(2)):
            raise VectorOperationError
        return row[0]
