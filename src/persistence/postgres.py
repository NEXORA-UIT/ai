"""PostgreSQL connection setup for AI document reads and writes."""

import os
from collections.abc import Callable
from typing import Protocol

import psycopg
from pgvector.psycopg import register_vector

from src.database_config import DATABASE_STORAGE_CONNECT_TIMEOUT_SECONDS
from src.ingestion.exceptions import DatabaseConnectionError


class ConnectionProvider(Protocol):
    """Supply a connection that the caller owns and closes."""

    def get_connection(self) -> psycopg.Connection: ...


def connect_pgvector(
    *,
    database_url: str | None = None,
    connection_factory: Callable[[], psycopg.Connection] | None = None,
) -> psycopg.Connection:
    try:
        if connection_factory is not None:
            connection = connection_factory()
        else:
            url = database_url or os.environ.get("DATABASE_URL")
            if not url:
                raise DatabaseConnectionError("DATABASE_URL is not configured.")
            connection = psycopg.connect(
                url, connect_timeout=DATABASE_STORAGE_CONNECT_TIMEOUT_SECONDS
            )
        if isinstance(connection, psycopg.Connection):
            try:
                register_vector(connection)
            except psycopg.Error:
                connection.close()
                raise
        return connection
    except psycopg.Error as exc:
        raise DatabaseConnectionError(
            f"Failed to connect to PostgreSQL: {exc}"
        ) from exc
