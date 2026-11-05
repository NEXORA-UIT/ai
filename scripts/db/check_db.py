"""Check PostgreSQL and pgvector without modifying the database."""

import os

import psycopg

from src.database import MissingVectorExtension, VectorOperationError, probe_pgvector


def main() -> int:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL is required.")
        return 1
    try:
        version = probe_pgvector(database_url)
    except MissingVectorExtension:
        print("The vector extension is not installed.")
        return 1
    except VectorOperationError:
        print("The vector operation failed.")
        return 1
    except psycopg.Error:
        print("PostgreSQL connection or database query failed.")
        return 1
    except Exception:
        print("PostgreSQL/pgvector check failed.")
        return 1
    print(f"PostgreSQL/pgvector check passed (vector {version}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
