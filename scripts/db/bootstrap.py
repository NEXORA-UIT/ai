"""Bootstrap the schema on a confirmed local development database."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psycopg

from scripts.local_env import load_local_env
from src.persistence.config import (
    DATABASE_BOOTSTRAP_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_POSTGRES_PORT,
)


def parse_safe_db_info(url: str | None) -> dict[str, Any]:
    """Extract non-sensitive connection info (host, port, dbname) safely."""
    if not url:
        return {"configured": False}
    try:
        parsed = urlparse(url)
        return {
            "configured": True,
            "host": parsed.hostname or "localhost",
            "port": parsed.port or DEFAULT_POSTGRES_PORT,
            "database": parsed.path.lstrip("/"),
        }
    except Exception:
        return {"configured": True, "parse_error": True}


def run_bootstrap(db_url: str, confirm_db_name: str) -> int:
    """Run bootstrap SQL DDL on explicitly confirmed database."""
    info = parse_safe_db_info(db_url)
    actual_db = info.get("database")
    if actual_db != confirm_db_name:
        print(
            f"ERROR: Database name confirmation mismatch! "
            f"Expected '{confirm_db_name}', but URL connects to '{actual_db}'.",
            file=sys.stderr,
        )
        return 1

    sql_path = Path(__file__).with_name("bootstrap_dev_db.sql")
    if not sql_path.exists():
        print(f"ERROR: Bootstrap SQL file not found at {sql_path}", file=sys.stderr)
        return 1

    sql_content = sql_path.read_text(encoding="utf-8")
    host = info.get("host")
    port = info.get("port")
    print(f"Running bootstrap DDL on host={host}:{port}, db={actual_db}...")

    try:
        with psycopg.connect(
            db_url,
            autocommit=True,
            connect_timeout=DATABASE_BOOTSTRAP_CONNECT_TIMEOUT_SECONDS,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database();")
                row = cur.fetchone()
                if row is None or row[0] != confirm_db_name:
                    print(
                        "ERROR: Connected database does not match confirmation.",
                        file=sys.stderr,
                    )
                    return 1
                cur.execute(sql_content)
        print("SUCCESS: Bootstrap DDL applied successfully.")
        return 0
    except psycopg.Error as exc:
        print(f"ERROR: Bootstrap failed: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-db-name", required=True)
    parser.add_argument("--database-url")
    args = parser.parse_args(argv)
    load_local_env(Path(__file__).resolve().parents[2] / ".env")
    database_url = (
        args.database_url
        or os.environ.get("DATABASE_URL")
        or os.environ.get("TEST_DATABASE_URL")
    )
    if not database_url:
        parser.error("DATABASE_URL or --database-url is required")
    return run_bootstrap(database_url, args.confirm_db_name)


if __name__ == "__main__":
    sys.exit(main())
