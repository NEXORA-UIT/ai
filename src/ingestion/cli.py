"""Compatibility entrypoint for the previous ingestion and bootstrap commands."""

import sys

from scripts.ingestion.cli import main as ingest_main


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--bootstrap-db" in args:
        from scripts.db.bootstrap import main as bootstrap_main

        return bootstrap_main([arg for arg in args if arg != "--bootstrap-db"])
    if "-h" in args or "--help" in args:
        print("Legacy --bootstrap-db moved to python -m scripts.db.bootstrap.")
    return ingest_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
