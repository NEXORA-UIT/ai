"""Compatibility entrypoint. Prefer python -m scripts.ingestion.preview."""

from scripts.ingestion.preview import main

if __name__ == "__main__":
    raise SystemExit(main())
