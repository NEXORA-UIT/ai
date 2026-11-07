"""Pytest root configuration and environment setup."""

from pathlib import Path

from scripts.local_env import load_local_env


def pytest_configure(config) -> None:
    """Load .env file into os.environ if present, before running tests."""
    load_local_env(Path(__file__).parent.parent / ".env")
