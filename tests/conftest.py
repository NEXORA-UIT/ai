"""Pytest root configuration and environment setup."""

import os
from pathlib import Path


def pytest_configure(config) -> None:
    """Load .env file into os.environ if present, before running tests."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v
