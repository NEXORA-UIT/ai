"""Load the repository's simple .env format for local entrypoints."""

import os
from pathlib import Path


def load_local_env(path: Path) -> None:
    """Apply local defaults while leaving the caller's environment in control."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key := key.strip():
            os.environ.setdefault(key, value.strip().strip("'\""))
