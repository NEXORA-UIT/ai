import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("module", "expected_flag"),
    [
        ("src.ingestion.cli", "--bootstrap-db"),
        ("src.ingestion.preview", "--export-vectors"),
        ("evaluation.run_retrieval_cli", "--expected-database-name"),
    ],
)
def test_public_cli_help_starts_without_external_services(module, expected_flag):
    result = subprocess.run(
        [sys.executable, "-B", "-m", module, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert expected_flag in result.stdout
