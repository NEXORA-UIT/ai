import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from src.retrieval.dense import search_by_vector
from src.retrieval.sparse import search_sparse

_BOARD = "11111111-1111-4111-8111-111111111111"


def test_retrieval_imports_do_not_load_document_writer():
    script = (
        "import sys; "
        "import src.retrieval.dense, src.retrieval.sparse; "
        "assert 'src.persistence.document_storage' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "search",
    [
        lambda storage: search_by_vector([1.0] + [0.0] * 1023, _BOARD, storage=storage),
        lambda storage: search_sparse("question", _BOARD, storage=storage),
    ],
    ids=["dense", "sparse"],
)
def test_database_read_closes_connection_on_query_failure(search):
    storage = MagicMock()
    connection = storage.get_connection.return_value
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.execute.side_effect = RuntimeError("database read failed")

    with pytest.raises(RuntimeError, match="database read failed"):
        search(storage)

    connection.close.assert_called_once_with()
