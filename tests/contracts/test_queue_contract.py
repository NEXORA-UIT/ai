import json
from pathlib import Path

from nexora_ai.contracts.queue import QUEUE_PROTOCOL


def test_committed_queue_protocol_matches_python_source_of_truth():
    path = Path(__file__).parents[2] / "schemas" / "v1" / "queue-contracts.json"
    assert json.loads(path.read_text(encoding="utf-8")) == QUEUE_PROTOCOL
