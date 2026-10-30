import json
from pathlib import Path

from nexora_ai.contracts.schema import SCHEMAS


def test_committed_json_schemas_match_pydantic_models():
    root = Path(__file__).parents[2] / "schemas" / "v1"
    for name, model in SCHEMAS.items():
        path = root / f"{name}.schema.json"
        assert json.loads(path.read_text(encoding="utf-8")) == model.model_json_schema(
            by_alias=True
        )
