"""Export payload JSON Schemas and the shared BullMQ protocol manifest."""

import json
from pathlib import Path

from nexora_ai.contracts.queue import QUEUE_PROTOCOL
from nexora_ai.contracts.schema import SCHEMAS


def main() -> None:
    output = Path(__file__).parents[1] / "schemas" / "v1"
    output.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMAS.items():
        schema = model.model_json_schema(by_alias=True)
        (output / f"{name}.schema.json").write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (output / "queue-contracts.json").write_text(
        json.dumps(QUEUE_PROTOCOL, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
