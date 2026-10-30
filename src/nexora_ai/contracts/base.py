"""Wire-format rules for Express ↔ Python messages."""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict
from pydantic.alias_generators import to_camel


class WireModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=False,
        extra="forbid",
    )


def _require_integer(value: object) -> object:
    if type(value) is not int:
        raise ValueError("must be an integer")
    return value


SchemaVersion = Annotated[Literal[1], BeforeValidator(_require_integer)]
