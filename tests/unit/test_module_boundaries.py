from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "module_name",
    (
        "nexora_ai.rag",
        "nexora_ai.agents",
    ),
)
def test_ai_feature_modules_have_separate_importable_boundaries(
    module_name: str,
) -> None:
    module = import_module(module_name)

    assert module.__name__ == module_name
