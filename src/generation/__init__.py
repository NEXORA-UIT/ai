"""Provider-neutral grounded answer interfaces."""

from src.generation.answer import (
    AnswerProviderError,
    EvidenceScopeError,
    answer_with_sources,
)
from src.generation.provider import DeepSeekAnswerClient, DeepSeekProviderError
from src.generation.schemas import AnswerResult, Citation, GroundedPrompt

__all__ = [
    "AnswerProviderError",
    "AnswerResult",
    "Citation",
    "DeepSeekAnswerClient",
    "DeepSeekProviderError",
    "EvidenceScopeError",
    "GroundedPrompt",
    "answer_with_sources",
]
