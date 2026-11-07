"""Typed provider-neutral answer and source citation records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Citation:
    """Public source location for a chunk used by a grounded answer."""

    chunk_id: str
    document_id: str
    file_name: str
    page_number: int | None
    start_offset: int | None
    end_offset: int | None


@dataclass(frozen=True)
class AnswerResult:
    """Answer text with verified citations or a deterministic refusal."""

    text: str
    citations: tuple[Citation, ...]
    refused: bool


@dataclass(frozen=True)
class GroundedPrompt:
    """Trusted instructions and untrusted document context kept separate."""

    system_instructions: str
    question: str
    document_context: str


class AnswerClient(Protocol):
    """Provider-neutral boundary for a structured answer completion."""

    def complete(self, prompt: GroundedPrompt) -> dict[str, object]:
        """Return answer text and cited chunk IDs for the supplied prompt."""
        ...
