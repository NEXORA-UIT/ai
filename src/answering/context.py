"""Build bounded prompts with document data isolated from system policy."""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from src.answering.config import DEFAULT_CONTEXT_CHAR_LIMIT
from src.answering.schemas import GroundedPrompt
from src.retrieval.schemas import RerankedChunk


@lru_cache(maxsize=1)
def load_system_instructions() -> str:
    """Load the reviewed grounded-answer instructions from the repository."""
    prompt_path = Path(__file__).with_name("system_prompt.md")
    return prompt_path.read_text(encoding="utf-8").strip()


def _serialize_context(chunks: Sequence[RerankedChunk]) -> str:
    return json.dumps(
        [
            {"chunk_id": chunk.source.chunk_id, "content": chunk.source.content}
            for chunk in chunks
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_grounded_prompt(
    question: str,
    chunks: Sequence[RerankedChunk],
    *,
    max_context_chars: int = DEFAULT_CONTEXT_CHAR_LIMIT,
) -> tuple[GroundedPrompt, tuple[RerankedChunk, ...]]:
    """Keep whole chunks in ranked order while enforcing a character budget."""
    if isinstance(max_context_chars, bool) or not isinstance(max_context_chars, int):
        raise ValueError("max_context_chars must be a positive integer")
    if max_context_chars < 1:
        raise ValueError("max_context_chars must be a positive integer")

    included: list[RerankedChunk] = []
    for chunk in chunks:
        proposed = [*included, chunk]
        if len(_serialize_context(proposed)) > max_context_chars:
            break
        included.append(chunk)

    document_context = _serialize_context(included) if included else ""
    prompt = GroundedPrompt(
        system_instructions=load_system_instructions(),
        question=question,
        document_context=document_context,
    )
    return prompt, tuple(included)
