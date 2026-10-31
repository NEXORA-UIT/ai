"""Tokenizer abstractions and wrappers for token-accurate chunking."""

from typing import Protocol

from tokenizers import Tokenizer

from src.ingestion.exceptions import TokenizerLoadError

DEFAULT_BGE_M3_MODEL_ID = "BAAI/bge-m3"
DEFAULT_BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


class TokenCounter(Protocol):
    """Protocol for token counters used in recursive chunking."""

    @property
    def name(self) -> str:
        """Name or identifier of the tokenizer."""
        ...

    @property
    def revision(self) -> str | None:
        """Model commit revision hash, or None for heuristic tokenizers."""
        ...

    def count_tokens(self, text: str, add_special_tokens: bool = False) -> int:
        """Count the number of tokens in the given text."""
        ...

    def encode_ids(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token ID sequence."""
        ...


class WhitespaceTokenCounter:
    """Fast whitespace-based token counter for offline testing and experiments."""

    @property
    def name(self) -> str:
        return "whitespace"

    @property
    def revision(self) -> str | None:
        return None

    def count_tokens(self, text: str, add_special_tokens: bool = False) -> int:
        words = text.split()
        count = len(words)
        return count + 2 if (add_special_tokens and count > 0) else count

    def encode_ids(self, text: str, add_special_tokens: bool = False) -> list[int]:
        words = text.split()
        count = len(words)
        if add_special_tokens and count > 0:
            return list(range(count + 2))
        return list(range(count))


class BgeM3Tokenizer:
    """Wrapper around Hugging Face tokenizers for BAAI/bge-m3.

    Loads only the tokenizer metadata/vocabulary without model weights.
    Explicitly disables padding and truncation to count full documents.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_BGE_M3_MODEL_ID,
        revision: str = DEFAULT_BGE_M3_REVISION,
    ) -> None:
        self.model_id = model_id
        self._revision = revision
        try:
            self._tokenizer = Tokenizer.from_pretrained(model_id, revision=revision)
        except Exception as exc:
            raise TokenizerLoadError(
                f"Failed to load tokenizer '{model_id}' (revision: {revision}): {exc}"
            ) from exc

        # Explicitly ensure tokenizer counts all tokens without truncation or padding
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()

    @property
    def name(self) -> str:
        return self.model_id

    @property
    def revision(self) -> str:
        return self._revision

    def count_tokens(self, text: str, add_special_tokens: bool = False) -> int:
        """Return token count for text. Returns 0 for empty or whitespace-only text."""
        if not text or text.isspace():
            return 0
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return len(encoding.ids)

    def encode_ids(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs list."""
        if not text or text.isspace():
            return []
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return list(encoding.ids)
