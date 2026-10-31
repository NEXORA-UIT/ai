"""Unit tests for TokenCounter protocol and tokenizer wrappers."""

import pytest

from src.ingestion.exceptions import TokenizerLoadError
from src.ingestion.tokenizer import (
    DEFAULT_BGE_M3_MODEL_ID,
    DEFAULT_BGE_M3_REVISION,
    BgeM3Tokenizer,
    TokenCounter,
    WhitespaceTokenCounter,
)


def test_whitespace_token_counter():
    counter: TokenCounter = WhitespaceTokenCounter()
    text = "Hệ thống quản lý Nexora"
    assert counter.count_tokens(text) == 5
    assert counter.name == "whitespace"
    assert counter.revision is None


def test_whitespace_token_counter_empty_text():
    counter = WhitespaceTokenCounter()
    assert counter.count_tokens("") == 0
    assert counter.count_tokens("   \n\t  ") == 0


def test_tokenizer_load_error_on_invalid_model():
    with pytest.raises(TokenizerLoadError, match="Failed to load tokenizer"):
        BgeM3Tokenizer(
            model_id="non_existent/model_repo_xyz_123",
            revision="invalid_revision",
        )


@pytest.mark.integration
def test_bge_m3_tokenizer_token_counting():
    tokenizer = BgeM3Tokenizer()
    assert tokenizer.name == DEFAULT_BGE_M3_MODEL_ID
    assert tokenizer.revision == DEFAULT_BGE_M3_REVISION

    text = "Hệ thống Nexora AI"
    # "Hệ", "thống", "Ne", "xor", "a", "AI" -> 6 content tokens
    count = tokenizer.count_tokens(text)
    assert count == 6

    # Empty text
    assert tokenizer.count_tokens("") == 0


@pytest.mark.integration
def test_bge_m3_tokenizer_with_special_tokens():
    tokenizer = BgeM3Tokenizer()
    text = "Hệ thống"
    count_raw = tokenizer.count_tokens(text, add_special_tokens=False)
    count_special = tokenizer.count_tokens(text, add_special_tokens=True)
    # BGE-M3 prepends <s> and appends </s>
    assert count_special == count_raw + 2


@pytest.mark.integration
def test_bge_m3_tokenizer_no_truncation_on_long_text():
    """Verify BgeM3Tokenizer does not truncate or pad counts on long text."""
    tokenizer = BgeM3Tokenizer()
    # 10,000 repetitions of word -> over 10,000 tokens (exceeds 8192 context limit)
    words = ["từ"] * 10000
    long_text = " ".join(words)
    count = tokenizer.count_tokens(long_text)
    # Must count all words without truncating to 8192 or 512
    assert count >= 10000
