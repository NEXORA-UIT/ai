"""Unit tests for BgeM3Embedder using fake/mock test doubles."""

import math
from typing import Any

import numpy as np
import pytest

from src.embedding.embedder import BgeM3Embedder
from src.embedding.exceptions import (
    DeviceUnavailableError,
    InferenceError,
    ModelLoadError,
    TokenBudgetExceededError,
    VectorValidationError,
)
from src.ingestion.schemas import Chunk


class FakeTokenizer:
    """Mock tokenizer providing realistic token counts including special tokens."""

    def __init__(self, max_tokens: int = 8192):
        self.max_tokens = max_tokens

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        padding: bool = False,
    ) -> list[int]:
        words = text.split()
        count = len(words)
        # Adds <s> and </s> when add_special_tokens is True
        total = count + 2 if add_special_tokens else count
        return list(range(total))


class FakeSentenceTransformer:
    """Configurable test double simulating SentenceTransformer behavior."""

    def __init__(
        self,
        dimension: int = 1024,
        max_seq_length: int = 8192,
        raise_on_encode: bool = False,
        corrupt_dimension: int | None = None,
        return_nan: bool = False,
        return_inf: bool = False,
        return_all_zeros: bool = False,
        unnormalized: bool = False,
        wrong_count: int | None = None,
        custom_raw_output: Any = None,
    ):
        self.dimension = dimension
        self.max_seq_length = max_seq_length
        self.tokenizer: Any = FakeTokenizer(max_seq_length)
        self.raise_on_encode = raise_on_encode
        self.corrupt_dimension = corrupt_dimension
        self.return_nan = return_nan
        self.return_inf = return_inf
        self.return_all_zeros = return_all_zeros
        self.unnormalized = unnormalized
        self.wrong_count = wrong_count
        self.custom_raw_output = custom_raw_output
        self.encode_calls: list[list[str]] = []

    def encode(
        self,
        texts: list[str],
        batch_size: int = 8,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> Any:
        self.encode_calls.append(list(texts))
        if self.raise_on_encode:
            raise RuntimeError("Simulated forward pass failure")

        if self.custom_raw_output is not None:
            return self.custom_raw_output

        count = self.wrong_count if self.wrong_count is not None else len(texts)
        vectors = []
        for _i, text in enumerate(texts[:count]):
            dim = self.corrupt_dimension if self.corrupt_dimension else self.dimension
            if self.return_nan:
                vec = np.full(dim, np.nan, dtype=np.float32)
            elif self.return_inf:
                vec = np.full(dim, np.inf, dtype=np.float32)
            elif self.return_all_zeros:
                vec = np.zeros(dim, dtype=np.float32)
            else:
                seed = hash(text) & 0xFFFFFFFF
                rng = np.random.default_rng(seed)
                raw = rng.standard_normal(dim).astype(np.float32)
                if self.unnormalized:
                    vec = raw * 5.0
                else:
                    norm = np.linalg.norm(raw)
                    vec = raw / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def test_embed_texts_empty_list_does_not_load_model():
    """embed_texts([]) must return [] immediately without loading the model."""
    embedder = BgeM3Embedder(device="cpu")
    # Verify underlying model reference is initially None
    assert embedder._model is None
    res = embedder.embed_texts([])
    assert res == []
    # Verify model was never loaded
    assert embedder._model is None


def test_embed_texts_empty_or_whitespace_string_raises():
    """Empty or whitespace-only strings must raise ValueError with index."""
    fake_model = FakeSentenceTransformer()
    embedder = BgeM3Embedder(model=fake_model)

    with pytest.raises(ValueError, match="Input text at index 1 is empty"):
        embedder.embed_texts(["Valid text", "   ", "Another valid text"])

    with pytest.raises(ValueError, match="Input text at index 0 is empty"):
        embedder.embed_texts([""])


def test_embed_texts_invalid_type_raises():
    """Non-list input or non-string element must raise TypeError."""
    fake_model = FakeSentenceTransformer()
    embedder = BgeM3Embedder(model=fake_model)

    with pytest.raises(TypeError, match="texts must be a list"):
        embedder.embed_texts("not a list")  # type: ignore

    with pytest.raises(TypeError, match="Input text at index 1 must be a string"):
        embedder.embed_texts(["valid", 12345])  # type: ignore


def test_invalid_batch_size_raises():
    """batch_size < 1 must raise ValueError."""
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        BgeM3Embedder(batch_size=0)

    fake_model = FakeSentenceTransformer()
    embedder = BgeM3Embedder(model=fake_model, batch_size=8)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        embedder.embed_texts(["hello"], batch_size=-2)


def test_batching_exact_order_and_partial_last_batch():
    """Inputs must be batched according to batch_size and preserve exact order."""
    fake_model = FakeSentenceTransformer()
    embedder = BgeM3Embedder(model=fake_model, batch_size=2)

    texts = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    vectors = embedder.embed_texts(texts, batch_size=2)

    assert len(vectors) == 5
    # Verify batches: [2, 2, 1]
    assert len(fake_model.encode_calls) == 3
    assert fake_model.encode_calls[0] == ["doc1", "doc2"]
    assert fake_model.encode_calls[1] == ["doc3", "doc4"]
    assert fake_model.encode_calls[2] == ["doc5"]

    # Check each vector has length 1024 and norm approx 1.0
    for vec in vectors:
        assert len(vec) == 1024
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-4


def test_token_budget_exceeded_before_inference():
    """Text exceeding max_seq_length must be rejected before calling model.encode."""
    fake_model = FakeSentenceTransformer(max_seq_length=10)
    embedder = BgeM3Embedder(model=fake_model, max_seq_length=10)

    # 15 words exceeds 10 tokens limit
    long_text = "word " * 15
    with pytest.raises(TokenBudgetExceededError, match="exceeds maximum token budget"):
        embedder.embed_texts(["short text", long_text])

    # Ensure model.encode was never called due to early rejection
    assert len(fake_model.encode_calls) == 0


def test_output_validation_wrong_vector_count():
    """Batch returning mismatched vector count must raise VectorValidationError."""
    fake_model = FakeSentenceTransformer(wrong_count=1)
    embedder = BgeM3Embedder(model=fake_model, batch_size=2)

    with pytest.raises(
        VectorValidationError, match=r"Inference output must have shape \(2, 1024\)"
    ):
        embedder.embed_texts(["doc1", "doc2"])


def test_output_validation_wrong_dimension():
    """Vectors not having 1024 dimensions must raise VectorValidationError."""
    fake_model = FakeSentenceTransformer(corrupt_dimension=512)
    embedder = BgeM3Embedder(model=fake_model)

    with pytest.raises(
        VectorValidationError, match=r"Inference output must have shape \(1, 1024\)"
    ):
        embedder.embed_texts(["doc1"])


def test_output_validation_nan():
    """Vectors containing NaN values must raise VectorValidationError."""
    fake_model = FakeSentenceTransformer(return_nan=True)
    embedder = BgeM3Embedder(model=fake_model)

    with pytest.raises(VectorValidationError, match="contains NaN or Infinity"):
        embedder.embed_texts(["doc1"])


def test_output_validation_inf():
    """Vectors containing Infinity values must raise VectorValidationError."""
    fake_model = FakeSentenceTransformer(return_inf=True)
    embedder = BgeM3Embedder(model=fake_model)

    with pytest.raises(VectorValidationError, match="contains NaN or Infinity"):
        embedder.embed_texts(["doc1"])


def test_output_validation_all_zeros():
    """All-zero vectors must raise VectorValidationError."""
    fake_model = FakeSentenceTransformer(return_all_zeros=True)
    embedder = BgeM3Embedder(model=fake_model)

    with pytest.raises(VectorValidationError, match="all-zero vector"):
        embedder.embed_texts(["doc1"])


def test_output_validation_unnormalized():
    """Vectors with L2 norm not equal to 1.0 must raise VectorValidationError."""
    fake_model = FakeSentenceTransformer(unnormalized=True)
    embedder = BgeM3Embedder(model=fake_model)

    with pytest.raises(VectorValidationError, match="invalid L2 norm"):
        embedder.embed_texts(["doc1"])


def test_inference_error_wrapping():
    """Inference failures must be wrapped into InferenceError preserving cause."""
    fake_model = FakeSentenceTransformer(raise_on_encode=True)
    embedder = BgeM3Embedder(model=fake_model)

    with pytest.raises(
        InferenceError, match="Inference failed during batch encoding"
    ) as exc_info:
        embedder.embed_texts(["doc1"])
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_cuda_device_unavailable_raises():
    """Explicitly requesting CUDA when unavailable must raise DeviceUnavailableError."""
    # Since torch.cuda.is_available() is False on CPU torch, this must raise immediately
    with pytest.raises(DeviceUnavailableError, match="CUDA was explicitly requested"):
        BgeM3Embedder(device="cuda")


def test_unsupported_device_raises():
    """Specifying an unknown device must raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported device: 'tpu'"):
        BgeM3Embedder(device="tpu")


def test_embed_chunks_success():
    """embed_chunks properly pairs Chunk instances with dense vectors."""
    fake_model = FakeSentenceTransformer()
    embedder = BgeM3Embedder(model=fake_model)

    chunks = [
        Chunk(
            chunk_index=0,
            content="First chunk content",
            token_count=3,
            start_offset=0,
            end_offset=19,
            page_number=1,
            metadata={"source": "test"},
        ),
        Chunk(
            chunk_index=1,
            content="Second chunk content",
            token_count=3,
            start_offset=20,
            end_offset=40,
            page_number=2,
            metadata={"source": "test"},
        ),
    ]

    embedded = embedder.embed_chunks(chunks)
    assert len(embedded) == 2
    assert embedded[0].chunk == chunks[0]
    assert embedded[1].chunk == chunks[1]
    assert len(embedded[0].dense_embedding) == 1024
    assert len(embedded[1].dense_embedding) == 1024
    assert embedded[0].chunk.page_number == 1
    assert embedded[1].chunk.page_number == 2


def test_embed_chunks_empty_returns_empty():
    """embed_chunks([]) returns [] without errors."""
    embedder = BgeM3Embedder(device="cpu")
    assert embedder.embed_chunks([]) == []


def test_embed_chunks_invalid_element_type():
    """Passing non-Chunk elements must raise TypeError."""
    fake_model = FakeSentenceTransformer()
    embedder = BgeM3Embedder(model=fake_model)
    with pytest.raises(TypeError, match="is not a Chunk"):
        embedder.embed_chunks(["not a chunk"])  # type: ignore


# ---------------------------------------------------------------------------
# Regression tests: Token budget effective limit & Model validation
# ---------------------------------------------------------------------------


def test_effective_limit_configured_lower_than_model():
    """When configured max_seq_length < model max_seq_length, use configured."""
    fake_model = FakeSentenceTransformer(max_seq_length=100)
    # Configured is 10, model has 100 -> effective is 10
    embedder = BgeM3Embedder(model=fake_model, max_seq_length=10)

    # 15 words -> 17 tokens > 10
    long_text = "word " * 15
    with pytest.raises(TokenBudgetExceededError, match="exceeds maximum token budget"):
        embedder.embed_texts([long_text])

    # Model inference must not be called when rejected
    assert len(fake_model.encode_calls) == 0


def test_effective_limit_model_lower_than_configured():
    """When model max_seq_length < configured max_seq_length, use model limit."""
    fake_model = FakeSentenceTransformer(max_seq_length=10)
    # Configured is 100, model has 10 -> effective is 10
    embedder = BgeM3Embedder(model=fake_model, max_seq_length=100)

    long_text = "word " * 15
    with pytest.raises(TokenBudgetExceededError, match="exceeds maximum token budget"):
        embedder.embed_texts([long_text])

    assert len(fake_model.encode_calls) == 0


def test_missing_tokenizer_raises_model_load_error():
    """When model has no valid tokenizer, raise ModelLoadError."""
    fake_model = FakeSentenceTransformer()
    fake_model.tokenizer = None
    embedder = BgeM3Embedder(model=fake_model)

    with pytest.raises(ModelLoadError, match="does not provide a valid tokenizer"):
        embedder.embed_texts(["Valid text"])


@pytest.mark.parametrize("invalid_limit", [None, "8192", 0, -5])
def test_invalid_model_max_seq_length_raises_model_load_error(invalid_limit):
    """When model has an invalid max_seq_length, raise ModelLoadError."""
    fake_model = FakeSentenceTransformer()
    fake_model.max_seq_length = invalid_limit
    embedder = BgeM3Embedder(model=fake_model)

    with pytest.raises(ModelLoadError, match="valid integer max_seq_length"):
        embedder.embed_texts(["Valid text"])


# ---------------------------------------------------------------------------
# Regression tests: Strict output shape & format validation
# ---------------------------------------------------------------------------


def test_output_validation_scalar_raises():
    """Scalar output from model must raise VectorValidationError."""
    fake_model = FakeSentenceTransformer(custom_raw_output=42.0)
    embedder = BgeM3Embedder(model=fake_model)

    with pytest.raises(VectorValidationError, match="must have shape"):
        embedder.embed_texts(["doc1"])


def test_output_validation_1d_raises():
    """1D output from model must raise VectorValidationError."""
    fake_model = FakeSentenceTransformer(custom_raw_output=np.zeros(1024))
    embedder = BgeM3Embedder(model=fake_model)

    with pytest.raises(VectorValidationError, match="must have shape"):
        embedder.embed_texts(["doc1"])


def test_output_validation_3d_raises():
    """3D output from model must raise VectorValidationError."""
    fake_model = FakeSentenceTransformer(custom_raw_output=np.zeros((1, 1, 1024)))
    embedder = BgeM3Embedder(model=fake_model)

    with pytest.raises(VectorValidationError, match="must have shape"):
        embedder.embed_texts(["doc1"])


def test_output_validation_non_numeric_raises():
    """Non-numeric output from model must raise VectorValidationError."""
    fake_model = FakeSentenceTransformer(custom_raw_output=[["non_numeric_str"] * 1024])
    embedder = BgeM3Embedder(model=fake_model)

    with pytest.raises(
        VectorValidationError, match="cannot be converted to numeric array"
    ):
        embedder.embed_texts(["doc1"])
