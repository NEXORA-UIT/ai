"""Integration tests using the real BAAI/bge-m3 model weights on CPU.

These tests download/load the real model (~2.24 GB) and perform genuine inference.
They are marked with @pytest.mark.real_model and skipped by default unless
the environment variable RUN_REAL_MODEL=1 is set.
"""

import math
import os

import numpy as np
import pytest

from src.embedding.embedder import BgeM3Embedder

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        os.getenv("RUN_REAL_MODEL", "0") != "1",
        reason=(
            "Real BGE-M3 test requires RUN_REAL_MODEL=1 to avoid "
            "loading weights during quick unit runs."
        ),
    ),
]


def _dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate dot product between two vectors (equivalent to cosine similarity)."""
    return float(np.dot(np.array(vec_a), np.array(vec_b)))


@pytest.fixture(scope="module")
def embedder() -> BgeM3Embedder:
    """Initialize embedder with real weights on CPU once for all integration tests."""
    return BgeM3Embedder(device="cpu", batch_size=4)


class TestRealBgeM3Model:
    """Test suite executing real inference on CPU using BAAI/bge-m3."""

    def test_real_embedding_vietnamese_english(self, embedder: BgeM3Embedder) -> None:
        """Validate output vector shape, finiteness, L2 norm, and semantic sanity."""
        texts = [
            # Text 0: Vietnamese project management
            "Nexora là hệ thống cộng tác nhóm và quản lý tiến độ dự án thông minh.",
            # Text 1: Semantically related Vietnamese text (task tracking)
            "Công cụ theo dõi công việc và phân chia nhiệm vụ cho các thành viên.",
            # Text 2: Semantically unrelated Vietnamese text (cooking recipe)
            "Hướng dẫn nấu món phở bò gia truyền với nước dùng đậm đà thơm ngon.",
            # Text 3: English text
            "Artificial intelligence streamlines engineering task tracking.",
        ]

        vectors = embedder.embed_texts(texts, batch_size=2)

        # 1. Output count and dimension check
        assert len(vectors) == 4
        for idx, vec in enumerate(vectors):
            assert len(vec) == 1024, f"Vector {idx} must have 1024 dimensions"
            assert all(isinstance(x, float) for x in vec)

            # 2. Finiteness check
            assert all(not (math.isnan(x) or math.isinf(x)) for x in vec)

            # 3. L2 norm check
            norm = math.sqrt(sum(x * x for x in vec))
            assert abs(norm - 1.0) < 1e-4, (
                f"Vector {idx} norm {norm} is not unit length"
            )

        # 4. Semantic similarity sanity check:
        # sim(Text 0, Text 1 [related]) > sim(Text 0, Text 2 [unrelated cooking])
        sim_related = _dot_product(vectors[0], vectors[1])
        sim_unrelated = _dot_product(vectors[0], vectors[2])
        assert sim_related > sim_unrelated, (
            f"Expected related similarity ({sim_related:.4f}) > "
            f"unrelated similarity ({sim_unrelated:.4f})"
        )

        # Cross-lingual workflow similarity should exceed unrelated cooking text
        sim_cross_lingual = _dot_product(vectors[0], vectors[3])
        assert sim_cross_lingual > sim_unrelated, (
            f"Cross-lingual similarity ({sim_cross_lingual:.4f}) should exceed "
            f"unrelated cooking text similarity ({sim_unrelated:.4f})"
        )

    def test_batching_determinism(self, embedder: BgeM3Embedder) -> None:
        """Inference across batch sizes must produce identical vectors."""
        samples = [
            "Kiểm tra tính nhất quán khi chia batch.",
            "Verify numerical determinism across different batch sizes.",
        ]

        vecs_batch_1 = embedder.embed_texts(samples, batch_size=1)
        vecs_batch_2 = embedder.embed_texts(samples, batch_size=2)

        for v1, v2 in zip(vecs_batch_1, vecs_batch_2, strict=True):
            diff = np.max(np.abs(np.array(v1) - np.array(v2)))
            assert diff < 1e-5, f"Batch size discrepancy detected: max diff = {diff}"
