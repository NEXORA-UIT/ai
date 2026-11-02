"""Tests for the lazily loaded BGE cross-encoder adapter."""

from __future__ import annotations

from importlib import import_module
from unittest.mock import MagicMock, call

import pytest


def test_bge_reranker_loads_approved_revision_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = import_module("src.reranking.config")
    module = import_module("src.reranking.cross_encoder")
    model = MagicMock()
    model.predict.return_value = [0.75]
    factory = MagicMock(return_value=model)
    monkeypatch.setattr(module, "_create_cross_encoder", factory)
    reranker = module.BgeReranker()

    assert reranker.score_pairs("question", ["passage"]) == [0.75]

    factory.assert_called_once_with(
        model_id=config.DEFAULT_RERANKER_MODEL_ID,
        revision=config.DEFAULT_RERANKER_MODEL_REVISION,
        device="cpu",
    )


def test_bge_reranker_reuses_model_and_preserves_pairs_across_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("src.reranking.cross_encoder")
    model = MagicMock()
    model.predict.side_effect = [[0.1, 0.2], [0.3], [0.4]]
    factory = MagicMock(return_value=model)
    monkeypatch.setattr(module, "_create_cross_encoder", factory)
    reranker = module.BgeReranker(batch_size=2)

    first_scores = reranker.score_pairs("first question", ["a", "b", "c"])
    second_scores = reranker.score_pairs("second question", ["d"])

    assert first_scores == [0.1, 0.2, 0.3]
    assert second_scores == [0.4]
    factory.assert_called_once()
    assert model.predict.call_args_list == [
        call(
            [["first question", "a"], ["first question", "b"]],
            batch_size=2,
            show_progress_bar=False,
        ),
        call(
            [["first question", "c"]],
            batch_size=2,
            show_progress_bar=False,
        ),
        call(
            [["second question", "d"]],
            batch_size=2,
            show_progress_bar=False,
        ),
    ]


def test_empty_passage_list_returns_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("src.reranking.cross_encoder")
    factory = MagicMock()
    monkeypatch.setattr(module, "_create_cross_encoder", factory)
    reranker = module.BgeReranker()

    assert reranker.score_pairs("question", []) == []

    factory.assert_not_called()


def test_cross_encoder_inference_failure_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("src.reranking.cross_encoder")
    model = MagicMock()
    model.predict.side_effect = RuntimeError("backend unavailable")
    monkeypatch.setattr(module, "_create_cross_encoder", MagicMock(return_value=model))
    reranker = module.BgeReranker()

    with pytest.raises(module.RerankerInferenceError, match="scoring failed"):
        reranker.score_pairs("question", ["passage"])


def test_cross_encoder_model_load_failure_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("src.reranking.cross_encoder")
    factory = MagicMock(side_effect=RuntimeError("model unavailable"))
    monkeypatch.setattr(module, "_create_cross_encoder", factory)
    reranker = module.BgeReranker()

    with pytest.raises(module.RerankerLoadError, match="loading failed"):
        reranker.score_pairs("question", ["passage"])


def test_default_reranker_factory_returns_one_lazy_instance() -> None:
    module = import_module("src.reranking.cross_encoder")
    module.get_default_reranker.cache_clear()
    try:
        first = module.get_default_reranker()
        second = module.get_default_reranker()
    finally:
        module.get_default_reranker.cache_clear()

    assert first is second
