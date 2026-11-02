"""Deterministic latency summaries shared by evaluation runners."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencySummary:
    """Interpolated latency percentiles, in milliseconds."""

    p50_ms: float
    p95_ms: float
    sample_count: int


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] + fraction * (
        sorted_values[upper_index] - sorted_values[lower_index]
    )


def latency_summary(samples_ms: Sequence[float]) -> LatencySummary:
    """Return nearest-rank-interpolated p50/p95 with input validation."""
    if any(isinstance(sample, bool) for sample in samples_ms):
        raise ValueError("latency samples must be finite and nonnegative")
    try:
        samples = [float(sample) for sample in samples_ms]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("latency samples must be numeric") from exc
    if any(not math.isfinite(sample) or sample < 0 for sample in samples):
        raise ValueError("latency samples must be finite and nonnegative")
    if not samples:
        raise ValueError("latency_summary requires at least one sample")
    ordered = sorted(samples)
    return LatencySummary(
        p50_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
        sample_count=len(ordered),
    )
