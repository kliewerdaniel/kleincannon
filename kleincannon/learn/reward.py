"""Reward engine — a configurable, weighted score over platform metrics.

No single hardcoded objective: the active preset (and its weights) come from
learn.json. Metrics are normalised against a reference value so that, e.g.,
views (thousands) and followers_gained (tens) contribute on a comparable scale.
The latest metric snapshot for an experience is what gets scored; the DB keeps
the full history, so this function can be re-applied to any point in time.

The interface (`score(metrics, preset=None)`) is stable; swapping in a learned
reward (e.g. a fitted value function) later means implementing the same call.
"""
from __future__ import annotations

from typing import Any

from . import learn_config


# Canonical metric keys the engine knows about. Platforms return a superset;
# unknown keys are ignored. Missing keys score as 0.
KNOWN_METRICS = [
    "views", "likes", "comments", "shares", "saves", "followers_gained",
    "watch_time", "completion_rate", "profile_visits",
]


def _normalise(metric: str, value: float) -> float:
    ref = learn_config.reward_reference.get(metric, 1.0)
    if ref <= 0:
        return float(value)
    # Soft saturating normalisation: x/ref clipped, then log-compressed so a
    # viral outlier doesn't dominate the whole history.
    import math
    raw = value / ref
    return float(math.log1p(max(0.0, raw)))


def score(metrics: dict[str, float], preset: str | None = None) -> float:
    """Weighted, normalised reward for one metric snapshot.

    Returns a non-negative scalar. Higher = better per the active preset.
    """
    weights = learn_config.reward_presets.get(
        preset or learn_config.active_reward_preset,
        learn_config.reward_presets["balanced_growth"],
    )
    total_w = sum(weights.values()) or 1.0
    acc = 0.0
    for metric, w in weights.items():
        val = float(metrics.get(metric, 0.0) or 0.0)
        acc += (w / total_w) * _normalise(metric, val)
    return round(acc, 5)


def score_series(snapshots: list[dict[str, float]], preset: str | None = None) -> list[float]:
    return [score(s, preset) for s in snapshots]


def best_preset_for(metric_focus: str) -> str:
    """Convenience: pick the preset whose top weight is `metric_focus`."""
    for name, weights in learn_config.reward_presets.items():
        if metric_focus in weights and weights[metric_focus] >= max(weights.values()):
            return name
    return learn_config.active_reward_preset


def explain(metrics: dict[str, float], preset: str | None = None) -> dict[str, Any]:
    """Breakdown of how the reward was composed (for the dashboard)."""
    weights = learn_config.reward_presets.get(
        preset or learn_config.active_reward_preset,
        learn_config.reward_presets["balanced_growth"],
    )
    total_w = sum(weights.values()) or 1.0
    parts = {}
    for metric, w in weights.items():
        val = float(metrics.get(metric, 0.0) or 0.0)
        parts[metric] = {
            "raw": val,
            "weight": round(w / total_w, 4),
            "normalised": round(_normalise(metric, val), 4),
            "contribution": round((w / total_w) * _normalise(metric, val), 4),
        }
    return {"preset": preset or learn_config.active_reward_preset,
            "total": score(metrics, preset), "parts": parts}
