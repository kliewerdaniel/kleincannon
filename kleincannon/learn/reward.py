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


# ---------------------------------------------------------------------------
# P2 — closed-deal (conversion) reward + blended objective
#
# The optimizer's true north-star (decision #7) is a paid deal CLOSED, not
# vanity engagement. We keep the dense engagement reward as a *shaping prior*
# and blend in the sparse, delayed conversion reward. Weights are config
# (learn_config.reward_blend); W_conv grows as attributed-deal count rises so
# the loop naturally shifts from "maximize views" to "maximize deals" once it
# has signal. Fail-closed: with zero attributed deals, W_conv -> 0 and the
# blended reward reduces to pure engagement. No fabricated conversion.
# ---------------------------------------------------------------------------
def reward_conversion(value: float, confidence: float = 1.0) -> float:
    """Normalise an attributed deal's value into reward space.

    Range-scaled against the engagement reference so a single deal doesn't
    dominate the dense engagement history by orders of magnitude. Confidence
    (attribution quality) down-weights an uncertain attribution — a guessed
    source can't corrupt the model.
    """
    ref = learn_config.reward_reference.get("views", 5000.0) * 5.0  # deal ~= 5k views
    if ref <= 0:
        return float(value)
    import math
    norm = math.log1p(max(0.0, value / ref))
    return round(float(norm) * max(0.0, min(1.0, confidence)), 5)


def _attributed_deal_count() -> int:
    try:
        from . import db
        store = db.open_db()
        n = store.conversion_count()
        store.close()
        return n
    except Exception:  # noqa: BLE001 — never block reward on a DB hiccup
        return 0


def blend_weights(attributed_deals: int | None = None) -> tuple[float, float]:
    """Return (W_eng, W_conv) for the current deal-count regime.

    W_eng starts at W_eng_initial and decays toward a floor as attributed deals
    accumulate past `min_attributed_deals`; W_conv grows symmetrically up to
    W_conv_max. Deterministic given the configuration + deal count (reproducible).
    """
    b = learn_config.reward_blend
    w_eng0 = float(b.get("W_eng_initial", 1.0))
    w_conv_max = float(b.get("W_conv_max", 1.0))
    min_deals = max(1, int(b.get("min_attributed_deals", 8)))
    decay = float(b.get("decay", 0.85))
    n = _attributed_deal_count() if attributed_deals is None else int(attributed_deals or 0)
    # progress 0..1 as deals go from 0 -> min_deals (clamped at 1 beyond)
    progress = min(1.0, n / min_deals)
    w_conv = w_conv_max * (1.0 - (decay ** n))  # approaches w_conv_max, never exceeds
    # keep a small engagement floor so the dense prior never vanishes entirely
    w_eng = max(w_eng0 * 0.15, w_eng0 * (1.0 - progress))
    return round(w_eng, 5), round(w_conv, 5)


def blended_reward(metrics: dict[str, float], deal_value: float = 0.0,
                   deal_confidence: float = 1.0, attributed_deals: int | None = None,
                   preset: str | None = None) -> float:
    """The objective the bandit is actually trained on.

    reward = W_eng * reward_engagement(metrics) + W_conv * reward_conversion(value)
    """
    w_eng, w_conv = blend_weights(attributed_deals)
    eng = score(metrics, preset)
    conv = reward_conversion(deal_value, deal_confidence) if deal_value > 0 else 0.0
    return round(w_eng * eng + w_conv * conv, 5)


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
