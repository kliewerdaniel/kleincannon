"""Candidate optimization + evolutionary prompt mutation.

Given a base Episode, this module asks the learning engine for the best next
generation config, then *mutates* that config across the spec's variation
dimensions to produce a population of candidates. The bandit scores and
selects the best to publish; the rest are archived as training data (so every
attempt — even the losers — teaches the model).

Lineage: each candidate records `parent_id` + `variation`, so the subsystem
can trace a winning strategy back through its ancestors (the "intelligence
is the accumulated decisions" idea made literal).

Pure logic — no network, no pipeline mutation.
"""
from __future__ import annotations

import copy
import json
import uuid
from typing import Any

from . import learn_config
from .features import extract
from .engine import get_engine, ContextualBandit
from ..episode import Episode


# Variation functions: each takes a meta dict and returns a mutated copy.
# These are *config-space* mutations (the narrative params), not the engine
# internals. They mirror what a creator can actually change between posts.
def _v_shorter(m: dict) -> dict:
    m = copy.deepcopy(m)
    script = m.get("script", "")
    words = script.split()
    if len(words) > 12:
        m["script"] = " ".join(words[:int(len(words) * 0.7)])
    return m


def _v_longer(m: dict) -> dict:
    m = copy.deepcopy(m)
    script = m.get("script", "")
    # append a curiosity close (no burned CTA)
    m["script"] = (script + " And that changes everything.").strip()
    return m


def _v_more_emotional(m: dict) -> dict:
    m = copy.deepcopy(m)
    # Tag the emotion boost WITHOUT clobbering the real visual_style (the style
    # catalog/learner owns that field). The bandit still sees the emotion intent
    # via this flag, and video_features() keeps the style one-hot intact.
    m["_emotion_boost"] = 1.0
    return m


def _v_more_curiosity(m: dict) -> dict:
    m = copy.deepcopy(m)
    m["hook"] = "You won't believe what " + (m.get("hook", "") or m.get("topic", ""))
    return m


def _v_stronger_hook(m: dict) -> dict:
    m = copy.deepcopy(m)
    m["hook"] = "Stop scrolling. " + (m.get("hook", "") or "")
    return m


def _v_more_urgency(m: dict) -> dict:
    m = copy.deepcopy(m)
    m["caption"] = (m.get("caption", "") + " Do this today.").strip()
    return m


def _v_different_cta(m: dict) -> dict:
    m = copy.deepcopy(m)
    # CTA stays OFF the video; we vary the *post* caption call-to-action instead.
    m["caption"] = (m.get("caption", "") + " Follow for part 2.").strip()
    return m


def _v_different_pacing(m: dict) -> dict:
    m = copy.deepcopy(m)
    # pacing is derived from script length / duration; nudge duration bucket
    try:
        m["video_length"] = round(float(m.get("video_length", 30)) * 1.2, 2)
    except (TypeError, ValueError):
        pass
    return m


VARIATIONS = {
    "shorter": _v_shorter,
    "longer": _v_longer,
    "more_emotional": _v_more_emotional,
    "more_curiosity": _v_more_curiosity,
    "stronger_hook": _v_stronger_hook,
    "more_urgency": _v_more_urgency,
    "different_cta": _v_different_cta,
    "different_pacing": _v_different_pacing,
}


def mutate(meta: dict[str, Any], variation: str) -> dict[str, Any]:
    fn = VARIATIONS.get(variation, _v_different_pacing)
    m = fn(meta)
    m["variation"] = variation
    return m


def base_meta_from_episode(ep: Episode, *,
                           niche: str = "", hashtags=None,
                           caption: str = "", thumbnail_style: str = "",
                           **extra) -> dict[str, Any]:
    from .metadata import capture
    return capture(ep, niche=niche, hashtags=hashtags, caption=caption,
                   thumbnail_style=thumbnail_style, **extra)


def optimize(ep: Episode, *, engine: ContextualBandit | None = None,
             parent_id: str | None = None, niche: str = "",
             hashtags=None, caption: str = "", **extra) -> list[dict[str, Any]]:
    """Produce a population of candidate metas, scored, with the best first.

    The base config is the engine's current best guess for this context; each
    variation spawns one mutated candidate. Every candidate gets a fresh id and
    records its parent + variation for lineage.
    """
    engine = engine or get_engine()
    base = base_meta_from_episode(ep, niche=niche, hashtags=hashtags,
                                  caption=caption, **extra)
    variations = learn_config.mutation_variations or list(VARIATIONS.keys())

    candidates: list[dict[str, Any]] = []
    for v in variations:
        m = mutate(base, v)
        m["_parent_id"] = parent_id
        m["_candidate_id"] = uuid.uuid4().hex[:12]
        mean, std = engine.predict(m)
        m["_predicted_reward"] = round(mean, 5)
        m["_uncertainty"] = round(std, 5)
        candidates.append(m)

    # rank by predicted reward (UCB would re-add alpha*std; here we sort on mean
    # for the final pick, but keep uncertainty visible in the dashboard)
    candidates.sort(key=lambda c: c["_predicted_reward"], reverse=True)
    return candidates


def pick_publish(candidates: list[dict[str, Any]], top_k: int | None = None
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split candidates into (publish, archive) by predicted reward."""
    top_k = top_k if top_k is not None else learn_config.candidate_publish_top_k
    top_k = max(1, min(top_k, len(candidates)))
    publish = candidates[:top_k]
    archive = candidates[top_k:]
    return publish, archive


def summarize_population(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"count": 0, "best": None, "spread": 0.0}
    preds = [c["_predicted_reward"] for c in candidates]
    return {
        "count": len(candidates),
        "best": {
            "variation": candidates[0].get("variation"),
            "predicted_reward": candidates[0]["_predicted_reward"],
            "uncertainty": candidates[0].get("_uncertainty"),
        },
        "worst": {
            "variation": candidates[-1].get("variation"),
            "predicted_reward": candidates[-1]["_predicted_reward"],
        },
        "spread": round(max(preds) - min(preds), 5),
        "mean_pred": round(sum(preds) / len(preds), 5),
    }
