"""Metadata capture — turn an Episode + upload context into a flat dict of
every generation parameter the spec calls out. Nothing is discarded; the
experience DB stores this verbatim so future models can learn from any field.

This module only READS the existing Episode manifest; it never mutates the
generation pipeline.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..episode import Episode
from .. import config
from . import learn_config


# Canonical list of captured fields (drives both extraction and the UI schema).
CAPTURE_FIELDS = [
    "topic", "niche", "purpose", "script", "title", "hook", "cta",
    "narration_model", "voice", "music", "visual_style", "image_prompts",
    "transition_style", "subtitle_style", "video_length", "posting_time",
    "posting_day", "hashtags", "caption", "thumbnail_style",
    "generation_model", "generation_temperature", "random_seed",
]


def capture(episode: Episode, *,
            niche: str = "", hashtags: list[str] | None = None,
            caption: str = "", thumbnail_style: str = "",
            music: str = "", title: str = "",
            **extra: Any) -> dict[str, Any]:
    """Build the full generation-parameter dict for an experience.

    `episode` carries most fields. The rest come from the upload decision /
    runtime overrides (e.g. seed, temperature may have been applied via
    config.push_overrides but aren't on the manifest — we pull them from
    learn_config / config when present).
    """
    beats = episode.beats
    script_text = episode.full_script
    image_prompts = {b.id: (b.image_prompt or "") for b in beats}

    # Pull any runtime overrides that were applied (seed/temperature/model)
    # from config module attributes if they exist.
    gen_model = getattr(config, "GENERATION_MODEL", "flux2-klein")
    gen_temp = getattr(config, "GENERATION_TEMPERATURE", 1.0)
    gen_seed = getattr(config, "GENERATION_SEED", None)
    transition = getattr(config, "TRANSITION_STYLE", "kenburns")
    subtitle_style = getattr(config, "SUBTITLE_STYLE", "karaoke")

    meta: dict[str, Any] = {
        "topic": episode.topic,
        "niche": niche or episode.purpose,
        "purpose": episode.purpose,
        "script": script_text,
        "title": title or episode.topic,
        "hook": episode.hook or (beats[0].text if beats else ""),
        "cta": episode.cta,
        "narration_model": "qwen3-tts-12hz-1.7b",
        "voice": episode.voice,
        "music": music or "none",
        "visual_style": episode.style_suffix[:80] if episode.style_suffix else "cinematic",
        "style_name": episode.style_name or "",
        "speed": round(float(episode.speed or 1.0), 4),
        "image_prompts": image_prompts,
        "transition_style": transition,
        "subtitle_style": subtitle_style,
        "video_length": round(episode.total_duration, 2),
        "posting_time": None,          # filled by uploader at post time
        "posting_day": None,           # filled by uploader at post time
        "hashtags": hashtags or list(learn_config.default_hashtags),
        "caption": caption or learn_config.default_caption,
        "thumbnail_style": thumbnail_style or "first_frame",
        "generation_model": gen_model,
        "generation_temperature": gen_temp,
        "random_seed": gen_seed,
    }
    meta.update(extra)
    # Deterministic content fingerprint (lineage / dedup).
    meta["_fingerprint"] = hashlib.sha1(
        json.dumps({k: meta[k] for k in ("topic", "script", "voice", "visual_style", "style_name")},
                   sort_keys=True, default=str).encode()).hexdigest()[:12]
    return meta


def from_episode_id(episode_id: str, **kw: Any) -> dict[str, Any]:
    ep = Episode.load(episode_id)
    return capture(ep, **kw)


def embed_posting_time(meta: dict[str, Any], posted_at: float) -> dict[str, Any]:
    """Fill posting_time / posting_day once the video is actually posted."""
    dt = datetime.fromtimestamp(posted_at)
    meta["posting_time"] = dt.strftime("%H:%M")
    meta["posting_day"] = dt.strftime("%A")
    return meta
