"""Stage 1 — the script. General-purpose: any topic, any purpose.

Two paths:
  * manual: you paste one sentence per beat (no LLM, instant, deterministic).
  * ai:     gemma4 writes a spoken monologue + splits it into beats.

No brand- or product-specific logic — this is a generic video generator.
"""
from __future__ import annotations

import json

from .. import config, llm
from ..episode import Episode, Beat

SYSTEM = """You write scripts for short-form vertical videos (TikTok / Reels / Shorts).

Given a TOPIC and an optional PURPOSE (what the video is for), write a tight,
spoken-word monologue and split it into beats. The voiceover is read aloud by a
TTS engine, so:
- Write the way people actually talk. Short sentences. Punchy. No #hashtags,
  no em-dashes, no lists, no "firstly/secondly".
- Each beat is ONE idea and lands on its own. 4-7 beats total, ~110-160 words.
- Open with a hook (a surprising claim or a feeling the viewer recognizes).
- End with the purpose fulfilled — a clear takeaway or a soft call to look closer.
- Never put visible text in the video; the script is audio-only.

Output STRICT JSON only:
{
  "hook": "one-line hook",
  "beats": ["sentence 1", "sentence 2", ...]
}"""


def from_text(topic: str, lines: list[str], purpose: str = "",
              voice: str | None = None) -> Episode:
    ep = Episode.new(topic, purpose=purpose)
    if voice:
        ep.voice = voice
    beats = [Beat(id=f"b{i + 1:02d}", text=t.strip())
             for i, t in enumerate(lines) if t.strip()]
    if not beats:
        raise SystemExit("no script lines provided")
    ep.beats = beats
    ep.save()
    print(f"[script] manual script: {len(beats)} beats -> {ep.id}")
    return ep


def from_ai(topic: str, purpose: str = "", beats: int = 6, voice: str | None = None
            ) -> Episode:
    llm.ensure_server()
    user = (f"TOPIC: {topic}\n"
            f"PURPOSE: {purpose}\n"
            f"BEATS: {beats}\n\n"
            f"Write a {beats}-beat script.")
    data = llm.chat_json(SYSTEM, user, temperature=0.9)
    raw_beats = data.get("beats") or []
    texts = [str(b).strip() for b in raw_beats if str(b).strip()]
    if not texts:
        raise SystemExit("model returned no beats")
    ep = from_text(topic, texts, purpose=purpose, voice=voice)
    ep.hook = (data.get("hook") or "").strip()
    ep.save()
    print(f"[script] ai script: {len(texts)} beats, hook set")
    return ep


def run(episode_id: str, manual_script: str | None = None, purpose: str = "",
        beats: int = 6, voice: str | None = None, use_manual: bool = True
        ) -> Episode:
    """CLI/web entry. If manual_script is given, use it; else try AI if LLM up."""
    if manual_script and manual_script.strip():
        lines = [ln.strip() for ln in manual_script.strip().splitlines() if ln.strip()]
        # detect comma-separated single line as fallback
        if len(lines) <= 1 and "," in (manual_script or ""):
            lines = [s.strip() for s in manual_script.split(",") if s.strip()]
        return from_text(topic_from_id(episode_id), lines, purpose=purpose, voice=voice)
    if llm.is_up():
        return from_ai(topic_from_id(episode_id), purpose=purpose, beats=beats, voice=voice)
    raise SystemExit("no manual script and LLM is unavailable — paste a script")


def topic_from_id(episode_id: str) -> str:
    # Derive a readable topic from an existing episode (used by web/manual paths).
    try:
        ep = Episode.load(episode_id)
        return ep.topic
    except Exception:
        return episode_id
