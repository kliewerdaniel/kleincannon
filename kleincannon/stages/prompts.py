"""Stage 4 — image prompts, with a style contract so the video doesn't drift.

General-purpose: writes one image prompt per beat for ANY topic. No product- or
brand-specific language. If the LLM is unavailable or slow, a deterministic local
generator takes over so the pipeline never soft-locks.
"""
from __future__ import annotations

import json
import re

from .. import llm
from ..episode import Episode

STYLE_SUFFIX = (
    "cinematic editorial photograph, vertical 9:16 composition, shot on 35mm, "
    "shallow depth of field, moody directional window light, desaturated teal and amber "
    "palette, film grain, high detail, absolutely no text, no writing, no signage, "
    "no readable screens, no numbers, no logos"
)

NEGATIVE = (
    "text, words, letters, typography, captions, subtitles, writing, numbers, digits, "
    "watermark, logo, signature, username, website URL, brand name, labels, "
    "readable screen, UI text, document text, spreadsheet cells, charts, graphs, "
    "book pages, newspaper, magazine text, handwriting, calligraphy, poster text, "
    "blurry, low quality, low resolution, pixelated, artifacts, "
    "cropped, out of frame, deformed hands, extra fingers, mutated, "
    "cartoon, 3d render, illustration"
)

SYSTEM = """You write image-generation prompts for a vertical short-form video.

Rules:
- One prompt per beat. Describe ONLY the subject, setting, and action — never style, camera, or lighting (those are appended automatically).
- Show the FEELING of the beat, not a literal illustration of its words.
- ABSOLUTELY NO TEXT. Never describe anything that would contain readable words, letters, or numbers: no signs, no screens with visible UI, no documents, spreadsheets, charts, graphs, whiteboards, books, papers, labels, posters, or phone displays showing text. The image model renders text as garbage. If a beat is about documents or data, depict it abstractly — stacks of blank paper, a blurred out-of-focus monitor glow, hands sorting unlabeled folders, an overwhelmed person at a desk — never anything legible.
- Vary composition deliberately across beats: alternate wide establishing, medium, close-up detail, over-the-shoulder. Never two identical framings in a row.
- Keep one consistent human subject across the video when a person appears: describe them the same way each time.
- 25-45 words each. Concrete nouns. No adjective soup.

Output STRICT JSON only: {"b01": "prompt", "b02": "prompt", ...}"""

# Phrases that tend to summon rendered text. Stripped from prompts as a last guard.
_TEXT_TRIGGERS = re.compile(
    r"\b(text|words?|letters?|numbers?|digits?|writing|written|caption|subtitle|"
    r"sign(?:age|s)?|label(?:s|led|ed)?|screen|monitor|display|spreadsheet|chart|"
    r"graph|document|paperwork|invoice|receipt|report|book|newspaper|magazine|"
    r"whiteboard|poster|billboard|logo|brand|dashboard|ui|interface|readable|"
    r"legible|typography|font)\b",
    re.IGNORECASE,
)


def _fallback_prompts(ep: Episode) -> dict[str, str]:
    """Deterministic prompts derived from the beat text — used when the LLM is
    unavailable or too slow. No external call, so the pipeline never soft-locks."""
    shots = [
        "wide establishing shot",
        "medium shot",
        "close-up detail",
        "over-the-shoulder angle",
        "low-angle heroic framing",
        "extreme close-up",
        "wide environmental context",
        "intimate portrait",
    ]
    out = {}
    for i, b in enumerate(ep.beats):
        angle = shots[i % len(shots)]
        subject = _TEXT_TRIGGERS.sub(" ", b.text)
        subject = re.sub(r"\s{2,}", " ", subject).strip().rstrip(".")
        subject = subject or f"a person reacting to the moment, beat {i + 1}"
        out[b.id] = f"{angle}: {subject}"
    return out


def run(episode_id: str, style_suffix: str = STYLE_SUFFIX,
        llm_timeout: float = 75.0) -> Episode:
    ep = Episode.load(episode_id)

    beats_payload = [{"id": b.id, "text": b.text} for b in ep.beats]
    user = (
        f"Video topic: {ep.topic}\n"
        f"Purpose: {ep.purpose}\n\n"
        f"Beats:\n{json.dumps(beats_payload, indent=2)}\n\n"
        f"Return one prompt per beat id. Exactly {len(ep.beats)} keys."
    )

    # Try the LLM, but never block the whole pipeline on a flaky model.
    prompts: dict[str, str] | None = None
    if llm.is_up():
        try:
            prompts = llm.chat_json(SYSTEM, user, temperature=0.85,
                                    timeout=int(llm_timeout))
        except Exception as e:  # noqa: BLE001
            print(f"[prompts] LLM failed ({type(e).__name__}); using local fallback")
            prompts = None

    if not prompts:
        prompts = _fallback_prompts(ep)
        print("[prompts] using deterministic local prompts (no LLM)")

    missing = [b.id for b in ep.beats if not prompts.get(b.id)]
    if missing:
        raise SystemExit(f"model skipped beats: {missing}")

    ep.style_suffix = style_suffix
    flagged = []
    for b in ep.beats:
        subject = prompts[b.id].strip().rstrip(".")
        leftover = sorted(set(m.lower() for m in _TEXT_TRIGGERS.findall(subject)))
        if leftover:
            for w in leftover:
                subject = re.sub(rf"\b{re.escape(w)}\b", " ", subject, flags=re.IGNORECASE)
            subject = re.sub(r"\s{2,}", " ", subject).strip().rstrip(".")
            flagged.append((b.id, leftover))
        b.image_prompt = f"{subject}. {style_suffix}"
    ep.save()

    print(f"[prompts] {len(ep.beats)} prompts written")
    for b in ep.beats:
        print(f"  {b.id}  {str(b.image_prompt)[:110]}…")
    if flagged:
        print("\n[prompts] NOTE — text-triggering words were found and STRIPPED from "
              "these beats' prompts (negative prompt adds a second layer of safety):")
        for bid, words in flagged:
            print(f"    {bid}: {', '.join(words)}")
    return ep
