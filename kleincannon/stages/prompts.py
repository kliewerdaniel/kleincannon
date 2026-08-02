"""Stage 4 — image prompts, with a style contract so the video doesn't drift.

General-purpose: writes one image prompt per beat for ANY topic. No product- or
brand-specific language. If the LLM is unavailable or slow, a deterministic local
generator takes over so the pipeline never soft-locks.
"""
from __future__ import annotations

import json
import re

from .. import llm, config
from ..episode import Episode


# Note: STYLE_SUFFIX is kept only as a last-resort fallback if the style catalog
# is ever empty. The active look is chosen per-episode from config.STYLE_CATALOG
# (see _resolve_active_style below) so videos stop all looking the same blue.
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
    r"legible|typography|font|ticket|lottery|lotto|jackpot|raffle|gamble|casino|"
    r"bet|odds|prize|payout|fortune|millionaire|coin|cash|money|banknote|bank|"
    r"bill|dollar|cent|price|cost|pay|paid|tax|taxes|slip|card|scratch|"
    r"percent|million|billion|thousand|hundred|math|equation|score|statistic|"
    r"data|table|second|seconds|minute|minutes|hour|hours|year|years|day|days|"
    r"win|won|winning|chance|imagine|imagining|hoping|hope)\b",
    re.IGNORECASE,
)

# When a beat is about numbers, money, odds, or time-at-scale, literal prompts
# make FLUX.2-klein render garbled text on tickets/coins/screens/banknotes. We
# depict the feeling abstractly instead — always text-free.
_NUMERIC = re.compile(r"\b\d+(?:[.,]\d+)*\b|%|percent")
_DANGER = re.compile(
    r"\b(lottery|lotto|ticket|jackpot|raffle|gamble|casino|bet|odds|chance|"
    r"win|won|winning|prize|payout|fortune|millionaire|coin|cash|money|banknote|"
    r"bank|bill|dollar|cent|price|cost|pay|paid|tax|taxes|slip|scratch|"
    r"million|billion|thousand|hundred|math|equation|score|statistic|data|"
    r"map|maps|atlas|globe|world|country|countries|united states|continent|"
    r"forecast|prediction|"
    r"second|seconds|minute|minutes|hour|hours|year|years|day|days|"
    r"imagine|imagining|hoping|hope|lost|never|impossible|vain)\b",
    re.IGNORECASE,
)

# Text-free, emotion-led abstract scenes (vary by beat index for shot variety).
# These deliberately NEVER echo the beat's literal nouns — the image model
# renders any concrete subject (a map of the US, a coin, a ticket) literally and
# badly, so the no-LLM fallback sticks to atmospheric, topic-neutral cinematic
# shots that fit ANY explainer.
_ABSTRACT = [
    "a vast open sky over a quiet horizon, a lone small figure in silhouette, still and watchful",
    "a hand lifting toward a shaft of shifting light, a quiet moment of realization",
    "moving water over dark rock, caught mid-flow with soft motion blur",
    "a person standing at a window, looking out at an enormous changing sky",
    "a long empty road fading into haze, the overwhelming scale of distance",
    "a close face in low light, a slow breath, a quiet exhale",
    "an empty chair beside a rain-streaked window, the stillness after a decision",
    "two open hands, palms up, a soft quiet understanding",
]


# Styles the deterministic "auto" pick is allowed to choose. Lifestyle / fashion
# looks ("Vivid Editorial", "Soft Pastel") render as model/editorial shots that
# don't fit a reality-check explainer, so they're reserved for an EXPLICIT
# --style choice (or a bandit recommendation that sets ep.style directly) and
# never auto-selected. The canonical set lives in config.SAFE_AUTO_STYLES; the
# helper below reads it so prompts.py and episode.py never disagree.
def _safe_auto_styles():
    return set(getattr(config, "SAFE_AUTO_STYLES", []))


def _fallback_prompts(ep: Episode) -> dict[str, str]:
    """Deterministic prompts used when the LLM is unavailable or too slow.

    The LLM's job is to "show the FEELING of the beat, not a literal
    illustration of its words." With no LLM we cannot do that per-topic, so we
    map every beat to a topic-neutral abstract cinematic scene — never the
    beat's literal nouns, which ZImage Turbo would otherwise render as a map,
    a coin, or a ticket of garbled text. The style suffix (always a
    photographic look in the auto path) supplies the consistent mood.
    """
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
        out[b.id] = f"{angle}: {_ABSTRACT[i % len(_ABSTRACT)]}"
    return out


def _auto_style_name(episode_id: str) -> str:
    """Name of the catalog style a topic id deterministically maps to.

    The deterministic pick is constrained to SAFE_AUTO_STYLES (photographic /
    cinematic looks that fit a reality-check explainer); lifestyle/fashion
    styles like "Vivid Editorial" are never auto-selected — they're reserved for
    an explicit --style choice.
    """
    from ..episode import style_for_id
    safe = _safe_auto_styles()
    base = style_for_id(episode_id).get("name", "Moody Cinematic")
    if base in safe:
        return base
    # Fall back to the first safe style deterministically (stable per topic).
    return next(iter(safe)) if safe else "Moody Cinematic"


def _resolve_active_style(ep: Episode, style_arg: str) -> dict:
    """Pick the concrete {name, palette, suffix} for this episode.

    Resolution order:
      1. style_arg is a literal catalog name  -> that entry.
      2. style_arg is a custom literal string  -> used verbatim as the suffix
         (style_name recorded as "custom" so the bandit can tell it apart).
      3. style_arg == "auto" (or empty)        -> deterministic from ep.id via the
         catalog (stable per topic, varied across topics). A previously-resolved
         REAL catalog name is kept for re-run consistency; a stale "custom" legacy
         suffix is NOT reused (that would freeze the look and mislabel it).
    """
    from ..episode import resolve_style, style_for_id
    catalog_names = {entry["name"] for entry in config.STYLE_CATALOG}
    arg = (style_arg or "").strip()
    if arg and arg.lower() not in ("auto",):
        return resolve_style(arg)
    # auto: keep a real catalog name resolved on a prior run (consistency);
    # unless it's a lifestyle/fashion look (Vivid Editorial, Soft Pastel) that
    # should never be auto-reused — re-derive a safe cinematic look instead.
    if ep.style_name in catalog_names and ep.style_name in _safe_auto_styles():
        for entry in config.STYLE_CATALOG:
            if entry["name"] == ep.style_name:
                return dict(entry)
    return style_for_id(ep.id)


def run(episode_id: str, style_suffix: str = "", llm_timeout: float = 75.0) -> Episode:
    ep = Episode.load(episode_id)

    # ---- Resolve the active visual style ---------------------------------
    # Priority: explicit suffix arg > config.PROMPT_STYLE override > episode.style.
    # The chosen style is stored on the episode (name + suffix) so the look is
    # reproducible AND fed to the learning engine (so we can learn which styles win).
    chosen = _resolve_active_style(ep, style_suffix or config.PROMPT_STYLE)
    ep.style = chosen["name"] if chosen["name"] not in ("auto",) else ep.style
    ep.style_name = chosen["name"] if chosen["name"] != "auto" else _auto_style_name(ep.id)
    ep.style_suffix = chosen["suffix"] or STYLE_SUFFIX
    ep.save()

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

    ep.style_suffix = ep.style_suffix   # already resolved above
    flagged = []
    for b in ep.beats:
        subject = prompts[b.id].strip().rstrip(".")
        leftover = sorted(set(m.lower() for m in _TEXT_TRIGGERS.findall(subject)))
        if leftover:
            for w in leftover:
                subject = re.sub(rf"\b{re.escape(w)}\b", " ", subject, flags=re.IGNORECASE)
            subject = re.sub(r"\s{2,}", " ", subject).strip().rstrip(".")
            flagged.append((b.id, leftover))
        b.image_prompt = f"{subject}. {ep.style_suffix}"
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
